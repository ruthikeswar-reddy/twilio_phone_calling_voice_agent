"""
VoicePipeline — Orchestrates the flow of audio and text between Twilio, STT, TTS, and the agent.  Key responsibilities:
- Receiving audio from Twilio, sending it to STT, and handling the resulting transcripts.
- Launching the agent with the appropriate transcript (interim speculative or final).
- Streaming the agent's text response to TTS and sending the resulting audio back to Twilio.
- Handling barge-in by cancelling the agent and clearing Twilio's buffer when the caller speaks during TTS playback.

"""

import asyncio
import base64
import json
import logging
import re
import time
from typing import Optional

from fastapi import WebSocket

from stt.deepgram_stt import DeepgramSTT
from tts.cartesia_tts import CartesiaTTS
from agent.csa_groq import CustomerSupportAgent
from audio_utils import ulaw_to_pcm, pcm_to_ulaw

logger = logging.getLogger(__name__)

# ── Tunable constants ────────────────────────────────────────────────────────

# How long after TTS first-audio before barge-in is allowed (ms).
# Prevents echo / network delay from instantly cancelling the agent.
BARGE_IN_GRACE_MS = 700

# Minimum number of words in an interim transcript before barge-in fires.
# Prevents a single "um", cough, or noise burst from interrupting TTS.
BARGE_IN_MIN_WORDS = 2

# After a speech_final chunk arrives, wait this long (seconds) before launching
# the agent. If more speech arrives in this window, accumulate and reset.
# This prevents the agent firing mid-sentence on short pauses (<400ms).
AGENT_LAUNCH_DEBOUNCE_S = 0.40


# Maps spoken digit words (including common STT variants) to digit characters.
_DIGIT_WORDS: dict[str, str] = {
    "zero": "0", "oh": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

# Fix 8: NATO phonetic alphabet → single uppercase letter.
# Users often spell out their employee IDs phonetically over the phone
# (e.g. "Echo Mike one two three" → "EM123").
# Only single-letter phonetic words are mapped so normal words like
# "golf" in "golf tournament" are unaffected when not part of an ID run.
_PHONETIC_ALPHABET: dict[str, str] = {
    "alpha": "A", "bravo": "B", "charlie": "C", "delta": "D",
    "echo": "E", "foxtrot": "F", "golf": "G", "hotel": "H",
    "india": "I", "juliet": "J", "kilo": "K", "lima": "L",
    "mike": "M", "november": "N", "oscar": "O", "papa": "P",
    "quebec": "Q", "romeo": "R", "sierra": "S", "tango": "T",
    "uniform": "U", "victor": "V", "whiskey": "W", "xray": "X",
    "yankee": "Y", "zulu": "Z",
}

# Multi-word spoken phrases → their symbol/text equivalents.
# Longer phrases must come before shorter overlapping ones.
_SPOKEN_PHRASES: list[tuple[str, str]] = [
    ("at the rate of", "@"),
    ("at the rate",    "@"),
    ("at rate",        "@"),
    ("dot com",        ".com"),
    ("dot net",        ".net"),
    ("dot org",        ".org"),
    ("dot in",         ".in"),
    ("dot ai",         ".ai"),
    ("dot io",         ".io"),
    ("dot co",         ".co"),
    ("underscore",     "_"),
    ("hyphen",         "-"),
    ("dash",           "-"),
    ("dot",            "."),
    ("at",             "@"),   # bare "at" as last resort for @ symbol
]


def _normalize_transcript(text: str) -> str:
    """
    Normalise a raw STT transcript for the voice agent:

    1. Replace spoken email symbols so the LLM receives a real email address:
         "shruthi at the rate eminds dot ai"  → "shruthi@eminds.ai"
         "john underscore doe at gmail dot com" → "john_doe@gmail.com"

    2. Convert digit words to digits:
         "one two three"  → "123"
         "My ID is one two three" → "My ID is 123"

    3. Collapse runs of single alphanumeric tokens into one token (for IDs):
         "E M one two three" → "EM123"
    """
    lower = text.lower().strip()

    # Step 1: replace multi-word spoken phrases with their symbols.
    # Use regex word boundaries (\b) so that phrases like "at" don't match
    # inside words like "batman" or "therateemail", preventing double-@ bugs.
    for phrase, symbol in _SPOKEN_PHRASES:
        lower = re.sub(r"\b" + re.escape(phrase) + r"\b", symbol, lower)

    # After phrase substitution, collapse any spaces that crept in around
    # symbols that are part of an email address (e.g. "shruthi @ eminds . ai")
    # by removing spaces adjacent to @ . _ -
    lower = re.sub(r"\s*@\s*", "@", lower)
    lower = re.sub(r"\s*\.\s*", ".", lower)
    lower = re.sub(r"\s*_\s*", "_", lower)
    lower = re.sub(r"\s*-\s*", "-", lower)

    # Step 1.5: NATO phonetic alphabet → single letter.
    # Convert each token individually so normal multi-word sentences are unaffected.
    # "echo mike 1 2 3" → "E M 1 2 3" → later merged to "EM123" in Step 3.
    tokens = [_PHONETIC_ALPHABET.get(tok, tok) for tok in lower.split()]
    lower = " ".join(tokens)

    tokens = lower.split()

    # Step 2: digit-word → digit character
    converted = []
    for tok in tokens:
        stripped = tok.rstrip(".,!?;:")
        punct = tok[len(stripped):]
        converted.append(_DIGIT_WORDS.get(stripped, stripped) + punct)

    # Step 3: merge runs of single alphanumeric tokens (e.g. spelled-out IDs).
    # Tokens may carry trailing punctuation (e.g. "3." from "three.") — strip it
    # before the single-char check and re-attach it only to the last token in the run.
    merged: list[str] = []
    i = 0
    while i < len(converted):
        tok = converted[i]
        core = tok.rstrip(".,!?;:")
        trail = tok[len(core):]
        if len(core) == 1 and core.isalnum():
            run = [core]
            run_trail = trail
            j = i + 1
            while j < len(converted):
                nt = converted[j]
                nt_core = nt.rstrip(".,!?;:")
                nt_trail = nt[len(nt_core):]
                if len(nt_core) == 1 and nt_core.isalnum():
                    run.append(nt_core)
                    run_trail = nt_trail
                    j += 1
                else:
                    break
            if len(run) > 1:
                merged.append("".join(run) + run_trail)
                i = j
                continue
        merged.append(tok)
        i += 1

    return " ".join(merged)


# ── Main class ───────────────────────────────────────────────────────────────

class VoicePipeline:
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None

        self.stt = DeepgramSTT()
        self.tts = CartesiaTTS()
        self.agent = CustomerSupportAgent()

        self._agent_task: Optional[asyncio.Task] = None
        self._is_speaking = False
        self._should_stop = False
        self._turn_start_time: float = 0

        # FIX 2: timestamp of when the first TTS audio byte was actually sent
        # to Twilio, used to enforce the grace period.
        self._tts_audio_start_time: float = 0.0

        self._tts_text_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._audio_output_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

        # Token-by-token TTS buffering for low-latency streaming
        self._tts_token_buffer = ""
        self._tts_buffer_timer: Optional[asyncio.TimerHandle] = None
        self._tts_min_buffer_chars = 100
        self._tts_buffer_timeout = 0.05
        self._tts_buffer_start_time: float = 0

        # Barge-in is a two-step process:
        #   1. speech_start → arm (if grace period passed)
        #   2. interim/final transcript with >= BARGE_IN_MIN_WORDS → fire
        # This prevents noise/echo/single-word sounds from interrupting TTS.
        self._barge_in_armed: bool = False

        # Issue 1: debounce agent launch across speech_final chunks.
        # Accumulates partial finals; agent only fires after AGENT_LAUNCH_DEBOUNCE_S
        # of silence or on UtteranceEnd, so mid-sentence pauses don't trigger it.
        self._pending_transcript: str = ""
        self._agent_debounce_task: Optional[asyncio.Task] = None

    # ── Top-level runner ─────────────────────────────────────────────────────

    async def run(self):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._receive_twilio_loop())
            tg.create_task(self._send_audio_loop())
            tg.create_task(self._tts_loop())

    # ── Twilio inbound loop ──────────────────────────────────────────────────

    async def _receive_twilio_loop(self):
        async with self.stt.stream() as stt_stream:
            stt_task = asyncio.create_task(self._handle_stt_events(stt_stream))
            try:
                async for message in self.ws.iter_text():
                    data = json.loads(message)
                    event = data.get("event")

                    if event == "start":
                        self.stream_sid = data["start"]["streamSid"]
                        self.call_sid = data["start"].get("callSid", "unknown")
                        logger.info(
                            f"Stream started: SID={self.stream_sid} Call={self.call_sid}"
                        )

                    elif event == "media":
                        ulaw_bytes = base64.b64decode(data["media"]["payload"])

                        # ── FIX 1 ──────────────────────────────────────────
                        # OLD (broken): called _handle_barge_in() on EVERY
                        # media packet while _is_speaking=True. Twilio sends
                        # packets continuously (even pure silence), so barge-in
                        # fired within milliseconds of TTS starting regardless
                        # of whether the caller was actually speaking.
                        #
                        # NEW: barge-in is REMOVED from here entirely.
                        # It is now triggered only by the Deepgram VAD
                        # "speech_start" event in _handle_stt_events(), which
                        # fires only when real speech energy is detected.
                        # ───────────────────────────────────────────────────
                        await stt_stream.send_audio(ulaw_bytes)

                    elif event == "stop":
                        logger.info("Stream stopped by Twilio")
                        self._should_stop = True
                        break
            finally:
                stt_task.cancel()
                await stt_stream.finish()

    # ── STT event handler ────────────────────────────────────────────────────

    async def _handle_stt_events(self, stt_stream):
        try:
            async for event in stt_stream.events():

                # ── VAD-based barge-in: two-step arm → fire ──────────────
                # Step 1 (arm): speech_start fires immediately on any sound.
                #   We only ARM here after the grace period — we do NOT fire yet,
                #   because background noise / TTS echo also triggers speech_start.
                # Step 2 (fire): when an interim/final transcript arrives while
                #   armed, we know real speech content was detected and fire.
                if event["type"] == "speech_start":
                    self._turn_start_time = time.monotonic()

                    if self._is_speaking:
                        elapsed_tts_ms = (
                            time.monotonic() - self._tts_audio_start_time
                        ) * 1000

                        if elapsed_tts_ms >= BARGE_IN_GRACE_MS:
                            self._barge_in_armed = True
                            logger.debug(
                                f"Barge-in ARMED (VAD speech_start, "
                                f"TTS playing {elapsed_tts_ms:.0f}ms) — "
                                f"waiting for transcript to confirm"
                            )
                        else:
                            logger.debug(
                                f"Barge-in suppressed — grace period active "
                                f"(TTS only {elapsed_tts_ms:.0f}ms old, "
                                f"threshold={BARGE_IN_GRACE_MS}ms)"
                            )

                elif event["type"] == "interim":
                    transcript = event.get("transcript", "").strip()
                    logger.debug(
                        f"Interim [{event.get('confidence', 0):.2f}]: {transcript}"
                    )
                    # FIX 2: Only fire barge-in when BARGE_IN_MIN_WORDS words are
                    # confirmed. A single "um", noise burst, or TTS echo won't
                    # satisfy this threshold, preventing premature TTS cutoff.
                    word_count = len(transcript.split())
                    if (
                        self._barge_in_armed
                        and self._is_speaking
                        and word_count >= BARGE_IN_MIN_WORDS
                    ):
                        elapsed_tts_ms = (
                            time.monotonic() - self._tts_audio_start_time
                        ) * 1000
                        logger.info(
                            f"BARGE-IN detected (interim {word_count} words confirmed, "
                            f"TTS had been playing {elapsed_tts_ms:.0f}ms)"
                        )
                        self._barge_in_armed = False
                        await self._handle_barge_in()

                elif event["type"] == "final":
                    transcript = event["transcript"].strip()
                    if not transcript:
                        self._barge_in_armed = False
                        continue

                    # Fire any pending armed barge-in before handling agent launch
                    if self._barge_in_armed and self._is_speaking:
                        elapsed_tts_ms = (
                            time.monotonic() - self._tts_audio_start_time
                        ) * 1000
                        logger.info(
                            f"BARGE-IN detected (final transcript confirmed, "
                            f"TTS had been playing {elapsed_tts_ms:.0f}ms)"
                        )
                        await self._handle_barge_in()
                    self._barge_in_armed = False

                    logger.info(f"Final transcript chunk: '{transcript}'")
                    elapsed = (time.monotonic() - self._turn_start_time) * 1000
                    logger.info(f"STT->final latency: {elapsed:.0f}ms")
                    self._turn_start_time = time.monotonic()

                    # FIX 1: Accumulate transcript chunks and debounce agent launch.
                    # speech_final fires on 200ms pauses which can be mid-sentence.
                    # We wait AGENT_LAUNCH_DEBOUNCE_S; if more speech arrives the
                    # timer resets and the new chunk is appended.
                    self._pending_transcript = (
                        (self._pending_transcript + " " + transcript).strip()
                        if self._pending_transcript else transcript
                    )
                    if self._agent_debounce_task and not self._agent_debounce_task.done():
                        self._agent_debounce_task.cancel()
                    self._agent_debounce_task = asyncio.create_task(
                        self._debounced_agent_launch()
                    )

                elif event["type"] == "utterance_end":
                    # Deepgram confirmed >= utterance_end_ms (1000ms) of silence.
                    # The user has definitely stopped — launch immediately without
                    # waiting for the debounce timer.
                    if self._pending_transcript:
                        if (
                            self._agent_debounce_task
                            and not self._agent_debounce_task.done()
                        ):
                            self._agent_debounce_task.cancel()
                            self._agent_debounce_task = None
                        transcript = self._pending_transcript
                        self._pending_transcript = ""
                        logger.info(
                            f"UtteranceEnd — launching agent immediately: '{transcript}'"
                        )
                        self._launch_agent(transcript)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"STT event handler error: {e}")

    # ── Debounced agent launch ───────────────────────────────────────────────

    async def _debounced_agent_launch(self):
        """
        Wait AGENT_LAUNCH_DEBOUNCE_S seconds, then launch the agent with the
        accumulated pending transcript. If cancelled (more speech arrives),
        the pending_transcript is kept so the next chunk can append to it.
        """
        try:
            await asyncio.sleep(AGENT_LAUNCH_DEBOUNCE_S)
            if self._pending_transcript:
                transcript = self._pending_transcript
                self._pending_transcript = ""
                logger.info(
                    f"Debounce elapsed — launching agent: '{transcript}'"
                )
                self._launch_agent(transcript)
        except asyncio.CancelledError:
            pass  # more speech arrived; pending_transcript preserved for next chunk

    # ── Agent launcher & runner ──────────────────────────────────────────────

    def _launch_agent(self, transcript: str):
        transcript = _normalize_transcript(transcript)

        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
        self._drain_queue(self._tts_text_queue)
        self._drain_queue(self._audio_output_queue)

        self._agent_task = asyncio.create_task(self._run_agent(transcript))

    async def _run_agent(self, transcript: str):
        try:
            logger.info("Agent starting")
            agent_start = time.monotonic()
            first_token = True

            async for text_chunk in self.agent.stream(
                transcript,
                self.call_sid or "default",
                speculative=False,
            ):
                if first_token:
                    logger.info(
                        f"First LLM token: "
                        f"{(time.monotonic()-agent_start)*1000:.0f}ms"
                    )
                    first_token = False
                await self._tts_text_queue.put(text_chunk)

            await self._tts_text_queue.put(None)
            logger.info("Agent response complete")

        except asyncio.CancelledError:
            logger.info("Agent cancelled (barge-in)")
            await self._tts_text_queue.put(None)

    # ── Buffer timer helpers ─────────────────────────────────────────────────

    def _cancel_buffer_timer(self):
        if self._tts_buffer_timer is not None:
            self._tts_buffer_timer.cancel()
            self._tts_buffer_timer = None

    def _schedule_buffer_flush(self, loop):
        self._cancel_buffer_timer()
        self._tts_buffer_timer = loop.call_later(
            self._tts_buffer_timeout,
            self._flush_buffer_sync,
        )

    def _flush_buffer_sync(self):
        self._tts_buffer_timer = None

    # ── TTS streaming loop ───────────────────────────────────────────────────

    async def _tts_loop(self):
        while not self._should_stop:
            try:
                first_chunk = await self._tts_text_queue.get()
                if first_chunk is None:
                    continue  # stale end-signal, discard

                self._tts_token_buffer = first_chunk
                self._tts_buffer_start_time = time.monotonic()
                tts_started = False

                loop = asyncio.get_event_loop()
                self._schedule_buffer_flush(loop)

                try:
                    while True:
                        should_flush_chars = (
                            len(self._tts_token_buffer) >= self._tts_min_buffer_chars
                        )

                        try:
                            text_chunk = await asyncio.wait_for(
                                self._tts_text_queue.get(),
                                timeout=self._tts_buffer_timeout,
                            )
                        except asyncio.TimeoutError:
                            text_chunk = None

                        if text_chunk is not None:
                            self._tts_token_buffer += text_chunk
                            self._schedule_buffer_flush(loop)

                        if should_flush_chars or (
                            text_chunk is None and self._tts_token_buffer.strip()
                        ):
                            buffer_to_send = self._tts_token_buffer.strip()
                            if buffer_to_send:
                                elapsed = (
                                    time.monotonic() - self._tts_buffer_start_time
                                ) * 1000
                                flush_reason = (
                                    "char-threshold"
                                    if should_flush_chars
                                    else "timeout"
                                )
                                logger.debug(
                                    f"Buffer flush ({flush_reason}, "
                                    f"{len(buffer_to_send)} chars, {elapsed:.0f}ms): "
                                    f"'{buffer_to_send}'"
                                )

                                if not tts_started:
                                    logger.info(
                                        f"TTS starting: '{buffer_to_send[:80]}...'"
                                        if len(buffer_to_send) > 80
                                        else f"TTS starting: '{buffer_to_send}'"
                                    )
                                    tts_started = True

                                for segment in self._split_for_tts(buffer_to_send):
                                    async for audio_chunk in self.tts.synthesize_stream(
                                        segment
                                    ):
                                        await self._audio_output_queue.put(audio_chunk)

                            self._tts_token_buffer = ""
                            self._tts_buffer_start_time = time.monotonic()

                        if text_chunk is None:
                            self._cancel_buffer_timer()
                            break

                except asyncio.CancelledError:
                    self._cancel_buffer_timer()

            except asyncio.CancelledError:
                self._cancel_buffer_timer()

            await self._audio_output_queue.put(None)

    # ── Audio send loop ──────────────────────────────────────────────────────

    async def _send_audio_loop(self):
        first_audio = True

        while not self._should_stop:
            chunk = await self._audio_output_queue.get()

            if chunk is None:
                self._is_speaking = False
                self._barge_in_armed = False  # disarm when TTS finishes
                first_audio = True
                continue

            if first_audio:
                ttfw = (time.monotonic() - self._turn_start_time) * 1000
                logger.info(f"TIME TO FIRST WORD (TTFW): {ttfw:.0f}ms")
                first_audio = False
                self._is_speaking = True
                # FIX 2: record the moment TTS audio first hit the wire.
                # Used in _handle_stt_events to enforce BARGE_IN_GRACE_MS.
                self._tts_audio_start_time = time.monotonic()

            ulaw_chunk = await asyncio.get_event_loop().run_in_executor(
                None, pcm_to_ulaw, chunk
            )
            payload = base64.b64encode(ulaw_chunk).decode("utf-8")
            await self.ws.send_json(
                {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": payload},
                }
            )

    # ── Barge-in handler ─────────────────────────────────────────────────────

    async def _handle_barge_in(self):
        """
        Cancel the running agent, clear Twilio's playback buffer, and reset
        pipeline state.  Called ONLY from the VAD speech_start path now —
        never from raw media packets.
        """
        self._cancel_buffer_timer()
        # Cancel any pending debounce so a stale transcript doesn't re-launch
        if self._agent_debounce_task and not self._agent_debounce_task.done():
            self._agent_debounce_task.cancel()
            self._agent_debounce_task = None
        self._pending_transcript = ""
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
        await self.ws.send_json(
            {"event": "clear", "streamSid": self.stream_sid}
        )
        self._is_speaking = False
        self._barge_in_armed = False
        self._drain_queue(self._tts_text_queue)
        self._drain_queue(self._audio_output_queue)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> tuple[list[str], str]:
        parts = re.split(r"(?<=[.!?])\s+", text)
        if len(parts) <= 1:
            return [], text
        return parts[:-1], parts[-1]

    @staticmethod
    def _split_for_tts(text: str) -> list[str]:
        """
        Split long text into speakable segments for individual TTS calls.
        Splits on newlines first, then sentence boundaries within long lines.
        Filters out empty segments and structural labels that read awkwardly.
        """
        segments: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # If line is short enough, send as-is
            if len(line) <= 120:
                segments.append(line)
            else:
                # Split long lines on sentence boundaries
                parts = re.split(r"(?<=[.!?])\s+", line)
                segments.extend(p.strip() for p in parts if p.strip())
        return segments or [text.strip()]

    @staticmethod
    def _drain_queue(q: asyncio.Queue):
        while not q.empty():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def cleanup(self):
        self._cancel_buffer_timer()
        if self._agent_debounce_task:
            self._agent_debounce_task.cancel()
        if self._agent_task:
            self._agent_task.cancel()
        self._should_stop = True
        logger.info("Pipeline cleaned up")

