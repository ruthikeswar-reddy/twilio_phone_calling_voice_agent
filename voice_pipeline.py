"""
VoicePipeline — Fixed with 5 targeted patches:

  FIX 1 — BARGE-IN ROOT CAUSE:
    Removed per-media-packet barge-in check in _receive_twilio_loop.
    Old code called _handle_barge_in() on EVERY incoming Twilio packet
    while _is_speaking=True, including pure silence packets. Since Twilio
    streams continuously, this fired barge-in within ~8ms of TTS starting.
    New: barge-in is only triggered by Deepgram's VAD "speech_start" event,
    meaning real speech energy detected on the line.

  FIX 2 — BARGE-IN GRACE PERIOD:
    Even with VAD-based barge-in, added a 500ms grace window after TTS
    audio starts. The phone network/Twilio can echo back the agent's own
    audio briefly. If VAD fires within 500ms of TTS start, it's suppressed.
    Configurable via BARGE_IN_GRACE_MS.

  FIX 3 — SPECULATIVE DEDUPLICATION:
    Interim results fire on every STT word chunk. Without dedup, the same
    phrase (e.g. "What can you do for me?") triggered 3 LLM calls in logs.
    Fixed: track _last_speculative_text; skip if normalized text matches last.
    Also track _last_speculative_reset_on_final to clear the guard on each
    final transcript so the next utterance always works.

  FIX 4 — SPECULATIVE vs FINAL CONFLICT RESOLUTION:
    When speculative agent is mid-flight and final transcript arrives:
      - similarity >= 0.80 → transcripts match, let speculative complete
      - similarity <  0.80 → transcripts diverge, cancel & restart with final
    This prevents two separate responses playing for one utterance.
    Uses difflib.SequenceMatcher for lightweight text comparison.

  FIX 5 — TRACK _is_speculative_active FLAG:
    _launch_agent() now records whether it was launched speculatively.
    Once the first LLM token arrives we clear the flag — at that point
    we've committed to the response and won't let a divergent final cancel it.
"""

import asyncio
import base64
import difflib
import json
import logging
import re
import time
from typing import AsyncIterator, Optional

from fastapi import WebSocket

from stt.deepgram_stt import DeepgramSTT
from tts.cartesia_tts import CartesiaTTS
from agent.langgraph_agent import LangGraphAgent
from audio_utils import ulaw_to_pcm, pcm_to_ulaw

logger = logging.getLogger(__name__)

# ── Tunable constants ────────────────────────────────────────────────────────

MIN_TRANSCRIPT_WORDS = 3
INTERIM_CONFIDENCE_THRESHOLD = 0.85

# FIX 2: How long after TTS first-audio before barge-in is allowed (ms).
# Prevents echo / network delay from instantly cancelling the agent.
BARGE_IN_GRACE_MS = 500

# FIX 4: Minimum similarity ratio (0-1) between speculative and final
# transcript for the speculative response to be considered "correct enough"
# and allowed to continue without restarting.
SPECULATIVE_SIMILARITY_THRESHOLD = 0.80


# ── Utility ──────────────────────────────────────────────────────────────────

def _text_similarity(a: str, b: str) -> float:
    """
    Normalised SequenceMatcher ratio between two strings (case-insensitive).
    Returns 0.0 (totally different) … 1.0 (identical).
    """
    return difflib.SequenceMatcher(
        None,
        a.strip().lower(),
        b.strip().lower(),
    ).ratio()


# ── Main class ───────────────────────────────────────────────────────────────

class VoicePipeline:
    def __init__(self, websocket: WebSocket):
        self.ws = websocket
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None

        self.stt = DeepgramSTT()
        self.tts = CartesiaTTS()
        self.agent = LangGraphAgent()

        self._agent_task: Optional[asyncio.Task] = None
        self._is_speaking = False
        self._should_stop = False
        self._turn_start_time: float = 0

        # FIX 2: timestamp of when the first TTS audio byte was actually sent
        # to Twilio, used to enforce the grace period.
        self._tts_audio_start_time: float = 0.0

        self._tts_text_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._audio_output_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

        # FIX 3: speculative dedup state
        # Stores the normalised (lower-stripped) text of the last speculative
        # launch so we can skip identical interim events.
        self._last_speculative_text: str = ""

        # FIX 4 & 5: speculative conflict-resolution state
        self._current_agent_text: str = ""       # transcript the running agent was given
        self._is_speculative_active: bool = False # True only until first LLM token arrives

        # Token-by-token TTS buffering for low-latency streaming
        self._tts_token_buffer = ""
        self._tts_buffer_timer: Optional[asyncio.TimerHandle] = None
        self._tts_min_buffer_chars = 100
        self._tts_buffer_timeout = 0.05
        self._tts_buffer_start_time: float = 0

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

                # ── FIX 1 + FIX 2: VAD-based barge-in with grace period ───
                if event["type"] == "speech_start":
                    self._turn_start_time = time.monotonic()

                    if self._is_speaking:
                        elapsed_tts_ms = (
                            time.monotonic() - self._tts_audio_start_time
                        ) * 1000

                        if elapsed_tts_ms >= BARGE_IN_GRACE_MS:
                            # Real speech detected after grace period — valid barge-in
                            logger.info(
                                f"BARGE-IN detected (VAD speech_start, "
                                f"TTS had been playing {elapsed_tts_ms:.0f}ms)"
                            )
                            await self._handle_barge_in()
                        else:
                            # Too soon after TTS started — likely echo / network artefact
                            logger.debug(
                                f"Barge-in suppressed — grace period active "
                                f"(TTS only {elapsed_tts_ms:.0f}ms old, "
                                f"threshold={BARGE_IN_GRACE_MS}ms)"
                            )

                # ── FIX 3: Speculative dedup ──────────────────────────────
                elif event["type"] == "interim":
                    transcript = event["transcript"]
                    confidence = event.get("confidence", 0)
                    words = len(transcript.split())
                    logger.debug(f"Interim [{confidence:.2f}]: {transcript}")

                    normalised = transcript.strip().lower()
                    is_duplicate = (normalised == self._last_speculative_text)

                    if (
                        words >= MIN_TRANSCRIPT_WORDS
                        and confidence >= INTERIM_CONFIDENCE_THRESHOLD
                        and not self._is_speaking
                        and (self._agent_task is None or self._agent_task.done())
                        and not is_duplicate   # FIX 3: skip identical re-trigger
                    ):
                        logger.info(f"Speculative start on interim: '{transcript}'")
                        self._last_speculative_text = normalised
                        self._turn_start_time = time.monotonic()
                        self._launch_agent(transcript, speculative=True)

                # ── FIX 4: Speculative vs Final conflict resolution ────────
                elif event["type"] == "final":
                    transcript = event["transcript"].strip()
                    if not transcript:
                        continue

                    logger.info(f"Final transcript: '{transcript}'")
                    elapsed = (time.monotonic() - self._turn_start_time) * 1000
                    logger.info(f"STT->final latency: {elapsed:.0f}ms")

                    # Reset dedup guard so the NEXT utterance starts fresh
                    self._last_speculative_text = ""

                    agent_running = (
                        self._agent_task and not self._agent_task.done()
                    )

                    if agent_running and self._is_speculative_active:
                        # A speculative agent is mid-flight. Check if the final
                        # transcript is close enough to let it continue.
                        similarity = _text_similarity(
                            self._current_agent_text, transcript
                        )
                        logger.info(
                            f"Speculative similarity check: {similarity:.2f} "
                            f"('{self._current_agent_text}' vs '{transcript}')"
                        )

                        if similarity >= SPECULATIVE_SIMILARITY_THRESHOLD:
                            # Close enough — speculative was right, no restart
                            logger.info(
                                f"Speculative matched final "
                                f"(similarity={similarity:.2f}) — continuing"
                            )
                            # Mark as no longer speculative so it won't be
                            # cancelled by a second final (edge case)
                            self._is_speculative_active = False
                        else:
                            # Transcript diverged significantly — cancel and
                            # relaunch with the correct final text
                            logger.info(
                                f"Speculative diverged from final "
                                f"(similarity={similarity:.2f}) — restarting"
                            )
                            self._turn_start_time = time.monotonic()
                            self._launch_agent(transcript, speculative=False)

                    elif agent_running and not self._is_speculative_active:
                        # A non-speculative (final) agent is already running —
                        # this final is a duplicate, ignore it.
                        logger.debug(
                            "Final agent already running — duplicate final ignored"
                        )

                    else:
                        # No agent running at all — start fresh with final
                        self._turn_start_time = time.monotonic()
                        self._launch_agent(transcript, speculative=False)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"STT event handler error: {e}")

    # ── Agent launcher & runner ──────────────────────────────────────────────

    def _launch_agent(self, transcript: str, speculative: bool = False):
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
        self._drain_queue(self._tts_text_queue)
        self._drain_queue(self._audio_output_queue)

        # FIX 4 & 5: record what text this agent is working on and whether
        # it is speculative, so the final-transcript handler can compare.
        self._current_agent_text = transcript
        self._is_speculative_active = speculative

        self._agent_task = asyncio.create_task(
            self._run_agent(transcript, speculative)
        )

    async def _run_agent(self, transcript: str, speculative: bool):
        try:
            logger.info(
                f"Agent starting ({'speculative' if speculative else 'final'})"
            )
            agent_start = time.monotonic()
            first_token = True

            async for text_chunk in self.agent.stream(transcript):
                if first_token:
                    logger.info(
                        f"First LLM token: "
                        f"{(time.monotonic()-agent_start)*1000:.0f}ms"
                    )
                    first_token = False
                    self._is_speculative_active = False
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
                                    f"'{buffer_to_send[:60]}'"
                                )

                                if not tts_started:
                                    logger.info(
                                        f"TTS starting: '{buffer_to_send[:60]}'"
                                    )
                                    tts_started = True

                                async for audio_chunk in self.tts.synthesize_stream(
                                    buffer_to_send
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
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
        await self.ws.send_json(
            {"event": "clear", "streamSid": self.stream_sid}
        )
        self._is_speaking = False
        self._is_speculative_active = False
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
    def _drain_queue(q: asyncio.Queue):
        while not q.empty():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def cleanup(self):
        self._cancel_buffer_timer()
        if self._agent_task:
            self._agent_task.cancel()
        self._should_stop = True
        logger.info("Pipeline cleaned up")

