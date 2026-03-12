"""
VoicePipeline: orchestrates the full STT -> Agent -> TTS loop.

Fixes vs original:
  - _tts_loop: rewrote to correctly restart per-turn (original had unreachable code)
  - _send_audio_loop: moved first_audio reset inside loop (was unreachable after while exits)
  - Added barge-in stability (drain queues before cancel)
"""

import asyncio
import base64
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

MIN_TRANSCRIPT_WORDS = 3
INTERIM_CONFIDENCE_THRESHOLD = 0.85


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

        self._tts_text_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self._audio_output_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

        # Token-by-token TTS buffering for low-latency streaming
        self._tts_token_buffer = ""
        self._tts_buffer_timer: Optional[asyncio.TimerHandle] = None
        self._tts_min_buffer_chars = 100      # Flush when hitting 100 chars
        self._tts_buffer_timeout = 0.05        # Flush after 50ms idle (0.05 seconds)
        self._tts_buffer_start_time: float = 0  # Track when buffer started accumulating

    async def run(self):
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._receive_twilio_loop())
            tg.create_task(self._send_audio_loop())
            tg.create_task(self._tts_loop())

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
                        logger.info(f"Stream started: SID={self.stream_sid} Call={self.call_sid}")

                    elif event == "media":
                        ulaw_bytes = base64.b64decode(data["media"]["payload"])
                        if self._is_speaking:
                            await self._handle_barge_in()
                        await stt_stream.send_audio(ulaw_bytes)

                    elif event == "stop":
                        logger.info("Stream stopped by Twilio")
                        self._should_stop = True
                        break
            finally:
                stt_task.cancel()
                await stt_stream.finish()

    async def _handle_stt_events(self, stt_stream):
        try:
            async for event in stt_stream.events():
                if event["type"] == "interim":
                    transcript = event["transcript"]
                    confidence = event.get("confidence", 0)
                    words = len(transcript.split())
                    logger.debug(f"Interim [{confidence:.2f}]: {transcript}")

                    if (
                        words >= MIN_TRANSCRIPT_WORDS
                        and confidence >= INTERIM_CONFIDENCE_THRESHOLD
                        and not self._is_speaking
                        and (self._agent_task is None or self._agent_task.done())
                    ):
                        logger.info(f"Speculative start on interim: '{transcript}'")
                        self._turn_start_time = time.monotonic()
                        self._launch_agent(transcript, speculative=True)

                elif event["type"] == "final":
                    transcript = event["transcript"].strip()
                    if not transcript:
                        continue
                    logger.info(f"Final transcript: '{transcript}'")
                    elapsed = (time.monotonic() - self._turn_start_time) * 1000
                    logger.info(f"STT->final latency: {elapsed:.0f}ms")

                    if not (self._agent_task and not self._agent_task.done()):
                        self._turn_start_time = time.monotonic()
                        self._launch_agent(transcript, speculative=False)

                elif event["type"] == "speech_start":
                    self._turn_start_time = time.monotonic()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"STT event handler error: {e}")

    def _launch_agent(self, transcript: str, speculative: bool = False):
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
        self._drain_queue(self._tts_text_queue)
        self._drain_queue(self._audio_output_queue)
        self._agent_task = asyncio.create_task(self._run_agent(transcript, speculative))

    async def _run_agent(self, transcript: str, speculative: bool):
        try:
            logger.info(f"Agent starting ({'speculative' if speculative else 'final'})")
            agent_start = time.monotonic()
            first_token = True

            async for text_chunk in self.agent.stream(transcript):
                if first_token:
                    logger.info(f"First LLM token: {(time.monotonic()-agent_start)*1000:.0f}ms")
                    first_token = False
                await self._tts_text_queue.put(text_chunk)

            await self._tts_text_queue.put(None)
            logger.info("Agent response complete")

        except asyncio.CancelledError:
            logger.info("Agent cancelled (barge-in)")
            await self._tts_text_queue.put(None)

    def _cancel_buffer_timer(self):
        """Cancel any pending buffer timer."""
        if self._tts_buffer_timer is not None:
            self._tts_buffer_timer.cancel()
            self._tts_buffer_timer = None

    def _schedule_buffer_flush(self, loop):
        """Schedule buffer flush after timeout if no new tokens arrive."""
        self._cancel_buffer_timer()
        self._tts_buffer_timer = loop.call_later(
            self._tts_buffer_timeout,
            self._flush_buffer_sync
        )

    def _flush_buffer_sync(self):
        """Synchronous callback to flush buffer (called from timer)."""
        self._tts_buffer_timer = None
        # Signal that buffer should flush on next await point
        # This is set to None to signal the async code

    # Token-by-token TTS loop for minimum latency
    async def _tts_loop(self):
        while not self._should_stop:
            try:
                # Wait for first token of a new turn
                first_chunk = await self._tts_text_queue.get()
                if first_chunk is None:
                    continue  # stale end-signal, discard

                # Reset buffer and timer for new turn
                self._tts_token_buffer = first_chunk
                self._tts_buffer_start_time = time.monotonic()
                tts_started = False

                loop = asyncio.get_event_loop()
                self._schedule_buffer_flush(loop)

                try:
                    while True:
                        # Check if we should flush based on buffer size
                        should_flush_chars = len(self._tts_token_buffer) >= self._tts_min_buffer_chars

                        # Try to get next token with timeout to trigger flush
                        try:
                            text_chunk = await asyncio.wait_for(
                                self._tts_text_queue.get(), timeout=self._tts_buffer_timeout
                            )
                        except asyncio.TimeoutError:
                            text_chunk = None  # Timeout = end of token stream or pause

                        # Handle new token
                        if text_chunk is not None:
                            self._tts_token_buffer += text_chunk
                            # Reschedule timer to give more time for batching
                            self._schedule_buffer_flush(loop)

                        # Flush in two cases:
                        # 1. Buffer reached 100+ chars (don't wait for timeout)
                        # 2. Timeout expired (no new tokens in 50ms)
                        if should_flush_chars or (text_chunk is None and self._tts_token_buffer.strip()):
                            buffer_to_send = self._tts_token_buffer.strip()
                            if buffer_to_send:
                                elapsed = (time.monotonic() - self._tts_buffer_start_time) * 1000
                                flush_reason = "char-threshold" if should_flush_chars else "timeout"
                                logger.debug(f"Buffer flush ({flush_reason}, {len(buffer_to_send)} chars, {elapsed:.0f}ms): '{buffer_to_send[:60]}'")

                                if not tts_started:
                                    logger.info(f"TTS starting: '{buffer_to_send[:60]}'")
                                    tts_started = True

                                # Stream TTS for accumulated tokens
                                async for audio_chunk in self.tts.synthesize_stream(buffer_to_send):
                                    await self._audio_output_queue.put(audio_chunk)

                            # Reset buffer for next batch
                            self._tts_token_buffer = ""
                            self._tts_buffer_start_time = time.monotonic()

                        # End of turn (all tokens received)
                        if text_chunk is None:
                            self._cancel_buffer_timer()
                            break

                except asyncio.CancelledError:
                    self._cancel_buffer_timer()
                    pass

            except asyncio.CancelledError:
                self._cancel_buffer_timer()
                pass

            await self._audio_output_queue.put(None)  # signal end of this turn's audio

    @staticmethod
    def _split_sentences(text: str) -> tuple[list[str], str]:
        parts = re.split(r'(?<=[.!?])\s+', text)
        if len(parts) <= 1:
            return [], text
        return parts[:-1], parts[-1]

    # FIX: first_audio reset is now INSIDE the loop (was unreachable at end of while)
    async def _send_audio_loop(self):
        first_audio = True

        while not self._should_stop:
            chunk = await self._audio_output_queue.get()

            if chunk is None:
                self._is_speaking = False
                first_audio = True   # FIX: reset here, after turn ends
                continue

            if first_audio:
                ttfw = (time.monotonic() - self._turn_start_time) * 1000
                logger.info(f"TIME TO FIRST WORD (TTFW): {ttfw:.0f}ms")
                first_audio = False
                self._is_speaking = True

            ulaw_chunk = await asyncio.get_event_loop().run_in_executor(
                None, pcm_to_ulaw, chunk
            )
            payload = base64.b64encode(ulaw_chunk).decode("utf-8")
            await self.ws.send_json({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": payload},
            })

    async def _handle_barge_in(self):
        logger.info("BARGE-IN detected")
        self._cancel_buffer_timer()  # Cancel any pending buffer flush
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
        await self.ws.send_json({"event": "clear", "streamSid": self.stream_sid})
        self._is_speaking = False
        self._drain_queue(self._tts_text_queue)
        self._drain_queue(self._audio_output_queue)

    @staticmethod
    def _drain_queue(q: asyncio.Queue):
        while not q.empty():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def cleanup(self):
        self._cancel_buffer_timer()  # Cancel any pending buffer flush
        if self._agent_task:
            self._agent_task.cancel()
        self._should_stop = True
        logger.info("Pipeline cleaned up")
