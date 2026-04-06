"""
Deepgram Nova-3 Streaming STT
- Best-in-class accuracy and latency (~80-150ms to transcript)
- Accepts μ-law 8kHz directly (no conversion needed!)
- Uses WebSocket streaming for real-time interim + final results
- Endpointing: VAD-based utterance detection

FIX: Correctly distinguish is_final vs speech_final.

  is_final=True    → Deepgram has committed a chunk to its transcript, but
                     the utterance is NOT over. More speech may follow. These
                     should be treated as high-confidence interims, NOT full
                     finals — otherwise multiple "final" events fire per
                     utterance, causing the agent to restart mid-reply.

  speech_final=True → The utterance is definitively complete (endpointing
                      threshold reached). This is the correct trigger for a
                      full "final" agent launch.

  So the correct logic is:
    speech_final=True  → emit type="final"   (utterance done, run agent)
    is_final=True only → emit type="interim" (chunk committed, keep accumulating)
    neither            → emit type="interim" (in-progress chunk)
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator
from dotenv import load_dotenv

import websockets

logger = logging.getLogger(__name__)
load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
logger.info(f"Deepgram API Key loaded: {'Yes' if DEEPGRAM_API_KEY else 'No'}")

# Deepgram Nova-3 WebSocket URL
# mulaw encoding accepted natively → zero conversion overhead
DEEPGRAM_WS_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-3"              # Best accuracy + speed
    "&language=en-US"
    "&encoding=mulaw"            # Twilio native format
    "&sample_rate=8000"          # Twilio native rate
    "&channels=1"
    "&punctuate=true"
    "&interim_results=true"      # Enable for speculative agent start
    "&endpointing=200"           # 200ms silence = utterance end (speech_final trigger)
    "&utterance_end_ms=1000"      # It should be minimum of   1000ms anything less than that we get error from deepgram and total pipeline fails
    "&numerals=true"                                  # Cuts the worst-case STT latency spike by ~500ms
    "&smart_format=true"         # Better formatting
    "&vad_events=true"           # Speech start/end events — required for FIX 1
    "&no_delay=true"             # Minimize processing delay
)


class DeepgramSTTStream:
    """
    Active streaming session. Use as context manager via DeepgramSTT.stream()
    """

    def __init__(self):
        self._ws = None
        self._event_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._receiver_task: asyncio.Task | None = None
        self._closed = False

        # Accumulate is_final chunks into one coherent utterance.
        # Only emitted as "final" when speech_final=True arrives.
        self._utterance_buffer: str = ""

    async def connect(self):
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        self._ws = await websockets.connect(
            DEEPGRAM_WS_URL,
            additional_headers=headers,
            ping_interval=10,
            ping_timeout=5,
            max_size=2**20,
        )
        self._receiver_task = asyncio.create_task(self._receive_loop())
        logger.info("🎤 Deepgram WebSocket connected (Nova-3, μ-law 8kHz)")

    async def send_audio(self, ulaw_bytes: bytes):
        """Forward raw μ-law audio bytes directly to Deepgram."""
        if self._ws and not self._closed:
            try:
                await self._ws.send(ulaw_bytes)
            except Exception as e:
                logger.error(f"Deepgram send error: {e}")

    async def finish(self):
        """Signal end of audio stream to Deepgram."""
        if self._ws and not self._closed:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
        self._closed = True

    def events(self) -> AsyncIterator[dict]:
        """Async iterator that yields transcript events."""
        return self._event_iterator()

    async def _event_iterator(self):
        while True:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=30.0
                )
                if event.get("type") == "CLOSED":
                    break
                yield event
            except asyncio.TimeoutError:
                break

    async def _receive_loop(self):
        """
        Reads messages from Deepgram WebSocket and normalises them into our
        standard event format.

        Event type mapping:
          SpeechStarted          → speech_start  (used for VAD-based barge-in)
          UtteranceEnd           → utterance_end
          Results speech_final   → final         (complete utterance, run agent)
          Results is_final only  → interim       (FIX: was incorrectly "final")
          Results neither        → interim
        """
        try:
            async for raw_message in self._ws:
                data = json.loads(raw_message)
                msg_type = data.get("type", "")

                if msg_type == "Results":
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    if not alternatives:
                        continue

                    alt = alternatives[0]
                    transcript = alt.get("transcript", "").strip()
                    confidence = alt.get("confidence", 0)
                    is_final = data.get("is_final", False)
                    speech_final = data.get("speech_final", False)

                    if not transcript:
                        # Even empty speech_final should flush the buffer if
                        # we have accumulated content.
                        if speech_final and self._utterance_buffer:
                            await self._event_queue.put({
                                "type": "final",
                                "transcript": self._utterance_buffer,
                                "confidence": confidence,
                            })
                            self._utterance_buffer = ""
                        continue

                    if speech_final:
                        # ── FIX: utterance complete ────────────────────────
                        # Merge any previously buffered is_final chunks with
                        # this last segment for the most complete transcript.
                        if self._utterance_buffer:
                            full_transcript = (
                                self._utterance_buffer.rstrip() + " " + transcript
                            ).strip()
                        else:
                            full_transcript = transcript

                        await self._event_queue.put({
                            "type": "final",
                            "transcript": full_transcript,
                            "confidence": confidence,
                        })
                        self._utterance_buffer = ""  # reset for next utterance

                    elif is_final:
                        # ── FIX: chunk committed, but utterance not done ───
                        # Accumulate into buffer and emit as a high-confidence
                        # interim so the speculative agent can still start early.
                        self._utterance_buffer = (
                            self._utterance_buffer.rstrip() + " " + transcript
                        ).strip()

                        await self._event_queue.put({
                            "type": "interim",
                            "transcript": self._utterance_buffer,
                            "confidence": min(confidence + 0.05, 1.0),  # slight boost
                        })

                    else:
                        # In-progress partial result
                        # Show full picture: buffered + current partial
                        display = (
                            self._utterance_buffer.rstrip() + " " + transcript
                        ).strip() if self._utterance_buffer else transcript

                        await self._event_queue.put({
                            "type": "interim",
                            "transcript": display,
                            "confidence": confidence,
                        })

                elif msg_type == "SpeechStarted":
                    await self._event_queue.put({"type": "speech_start"})

                elif msg_type == "UtteranceEnd":
                    # UtteranceEnd fires after utterance_end_ms of silence.
                    # If we still have buffered is_final content, flush it as final.
                    if self._utterance_buffer:
                        logger.debug(
                            f"UtteranceEnd flushing buffer: '{self._utterance_buffer}'"
                        )
                        await self._event_queue.put({
                            "type": "final",
                            "transcript": self._utterance_buffer,
                            "confidence": 0.9,
                        })
                        self._utterance_buffer = ""

                    await self._event_queue.put({"type": "utterance_end"})

                elif msg_type in ("Metadata", "KeepAlive"):
                    pass  # Ignore

        except websockets.exceptions.ConnectionClosed:
            logger.info("Deepgram connection closed")
        except Exception as e:
            logger.error(f"Deepgram receive error: {e}")
        finally:
            await self._event_queue.put({"type": "CLOSED"})

    async def close(self):
        if self._receiver_task:
            self._receiver_task.cancel()
        if self._ws:
            await self._ws.close()


class DeepgramSTT:
    @asynccontextmanager
    async def stream(self):
        stt_stream = DeepgramSTTStream()
        await stt_stream.connect()
        try:
            yield stt_stream
        finally:
            await stt_stream.close()

