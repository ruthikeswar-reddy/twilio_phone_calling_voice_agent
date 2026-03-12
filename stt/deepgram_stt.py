"""
Deepgram Nova-3 Streaming STT
- Best-in-class accuracy and latency (~80-150ms to transcript)
- Accepts μ-law 8kHz directly (no conversion needed!)
- Uses WebSocket streaming for real-time interim + final results
- Endpointing: VAD-based utterance detection
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
    "&endpointing=200"           # 200ms silence = utterance end (tune as needed)
    "&utterance_end_ms=1000"     # Finalize after 1s silence
    "&smart_format=true"         # Better formatting
    "&vad_events=true"           # Speech start/end events
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
                # Send close stream message
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
        Reads messages from Deepgram WebSocket and normalizes
        them into our standard event format.
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
                        continue

                    if speech_final or is_final:
                        await self._event_queue.put({
                            "type": "final",
                            "transcript": transcript,
                            "confidence": confidence,
                        })
                    else:
                        await self._event_queue.put({
                            "type": "interim",
                            "transcript": transcript,
                            "confidence": confidence,
                        })

                elif msg_type == "SpeechStarted":
                    await self._event_queue.put({"type": "speech_start"})

                elif msg_type == "UtteranceEnd":
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
