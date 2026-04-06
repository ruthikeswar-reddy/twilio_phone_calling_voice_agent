"""
Cartesia Sonic Streaming TTS
- Lowest latency TTS available: ~80-120ms to first audio chunk
- Supports streaming (chunks available before full synthesis complete)
- Returns PCM 16-bit audio which we convert to μ-law for Twilio

Alternative: ElevenLabs Turbo v2.5 (slightly higher latency ~200-300ms
but better voice quality). See commented code at bottom.
"""

import asyncio
import logging
import os
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
CARTESIA_API_URL = "https://api.cartesia.ai/tts/bytes"

# Cartesia WebSocket streaming endpoint for lowest latency
CARTESIA_WS_URL = "wss://api.cartesia.ai/tts/websocket"
CARTESIA_VERSION = "2024-06-10"


class CartesiaTTS:
    """
    Streams TTS audio from Cartesia's Sonic model via their WebSocket API.
    Yields PCM audio chunks as they arrive (before full synthesis done).
    """

    def __init__(self):
        self.voice_id = os.getenv("CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091")
        # PCM 16-bit at 8kHz for direct Twilio compatibility after μ-law encoding
        self.output_format = {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": 8000,  # Match Twilio 8kHz
        }

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """
        Streams raw PCM audio bytes for the given text.
        Yields chunks as they arrive from Cartesia.
        """
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
        ) as client:
            headers = {
                "Cartesia-Version": CARTESIA_VERSION,
                "X-API-Key": CARTESIA_API_KEY,
                "Content-Type": "application/json",
            }
            payload = {
                "model_id": "sonic-english",   # Fastest model
                "transcript": text,
                "voice": {
                    "mode": "id",
                    "id": self.voice_id,
                },
                "output_format": self.output_format,
                "language": "en",
            }

            async with client.stream(
                "POST",
                CARTESIA_API_URL,
                headers=headers,
                json=payload,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    logger.error(
                        f"Cartesia error {response.status_code}: {error_body.decode()}"
                    )
                    return

                # Yield PCM chunks as they arrive
                chunk_count = 0
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    if chunk:
                        if chunk_count == 0:
                            logger.debug(f"🔊 First TTS audio chunk received ({len(chunk)} bytes)")
                        chunk_count += 1
                        yield chunk

                logger.debug(f"TTS complete: {chunk_count} chunks for '{text[:40]}...'")


# ─────────────────────────────────────────────────────────────────────
# ALTERNATIVE: ElevenLabs Turbo v2.5
# Better voice quality, ~200ms to first chunk
# Uncomment and set ELEVENLABS_API_KEY to use instead of Cartesia
# ─────────────────────────────────────────────────────────────────────

# import httpx
# ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
# ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# class ElevenLabsTTS:
#     async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
#         url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream"
#         headers = {
#             "xi-api-key": ELEVENLABS_API_KEY,
#             "Content-Type": "application/json",
#         }
#         payload = {
#             "text": text,
#             "model_id": "eleven_turbo_v2_5",
#             "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
#             "output_format": "ulaw_8000",   # Native μ-law! No conversion needed
#             "optimize_streaming_latency": 4, # Max latency optimization
#         }
#         async with httpx.AsyncClient(timeout=10.0) as client:
#             async with client.stream("POST", url, headers=headers, json=payload) as response:
#                 async for chunk in response.aiter_bytes(chunk_size=4096):
#                     if chunk:
#                         yield chunk  # Already μ-law, skip pcm_to_ulaw in pipeline
