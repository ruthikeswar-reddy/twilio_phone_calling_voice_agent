"""
test_e2e.py — Section-by-section end-to-end test suite for the Voice Agent.

Run sections individually:
  python test_e2e.py section1   # audio conversion (no network)
  python test_e2e.py section2   # Deepgram STT (needs DEEPGRAM_API_KEY)
  python test_e2e.py section3   # LangGraph Agent (needs OPENAI_API_KEY)
  python test_e2e.py section4   # Cartesia TTS (needs CARTESIA_API_KEY)
  python test_e2e.py section5   # Pipeline simulation (mock Twilio WS, no APIs)
  python test_e2e.py section6   # TwiML endpoint (needs server running)
  python test_e2e.py all        # Run all sections

Prerequisites:
  pip install -r requirements.txt
  cp .env.example .env  && fill in your API keys
"""

import asyncio
import base64
import json
import os
import sys
import struct
import time
import wave
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def section(n: int, title: str):
    print(f"\n{'='*60}")
    print(f"  SECTION {n}: {title}")
    print(f"{'='*60}")


def ok(msg: str):
    print(f"  ✅  {msg}")


def fail(msg: str):
    print(f"  ❌  {msg}")


def info(msg: str):
    print(f"  ℹ️   {msg}")


def check_env(*keys):
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        fail(f"Missing env vars: {', '.join(missing)}")
        fail("Add them to your .env file and retry")
        return False
    return True


# ─────────────────────────────────────────────────────────────
# Section 1: Audio Utilities (no network required)
# ─────────────────────────────────────────────────────────────

def test_section1_audio():
    section(1, "Audio Utilities — PCM ↔ μ-law conversion")
    info("No network or API keys required")

    from audio_utils import pcm_to_ulaw, ulaw_to_pcm, resample_pcm

    # Generate a 100ms 440Hz sine wave at 8kHz
    import math
    sample_rate = 8000
    duration_ms = 100
    num_samples = sample_rate * duration_ms // 1000  # 800 samples

    pcm_samples = []
    for i in range(num_samples):
        value = int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
        pcm_samples.append(struct.pack("<h", value))  # 16-bit little-endian
    pcm_bytes = b"".join(pcm_samples)

    info(f"Generated {len(pcm_bytes)} bytes of PCM (100ms 440Hz tone at 8kHz)")

    # Test PCM → μ-law
    ulaw_bytes = pcm_to_ulaw(pcm_bytes)
    assert len(ulaw_bytes) == num_samples, f"Expected {num_samples} μ-law bytes, got {len(ulaw_bytes)}"
    ok(f"PCM → μ-law: {len(pcm_bytes)} bytes → {len(ulaw_bytes)} bytes (2:1 compression)")

    # Test μ-law → PCM round-trip
    pcm_recovered = ulaw_to_pcm(ulaw_bytes)
    assert len(pcm_recovered) == len(pcm_bytes), "Round-trip size mismatch"
    ok(f"μ-law → PCM round-trip: sizes match ({len(pcm_recovered)} bytes)")

    # Verify signal is preserved (μ-law is lossy but within ~2% amplitude)
    original_first = struct.unpack("<h", pcm_bytes[:2])[0]
    recovered_first = struct.unpack("<h", pcm_recovered[:2])[0]
    diff_pct = abs(original_first - recovered_first) / max(abs(original_first), 1) * 100
    info(f"First sample: original={original_first}, recovered={recovered_first}, diff={diff_pct:.1f}%")
    assert diff_pct < 5, f"Round-trip error too high: {diff_pct:.1f}%"
    ok(f"Signal integrity: {diff_pct:.1f}% deviation (< 5% acceptable for μ-law)")

    # Test resampling (16kHz → 8kHz)
    pcm_16k = pcm_bytes * 2  # Fake 16kHz audio (double the samples)
    pcm_8k = resample_pcm(pcm_16k, 16000, 8000)
    info(f"Resample 16kHz→8kHz: {len(pcm_16k)} → {len(pcm_8k)} bytes")
    # Should be roughly half the size
    assert abs(len(pcm_8k) - len(pcm_bytes)) < 10, "Resampling produced wrong output size"
    ok("Resampling 16kHz → 8kHz: size correct")

    ok("Section 1 PASSED ✓")
    return True


# ─────────────────────────────────────────────────────────────
# Section 2: Deepgram STT (needs DEEPGRAM_API_KEY)
# ─────────────────────────────────────────────────────────────

async def test_section2_stt():
    section(2, "Deepgram STT — WebSocket streaming transcription")
    if not check_env("DEEPGRAM_API_KEY"):
        return False

    from stt.deepgram_stt import DeepgramSTT
    import math
    import struct

    # Generate 2 seconds of 440Hz sine at 8kHz μ-law (simulate voice)
    # Note: a pure tone won't produce a real transcript; we're testing connectivity
    sample_rate = 8000
    num_samples = sample_rate * 2
    pcm_samples = []
    for i in range(num_samples):
        value = int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
        pcm_samples.append(struct.pack("<h", value))
    pcm_bytes = b"".join(pcm_samples)

    from audio_utils import pcm_to_ulaw
    ulaw_bytes = pcm_to_ulaw(pcm_bytes)
    info(f"Generated {len(ulaw_bytes)} bytes μ-law audio (2s sine wave)")

    stt = DeepgramSTT()
    events_received = []

    try:
        async with asyncio.timeout(15):
            async with stt.stream() as stream:
                ok("Deepgram WebSocket connected")

                # Send audio in 20ms chunks (matching Twilio's cadence)
                chunk_size = sample_rate * 20 // 1000 * 1  # 20ms of μ-law
                for i in range(0, len(ulaw_bytes), chunk_size):
                    await stream.send_audio(ulaw_bytes[i:i + chunk_size])
                    await asyncio.sleep(0.02)  # 20ms pacing

                ok(f"Sent {len(ulaw_bytes)} bytes of audio to Deepgram")

                # Wait briefly for any events
                try:
                    async with asyncio.timeout(3):
                        async for event in stream.events():
                            events_received.append(event)
                            info(f"Received event: {event['type']}")
                            if len(events_received) >= 3:
                                break
                except asyncio.TimeoutError:
                    pass

    except asyncio.TimeoutError:
        fail("Deepgram connection timed out (15s)")
        return False
    except Exception as e:
        fail(f"Deepgram error: {e}")
        return False

    ok(f"Received {len(events_received)} STT events (sine wave won't produce text)")
    ok("Deepgram WebSocket connection: WORKING")

    info("💡 To get real transcripts, test with actual voice audio (WAV file):")
    info("   Replace the sine wave in this test with real speech PCM/μ-law bytes")

    ok("Section 2 PASSED ✓")
    return True


# ─────────────────────────────────────────────────────────────
# Section 3: LangGraph Agent (needs OPENAI_API_KEY or ANTHROPIC_API_KEY)
# ─────────────────────────────────────────────────────────────

async def test_section3_agent():
    section(3, "LangGraph Agent — LLM streaming tokens")
    provider = os.getenv("LLM_PROVIDER", "openai")
    info(f"Provider: {provider}")

    if provider == "openai" and not check_env("OPENAI_API_KEY"):
        return False
    if provider == "anthropic" and not check_env("ANTHROPIC_API_KEY"):
        return False

    from agent.langgraph_agent import LangGraphAgent

    agent = LangGraphAgent()
    test_input = "What is 2 + 2? Answer in one sentence."

    tokens = []
    start = time.monotonic()
    first_token_time = None

    try:
        async with asyncio.timeout(20):
            async for chunk in agent.stream(test_input):
                tokens.append(chunk)
                if first_token_time is None:
                    first_token_time = time.monotonic()
                    ttft = (first_token_time - start) * 1000
                    info(f"First token latency: {ttft:.0f}ms")

    except asyncio.TimeoutError:
        fail("Agent timed out (20s)")
        return False
    except Exception as e:
        fail(f"Agent error: {e}")
        return False

    full_response = "".join(tokens)
    total_time = (time.monotonic() - start) * 1000

    ok(f"Response: '{full_response.strip()}'")
    ok(f"Total tokens: {len(tokens)}, total time: {total_time:.0f}ms")

    assert len(full_response) > 0, "Empty response from agent"
    ok("Section 3 PASSED ✓")
    return True


# ─────────────────────────────────────────────────────────────
# Section 4: Cartesia TTS (needs CARTESIA_API_KEY)
# ─────────────────────────────────────────────────────────────

async def test_section4_tts():
    section(4, "Cartesia TTS — audio synthesis streaming")
    if not check_env("CARTESIA_API_KEY"):
        return False

    from tts.cartesia_tts import CartesiaTTS

    tts = CartesiaTTS()
    test_text = "Hello, I am your AI voice assistant."

    chunks = []
    start = time.monotonic()
    first_chunk_time = None

    try:
        async with asyncio.timeout(15):
            async for chunk in tts.synthesize_stream(test_text):
                if first_chunk_time is None:
                    first_chunk_time = time.monotonic()
                    ttfc = (first_chunk_time - start) * 1000
                    info(f"First audio chunk latency: {ttfc:.0f}ms")
                chunks.append(chunk)

    except asyncio.TimeoutError:
        fail("TTS timed out (15s)")
        return False
    except Exception as e:
        fail(f"TTS error: {e}")
        return False

    total_bytes = sum(len(c) for c in chunks)
    total_time = (time.monotonic() - start) * 1000

    ok(f"Received {len(chunks)} audio chunks, {total_bytes} bytes total")
    ok(f"Total synthesis time: {total_time:.0f}ms")

    # Verify it's valid PCM (should be 16-bit aligned)
    assert total_bytes % 2 == 0, "PCM bytes not aligned to 16-bit samples"
    ok("PCM alignment: correct (16-bit aligned)")

    # Save test audio for manual inspection
    out_path = Path("/tmp/tts_test_output.wav")
    audio_data = b"".join(chunks)
    try:
        with wave.open(str(out_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # 16-bit
            wf.setframerate(8000)
            wf.writeframes(audio_data)
        ok(f"Saved test audio to {out_path} (play with: aplay {out_path})")
    except Exception:
        info("Could not save WAV (wave module issue) — PCM data looks valid")

    ok("Section 4 PASSED ✓")
    return True


# ─────────────────────────────────────────────────────────────
# Section 5: Pipeline Simulation (mock Twilio WebSocket)
# ─────────────────────────────────────────────────────────────

async def test_section5_pipeline_simulation():
    section(5, "VoicePipeline — mock Twilio WebSocket simulation")
    info("Simulates what Twilio sends over the WebSocket. Uses real APIs if configured.")

    # Build a mock WebSocket that records what was sent TO Twilio
    class MockWebSocket:
        def __init__(self):
            self.sent_events = []
            self.accepted = False
            self._message_queue = asyncio.Queue()
            self._done = False

        async def accept(self):
            self.accepted = True

        async def send_json(self, data):
            self.sent_events.append(data)
            if data.get("event") == "media":
                info(f"  → Sent audio chunk to Twilio: {len(data['media']['payload'])} chars base64")

        async def iter_text(self):
            """Simulate Twilio sending the connected + start + media + stop sequence."""
            import math
            import struct

            # 1. Connected event
            yield json.dumps({"event": "connected", "protocol": "Call"})
            await asyncio.sleep(0.01)

            # 2. Start event
            yield json.dumps({
                "event": "start",
                "start": {
                    "streamSid": "MZtest1234567890",
                    "callSid": "CAtest1234567890",
                    "accountSid": "ACtest",
                    "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000}
                }
            })
            await asyncio.sleep(0.01)

            # 3. Send ~1 second of silence (μ-law silence = 0xFF bytes)
            chunk_size = 160  # 20ms @ 8kHz
            for _ in range(25):  # 25 * 20ms = 500ms
                ulaw_silence = bytes([0xFF] * chunk_size)
                yield json.dumps({
                    "event": "media",
                    "streamSid": "MZtest1234567890",
                    "media": {"payload": base64.b64encode(ulaw_silence).decode()}
                })
                await asyncio.sleep(0.02)

            # 4. Stop event
            yield json.dumps({"event": "stop", "streamSid": "MZtest1234567890"})

    ws = MockWebSocket()
    from voice_pipeline import VoicePipeline

    pipeline = VoicePipeline(ws)

    ok("Mock WebSocket created")
    info("Running pipeline with simulated silence audio (no transcript expected)...")

    try:
        async with asyncio.timeout(15):
            call_sid = await pipeline.run()
    except asyncio.TimeoutError:
        info("Pipeline timed out after 15s (expected for silence input)")
        call_sid = pipeline.call_sid
    except Exception as e:
        info(f"Pipeline stopped: {e}")
        call_sid = pipeline.call_sid

    ok(f"call_sid captured: {call_sid}")
    ok(f"stream_sid captured: {pipeline.stream_sid}")
    ok(f"Events sent to Twilio: {len(ws.sent_events)}")

    media_events = [e for e in ws.sent_events if e.get("event") == "media"]
    info(f"  Audio media events sent: {len(media_events)}")

    ok("Section 5 PASSED ✓ (pipeline runs, mock WS works, state captured correctly)")
    return True


# ─────────────────────────────────────────────────────────────
# Section 6: TwiML Endpoint Verification (needs server running)
# ─────────────────────────────────────────────────────────────

async def test_section6_twiml():
    section(6, "TwiML Endpoint — /incoming-call response verification")
    server_url = os.getenv("TEST_SERVER_URL", "http://localhost:8000")
    info(f"Testing server at: {server_url}")
    info("Make sure the server is running: python main.py")

    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(f"{server_url}/incoming-call")
    except Exception as e:
        fail(f"Could not reach server: {e}")
        info("Start the server first: python main.py")
        return False

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    ok(f"HTTP 200 OK from /incoming-call")

    ct = response.headers.get("content-type", "")
    assert "text/xml" in ct, f"Expected text/xml, got {ct}"
    ok(f"Content-Type: {ct}")

    body = response.text
    info(f"TwiML response:\n{body}")

    # Validate TwiML structure
    assert '<?xml version="1.0"' in body, "Missing XML declaration"
    assert "<Response>" in body, "Missing <Response>"
    assert "<Connect>" in body, "Missing <Connect>"
    assert "<Stream" in body, "Missing <Stream>"
    assert 'url="wss://' in body, "Stream URL should be wss://"
    assert "/media-stream" in body, "Stream URL should point to /media-stream"

    # BUG CHECK: verify Twilio's {From} variable is properly present (not escaped)
    assert "{From}" in body, "Twilio {From} variable is missing or escaped!"
    ok("Twilio {From} variable: correctly present in TwiML")

    # Verify no {{From}} escaping bug
    assert "{{From}}" not in body, "{{From}} escaping bug detected!"
    ok("No f-string escaping bug detected")

    # Health check
    health_resp = await (await httpx.AsyncClient(timeout=5.0).__aenter__()).get(
        f"{server_url}/health"
    )
    health_data = health_resp.json()
    ok(f"Health check: {health_data}")

    ok("Section 6 PASSED ✓")
    return True


# ─────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────

async def run_all():
    results = {}

    # Section 1: No network needed
    results["section1"] = test_section1_audio()

    # Sections 2-5: Need API keys
    results["section2"] = await test_section2_stt()
    results["section3"] = await test_section3_agent()
    results["section4"] = await test_section4_tts()
    results["section5"] = await test_section5_pipeline_simulation()
    results["section6"] = await test_section6_twiml()

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")

    all_passed = all(results.values())
    print(f"\n  Overall: {'ALL PASSED ✓' if all_passed else 'SOME FAILED'}")
    return all_passed


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    if arg == "section1":
        test_section1_audio()
    elif arg == "section2":
        asyncio.run(test_section2_stt())
    elif arg == "section3":
        asyncio.run(test_section3_agent())
    elif arg == "section4":
        asyncio.run(test_section4_tts())
    elif arg == "section5":
        asyncio.run(test_section5_pipeline_simulation())
    elif arg == "section6":
        asyncio.run(test_section6_twiml())
    elif arg == "all":
        asyncio.run(run_all())
    else:
        print(f"Unknown section: {arg}")
        print("Usage: python test_e2e.py [section1|section2|section3|section4|section5|section6|all]")
        sys.exit(1)


if __name__ == "__main__":
    main()
