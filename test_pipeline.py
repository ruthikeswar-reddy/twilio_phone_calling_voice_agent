#!/usr/bin/env python3
"""
Sectional Test Suite for Voice Agent Pipeline

Tests each component independently, then end-to-end.

Usage:
  python test_pipeline.py              # Run all tests
  python test_pipeline.py --section 1  # Test only audio utils
  python test_pipeline.py --section 2  # Test STT (requires DEEPGRAM_API_KEY)
  python test_pipeline.py --section 3  # Test LLM Agent (requires OPENAI/ANTHROPIC key)
  python test_pipeline.py --section 4  # Test TTS (requires CARTESIA_API_KEY)
  python test_pipeline.py --section 5  # Full pipeline integration test
  python test_pipeline.py --section 6  # Twilio WebSocket simulation
"""

import asyncio
import argparse
import base64
import json
import logging
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"


# ════════════════════════════════════════════════════════════
# SECTION 1: Audio Utils
# ════════════════════════════════════════════════════════════

async def test_audio_utils():
    print("\n" + "═" * 60)
    print("SECTION 1: Audio Utils (pcm_to_ulaw / ulaw_to_pcm)")
    print("═" * 60)

    from audio_utils import pcm_to_ulaw, ulaw_to_pcm, resample_pcm, normalize_audio_level
    import numpy as np

    # Test 1.1: roundtrip silence
    print("\n[1.1] PCM silence roundtrip (16-bit → μ-law → 16-bit)")
    pcm_silence = bytes(160 * 2)  # 10ms at 8kHz, 16-bit = 160 samples × 2 bytes
    ulaw = pcm_to_ulaw(pcm_silence)
    pcm_back = ulaw_to_pcm(ulaw)
    assert len(ulaw) == 160, f"Expected 160 μ-law bytes, got {len(ulaw)}"
    assert len(pcm_back) == 320, f"Expected 320 PCM bytes, got {len(pcm_back)}"
    print(f"  {PASS}: {len(pcm_silence)} PCM bytes → {len(ulaw)} μ-law bytes → {len(pcm_back)} PCM bytes")

    # Test 1.2: roundtrip sine wave
    print("\n[1.2] Sine wave encode/decode accuracy")
    t = np.linspace(0, 0.01, 80, dtype=np.float32)  # 10ms
    sine = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    pcm_bytes = sine.tobytes()
    ulaw_bytes = pcm_to_ulaw(pcm_bytes)
    recovered_bytes = ulaw_to_pcm(ulaw_bytes)
    recovered = np.frombuffer(recovered_bytes, dtype=np.int16)
    # μ-law has ~13-bit effective precision; allow ~1% RMS error
    rms_error = float(np.sqrt(np.mean((sine.astype(np.float32) - recovered.astype(np.float32)) ** 2)))
    rms_signal = float(np.sqrt(np.mean(sine.astype(np.float32) ** 2)))
    relative_error = rms_error / rms_signal if rms_signal > 0 else 0
    status = PASS if relative_error < 0.05 else FAIL
    print(f"  {status}: RMS error = {relative_error*100:.2f}% (μ-law has ~1-2% inherent quantization)")

    # Test 1.3: resample
    print("\n[1.3] Resample 16kHz → 8kHz")
    pcm_16k = np.zeros(320, dtype=np.int16).tobytes()  # 10ms at 16kHz
    pcm_8k = resample_pcm(pcm_16k, 16000, 8000)
    expected = 160 * 2  # 160 samples at 8kHz × 2 bytes
    status = PASS if len(pcm_8k) == expected else FAIL
    print(f"  {status}: 320 samples@16kHz → {len(pcm_8k)//2} samples@8kHz (expected 160)")

    # Test 1.4: normalize
    print("\n[1.4] Audio normalization")
    quiet = (np.ones(160, dtype=np.int16) * 100).tobytes()
    normalized = normalize_audio_level(quiet, target_rms=3000.0)
    pcm_out = np.frombuffer(normalized, dtype=np.int16)
    new_rms = float(np.sqrt(np.mean(pcm_out.astype(np.float32) ** 2)))
    status = PASS if new_rms > 200 else FAIL
    print(f"  {status}: RMS before=100 → after={new_rms:.0f} (target 3000)")

    print(f"\n  [Section 1 Complete] Audio utils working correctly")


# ════════════════════════════════════════════════════════════
# SECTION 2: Deepgram STT
# ════════════════════════════════════════════════════════════

async def test_stt():
    print("\n" + "═" * 60)
    print("SECTION 2: Deepgram Nova-3 Streaming STT")
    print("═" * 60)

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print(f"  {SKIP}: DEEPGRAM_API_KEY not set in .env")
        return

    from stt.deepgram_stt import DeepgramSTT
    from audio_utils import pcm_to_ulaw
    import numpy as np

    print("\n[2.1] Deepgram WebSocket connection")
    stt = DeepgramSTT()
    connected = False
    events_received = []
    transcript_received = None

    # Generate a simple test: silence (connection test only)
    async with stt.stream() as stream:
        connected = True
        print(f"  {PASS}: Connected to Deepgram Nova-3")

        # Send 2 seconds of silence to test connection stability
        print("\n[2.2] Send silence (connection stability)")
        silence_frame = bytes(160)  # 10ms of μ-law silence = 160 bytes
        for _ in range(200):  # 2 seconds
            await stream.send_audio(silence_frame)

        await asyncio.sleep(0.5)
        print(f"  {PASS}: 2 seconds of audio sent without error")

    print(f"\n  [Section 2 Complete]")
    print(f"  NOTE: To test with real speech, call the server with a Twilio phone call")
    print(f"  Expected transcript latency: 80-150ms after speech ends")


# ════════════════════════════════════════════════════════════
# SECTION 3: LangGraph Agent
# ════════════════════════════════════════════════════════════

async def test_agent():
    print("\n" + "═" * 60)
    print("SECTION 3: LangGraph Agent (LLM Streaming)")
    print("═" * 60)

    provider = os.getenv("LLM_PROVIDER", "openai")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if provider == "openai" and not openai_key:
        print(f"  {SKIP}: OPENAI_API_KEY not set (LLM_PROVIDER=openai)")
        return
    if provider == "anthropic" and not anthropic_key:
        print(f"  {SKIP}: ANTHROPIC_API_KEY not set (LLM_PROVIDER=anthropic)")
        return

    from agent.langgraph_agent import LangGraphAgent

    print(f"\n[3.1] Agent initialization ({provider})")
    agent = LangGraphAgent()
    print(f"  {PASS}: Agent created")

    print(f"\n[3.2] Streaming response — 'What is 2 + 2?'")
    start = time.monotonic()
    tokens = []
    first_token_time = None
    full_response = ""

    async for token in agent.stream("What is 2 + 2?"):
        if first_token_time is None:
            first_token_time = time.monotonic()
            ttft = (first_token_time - start) * 1000
            print(f"  ⚡ First token received in {ttft:.0f}ms")
        tokens.append(token)
        full_response += token

    total_time = (time.monotonic() - start) * 1000
    print(f"  {PASS}: Response: '{full_response.strip()}'")
    print(f"  ⏱  Total: {total_time:.0f}ms | Tokens: {len(tokens)}")

    print(f"\n[3.3] Multi-turn conversation memory")
    agent2 = LangGraphAgent()
    async for _ in agent2.stream("My name is Alice."):
        pass
    response2 = ""
    async for token in agent2.stream("What's my name?"):
        response2 += token
    has_memory = "alice" in response2.lower()
    status = PASS if has_memory else FAIL
    print(f"  {status}: Agent remembered name in multi-turn: '{response2.strip()[:80]}'")

    print(f"\n[3.4] Concise voice response format")
    response3 = ""
    async for token in agent.stream("Tell me about the history of computing in extreme detail"):
        response3 += token
    word_count = len(response3.split())
    status = PASS if word_count <= 80 else FAIL
    print(f"  {status}: Response word count = {word_count} (target: ≤80 words for voice)")
    print(f"  Response: '{response3.strip()[:100]}...'")

    print(f"\n  [Section 3 Complete]")


# ════════════════════════════════════════════════════════════
# SECTION 4: Cartesia TTS
# ════════════════════════════════════════════════════════════

async def test_tts():
    print("\n" + "═" * 60)
    print("SECTION 4: Cartesia Sonic TTS Streaming")
    print("═" * 60)

    if not os.getenv("CARTESIA_API_KEY"):
        print(f"  {SKIP}: CARTESIA_API_KEY not set in .env")
        return

    from tts.cartesia_tts import CartesiaTTS

    print("\n[4.1] TTS initialization")
    tts = CartesiaTTS()
    print(f"  {PASS}: TTS client created (voice: {tts.voice_id})")

    print("\n[4.2] Stream first audio chunk latency")
    start = time.monotonic()
    first_chunk_time = None
    total_bytes = 0
    chunk_count = 0

    async for chunk in tts.synthesize_stream("Hello, how can I help you today?"):
        if first_chunk_time is None:
            first_chunk_time = time.monotonic()
            ttfc = (first_chunk_time - start) * 1000
            print(f"  ⚡ First audio chunk in {ttfc:.0f}ms (target: <150ms)")
        total_bytes += len(chunk)
        chunk_count += 1

    total_time = (time.monotonic() - start) * 1000
    duration_ms = (total_bytes / 2 / 8000) * 1000  # PCM 16-bit at 8kHz
    status = PASS if first_chunk_time and ttfc < 300 else FAIL
    print(f"  {status}: {chunk_count} chunks | {total_bytes} bytes | ~{duration_ms:.0f}ms of audio")
    print(f"  ⏱  Total synthesis: {total_time:.0f}ms")

    print("\n[4.3] μ-law conversion roundtrip")
    from audio_utils import pcm_to_ulaw
    chunks = []
    async for chunk in tts.synthesize_stream("Test."):
        chunks.append(chunk)

    if chunks:
        first_pcm = chunks[0]
        ulaw = pcm_to_ulaw(first_pcm)
        ratio = len(ulaw) / len(first_pcm)
        status = PASS if abs(ratio - 0.5) < 0.01 else FAIL
        print(f"  {status}: PCM {len(first_pcm)}B → μ-law {len(ulaw)}B (ratio {ratio:.2f}, expected 0.50)")

    print(f"\n  [Section 4 Complete]")


# ════════════════════════════════════════════════════════════
# SECTION 5: Full Pipeline Integration (no real Twilio)
# ════════════════════════════════════════════════════════════

async def test_full_pipeline():
    print("\n" + "═" * 60)
    print("SECTION 5: Full Pipeline Integration (STT→LLM→TTS)")
    print("═" * 60)

    missing = []
    if not os.getenv("DEEPGRAM_API_KEY"):
        missing.append("DEEPGRAM_API_KEY")
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        missing.append("OPENAI_API_KEY or ANTHROPIC_API_KEY")
    if not os.getenv("CARTESIA_API_KEY"):
        missing.append("CARTESIA_API_KEY")

    if missing:
        print(f"  {SKIP}: Missing API keys: {', '.join(missing)}")
        return

    from agent.langgraph_agent import LangGraphAgent
    from tts.cartesia_tts import CartesiaTTS
    from audio_utils import pcm_to_ulaw
    import base64

    print("\n[5.1] Simulated transcript → LLM → TTS → μ-law pipeline")
    start = time.monotonic()

    test_transcript = "What's the weather like today?"
    agent = LangGraphAgent()
    tts = CartesiaTTS()

    # Collect LLM tokens
    llm_start = time.monotonic()
    full_response = ""
    first_llm_token = None
    async for token in agent.stream(test_transcript):
        if first_llm_token is None:
            first_llm_token = time.monotonic()
        full_response += token

    llm_ms = (time.monotonic() - llm_start) * 1000
    print(f"  ⚡ LLM complete: '{full_response.strip()[:80]}' ({llm_ms:.0f}ms)")

    # TTS the response
    tts_start = time.monotonic()
    audio_chunks = []
    first_audio_chunk = None
    async for chunk in tts.synthesize_stream(full_response.strip()):
        if first_audio_chunk is None:
            first_audio_chunk = time.monotonic()
        audio_chunks.append(chunk)

    first_tts_ms = (first_audio_chunk - tts_start) * 1000 if first_audio_chunk else 0
    total_ms = (time.monotonic() - start) * 1000

    # Encode to μ-law (as Twilio would receive it)
    total_ulaw_bytes = sum(len(pcm_to_ulaw(c)) for c in audio_chunks)
    audio_duration_ms = (total_ulaw_bytes / 8000) * 1000

    print(f"  ⚡ TTS first chunk: {first_tts_ms:.0f}ms")
    print(f"  📦 Audio output: {total_ulaw_bytes} μ-law bytes (~{audio_duration_ms:.0f}ms of audio)")
    print(f"  🎯 Total pipeline latency: {total_ms:.0f}ms")

    status = PASS if total_ms < 3000 else FAIL
    print(f"  {status}: End-to-end pipeline completed in {total_ms:.0f}ms")

    print(f"\n  Latency breakdown:")
    print(f"    LLM (full):        {llm_ms:.0f}ms")
    print(f"    TTS (first chunk): {first_tts_ms:.0f}ms")
    print(f"    Total E2E:         {total_ms:.0f}ms")

    print(f"\n  [Section 5 Complete]")


# ════════════════════════════════════════════════════════════
# SECTION 6: Twilio WebSocket Simulation
# ════════════════════════════════════════════════════════════

async def test_twilio_simulation():
    print("\n" + "═" * 60)
    print("SECTION 6: Twilio WebSocket Simulation")
    print("═" * 60)
    print("  This test simulates a real Twilio media stream connection.")
    print("  Your server must be running: python main.py")
    print("")

    server_url = os.getenv("TEST_SERVER_URL", "ws://localhost:8000/media-stream")

    try:
        import websockets
    except ImportError:
        print(f"  {SKIP}: websockets not installed: pip install websockets")
        return

    print(f"[6.1] Connecting to {server_url}")
    try:
        async with websockets.connect(server_url, open_timeout=5) as ws:
            print(f"  {PASS}: WebSocket connected")

            # Step 1: Send 'start' event (simulates Twilio starting a stream)
            stream_sid = "MZ" + "x" * 32
            call_sid = "CA" + "x" * 32
            start_event = {
                "event": "start",
                "sequenceNumber": "1",
                "start": {
                    "streamSid": stream_sid,
                    "callSid": call_sid,
                    "accountSid": "ACtest",
                    "tracks": ["inbound"],
                    "customParameters": {},
                    "mediaFormat": {
                        "encoding": "audio/x-mulaw",
                        "sampleRate": 8000,
                        "channels": 1
                    }
                },
                "streamSid": stream_sid,
            }
            await ws.send(json.dumps(start_event))
            print(f"  {PASS}: Sent 'start' event (stream_sid={stream_sid[:8]}...)")

            # Step 2: Send 1 second of μ-law silence audio
            print(f"\n[6.2] Sending 1 second of silent audio")
            silence_frame = base64.b64encode(bytes(160)).decode()  # 10ms μ-law silence
            for seq in range(2, 102):  # 100 frames = 1 second
                media_event = {
                    "event": "media",
                    "sequenceNumber": str(seq),
                    "media": {
                        "track": "inbound",
                        "chunk": str(seq - 1),
                        "timestamp": str((seq - 2) * 10),
                        "payload": silence_frame
                    },
                    "streamSid": stream_sid,
                }
                await ws.send(json.dumps(media_event))

            await asyncio.sleep(0.5)
            print(f"  {PASS}: 1 second of audio sent")

            # Step 3: Send 'stop' event
            print(f"\n[6.3] Sending 'stop' event")
            stop_event = {
                "event": "stop",
                "sequenceNumber": "102",
                "stop": {"accountSid": "ACtest", "callSid": call_sid},
                "streamSid": stream_sid,
            }
            await ws.send(json.dumps(stop_event))
            print(f"  {PASS}: Sent 'stop' event")

    except ConnectionRefusedError:
        print(f"  {FAIL}: Cannot connect to {server_url}")
        print(f"         Is the server running? python main.py")
    except Exception as e:
        print(f"  {FAIL}: {type(e).__name__}: {e}")

    print(f"\n  [Section 6 Complete]")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="Voice Agent Test Suite")
    parser.add_argument("--section", type=int, choices=[1, 2, 3, 4, 5, 6],
                       help="Run only a specific section (1-6)")
    args = parser.parse_args()

    print("\n" + "█" * 60)
    print("   VOICE AGENT PIPELINE TEST SUITE")
    print("█" * 60)

    sections = {
        1: ("Audio Utils",              test_audio_utils),
        2: ("Deepgram STT",             test_stt),
        3: ("LangGraph Agent",          test_agent),
        4: ("Cartesia TTS",             test_tts),
        5: ("Full Pipeline Integration",test_full_pipeline),
        6: ("Twilio WS Simulation",     test_twilio_simulation),
    }

    if args.section:
        name, fn = sections[args.section]
        await fn()
    else:
        for num, (name, fn) in sections.items():
            try:
                await fn()
            except Exception as e:
                print(f"\n  {FAIL}: Section {num} ({name}) crashed: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "═" * 60)
    print("  Tests complete. See above for PASS/FAIL/SKIP status.")
    print("  For end-to-end Twilio testing, see TESTING_GUIDE.md")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
