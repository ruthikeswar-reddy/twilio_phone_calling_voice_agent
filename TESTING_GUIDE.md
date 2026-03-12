# End-to-End Testing Guide: Voice Agent

## Prerequisites

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install pyngrok            # For local tunneling
pip install twilio             # For call transfer feature

# 2. Copy and fill in your .env
cp .env.example .env
# Edit .env with your actual API keys

# 3. Verify Python version
python --version               # 3.10+ required; 3.13 uses numpy for audioop
```

---

## Step 1: Test Individual Components

Run these first to catch API key or network issues before testing with Twilio.

```bash
# All sections
python test_pipeline.py

# Individual sections
python test_pipeline.py --section 1   # Audio utils (no API key needed)
python test_pipeline.py --section 2   # Deepgram STT (needs DEEPGRAM_API_KEY)
python test_pipeline.py --section 3   # LLM Agent (needs OPENAI/ANTHROPIC key)
python test_pipeline.py --section 4   # Cartesia TTS (needs CARTESIA_API_KEY)
python test_pipeline.py --section 5   # Full STT→LLM→TTS pipeline
python test_pipeline.py --section 6   # Twilio WebSocket simulation (run server first)
```

### Expected Section 1 Output (Audio Utils — always passes)
```
SECTION 1: Audio Utils (pcm_to_ulaw / ulaw_to_pcm)
[1.1] PCM silence roundtrip: ✅ PASS
[1.2] Sine wave accuracy:    ✅ PASS  (RMS error ~1-2%)
[1.3] Resample 16k→8k:       ✅ PASS
[1.4] Normalization:         ✅ PASS
```

### Expected Section 3 Output (LLM Agent)
```
[3.1] Agent initialization: ✅ PASS
[3.2] First token in ~300-800ms (gpt-4o-mini)
[3.2] Response: 'The answer is 4.'
[3.3] Multi-turn memory:    ✅ PASS
[3.4] Word count ≤ 80:      ✅ PASS
```

### Expected Section 4 Output (TTS)
```
[4.1] TTS init:             ✅ PASS
[4.2] First audio chunk in <150ms
[4.3] μ-law conversion:    ✅ PASS (ratio 0.50)
```

---

## Step 2: Start the Server Locally with ngrok

```bash
# Terminal 1: Start server with auto-ngrok tunnel
python main.py --ngrok

# You'll see output like:
#  ╔══════════════════════════════════════════════════╗
#  ║  Twilio Webhook URL:                              ║
#  ║  https://abc123.ngrok.io/incoming-call            ║
#  ╚══════════════════════════════════════════════════╝
```

Copy the `https://....ngrok.io/incoming-call` URL.

---

## Step 3: Configure Twilio

1. Go to **Twilio Console** → [console.twilio.com](https://console.twilio.com)
2. Navigate to **Phone Numbers** → **Manage** → **Active Numbers**
3. Click your phone number
4. Under **Voice Configuration**, set:
   - **"A call comes in"**: Webhook
   - **URL**: `https://abc123.ngrok.io/incoming-call`
   - **HTTP Method**: `POST`
5. Click **Save**

---

## Step 4: Test the Health Endpoint

```bash
# Confirm server is responding
curl http://localhost:8000/health

# Expected response:
# {"status":"ok","stt":"deepgram-nova-3","tts":"cartesia-sonic","llm":"openai"}

# Test the incoming-call TwiML generation
curl -X POST http://localhost:8000/incoming-call \
  -H "Host: abc123.ngrok.io"

# Expected: TwiML XML with wss://abc123.ngrok.io/media-stream
```

---

## Step 5: Test WebSocket Simulation (Before Real Call)

```bash
# Terminal 1: Server must be running
python main.py --ngrok

# Terminal 2: Run simulation
python test_pipeline.py --section 6

# Expected output:
# [6.1] Connecting:    ✅ PASS
# [6.2] Sending audio: ✅ PASS
# [6.3] Stop event:    ✅ PASS
```

---

## Step 6: Make a Real Test Call

1. Call your Twilio phone number from any phone
2. You should hear: *"Please wait while I connect you to our AI assistant."*
3. Wait for the AI to start listening (1–2 seconds)
4. Say something: *"What's 2 plus 2?"*
5. The AI should respond in **under 500ms** after you stop speaking

### What to Look for in Server Logs

```
🔴 Stream started: SID=MZxxx Call=CAxxx
🎤 Deepgram WebSocket connected (Nova-3, μ-law 8kHz)
🗣  User started speaking
🔤 Interim [0.91]: what is 2 plus  ← speculative start fires here
⚡ Speculative start on interim: 'what is 2 plus'
🤖 Agent starting (speculative)
⚡ First LLM token: 287ms
✅ Final transcript: 'What is 2 plus 2?'
🔊 TTS starting: 'The answer is 4.'
⚡ First audio chunk: 105ms
🎯 TIME TO FIRST WORD (TTFW): 412ms  ← target < 500ms
📵 Call disconnected
```

---

## Step 7: Test Call Transfer

```bash
# While a call is active (get call_sid from server logs or Twilio Console):
curl -X POST "http://localhost:8000/transfer-call/CAxxxxxxxxxxxx" \
  -G --data-urlencode "to_number=+15551234567"

# Expected: {"status":"transferred","call_sid":"CAxxxx","to":"+15551234567"}
```

**Note**: Transfer requires:
- `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` in `.env`
- `twilio` Python package: `pip install twilio`

---

## Latency Benchmarks

| Component | Target | Good | Needs Work |
|-----------|--------|------|------------|
| Deepgram STT (interim) | 80ms | <150ms | >300ms |
| LLM first token (gpt-4o-mini) | 200ms | <400ms | >600ms |
| Cartesia TTS first chunk | 80ms | <150ms | >250ms |
| **Total TTFW** | **<500ms** | **<700ms** | **>1000ms** |

---

## Common Issues

### "WebSocket connection failed"
- Server is not running, or ngrok tunnel expired
- Restart `python main.py --ngrok` and update Twilio webhook URL

### "DEEPGRAM_API_KEY not set"
- Add `DEEPGRAM_API_KEY=your_key` to `.env` file

### "audioop not found" (Python 3.13+)
- Fixed: `audio_utils.py` now uses numpy fallback automatically

### "Twilio returns 11200 error"
- Your webhook URL is not publicly accessible
- Make sure ngrok is running and URL is correct in Twilio Console

### TTS is too quiet / loud
- Adjust `target_rms` in `audio_utils.normalize_audio_level()` (default: 3000)

### Agent response is too long (bad for voice)
- Reduce `max_tokens` in `agent/langgraph_agent.py` (default: 300)
- Strengthen the system prompt: "Respond in exactly 1 sentence."

---

## FastRTC Alternative (When to Use It)

If you want to replace this architecture with FastRTC, it would look like this.
**WARNING**: This loses speculative LLM start and increases TTFW by ~300ms:

```python
# pip install fastrtc
from fastrtc import Stream, ReplyOnPause
from fastrtc.utils import audio_to_bytes

async def voice_handler(audio):
    # audio arrives AFTER silence — no speculative start possible
    transcript = await deepgram_stt(audio)
    response_text = await llm(transcript)
    audio_out = await cartesia_tts(response_text)
    yield audio_out   # much higher latency

stream = Stream(
    handler=ReplyOnPause(voice_handler),
    modality="audio",
    mode="send-receive"
)
stream.mount(app)   # adds /incoming-call and /media-stream automatically
```

Use FastRTC only if you prioritize simplicity over latency (<500ms is not needed).
