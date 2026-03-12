# 🎙️ Production Voice Agent

**Twilio → Deepgram Nova-3 → LangGraph → Cartesia Sonic**

A production-grade voice AI pipeline. Callers phone your Twilio number and speak with an LLM-powered agent. Target latency: **Time-To-First-Word < 500ms**.

---

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run locally
uvicorn main:app --reload --port 8000

# 4. Expose via ngrok
ngrok http 8000

# 5. Set Twilio webhook to:
# https://<your-ngrok-url>/incoming-call
```

---

## Full Documentation

Open **`docs/documentation.html`** in your browser for the complete guide including:
- Architecture diagram
- Latency budget breakdown
- Step-by-step installation
- Twilio setup instructions
- Provider comparison tables
- Troubleshooting guide

---

## Project Structure

```
voice-agent/
├── main.py                  # FastAPI app, Twilio webhook, WebSocket endpoint
├── voice_pipeline.py        # Core orchestrator (3 async loops)
├── audio_utils.py           # PCM ↔ μ-law conversion
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container build
├── .env.example             # Environment variable template
│
├── stt/
│   └── deepgram_stt.py      # Deepgram Nova-3 streaming STT
│
├── tts/
│   └── cartesia_tts.py      # Cartesia Sonic streaming TTS
│
├── agent/
│   └── langgraph_agent.py   # LangGraph StateGraph + streaming LLM
│
└── docs/
    └── documentation.html   # Full project documentation
```

---

## Required API Keys

| Service | Purpose | Get Key |
|---------|---------|---------|
| Twilio | Phone calls + Media Streams | https://twilio.com |
| Deepgram | Speech-to-Text (Nova-3) | https://deepgram.com |
| OpenAI | LLM (gpt-4o-mini) | https://platform.openai.com |
| Cartesia | Text-to-Speech (Sonic) | https://cartesia.ai |

---

## Latency Budget

| Stage | Latency |
|-------|---------|
| Deepgram Nova-3 STT | ~80–150ms |
| LangGraph + LLM TTFT | ~50–200ms |
| Cartesia TTS first chunk | ~80–120ms |
| **Total TTFW** | **~220–390ms ✅** |

*Speculative agent start on high-confidence Deepgram interim results saves 150–300ms.*
