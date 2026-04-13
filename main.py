"""
Production Voice Agent: Twilio → Deepgram STT → LangGraph → Cartesia TTS
Target: <500ms Time-To-First-Word (TTFW)

Architecture:
  Phone Call → Twilio Media Stream (WebSocket, μ-law 8kHz)
             → Deepgram Nova-3 Streaming STT (~80-150ms)
             → LangGraph Agent (streaming tokens)
             → Cartesia Sonic TTS (streaming PCM, ~80-120ms to first chunk)
             → μ-law encode → Twilio Media Stream → Phone

WHY WEBSOCKET (not WebRTC/FastRTC):
  Twilio Media Streams ONLY exposes a WebSocket interface to backend servers.
  FastRTC (HuggingFace) handles browser→server WebRTC via aiortc — it cannot
  speak Twilio's proprietary Media Streams JSON protocol.
  WebRTC between caller and Twilio is Twilio's internal infrastructure; your
  code never touches it. This WebSocket handler IS the correct architecture.

LOCAL TESTING (localhost IS fine with ngrok):
  $ ngrok http 8000
  Set https://<ngrok-url>/incoming-call as your Twilio voice webhook.
  You do NOT need to deploy to cloud just to test — ngrok tunnels Twilio
  traffic to your laptop.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from voice_pipeline import VoicePipeline  # Use the new Flux-based pipeline implementation

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_active_calls: dict[str, VoicePipeline] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Voice Agent starting up")
    logger.info(f"   LLM: {os.getenv('LLM_PROVIDER', 'openai')} / {os.getenv('GROQ_MODEL', 'gpt-4o-mini')}")
    logger.info(f"   Port: {os.getenv('PORT', 8000)}")
    yield
    logger.info(f"Voice Agent shutting down — {len(_active_calls)} calls were active")


app = FastAPI(title="Production Voice Agent", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_calls": len(_active_calls),
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
    }


@app.post("/incoming-call")
async def incoming_call(request: Request):
    """
    Twilio webhook for inbound calls.
    Returns TwiML that instructs Twilio to open a Media Stream WebSocket
    back to this server at /media-stream.

    HOW IT WORKS:
      1. Your Twilio number is configured with this URL as its voice webhook
      2. When someone calls, Twilio POSTs here and waits for TwiML XML
      3. The <Stream> TwiML tells Twilio to open a WebSocket to /media-stream
      4. Twilio streams μ-law 8kHz audio over that WebSocket in real time

    TWIML NOTE:
      {From} is a Twilio template variable — it must be a literal brace pair
      in the XML output. Use string concat (not f-string) to avoid escaping issues.
    """
    host = request.headers.get("host", "your-domain.com")
    ws_url = f"wss://{host}/media-stream"

    # BUGFIX: Do NOT use an f-string for the TwiML body — Python f-string
    # escaping will mangle Twilio's {From} variable syntax.
    twiml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Response>\n"
        '  <Say voice="alice">Hi I am your IT Support Assistant. How can I help you today</Say>\n'
        "  <Connect>\n"
        f'    <Stream url="{ws_url}">\n'
        '      <Parameter name="caller" value="{From}"/>\n'
        "    </Stream>\n"
        "  </Connect>\n"
        "</Response>"
    )

    logger.info(f"Incoming call — directing Media Stream to {ws_url}")
    return Response(content=twiml, media_type="text/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """
    WebSocket endpoint that Twilio Media Streams connects to.

    TWILIO MEDIA STREAMS PROTOCOL (JSON over WebSocket):
      Twilio → Server events:
        {"event": "connected", ...}
        {"event": "start", "start": {"streamSid": "MZ...", "callSid": "CA..."}, ...}
        {"event": "media", "streamSid": "MZ...", "media": {"payload": "<base64 ulaw>"}}
        {"event": "stop", ...}

      Server → Twilio events:
        {"event": "media", "streamSid": "MZ...", "media": {"payload": "<base64 ulaw>"}}
        {"event": "clear", "streamSid": "MZ..."}   ← interrupts playback (barge-in)

    WHY NOT FASTRTC:
      FastRTC wraps aiortc for browser WebRTC sessions. It cannot parse Twilio's
      JSON envelope protocol, handle streamSid, or respond with Twilio's expected
      event format. This WebSocket handler is the only supported integration method.
    """
    await websocket.accept()
    call_sid = "pending"
    pipeline = VoicePipeline(websocket)

    try:
        logger.info(" New Twilio Media Stream WebSocket connection")
        call_sid = await pipeline.run()
    except WebSocketDisconnect:
        logger.info(f"Call {call_sid} disconnected")
    except asyncio.TimeoutError:
        logger.warning(f"Call {call_sid} timed out")
    except Exception as e:
        # TaskGroup wraps task exceptions in ExceptionGroup — unwrap for readable logs
        errors = e.exceptions if isinstance(e, ExceptionGroup) else [e]
        for err in errors:
            logger.error(f"Pipeline error on call {call_sid}: {type(err).__name__}: {err}")
    finally:
        _active_calls.pop(call_sid, None)
        await pipeline.cleanup()


@app.get("/calls")
async def list_calls():
    """Dev: list active calls."""
    return {"active_calls": list(_active_calls.keys()), "count": len(_active_calls)}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting on :{port}")
    logger.info("For local Twilio testing: ngrok http %d", port)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=10,
    )
