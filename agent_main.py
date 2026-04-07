import asyncio
import json
import logging
import os
import uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()  # Load .env early

from agent.csa_groq import CustomerSupportAgent

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CSA Groq Agent API", description="Streaming chat endpoint for Customer Support Agent")

_active_agents: dict[str, CustomerSupportAgent] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

@app.post("/chat/csa", response_class=StreamingResponse)
async def chat_csa(request: ChatRequest) -> StreamingResponse:
    """
    Streaming chat endpoint for CSA Groq agent.
    
    Maintains conversation state via session_id.
    Streams agent responses token-by-token in SSE format.
    
    **Usage:**
    ```bash
    curl -X POST "http://localhost:8001/chat/csa" \\
      -H "Content-Type: application/json" \\
      -d '{"message": "VDI not working"}'
    ```
    """
    # Get or create agent instance for session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in _active_agents:
        _active_agents[session_id] = CustomerSupportAgent()
        logger.info(f"New agent session created: {session_id}")
    
    agent = _active_agents[session_id]
    
    async def event_stream() -> AsyncIterator[str]:
        """Generate SSE events from agent stream."""
        try:
            async for text_chunk in agent.stream(
                request.message, 
                session_id, 
                speculative=False
            ):
                if text_chunk.strip():  # Skip empty chunks
                    yield f"data: {json.dumps({'text': text_chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error for session {session_id}: {e}")
            yield f"data: {json.dumps({'error': 'Stream failed'})}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_stream(), 
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(_active_agents),
        "description": "CSA Groq Agent API ready"
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Reset conversation state for session."""
    if session_id in _active_agents:
        del _active_agents[session_id]
        logger.info(f"Session cleared: {session_id}")
        return {"message": "Session cleared"}
    raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions")
async def list_sessions():
    """List active sessions (dev only)."""
    return {
        "active_sessions": list(_active_agents.keys()),
        "count": len(_active_agents)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("AGENT_PORT", 8001))
    logger.info(f"Starting CSA Agent API on port {port}")
    logger.info("Test: curl -X POST 'http://localhost:8001/chat/csa' -d '{\"message\": \"VDI not working\"}'")
    uvicorn.run("agent_main:app", host="0.0.0.0", port=port, log_level="info")
