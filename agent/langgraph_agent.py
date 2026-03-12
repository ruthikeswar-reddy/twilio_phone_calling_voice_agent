"""
LangGraph Agent with streaming support.

Architecture:
  - Uses LangGraph's StateGraph for stateful multi-turn conversations
  - Streams tokens directly via .astream_events()
  - Maintains conversation history per session
  - Supports tool calling (web search, database lookup, etc.)

For voice, we optimize by:
  1. Keeping system prompt concise (fewer tokens = faster TTFT)
  2. Using claude-3-5-haiku or gpt-4o-mini for speed over gpt-4o
  3. max_tokens capped to prevent overly long responses
"""

import asyncio
import logging
import os
from typing import Annotated, AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# State definition
# ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ─────────────────────────────────────────────────────────────
# LLM setup — choose fast model for voice
# ─────────────────────────────────────────────────────────────

def _build_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "openai":
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),   # Fast, cheap, great for voice
            temperature=0.7,
            max_tokens=300,         # Keep responses short for voice
            streaming=True,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
            max_tokens=300,
            streaming=True,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


SYSTEM_PROMPT = """You are a helpful, concise voice assistant.
Keep responses SHORT (1-3 sentences max) and conversational.
Avoid bullet points, markdown, or lists — this is spoken audio.
Be warm, clear, and direct."""


# ─────────────────────────────────────────────────────────────
# Graph definition
# ─────────────────────────────────────────────────────────────

def _build_graph():
    llm = _build_llm()

    async def call_model(state: AgentState):
        messages = state["messages"]
        # Prepend system prompt if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    # Add tools here if needed:
    # llm_with_tools = llm.bind_tools([search_tool, calendar_tool])
    # Then add conditional edge for tool routing

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.set_entry_point("agent")
    graph.add_edge("agent", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────
# LangGraphAgent class
# ─────────────────────────────────────────────────────────────

class LangGraphAgent:
    def __init__(self):
        self._graph = _build_graph()
        self._conversation_history: list = []

    async def stream(self, user_input: str) -> AsyncIterator[str]:
        """
        Stream text tokens for the given user input.
        Yields string chunks as the LLM generates them.
        """
        # Add user message to history
        self._conversation_history.append(HumanMessage(content=user_input))

        # Build input state
        state = {"messages": self._conversation_history.copy()}

        full_response = ""

        try:
            async for event in self._graph.astream_events(state, version="v2"):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        text = chunk.content
                        full_response += text
                        yield text

        except Exception as e:
            logger.error(f"Agent streaming error: {e}")
            yield "I'm sorry, I encountered an error. Please try again."
            return

        # Save assistant response to history
        if full_response:
            self._conversation_history.append(
                AIMessage(content=full_response)
            )

        # Keep history manageable (last 10 turns)
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

    def reset(self):
        """Clear conversation history."""
        self._conversation_history = []
