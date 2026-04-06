import os
import re
from typing import Annotated, Dict, List, Optional, TypedDict, Any, AsyncIterator 

from pydantic import BaseModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import httpx
import asyncio
import logging
logger = logging.getLogger(__name__)


# ============================================================
# LLM
# ============================================================

def get_streaming_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
            temperature=0.1,
            max_tokens=300,
            streaming=True,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            max_tokens=300,
            streaming=True,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
            temperature=0.1,
            max_tokens=300,
            streaming=True,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

def get_structural_llm():
    """Non-streaming LLM for structured output (extraction, intent detection)"""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
            temperature=0.1,
            max_tokens=300,
            streaming=False,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            max_tokens=300,
            streaming=False,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
            temperature=0.1,
            max_tokens=300,
            streaming=False,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

llm = get_streaming_llm()
llm_struct = get_structural_llm()

# ===========================================================
# SERVICE NOW CREDENTIALS (load from environment variables)
# ===========================================================
SERVICENOW_URL = os.getenv("SERVICENOW_URL", "https://dev286942.service-now.com")
SERVICENOW_USERNAME = os.getenv("SERVICENOW_USERNAME", "")
SERVICENOW_PASSWORD = os.getenv("SERVICENOW_PASSWORD", "")

# ============================================================
# STATE INIT
# ============================================================

def ensure_state(state):
    state.setdefault("slots", {})
    state.setdefault("pending_field", None)
    state.setdefault("awaiting_confirmation", False)
    state.setdefault("confirmed", False)
    state.setdefault("field_to_update", None)
    return state


# ============================================================
# SCHEMA
# ============================================================

REQUIRED_FIELDS = [
    "detailed_description",
    "short_description",
    "employee_id",
    "employee_email",
    "location",
]


class ConfirmIntent(BaseModel):
    intent: str  # "confirm" | "change" | "unclear"

class TicketSlots(BaseModel):
    employee_id: Optional[str] = None
    employee_email: Optional[str] = None
    location: Optional[str] = None
    short_description: Optional[str] = None
    detailed_description: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[str] = None
    


class QuestionOutput(BaseModel):
    question: str


class TicketState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    slots: Dict[str, str]
    pending_field: Optional[str]
    awaiting_confirmation: bool
    confirmed: bool
    field_to_update: Optional[str] 


# ============================================================
# HELPERS
# ============================================================

def last_user(state):
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content.strip()
    return ""


def is_valid_email(email):
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))


def is_valid_employee_id(eid):
    return bool(re.match(r"^[A-Z0-9]{3,15}$", eid.upper()))


def is_weak(text):
    return not text or len(text.split()) < 5


def merge_slots(old, new):
    merged = dict(old)

    for k, v in new.model_dump().items():
        # Do NOT overwrite existing values
        if k in merged and merged[k]:
            continue

        if v and isinstance(v, str) and v.strip():
            merged[k] = v.strip()

    return merged


def enforce_description_rules(slots, original_text):
    text = original_text.strip().lower()

    is_short = len(text.split()) <= 7 and "," not in text

    # If short input → NEVER allow detailed_description
    if is_short and not slots.get("detailed_description"):
        slots["detailed_description"] = None

    # Prevent duplication
    if slots.get("short_description") and slots.get("detailed_description"):
        if slots["short_description"].strip().lower() == slots["detailed_description"].strip().lower():
            slots["detailed_description"] = None

    return slots


async def generate_short_description(text):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize into one short sentence (max 10 words)."),
        ("human", "{text}")
    ])
    chain = prompt | llm_struct
    res = await chain.ainvoke({"text": text})
    return res.content.strip()


def get_pending(slots):
    if not slots.get("detailed_description"):
        return "detailed_description"

    if not slots.get("short_description"):
        return "short_description"

    if slots.get("employee_email") and not is_valid_email(slots["employee_email"]):
        return "employee_email"

    if slots.get("employee_id") and not is_valid_employee_id(slots["employee_id"]):
        return "employee_id"

    for f in ["employee_id", "employee_email", "location"]:
        if not slots.get(f):
            return f

    return None


def summarize(slots):
    return (
        f"Employee ID: {slots.get('employee_id','')}\n"
        f"Email: {slots.get('employee_email','')}\n"
        f"Location: {slots.get('location','')}\n\n"
        f"Short Description of Issue: {slots.get('short_description','')}\n"
        f"Detailed Description of Issue: {slots.get('detailed_description','')}\n\n"
        f"Category: {slots.get('category','')}\n"
        f"Priority: {slots.get('priority','')}\n\n"
        "Is everything correct or would you like to change anything?"
    )


def fallback_question(field):
    return {
        "detailed_description": "Could you explain the issue in detail?",
        "short_description": "Could you give a short summary?",
        "employee_id": "May I have your employee ID?",
        "employee_email": "Could you share your company email?",
        "location": "Which location are you working from?"
    }.get(field, "Could you provide more details?")


PRIORITY_MAP = {
    "CRITICAL": "1",
    "HIGH": "2",
    "MEDIUM": "3",
    "LOW": "4"
}


class ServiceNowClient:
    def __init__(self):
        self.base_url = os.getenv("SERVICENOW_URL", SERVICENOW_URL)
        self.username = os.getenv("SERVICENOW_USERNAME", SERVICENOW_USERNAME)
        self.password = os.getenv("SERVICENOW_PASSWORD", SERVICENOW_PASSWORD)
        self.auth = (self.username, self.password) if self.username and self.password else None

    async def create_ticket(self, ticket_data: Dict[str, Any]) -> str:

        endpoint = f"{self.base_url}/api/now/table/incident"

        priority = PRIORITY_MAP.get(
            ticket_data.get("priority", "MEDIUM").upper(),
            "3"
        )

        payload = {
            "short_description": ticket_data.get("short_description"),
            "description": ticket_data.get("detailed_description"),
            "priority": priority,
            "caller_id": ticket_data.get("employee_email"),  # safer default
            "location": ticket_data.get("location"),
            "category": ticket_data.get("category"),
            "u_employee_id": ticket_data.get("employee_id"),  # optional custom field
        }
        logger.info(f"Creating ServiceNow ticket with payload: {payload}")
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        endpoint,
                        json=payload,
                        auth=self.auth,
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        },
                    )

                    if resp.status_code in [200, 201]:
                        return resp.json()["result"]["number"]

                    raise Exception(f"HTTP {resp.status_code}: {resp.text}")

            except Exception as e:
                logger.warning(f"ServiceNow attempt {attempt+1} failed: {e}")

                if attempt == 2:
                    raise

                await asyncio.sleep(2 ** attempt)  # exponential backoff

        raise Exception("ServiceNow ticket creation failed after retries")
snow_client = ServiceNowClient()

# ============================================================
# NEW: CATEGORY + PRIORITY INFERENCE
# ============================================================

async def infer_missing_fields(slots):
    if slots.get("category") and slots.get("priority"):
        return slots

    text = slots.get("detailed_description") or slots.get("short_description") or ""

    if not text:
        return slots

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Infer category and priority.\n"
         "Category: VDI, Network, Access, Hardware, Software\n"
         "Priority: Low, Medium, High, Critical\n"
         "Return only fields."),
        ("human", "{text}")
    ])

    chain = prompt | llm_struct.with_structured_output(TicketSlots)

    try:
        res = await chain.ainvoke({"text": text})

        if not slots.get("category") and res.category:
            slots["category"] = res.category

        if not slots.get("priority") and res.priority:
            slots["priority"] = res.priority
    except:
        pass

    return slots


# ============================================================
# NEW: FIELD DETECTION
# ============================================================

def detect_field_to_update(text):
    text = text.lower()

    if "employee id" in text:
        return "employee_id"
    if "email" in text:
        return "employee_email"
    if "location" in text:
        return "location"
    if "description" in text:
        return "detailed_description"

    return None

# ============================================================
# LLM CHAINS
# ============================================================

extract_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You extract structured ticket fields from user input.\n"
     "\n"
     "Fields:\n"
     "- short_description\n"
     "- detailed_description\n"
     "- category\n"
     "- priority\n"
     "\n"
     "CRITICAL RULES:\n"
     "- DO NOT generate, expand, or rewrite user input\n"
     "- DO NOT duplicate values across fields\n"
     "- detailed_description MUST be empty if user did not explicitly provide detailed explanation\n"
     "- NEVER copy short_description into detailed_description\n"
     "- Only fill fields that are explicitly present\n"
     "\n"
     "Behavior:\n"
     "- If input is short → fill ONLY short_description\n"
     "- If input is detailed → fill detailed_description and summarize into short_description\n"
     "\n"
     "Examples:\n"
     "Input: VDI not working\n"
     "Output:\n"
     "short_description: VDI not working\n"
     "detailed_description:\n"
     "\n"
     "Input: Cannot connect to VDI since morning, getting timeout error\n"
     "Output:\n"
     "short_description: Unable to connect to VDI\n"
     "detailed_description: Cannot connect to VDI since morning, getting timeout error\n"
     ),
    ("human", "{text}")
]) | llm_struct.with_structured_output(TicketSlots)


question_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You are a voice assistant helping raise a support ticket.\n"
     "Ask ONE polite, short, voice-friendly question.\n"
     "Max 12 words.\n"
     "No format questions.\n"
     "Be natural and conversational.\n"
     "Mention purpose if useful.\n"
     "\n"
     "Examples:\n"
     "- Could you briefly describe the issue you're facing?\n"
     "- Please share your employee ID for ticket creation\n"
     "- When did this issue start?\n"
     ),
    ("human", "Field: {field}")
]) | llm_struct.with_structured_output(QuestionOutput)


intent_chain = ChatPromptTemplate.from_messages([
    ("system", "Classify into greeting, unrelated, ticket. If issue → ticket"),
    ("human", "{text}")
]) | llm

confirm_intent_chain = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the user response after a ticket summary.\n"
     "Return one of:\n"
     "- confirm → user agrees everything is correct\n"
     "- change → user wants to modify something\n"
     "- unclear → not clear\n"
     "\n"
     "Examples:\n"
     "yes → confirm\n"
     "it's correct → confirm\n"
     "no changes needed → confirm\n"
     "change my email → change\n"
     "update location → change\n"
     "not sure → unclear\n"
     ),
    ("human", "{text}")
]) | llm_struct.with_structured_output(ConfirmIntent)

# ============================================================
# NODES
# ============================================================

async def classify(state):
    state = ensure_state(state)

    if state.get("awaiting_confirmation"):
        return "handle_confirm"

    if state.get("slots"):
        return "ticket"

    text = last_user(state).lower()

    keywords = ["not working", "issue", "problem", "error", "fail", "can't", "not able"]
    if any(k in text for k in keywords):
        return "ticket"

    res = await intent_chain.ainvoke({"text": text})

    if "greeting" in res.content:
        return "greeting"
    elif "unrelated" in res.content:
        return "unrelated"
    return "ticket"


async def greeting(state):
    return {"messages": [AIMessage(content="Hi! I'll help create your ServiceNow ticket. What issue are you facing?")]}


async def unrelated(state):
    return {"messages": [AIMessage(content="Hi! I'll help create your ServiceNow ticket. What issue are you facing?")]}


async def extract(state):
    state = ensure_state(state)

    text = last_user(state)

    # 🔥 Detect update intent DURING confirmation
    if state.get("awaiting_confirmation"):
        field = detect_field_to_update(text)
        if field:
            return {
                "field_to_update": field,
                "pending_field": field,
                "awaiting_confirmation": False,
                "messages": [AIMessage(content=f"Please provide new {field.replace('_',' ')}.")]
            }

    extracted = await extract_chain.ainvoke({"text": text})

    slots = dict(state.get("slots", {}))

    for k, v in extracted.model_dump().items():

        # 🔥 Allow overwrite ONLY for update
        if state.get("field_to_update") == k:
            if v and isinstance(v, str) and v.strip():
                slots[k] = v.strip()
            continue

        # 🔒 Prevent overwrite
        if k in slots and slots[k]:
            continue

        if v and isinstance(v, str) and v.strip():
            slots[k] = v.strip()

    state["field_to_update"] = None

    slots = enforce_description_rules(slots, text)

    # 🔥 infer category & priority
    slots = await infer_missing_fields(slots)

    if slots.get("detailed_description") and not slots.get("short_description"):
        try:
            slots["short_description"] = await generate_short_description(
                slots["detailed_description"]
            )
        except:
            pass

    pending = get_pending(slots)

    return {
        "slots": slots,
        "pending_field": pending,
        "awaiting_confirmation": False
    }



def route_after_extract(state):
    if state.get("pending_field"):
        return "ask"
    return "confirm"


async def ask(state):
    state = ensure_state(state)
    field = state.get("pending_field")

    try:
        res = await question_chain.ainvoke({"field": field})
        question = res.question.strip()

        if "format" in question.lower() or len(question.split()) > 12:
            question = fallback_question(field)

    except:
        question = fallback_question(field)

    return {"messages": [AIMessage(content=question)]}


async def confirm(state):
    state = ensure_state(state)
    return {
        "awaiting_confirmation": True,
        "messages": [AIMessage(content=summarize(state.get("slots", {})))]
    }


async def handle_confirm(state):
    state = ensure_state(state)
    text = last_user(state)

    try:
        res = await confirm_intent_chain.ainvoke({"text": text})
        intent = res.intent.lower().strip()
    except:
        intent = "unclear"

    # ✅ CONFIRM
    if intent == "confirm":
        return {
            "confirmed": True,
            "awaiting_confirmation": False,
            "messages": [AIMessage(content="Creating ticket...")]
        }

    # ✅ CHANGE
    if intent == "change":
        field = detect_field_to_update(text)

        if field:
            return {
                "confirmed": False,
                "field_to_update": field,
                "pending_field": field,
                "awaiting_confirmation": False,
                "messages": [AIMessage(content=f"Please provide new {field.replace('_',' ')}.")]
            }

        return {
            "confirmed": False,
            "messages": [AIMessage(content="What would you like to change?")]
        }

    # ✅ UNCLEAR
    return {
        "messages": [AIMessage(content="Please confirm or tell me what to change.")]
    }

def route_confirm(state):
    if state.get("confirmed"):
        return "create"
    return "extract"


async def create(state):
    state = ensure_state(state)
    slots = state.get("slots", {})

    try:
        ticket_id = await snow_client.create_ticket(slots)

        return {
            "messages": [
                AIMessage(
                    content=f"Ticket created successfully. Ticket ID: {ticket_id}"
                )
            ]
        }

    except Exception as e:
        return {
            "messages": [
                AIMessage(
                    content="Failed to create ticket. Please try again later."
                )
            ]
        }


# ============================================================
# GRAPH
# ============================================================
# ============================================================
# FIXED GRAPH
# ============================================================

def build_agent():
    g = StateGraph(TicketState)

    g.add_node("greeting", greeting)
    g.add_node("unrelated", unrelated)
    g.add_node("extract", extract)
    g.add_node("ask", ask)
    g.add_node("confirm", confirm)
    g.add_node("handle_confirm", handle_confirm)
    g.add_node("create", create)

    g.add_conditional_edges(START, classify, {
        "greeting": "greeting",
        "unrelated": "unrelated",
        "ticket": "extract",
        "handle_confirm": "handle_confirm"
    })

    g.add_edge("greeting", END)
    g.add_edge("unrelated", END)

    g.add_edge("confirm", END)

    g.add_conditional_edges("extract", route_after_extract, {
        "ask": "ask",
        "confirm": "confirm"
    })

    
    g.add_edge("ask", END)

    g.add_conditional_edges("handle_confirm", route_confirm, {
        "create": "create",
        "extract": "extract"
    })

    g.add_edge("create", END)

    return g.compile()


# ============================================================
# WRAPPER
# ============================================================

class CustomerSupportAgent:
    def __init__(self):
        self.app = build_agent()
        self._conversation_history: List[BaseMessage] = []

    async def stream(self, text: str, session_id: str) -> AsyncIterator[str]:
        """
        Stream ticket agent responses token-by-token for voice pipeline.
        Maintains conversation history across turns.

        Captures both:
        - Streaming LLM tokens (from LLM processing in greeting/extract nodes)
        - Static node responses (ask, confirm, handle_confirm, etc.) - ALWAYS SPOKEN

        GUARANTEE: All stateful messages from ask/confirm/handle_confirm are yielded for TTS.
        """
        # Add user input to history
        self._conversation_history.append(HumanMessage(content=text))

        # Prepare state with history
        state = {"messages": self._conversation_history.copy()}
        full_response = ""
        final_state = None
        had_streaming = False

        try:
            async for event in self.app.astream_events(
                state,
                config={"configurable": {"thread_id": session_id}},
                version="v2"
            ):
                # Capture streaming LLM tokens (greeting, extract nodes with streaming LLM)
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        text_chunk = chunk.content
                        full_response += text_chunk
                        had_streaming = True
                        logger.debug(f"Streaming token: {text_chunk[:50]}")
                        yield text_chunk  # Stream to TTS pipeline

                # Capture final graph state to get static node responses
                elif event["event"] == "on_chain_end" and event.get("name") == "LangGraph":
                    final_state = event.get("data", {}).get("output", {})
                    logger.debug(f"Graph finished with state keys: {final_state.keys() if final_state else 'None'}")

        except Exception as e:
            logger.error(f"Ticket agent streaming error: {e}")
            error_msg = "Sorry, I had trouble processing that. Could you repeat?"
            yield error_msg
            full_response = error_msg

        # CRITICAL: Always extract and speak the last AIMessage from graph state
        # This handles ask, confirm, handle_confirm, greeting, unrelated
        # Even if there was streaming, we need the FINAL message (e.g., confirm summary after extraction)
        if final_state:
            messages = final_state.get("messages", [])
            if messages and isinstance(messages[-1], AIMessage):
                last_message = messages[-1].content
                if last_message and last_message.strip():
                    # Only yield if:
                    # 1. No prior response, OR
                    # 2. This is a different message (confirm/handle_confirm after ask)
                    if not full_response or last_message != full_response:
                        full_response = last_message
                        logger.debug(f"Speaking static message from ask/confirm/handle_confirm: {last_message[:60]}")
                        yield last_message
                    elif not full_response:
                        # First message and no streaming occurred
                        full_response = last_message
                        logger.debug(f"Speaking final message: {last_message[:60]}")
                        yield last_message

        # Append full assistant response to history
        if full_response:
            self._conversation_history.append(AIMessage(content=full_response))

        # Keep history manageable (last 20 messages ~10 turns)
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]
    
    def reset(self):
        """Clear conversation history for new session."""
        self._conversation_history = []

    async def run_turn(self, text, session_id):
        """Legacy synchronous method - use stream() for voice."""
        state = {"messages": self._conversation_history + [HumanMessage(content=text)]}
        result = await self.app.ainvoke(
            state,
            config={"configurable": {"thread_id": session_id}}
        )
        self._conversation_history.append(result["messages"][-1])
        return result["messages"][-1].content
