import os
import re
from typing import Annotated, Dict, List, Optional, TypedDict, Any, AsyncIterator 

from pydantic import BaseModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
import httpx
import asyncio
import logging
logger = logging.getLogger(__name__)


# ============================================================
# LLM
# ============================================================

def get_streaming_llm():
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.1,
        max_tokens=300,
        streaming=True,
        api_key=os.getenv("GROQ_API_KEY"),
    )

def get_structural_llm():
    """Non-streaming LLM for structured output (extraction, intent detection)"""
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=0.1,
        max_tokens=300,
        streaming=False,
        api_key=os.getenv("GROQ_API_KEY"),
    )

llm = get_streaming_llm()
llm_struct = get_structural_llm()

import json
from pydantic import ValidationError

def parse_structured_json(content: str, model):
    """Parse LLM JSON response to Pydantic model for Groq.

    The LLM sometimes wraps the JSON in explanation text before the ```json block.
    Search for the block anywhere in the response, not just at the start.
    """
    content = content.strip()
    if '```json' in content:
        content = content.split('```json', 1)[1].split('```', 1)[0].strip()
    elif '```' in content:
        content = content.split('```', 1)[1].split('```', 1)[0].strip()
    else:
        # Try to find the first bare JSON object in the text
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group(0)
    try:
        data = json.loads(content)
        return model.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return model()

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

class ClassifyIntent(BaseModel):
    intent: str  # "greeting" | "unrelated" | "ticket"

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
        f"Here's the summary of your ticket:\n"
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
                        body = resp.text.strip()
                        if not body:
                            raise Exception(
                                f"HTTP {resp.status_code} but empty response body"
                            )
                        data = resp.json()
                        number = data.get("result", {}).get("number")
                        if not number:
                            raise Exception(
                                f"Ticket number missing in response: {body}"
                            )
                        return number

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
         "Infer the single best category and priority for this IT support issue.\n"
         "Category options: VDI, Network, Access, Hardware, Software\n"
         "Priority options: Low, Medium, High, Critical\n"
         "\n"
         "Return ONLY a JSON object with no explanation:\n"
         "{{\"category\": \"VDI\", \"priority\": \"Medium\"}}\n"),
        ("human", "{text}")
    ])

    async def call_llm(prompt_text):
        chain = prompt | llm_struct
        res = await chain.ainvoke({"text": prompt_text})
        return parse_structured_json(res.content, TicketSlots)

    try:
        res = await call_llm(text)

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
     "You extract structured ticket fields from user input. Extract ONLY what is explicitly present.\n"
     "\n"
     "Fields:\n"
     "- employee_id: alphanumeric employee identifier (e.g. EM123, 123, E001)\n"
     "- employee_email: work email address\n"
     "- location: office or city name\n"
     "- short_description: brief issue title (5-10 words)\n"
     "- detailed_description: full problem explanation (requires context, min ~20 words)\n"
     "- category: VDI, Network, Access, Hardware, or Software\n"
     "- priority: Low, Medium, High, or Critical\n"
     "\n"
     "RULES:\n"
     "- If input is ONLY an ID, number, email, or location → extract ONLY that field, set all others null\n"
     "- If input is a brief problem statement → extract ONLY short_description\n"
     "- If input is a detailed problem explanation → extract detailed_description AND short_description\n"
     "- NEVER copy short_description into detailed_description\n"
     "- NEVER invent or expand values beyond what the user said\n"
     "\n"
     "Return ONLY a JSON object with these keys (null if not present):\n"
     "employee_id, employee_email, location, short_description, detailed_description, category, priority\n"
     "\n"
     "Examples:\n"
     "Input: My employee ID is EM123\n"
     "{{\"employee_id\": \"EM123\", \"short_description\": null, \"detailed_description\": null}}\n"
     "\n"
     "Input: Its 123\n"
     "{{\"employee_id\": \"123\", \"short_description\": null, \"detailed_description\": null}}\n"
     "\n"
     "Input: abc@company.com\n"
     "{{\"employee_email\": \"abc@company.com\", \"employee_id\": null, \"short_description\": null}}\n"
     "\n"
     "Input: Hyderabad\n"
     "{{\"location\": \"Hyderabad\", \"employee_id\": null, \"short_description\": null}}\n"
     "\n"
     "Input: VDI not working\n"
     "{{\"short_description\": \"VDI not working\", \"detailed_description\": null, \"employee_id\": null}}\n"
     "\n"
     "Input: After entering credentials the VDI keeps loading and never opens, Sometimes its giving me, load on vdi machine is too much, try again later\n"
     "{{\"short_description\": \"VDI stuck loading after login\", \"detailed_description\": \"After entering credentials the VDI keeps loading and never opens, Sometimes its giving me, load on vdi machine is too much, try again later\", \"employee_id\": null}}\n"
     ),
    ("human", "{text}")
]) | llm_struct


question_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You are a voice assistant helping raise a support ticket.\n"
     "Ask ONE polite, short, voice-friendly question. Max 12 words.\n"
     "No format questions. Be natural and conversational.\n"
     "\n"
     "Return ONLY a JSON object: {{\"question\": \"your question here\"}}\n"
     "\n"
     "Examples:\n"
     "{{\"question\": \"Could you briefly describe the issue you're facing?\"}}\n"
     "{{\"question\": \"Please share your employee ID for ticket creation\"}}\n"
     ),
    ("human", "Field: {field}")
]) | llm_struct


intent_chain = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the user message into exactly one of: greeting, unrelated, ticket.\n"
     "Rules:\n"
     "- greeting: user says hello, hi, how are you, how can you help, etc.\n"
     "- ticket: user mentions a technical issue, problem, error, or request for help with a system\n"
     "- unrelated: anything else not related to IT support\n"
     "\n"
     "Return ONLY a JSON object: {{\"intent\": \"greeting\"}} or {{\"intent\": \"ticket\"}} or {{\"intent\": \"unrelated\"}}\n"
     "\n"
     "Examples:\n"
     "Hello, how can you help me? → {{\"intent\": \"greeting\"}}\n"
     "Hi there → {{\"intent\": \"greeting\"}}\n"
     "My VDI is not working → {{\"intent\": \"ticket\"}}\n"
     "What's the weather today? → {{\"intent\": \"unrelated\"}}\n"
     ),
    ("human", "{text}")
]) | llm_struct

confirm_intent_chain = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the user response after a ticket summary.\n"
     "Return ONLY a JSON object: {{\"intent\": \"confirm\"}} or {{\"intent\": \"change\"}} or {{\"intent\": \"unclear\"}}\n"
     "\n"
     "confirm → user agrees everything is correct\n"
     "change → user wants to modify something\n"
     "unclear → not clear\n"
     "\n"
     "Examples:\n"
     "yes → {{\"intent\": \"confirm\"}}\n"
     "it's correct → {{\"intent\": \"confirm\"}}\n"
     "change my email → {{\"intent\": \"change\"}}\n"
     "not sure → {{\"intent\": \"unclear\"}}\n"
     ),
    ("human", "{text}")
]) | llm_struct

# ============================================================
# NODES
# ============================================================

async def classify(state):
    state = ensure_state(state)
    logger.info("[NODE] classify — entry")

    if state.get("awaiting_confirmation"):
        logger.info("[NODE] classify → handle_confirm (awaiting confirmation)")
        return "handle_confirm"

    if state.get("slots") and any(state["slots"].values()):
        logger.info("[NODE] classify → ticket (slots already populated)")
        return "ticket"

    text = last_user(state).lower()

    keywords = ["not working", "issue", "problem", "error", "fail", "can't", "not able"]
    if any(k in text for k in keywords):
        logger.info("[NODE] classify → ticket (keyword match)")
        return "ticket"

    res = await intent_chain.ainvoke({"text": text})
    parsed = parse_structured_json(res.content, ClassifyIntent)
    intent = parsed.intent.lower().strip()
    logger.info(f"[NODE] classify — LLM intent='{intent}'")

    if intent == "greeting":
        logger.info("[NODE] classify → greeting (LLM intent)")
        return "greeting"
    elif intent == "unrelated":
        logger.info("[NODE] classify → unrelated (LLM intent)")
        return "unrelated"
    logger.info("[NODE] classify → ticket (LLM intent)")
    return "ticket"


async def greeting(state):
    logger.info("[NODE] greeting — sending welcome message")
    return {"messages": [AIMessage(content="Hi! I can take care of creating your ServiceNow ticket. Could you describe the issue you're facing?")]}


async def unrelated(state):
    logger.info("[NODE] unrelated — redirecting to ticket flow")
    return {"messages": [AIMessage(content="Hi! I can take care of creating your ServiceNow ticket. Could you describe the issue you're facing?")]}


async def extract(state):
    state = ensure_state(state)
    text = last_user(state)
    logger.info(f"[NODE] extract — user text: '{text}'")

    # Detect update intent DURING confirmation
    if state.get("awaiting_confirmation"):
        field = detect_field_to_update(text)
        if field:
            logger.info(f"[NODE] extract — update intent detected mid-confirmation, field='{field}'")
            return {
                "field_to_update": field,
                "pending_field": field,
                "awaiting_confirmation": False,
                "messages": [AIMessage(content=f"Please provide new {field.replace('_',' ')}.")]
            }

    res = await extract_chain.ainvoke({"text": text})
    extracted = parse_structured_json(res.content, TicketSlots)
    logger.info(f"[NODE] extract — extracted slots: {extracted.model_dump()}")

    slots = dict(state.get("slots", {}))

    for k, v in extracted.model_dump().items():
        if not (v and isinstance(v, str) and v.strip()):
            continue
        val = v.strip()

        # Validate before storing — reject obviously wrong values early
        if k == "employee_id" and not is_valid_employee_id(val):
            logger.info(f"[NODE] extract — skipping invalid employee_id: '{val}'")
            continue
        if k == "employee_email" and not is_valid_email(val):
            logger.info(f"[NODE] extract — skipping invalid employee_email: '{val}'")
            continue

        # Allow overwrite ONLY for explicit update request
        if state.get("field_to_update") == k:
            slots[k] = val
            logger.info(f"[NODE] extract — overwrite field '{k}' = '{val}'")
            continue

        # Prevent overwrite of existing valid values
        existing = slots.get(k)
        if existing:
            existing_valid = True
            if k == "employee_id":
                existing_valid = is_valid_employee_id(existing)
            elif k == "employee_email":
                existing_valid = is_valid_email(existing)
            if existing_valid:
                continue

        slots[k] = val

    state["field_to_update"] = None

    slots = enforce_description_rules(slots, text)
    slots = await infer_missing_fields(slots)

    if slots.get("detailed_description") and not slots.get("short_description"):
        try:
            slots["short_description"] = await generate_short_description(
                slots["detailed_description"]
            )
        except:
            pass

    pending = get_pending(slots)
    logger.info(f"[NODE] extract — slots after merge: {slots}")
    logger.info(f"[NODE] extract — pending_field='{pending}'")

    return {
        "slots": slots,
        "pending_field": pending,
        "awaiting_confirmation": False
    }


def route_after_extract(state):
    route = "ask" if state.get("pending_field") else "confirm"
    logger.info(f"[ROUTE] route_after_extract → '{route}' (pending_field='{state.get('pending_field')}')")
    return route


async def ask(state):
    state = ensure_state(state)
    field = state.get("pending_field")
    logger.info(f"[NODE] ask — prompting for field='{field}'")

    try:
        res = await question_chain.ainvoke({"field": field})
        parsed = parse_structured_json(res.content, QuestionOutput)
        question = parsed.question.strip()

        if "format" in question.lower() or len(question.split()) > 12:
            logger.info(f"[NODE] ask — LLM question rejected, using fallback for '{field}'")
            question = fallback_question(field)

    except:
        logger.warning(f"[NODE] ask — question_chain failed, using fallback for '{field}'")
        question = fallback_question(field)

    logger.info(f"[NODE] ask — question: '{question}'")
    return {"messages": [AIMessage(content=question)]}


async def confirm(state):
    state = ensure_state(state)
    logger.info("[NODE] confirm — presenting ticket summary to user")
    return {
        "awaiting_confirmation": True,
        "messages": [AIMessage(content=summarize(state.get("slots", {})))]
    }


async def handle_confirm(state):
    state = ensure_state(state)
    text = last_user(state)
    logger.info(f"[NODE] handle_confirm — user response: '{text}'")

    try:
        res = await confirm_intent_chain.ainvoke({"text": text})
        parsed = parse_structured_json(res.content, ConfirmIntent)
        intent = parsed.intent.lower().strip()
    except:
        logger.warning("[NODE] handle_confirm — confirm_intent_chain failed, defaulting to 'unclear'")
        intent = "unclear"

    logger.info(f"[NODE] handle_confirm — classified intent='{intent}'")

    if intent == "confirm":
        logger.info("[NODE] handle_confirm → confirmed, proceeding to create")
        return {
            "confirmed": True,
            "awaiting_confirmation": False,
            "messages": [AIMessage(content="Creating ticket...")]
        }

    if intent == "change":
        field = detect_field_to_update(text)
        if field:
            logger.info(f"[NODE] handle_confirm → change requested for field='{field}'")
            return {
                "confirmed": False,
                "field_to_update": field,
                "pending_field": field,
                "awaiting_confirmation": False,
                "messages": [AIMessage(content=f"Please provide new {field.replace('_',' ')}.")]
            }
        logger.info("[NODE] handle_confirm → change requested but field not detected")
        return {
            "confirmed": False,
            "messages": [AIMessage(content="What would you like to change?")]
        }

    logger.info("[NODE] handle_confirm → unclear, asking again")
    return {
        "messages": [AIMessage(content="Please confirm or tell me what to change.")]
    }


def route_confirm(state):
    if state.get("confirmed"):
        logger.info("[ROUTE] route_confirm → 'create'")
        return "create"
    # Change requested or unclear — end this turn, wait for next user message
    logger.info("[ROUTE] route_confirm → END (awaiting user input for change/unclear)")
    return END


async def create(state):
    state = ensure_state(state)
    slots = state.get("slots", {})
    logger.info(f"[NODE] create — attempting ServiceNow ticket creation with slots: {slots}")

    try:
        ticket_id = await snow_client.create_ticket(slots)
        logger.info(f"[NODE] create — ticket created successfully: {ticket_id}")
        return {
            "messages": [
                AIMessage(
                    content=f"Ticket created successfully. Ticket ID: {ticket_id}"
                )
            ]
        }

    except Exception as e:
        logger.error(f"[NODE] create — ServiceNow ticket creation failed: {e}")
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
        END: END,
    })

    g.add_edge("create", END)

    return g.compile()  # No checkpointer — state managed explicitly in CustomerSupportAgent


# ============================================================
# WRAPPER
# ============================================================

class CustomerSupportAgent:
    """
    Per-call stateful wrapper around the LangGraph ticket agent.

    State is managed explicitly here rather than via a MemorySaver checkpointer.
    This solves the speculative-run pollution problem: speculative runs READ the
    current state but never WRITE it back. Only final runs update persisted state.

    One instance is created per call (in VoicePipeline.__init__), so there is no
    cross-call contamination.
    """

    def __init__(self):
        self.app = build_agent()
        self._history: List[BaseMessage] = []
        self._slots: Dict = {}
        self._pending_field: Optional[str] = None
        self._awaiting_confirmation: bool = False
        self._confirmed: bool = False
        self._field_to_update: Optional[str] = None

    def _build_state(self, text: str) -> dict:
        """Snapshot current persisted state + new user message as graph input."""
        return {
            "messages": self._history + [HumanMessage(content=text)],
            "slots": dict(self._slots),
            "pending_field": self._pending_field,
            "awaiting_confirmation": self._awaiting_confirmation,
            "confirmed": self._confirmed,
            "field_to_update": self._field_to_update,
        }

    async def stream(self, text: str, session_id: str, speculative: bool = False) -> AsyncIterator[str]:  # session_id kept for call-site compatibility
        """
        Run the agent for one turn and yield the response text for TTS.

        speculative=True  → reads state, yields response, does NOT persist any changes.
        speculative=False → reads state, yields response, persists all state updates.
        """
        state = self._build_state(text)
        full_response = ""
        final_state = None

        try:
            async for event in self.app.astream_events(state, version="v2"):
                if event["event"] == "on_chain_end" and event.get("name") == "LangGraph":
                    final_state = event.get("data", {}).get("output", {})
                    logger.debug(f"Graph finished, state keys: {list(final_state.keys()) if final_state else 'None'}")

        except Exception as e:
            logger.error(f"Ticket agent streaming error: {e}")
            yield "Sorry, I had trouble processing that. Could you repeat?"
            return

        if final_state:
            messages = final_state.get("messages", [])
            if messages and isinstance(messages[-1], AIMessage):
                last_message = messages[-1].content
                if last_message and last_message.strip():
                    logger.info(f"[AGENT] Speaking: '{last_message}'")
                    full_response = last_message
                    yield last_message

            # Persist state only for confirmed (non-speculative) final turns
            if not speculative:
                self._slots = final_state.get("slots", self._slots)
                self._pending_field = final_state.get("pending_field")
                self._awaiting_confirmation = final_state.get("awaiting_confirmation", False)
                self._confirmed = final_state.get("confirmed", False)
                self._field_to_update = final_state.get("field_to_update")

                if full_response:
                    self._history.append(HumanMessage(content=text))
                    self._history.append(AIMessage(content=full_response))
                    if len(self._history) > 20:
                        self._history = self._history[-20:]
                logger.debug(
                    f"[AGENT] State persisted — slots={self._slots}, "
                    f"pending={self._pending_field}, awaiting={self._awaiting_confirmation}"
                )
            else:
                logger.debug("[AGENT] Speculative run — state NOT persisted")

        if not full_response:
            logger.warning("[AGENT] No response generated this turn")


graph = CustomerSupportAgent().app
