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
from dotenv import load_dotenv
load_dotenv()

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
    "employee_id",
    "detailed_description",
    "short_description",
    "first_observed",
    "troubleshooting_steps_tried",
    "priority",
]


class ConfirmIntent(BaseModel):
    intent: str  # "confirm" | "change" | "unclear"

class ClassifyIntent(BaseModel):
    intent: str  # "greeting" | "unrelated" | "ticket"

class TicketSlots(BaseModel):
    employee_id: Optional[str] = None
    short_description: Optional[str] = None
    detailed_description: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    priority: Optional[str] = None
    first_observed: Optional[str] = None
    troubleshooting_steps_tried: Optional[str] = None



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


def is_valid_employee_id(eid: str) -> bool:
    # Fix 4: strip hyphens and spaces before validation — users say "E-123" or "EM 123".
    # Also relax minimum length from 3 to 2 (IDs like "E1" are valid in some orgs)
    # and maximum from 15 to 20 to accommodate longer formats.
    normalized = re.sub(r"[\s\-]", "", eid.upper())
    return bool(re.match(r"^[A-Z0-9]{2,20}$", normalized))


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


def enforce_description_rules(slots, original_text, pending_field=None):
    text = original_text.strip().lower()

    # Fix 7: Only apply the short-input guard when we are NOT actively collecting
    # detailed_description. If pending_field == "detailed_description" the agent
    # explicitly asked for it — accept whatever the user says, even if brief.
    # Also tightened threshold from 7 to 4 words: "VPN is not connecting" (4 words)
    # is a valid description and was being incorrectly cleared before.
    if pending_field != "detailed_description":
        is_short = len(text.split()) <= 4 and "," not in text
        if is_short and not slots.get("detailed_description"):
            slots["detailed_description"] = None

    # Prevent duplication: if short_description and detailed_description are identical,
    # clear detailed_description so the LLM is asked for a fuller explanation.
    if slots.get("short_description") and slots.get("detailed_description"):
        if slots["short_description"].strip().lower() == slots["detailed_description"].strip().lower():
            slots["detailed_description"] = None

    return slots


async def generate_short_description(text):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize into one short sentence not missing the user provided context (max 10 words)."),
        ("human", "{text}")
    ])
    chain = prompt | llm_struct
    res = await chain.ainvoke({"text": text})
    return res.content.strip()


def get_pending(slots):
    # Validate existing employee_id if present but invalid
    if slots.get("employee_id") and not is_valid_employee_id(slots["employee_id"]):
        return "employee_id"

    if not slots.get("employee_id"):
        return "employee_id"

    if not slots.get("detailed_description"):
        return "detailed_description"

    if not slots.get("short_description"):
        return "short_description"

    if not slots.get("first_observed"):
        return "first_observed"

    if not slots.get("troubleshooting_steps_tried"):
        return "troubleshooting_steps_tried"

    if not slots.get("priority"):
        return "priority"

    return None


def summarize(slots):
    return (
        f"Summary of the information received for creating the ticket:\n"
        f"Employee ID: {slots.get('employee_id','')}\n"
        f"Short Description: {slots.get('short_description','')}\n"
        f"Observed Error: {slots.get('detailed_description','')}\n"
        f"First Observed: {slots.get('first_observed','')}\n"
        f"Troubleshooting Steps Tried: {slots.get('troubleshooting_steps_tried','')}\n"
        f"Priority: {slots.get('priority','')}\n\n"
        "Would you like me to add any other information or change any existing information?"
    )


def fallback_question(field):
    return {
        "employee_id": "Sure, I can help with that. Could you please share your Employee ID?",
        "detailed_description": "Thanks. Could you briefly describe the issue or error you're seeing?",
        "short_description": "Got it. Could you give a short summary of the issue?",
        "first_observed": "Understood. When did you first notice this issue?",
        "troubleshooting_steps_tried": "Have you tried any basic troubleshooting steps like restarting or reconnecting?",
        "priority": "Got it. Please confirm the priority — Low, Medium, or High?",
    }.get(field, "Got it. Could you provide more details?")


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

        troubleshooting = ticket_data.get("troubleshooting_steps_tried", "")
        first_observed = ticket_data.get("first_observed", "")
        detailed = ticket_data.get("detailed_description", "")
        full_description = detailed
        if first_observed:
            full_description += f"\n\nFirst Observed: {first_observed}"
        if troubleshooting:
            full_description += f"\n\nTroubleshooting Steps Tried: {troubleshooting}"

        payload = {
            "short_description": ticket_data.get("short_description"),
            "description": full_description,
            "priority": priority,
            "category": ticket_data.get("category"),
            "subcategory": ticket_data.get("subcategory"),
            "u_employee_id": ticket_data.get("employee_id"),
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
# CATEGORY + SUBCATEGORY INFERENCE
# ============================================================

# Valid ServiceNow category → subcategory mapping
CATEGORY_SUBCATEGORY_MAP = {
    "Network":         ["DHCP", "DNS", "IP Address", "VPN", "Wireless"],
    "Software":        ["Email", "Operating System"],
    "Hardware":        ["CPU", "Disk", "Monitor", "Keyboard", "Memory", "Mouse"],
    "Inquiry/Help":    ["Antivirus", "Email", "Internal Application"],
    "Database":        ["DB2", "MSSQL Server", "Oracle"],
    "Password Reset":  [],
}

_INFER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an IT support classifier. Given the issue description, infer the best category and subcategory.\n"
     "\n"
     "Available categories and their subcategories:\n"
     "- Network: DHCP, DNS, IP Address, VPN, Wireless\n"
     "- Software: Email, Operating System\n"
     "- Hardware: CPU, Disk, Monitor, Keyboard, Memory, Mouse\n"
     "- Inquiry/Help: Antivirus, Email, Internal Application\n"
     "- Database: DB2, MSSQL Server, Oracle\n"
     "- Password Reset: (no subcategory)\n"
     "\n"
     "Rules:\n"
     "- Choose ONLY from the categories and subcategories listed above.\n"
     "- If no subcategory fits, set subcategory to null.\n"
     "- For Password Reset issues, always set subcategory to null.\n"
     "\n"
     "Return ONLY a JSON object:\n"
     "{{\"category\": \"Network\", \"subcategory\": \"VPN\"}}\n"
     "\n"
     "Examples:\n"
     "Issue: unable to connect to VPN, authentication failed → {{\"category\": \"Network\", \"subcategory\": \"VPN\"}}\n"
     "Issue: cannot send or receive emails → {{\"category\": \"Software\", \"subcategory\": \"Email\"}}\n"
     "Issue: laptop keyboard not responding → {{\"category\": \"Hardware\", \"subcategory\": \"Keyboard\"}}\n"
     "Issue: forgot my Windows password → {{\"category\": \"Password Reset\", \"subcategory\": null}}\n"
     "Issue: Oracle database connection failing → {{\"category\": \"Database\", \"subcategory\": \"Oracle\"}}\n"
     "Issue: antivirus not updating → {{\"category\": \"Inquiry/Help\", \"subcategory\": \"Antivirus\"}}\n"
     ),
    ("human", "{text}")
])


async def infer_missing_fields(slots):
    """Infer category and subcategory from the issue description. Never asks the user."""
    if slots.get("category") and slots.get("subcategory") is not None:
        return slots

    text = slots.get("detailed_description") or slots.get("short_description") or ""
    if not text:
        return slots

    try:
        chain = _INFER_PROMPT | llm_struct
        res = await chain.ainvoke({"text": text})
        parsed = parse_structured_json(res.content, TicketSlots)

        if not slots.get("category") and parsed.category:
            # Validate category is in our allowed list
            if parsed.category in CATEGORY_SUBCATEGORY_MAP:
                slots["category"] = parsed.category
                logger.info(f"[INFER] category='{parsed.category}'")

        if parsed.subcategory:
            cat = slots.get("category", "")
            allowed_subs = CATEGORY_SUBCATEGORY_MAP.get(cat, [])
            if parsed.subcategory in allowed_subs:
                slots["subcategory"] = parsed.subcategory
                logger.info(f"[INFER] subcategory='{parsed.subcategory}'")
            else:
                slots["subcategory"] = None
        else:
            slots.setdefault("subcategory", None)

    except Exception as e:
        logger.warning(f"[INFER] category/subcategory inference failed: {e}")

    return slots


# ============================================================
# NEW: FIELD DETECTION
# ============================================================

def detect_field_to_update(text):
    # Fix 5: Expanded keyword coverage to handle natural language references.
    # Original list was too narrow — "the date", "the ID", "it was wrong" all failed.
    text = text.lower()

    if any(k in text for k in ["employee id", "emp id", "staff id", "my id", "the id", "id is", "id number"]):
        return "employee_id"
    if any(k in text for k in ["description", "issue", "error", "problem", "what i said", "the error"]):
        return "detailed_description"
    if any(k in text for k in ["priority", "urgent", "urgency", "critical", "severity"]):
        return "priority"
    if any(k in text for k in ["observed", "when", "date", "time", "started", "noticed", "first seen", "first time"]):
        return "first_observed"
    if any(k in text for k in ["troubleshoot", "steps", "tried", "attempt", "restart", "relog"]):
        return "troubleshooting_steps_tried"

    return None

# ============================================================
# LLM CHAINS
# ============================================================

extract_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You extract structured ticket fields from user input. Extract ONLY what is explicitly present.\n"
     "\n"
     "Fields:\n"
     "- employee_id: alphanumeric employee identifier (e.g. EM123, EMP10234, 123, E001)\n"
     "- short_description: brief issue title (5-10 words)\n"
     "- detailed_description: the error or problem the user is experiencing\n"
     "- category: VDI, Network, Access, Hardware, or Software\n"
     "- priority: Low, Medium, High, or Critical\n"
     "- first_observed: when the user first noticed the issue (e.g. 'this morning', 'yesterday', 'last week')\n"
     "- troubleshooting_steps_tried: steps the user has already attempted (e.g. 'restarted, reconnected')\n"
     "\n"
     "RULES:\n"
     "- If input is ONLY an ID or number → extract ONLY employee_id, set all others null\n"
     "- If input is a brief problem statement → extract ONLY short_description\n"
     "- If input is a detailed problem explanation → extract detailed_description AND short_description\n"
     "- If input describes when issue started → extract ONLY first_observed\n"
     "- If input describes steps already tried → extract ONLY troubleshooting_steps_tried\n"
     "- If input is a priority level (low/medium/high/critical) → extract ONLY priority\n"
     "- NEVER copy short_description into detailed_description\n"
     "- NEVER invent or expand values beyond what the user said\n"
     "\n"
     "Return ONLY a JSON object with these keys (null if not present):\n"
     "employee_id, short_description, detailed_description, category, priority, first_observed, troubleshooting_steps_tried\n"
     "\n"
     "Examples:\n"
     "Input: My employee ID is EMP10234\n"
     "{{\"employee_id\": \"EMP10234\", \"short_description\": null, \"detailed_description\": null, \"first_observed\": null, \"troubleshooting_steps_tried\": null}}\n"
     "\n"
     "Input: I'm getting an Authentication Failed error while connecting\n"
     "{{\"detailed_description\": \"Authentication Failed error while connecting\", \"short_description\": \"Authentication Failed error\", \"employee_id\": null, \"first_observed\": null, \"troubleshooting_steps_tried\": null}}\n"
     "\n"
     "Input: It started this morning\n"
     "{{\"first_observed\": \"this morning\", \"employee_id\": null, \"detailed_description\": null, \"troubleshooting_steps_tried\": null}}\n"
     "\n"
     "Input: Yes I've already tried restarting and reconnecting but it didn't work\n"
     "{{\"troubleshooting_steps_tried\": \"restarted, tried reconnecting\", \"employee_id\": null, \"detailed_description\": null, \"first_observed\": null}}\n"
     "\n"
     "Input: High priority\n"
     "{{\"priority\": \"High\", \"employee_id\": null, \"detailed_description\": null, \"first_observed\": null, \"troubleshooting_steps_tried\": null}}\n"
     "\n"
     "Input: It's blocking my work so High\n"
     "{{\"priority\": \"High\", \"employee_id\": null, \"detailed_description\": null, \"first_observed\": null, \"troubleshooting_steps_tried\": null}}\n"
     ),
    ("human", "{text}")
]) | llm_struct


question_chain = ChatPromptTemplate.from_messages([
    ("system",
     "You are an IT support voice assistant helping raise a support ticket.\n"
     "You will be given what the user just said and the next field to collect.\n"
     "\n"
     "Generate a response that:\n"
     "1. Starts with a brief, natural acknowledgment of what the user said (1-3 words).\n"
     "   Use phrases like: 'Sure, I can help with that.', 'Thanks.', 'Understood.', 'Got it.', 'Of course.'.\n"
     "   - For the very first message (greeting/issue report) use 'Sure, I can help with that.'\n"
     "   - For factual inputs (IDs, emails) use 'Thanks.'\n"
     "   - For problem descriptions use 'Understood.' or 'Got it.'\n"
     "   - For time/steps inputs use 'Got it.' or 'Noted.'\n"
     "2. Then asks ONE polite, short, voice-friendly question for the next field. Max 15 words.\n"
     "\n"
     "Field meanings:\n"
     "- employee_id: the caller's employee ID\n"
     "- detailed_description: the error or issue they are seeing\n"
     "- first_observed: when they first noticed the problem\n"
     "- troubleshooting_steps_tried: any steps they have already tried\n"
     "- priority: how urgent the issue is (Low, Medium, or High)\n"
     "\n"
     "Return ONLY a JSON object: {{\"question\": \"acknowledgment + question here\"}}\n"
     "\n"
     "Examples:\n"
     "User said: 'I'm unable to connect to VPN', Field: employee_id\n"
     "{{\"question\": \"Sure, I can help with that. Could you please share your Employee ID?\"}}\n"
     "\n"
     "User said: 'EMP10234', Field: detailed_description\n"
     "{{\"question\": \"Thanks. Could you briefly describe the issue or error you're seeing?\"}}\n"
     "\n"
     "User said: 'Getting Authentication Failed error', Field: first_observed\n"
     "{{\"question\": \"Understood. When did you first notice this issue?\"}}\n"
     "\n"
     "User said: 'It started this morning', Field: troubleshooting_steps_tried\n"
     "{{\"question\": \"Have you tried any basic troubleshooting steps like restarting or reconnecting?\"}}\n"
     "\n"
     "User said: 'Yes tried restarting but it didn't work', Field: priority\n"
     "{{\"question\": \"Got it. Please confirm the priority — Low, Medium, or High?\"}}\n"
     ),
    ("human", "User said: '{last_user_message}'\nField: {field}")
]) | llm_struct


intent_chain = ChatPromptTemplate.from_messages([
    ("system",
     "Classify the user message into exactly one of: greeting, unrelated, ticket.\n"
     "\n"
     "Rules:\n"
     "- greeting: user says hello, hi, how are you, how can you help, etc.\n"
     "- ticket: user mentions a technical issue, problem, error, or request for help with a system\n"
     "- ticket: user is replying to an agent question during an ongoing support conversation\n"
     "- unrelated: anything else not related to IT support AND not a reply to an agent question\n"
     "\n"
     "IMPORTANT: If a conversation context is provided below, the user is mid-conversation\n"
     "and their message is a direct reply to the agent's last question.\n"
     "In that case you MUST classify as 'ticket' regardless of what the message looks like,\n"
     "because bare answers like 'it's emp10234' or 'yesterday morning' only make sense\n"
     "as replies to a specific agent question, not as standalone statements.\n"
     "\n"
     "Return ONLY a JSON object: {{\"intent\": \"greeting\"}} or {{\"intent\": \"ticket\"}} or {{\"intent\": \"unrelated\"}}\n"
     "\n"
     "Examples (no context):\n"
     "Hello, how can you help me? → {{\"intent\": \"greeting\"}}\n"
     "Hi there → {{\"intent\": \"greeting\"}}\n"
     "My VDI is not working → {{\"intent\": \"ticket\"}}\n"
     "What's the weather today? → {{\"intent\": \"unrelated\"}}\n"
     "\n"
     "Examples (with context):\n"
     "Context: Agent asked 'Could you share your Employee ID?' | User: 'It's emp10234' → {{\"intent\": \"ticket\"}}\n"
     "Context: Agent asked 'When did you first notice this issue?' | User: 'yesterday morning' → {{\"intent\": \"ticket\"}}\n"
     "Context: Agent asked 'Could you describe the issue?' | User: 'VPN login fails' → {{\"intent\": \"ticket\"}}\n"
     ),
    ("human", "{context}User message: {text}")
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

# Fix 10: Strip leading filler / confirmation words before storing
# troubleshooting steps so "yes, restarting" becomes "restarting".
_TROUBLESHOOT_FILLER = re.compile(
    r"^(yes[,.]?\s*|no[,.]?\s*|sure[,.]?\s*|okay[,.]?\s*|ok[,.]?\s*|"
    r"well[,.]?\s*|so[,.]?\s*|i\s+have\s+|i've\s+|i\s+did\s+|"
    r"i\s+already\s+|already\s+|i\s+tried\s+(to\s+)?|tried\s+(to\s+)?|"
    r"i\s+have\s+tried\s+(to\s+)?|have\s+tried\s+(to\s+)?|i\s+)+",
    re.IGNORECASE,
)

def _strip_troubleshoot_filler(text: str) -> str:
    stripped = _TROUBLESHOOT_FILLER.sub("", text).strip().rstrip(".,")
    # Fall back to original if stripping removed everything
    return stripped if stripped else text.strip()


# ============================================================
# NODES
# ============================================================

_FAREWELL_KEYWORDS = [
    "bye", "goodbye", "see you", "take care",
    "thank you", "thanks", "have a good", "had a good",
    "that's all", "that is all", "no more", "nothing else",
    "all done", "i'm done", "im done", "have a nice",
]
_RESTART_KEYWORDS = [
    "start over", "start again", "begin again", "from scratch",
    "reset", "restart the conversation", "let's restart",
]

async def classify(state):
    state = ensure_state(state)
    logger.info("[NODE] classify — entry")

    text = last_user(state).lower()

    # Fix 6: Restart intent — must be checked first, before slots short-circuit,
    # because slots are populated mid-conversation and would bypass this otherwise.
    if any(k in text for k in _RESTART_KEYWORDS):
        logger.info("[NODE] classify → restart (keyword match)")
        return "restart"

    # Fix 1: Farewell intent — must be checked before slots short-circuit.
    # Without this, "goodbye" mid-conversation routes to ticket flow.
    if any(k in text for k in _FAREWELL_KEYWORDS):
        logger.info("[NODE] classify → farewell (keyword match)")
        return "farewell"

    if state.get("awaiting_confirmation"):
        logger.info("[NODE] classify → handle_confirm (awaiting confirmation)")
        return "handle_confirm"

    # Fix 2: Ticket already created — any follow-up message goes to farewell,
    # not back into the ticket flow. Slot data is cleared by the create node.
    if state.get("confirmed"):
        logger.info("[NODE] classify → farewell (ticket already created)")
        return "farewell"

    if state.get("slots") and any(state["slots"].values()):
        logger.info("[NODE] classify → ticket (slots already populated)")
        return "ticket"

    keywords = ["not working", "issue", "problem", "error", "fail", "can't", "not able"]
    if any(k in text for k in keywords):
        logger.info("[NODE] classify → ticket (keyword match)")
        return "ticket"

    # Build conversation context for the LLM so it can classify mid-conversation
    # replies correctly. Without context, "it's emp10234" looks unrelated to the LLM.
    # With context ("agent asked: Could you share your Employee ID?"), it's clearly
    # a direct answer and the LLM will classify it as "ticket".
    context = ""
    pending = state.get("pending_field")
    if pending:
        last_agent_q = ""
        for m in reversed(state.get("messages", [])):
            if isinstance(m, AIMessage) and m.content.strip():
                last_agent_q = m.content.strip()
                break
        if last_agent_q:
            context = f"Conversation context: Agent asked '{last_agent_q}' (collecting: {pending})\n"
        else:
            context = f"Conversation context: Agent is collecting '{pending}' from the user.\n"

    res = await intent_chain.ainvoke({"text": text, "context": context})
    parsed = parse_structured_json(res.content, ClassifyIntent)
    intent = parsed.intent.lower().strip()
    logger.info(f"[NODE] classify — LLM intent='{intent}' (context={'yes' if context else 'none'})")

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
    return {"messages": [AIMessage(content="Hi! I'm your IT Support Assistant. How can I help you today?")]}


async def unrelated(state):
    logger.info("[NODE] unrelated — redirecting to ticket flow")
    return {"messages": [AIMessage(content="Hi! I can take care of creating your ServiceNow ticket. Could you describe the issue you're facing?")]}


async def farewell(state):
    # Fix 1 & 2: single node handles both mid-conversation farewells and
    # post-ticket thank-yous so the agent always ends gracefully.
    logger.info("[NODE] farewell — ending conversation")
    return {"messages": [AIMessage(content="Thank you for reaching out. Goodbye, and have a great day!")]}


async def restart(state):
    # Fix 6: clear all collected data and invite the user to start fresh.
    # Returning empty slots and reset flags means classify will treat the
    # next message as a brand-new conversation.
    logger.info("[NODE] restart — clearing state and restarting")
    return {
        "slots": {},
        "pending_field": None,
        "awaiting_confirmation": False,
        "confirmed": False,
        "field_to_update": None,
        "messages": [AIMessage(content="Sure, let's start over. How can I help you today?")],
    }


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

    # Handle negative/nil answers for troubleshooting_steps_tried to break the loop
    _NO_STEPS_PHRASES = {"no", "none", "nothing", "nope", "n/a", "no steps", "haven't tried", "did not try", "not tried"}
    _pending = state.get("pending_field")
    if _pending == "troubleshooting_steps_tried":
        _text_norm = text.strip().lower().rstrip(".")
        if _text_norm in _NO_STEPS_PHRASES or _text_norm.startswith("no,") or _text_norm == "no i haven't":
            logger.info("[NODE] extract — user indicated no troubleshooting steps, setting to 'None'")
            slots = dict(state.get("slots", {}))
            slots["troubleshooting_steps_tried"] = "None"
            slots = enforce_description_rules(slots, text, pending_field=_pending)
            slots = await infer_missing_fields(slots)
            pending = get_pending(slots)
            return {"slots": slots, "pending_field": pending, "awaiting_confirmation": False}

    res = await extract_chain.ainvoke({"text": text})
    extracted = parse_structured_json(res.content, TicketSlots)
    logger.info(f"[NODE] extract — extracted slots: {extracted.model_dump()}")

    slots = dict(state.get("slots", {}))

    for k, v in extracted.model_dump().items():
        if not (v and isinstance(v, str) and v.strip()):
            continue
        val = v.strip()

        # Validate before storing — reject obviously wrong values early.
        # Fix 4: Also normalise employee_id (strip hyphens/spaces) before storing
        # so "E-123" and "E 123" are stored as "E123".
        if k == "employee_id":
            val = re.sub(r"[\s\-]", "", val.upper())
            if not is_valid_employee_id(val):
                logger.info(f"[NODE] extract — skipping invalid employee_id: '{val}'")
                continue

        # Allow overwrite ONLY for explicit update request
        if state.get("field_to_update") == k:
            slots[k] = val
            logger.info(f"[NODE] extract — overwrite field '{k}' = '{val}'")
            continue

        # Prevent overwrite of existing valid values.
        # Fix 3: Exception — if we are still actively collecting employee_id
        # (pending_field == "employee_id"), allow the user to correct themselves
        # ("wait, it's EMP456 not EMP123") without needing an explicit update request.
        existing = slots.get(k)
        if existing:
            existing_valid = True
            if k == "employee_id":
                existing_valid = is_valid_employee_id(existing)
                if existing_valid and _pending == "employee_id":
                    # Still in collection phase — user may be self-correcting
                    existing_valid = False
            if existing_valid:
                continue

        slots[k] = val

    state["field_to_update"] = None

    # ── Fix 1: Don't prefill description fields before employee_id is collected ──
    # The user's first message often describes the problem. If we store
    # detailed_description immediately, get_pending() will skip asking for it
    # explicitly. We only keep it once employee_id is confirmed in slots.
    if not slots.get("employee_id"):
        slots.pop("detailed_description", None)
        slots.pop("short_description", None)
        logger.info("[NODE] extract — employee_id not yet collected; discarding premature description fields")

    # Prefer raw text when LLM under-extracts troubleshooting steps.
    # Fix 10: Also strip leading filler words ("yes, I tried restarting" → "restarting").
    if _pending == "troubleshooting_steps_tried":
        raw_words = len(text.split())
        extracted_val = slots.get("troubleshooting_steps_tried", "")
        extracted_words = len(extracted_val.split()) if extracted_val else 0
        if not extracted_val or (raw_words >= 4 and extracted_words < raw_words * 0.5):
            cleaned = _strip_troubleshoot_filler(text)
            slots["troubleshooting_steps_tried"] = cleaned
            logger.info(
                f"[NODE] extract — raw-text override for troubleshooting_steps_tried "
                f"(extracted '{extracted_val}' → cleaned '{cleaned}')"
            )
        elif extracted_val:
            # Even when LLM extraction looks good, strip filler from what it returned
            slots["troubleshooting_steps_tried"] = _strip_troubleshoot_filler(extracted_val)

    slots = enforce_description_rules(slots, text, pending_field=_pending)
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
    user_text = last_user(state)
    logger.info(f"[NODE] ask — prompting for field='{field}'")

    try:
        res = await question_chain.ainvoke({"field": field, "last_user_message": user_text})
        parsed = parse_structured_json(res.content, QuestionOutput)
        question = parsed.question.strip()

        if "format" in question.lower() or len(question.split()) > 25:
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
        # Fix 9 & Fix 2: Reset all state after successful ticket creation.
        # - Clearing slots + confirmed means classify treats the next message fresh.
        # - The success message asks if there's another issue so the user can
        #   naturally start a second ticket or say goodbye.
        return {
            "slots": {},
            "pending_field": None,
            "awaiting_confirmation": False,
            "confirmed": False,
            "field_to_update": None,
            "messages": [
                AIMessage(
                    content=(
                        f"Thanks. Your ticket {ticket_id} has been created successfully. "
                        "Our IT team will reach out shortly. "
                        "Is there anything else I can help you with?"
                    )
                )
            ],
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
    g.add_node("farewell", farewell)   # Fix 1 & 2
    g.add_node("restart", restart)     # Fix 6
    g.add_node("extract", extract)
    g.add_node("ask", ask)
    g.add_node("confirm", confirm)
    g.add_node("handle_confirm", handle_confirm)
    g.add_node("create", create)

    g.add_conditional_edges(START, classify, {
        "greeting": "greeting",
        "unrelated": "unrelated",
        "farewell": "farewell",
        "restart": "restart",
        "ticket": "extract",
        "handle_confirm": "handle_confirm",
    })

    g.add_edge("greeting", END)
    g.add_edge("unrelated", END)
    g.add_edge("farewell", END)
    g.add_edge("restart", END)

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
