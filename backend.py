#https://excalidraw.com/#room=d4f0c6fd872cf33c3fb9,RFmfK1CX4bh3FI-iSANQcA
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid
from typing import Optional, Dict, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
import traceback
import logging

logging.basicConfig(level=logging.INFO)

# ===== Models =====
class ChatRequest(BaseModel):
    thread_id: str
    user_input: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    should_show_buttons: bool = False
    remaining_options: List[str] = []
    conversation_complete: bool = False
    state: Dict = {}

class NewConversationResponse(BaseModel):
    thread_id: str
    message: str

# ===== State Definition =====
class TicketState(TypedDict, total=False):
    user_input: str
    last_message: str
    category: Optional[str]
    first_sop_shown: bool
    wants_ticket: Optional[bool]
    category_confirmed: Optional[bool]
    other_categories_shown: bool
    mandatory_fields: Dict[str, str]
    current_field: Optional[str]
    ticket_created: bool
    conversation_complete: bool

# ===== Configuration =====
CATEGORY_OPTIONS = [
    "payroll", "hr", "it", "facilities", "finance",
    "security", "training", "travel", "procurement", "general"
]

SOP_DB = {
    "payroll": "If your payroll issue is about salary delay, please check ESS portal first. Contact payroll team if issue persists.",
    "hr": "For HR policy related issues, please check the HR handbook on the portal. Escalate to HR team for complex matters.",
    "it": "For IT issues, restart your system and clear cache before raising a ticket. Check IT self-service portal first.",
    "facilities": "For facility issues like AC, cleaning, or maintenance, check with floor coordinator first.",
    "finance": "For finance related queries, ensure you have proper documentation and approvals.",
    "security": "For security issues, contact security desk immediately. For access cards, visit security office.",
    "training": "For training requests, check the learning portal for available courses and schedules.",
    "travel": "For travel related issues, check travel policy and use approved travel booking system.",
    "procurement": "For procurement requests, ensure proper approval workflow and vendor compliance.",
    "general": "For general queries not covered in other categories, provide detailed information."
}

MANDATORY_FIELDS = {
    "payroll": ["employee_id", "issue_description", "pay_period"],
    "hr": ["employee_id", "policy_name", "issue_description"],
    "it": ["system_id", "issue_description", "error_code"],
    "facilities": ["location", "issue_description", "urgency"],
    "finance": ["cost_center", "amount", "issue_description"],
    "security": ["badge_number", "location", "issue_description"],
    "training": ["course_name", "preferred_date", "business_justification"],
    "travel": ["destination", "travel_dates", "purpose"],
    "procurement": ["vendor_name", "amount", "justification"],
    "general": ["department", "issue_description", "contact_details"]
}

# ===== Nodes =====
def show_first_category_sop(state: TicketState) -> TicketState:
    first_category = CATEGORY_OPTIONS[0]
    sop = SOP_DB.get(first_category, "No SOP available.")
    return {
        "category": first_category,
        "first_sop_shown": True,
        "last_message": (
            f"📘 {first_category.title()} SOP:\n\n{sop}\n\n"
            "Do you want to raise a ticket for this issue? (yes/no)"
        )
    }

def handle_ticket_decision(state: TicketState) -> TicketState:
    user_response = state.get("user_input", "").strip().lower()
    if user_response in ("yes", "y"):
        return {"wants_ticket": True}
    elif user_response in ("no", "n"):
        return {
            "wants_ticket": False,
            "conversation_complete": True,
            "last_message": "👍 Thank you! Have a great day."
        }
    else:
        return {"last_message": "Please respond with 'yes' or 'no'."}

def ask_category_confirmation(state: TicketState) -> TicketState:
    category = state.get("category", "Unknown")
    return {"last_message": f"Do you want to confirm {category.title()} as your category? (yes/no)"}

def handle_category_confirmation(state: TicketState) -> TicketState:
    user_response = state.get("user_input", "").strip().lower()
    if user_response in ("yes", "y"):
        return {"category_confirmed": True}
    elif user_response in ("no", "n"):
        return {"category_confirmed": False}
    else:
        return {"last_message": "Please respond with 'yes' or 'no'."}

def show_other_categories(state: TicketState) -> TicketState:
    return {
        "other_categories_shown": True,
        "last_message": "Please select from the following categories:"
    }

def handle_category_selection(state: TicketState) -> TicketState:
    selected = state.get("user_input", "").strip().lower()
    remaining = CATEGORY_OPTIONS[1:]
    if selected in remaining:
        return {"category": selected, "category_confirmed": None}
    return {"last_message": "Please select a valid option from the list."}

def get_next_field(state: TicketState) -> TicketState:
    category = state.get("category")
    if not category:
        return {"last_message": "Error: No category found."}
    required = MANDATORY_FIELDS.get(category, [])
    filled = state.get("mandatory_fields", {})
    for field in required:
        if field not in filled:
            return {
                "current_field": field,
                "last_message": f"Please provide your {field.replace('_',' ').title()}:"
            }
    return {"current_field": None}

def fill_field(state: TicketState) -> TicketState:
    current_field = state.get("current_field")
    if current_field:
        filled = dict(state.get("mandatory_fields", {}))
        filled[current_field] = state.get("user_input", "")
        return {"mandatory_fields": filled, "current_field": None}
    return {}

def create_ticket(state: TicketState) -> TicketState:
    ticket_id = f"TCKT-{uuid.uuid4().hex[:8].upper()}"
    category = state.get("category", "Unknown")
    return {
        "ticket_created": True,
        "conversation_complete": True,
        "last_message": (
            f"✅ Ticket Created Successfully!\n\n"
            f"**Ticket ID:** {ticket_id}\n"
            f"**Category:** {category.title()}\n"
            f"**Details:** {state.get('mandatory_fields', {})}\n\n"
            "You will receive updates via email. Thank you!"
        )
    }

def end_conversation(state: TicketState) -> TicketState:
    return {"conversation_complete": True}

def main_router(state: TicketState) -> TicketState:
    return {}

# ===== FIXED Routing =====
def route_main(state: TicketState) -> str:
    # 1) Show first category SOP
    if not state.get("first_sop_shown"):
        return "show_first_category_sop"
    
    # 2) Handle ticket decision (yes/no)
    if (state.get("first_sop_shown") and 
        state.get("wants_ticket") is None and 
        state.get("user_input")):
        return "handle_ticket_decision"
    
    # 3) If no to ticket → exit
    if state.get("wants_ticket") is False:
        return "end_conversation"
    
    # 4) If yes to ticket but no category confirmation yet
    if (state.get("wants_ticket") is True and 
        state.get("category_confirmed") is None and
        not state.get("other_categories_shown")):
        if state.get("user_input"):
            return "handle_category_confirmation"
        else:
            return "ask_category_confirmation"
    
    # 5) Category confirmed → collect fields
    if state.get("category_confirmed") is True:
        if state.get("current_field") and state.get("user_input"):
            return "fill_field"
        
        category = state.get("category")
        required = MANDATORY_FIELDS.get(category, []) if category else []
        filled = state.get("mandatory_fields", {}).keys()
        
        if all(field in filled for field in required):
            if not state.get("ticket_created"):
                return "create_ticket"
            return "end_conversation"
        return "get_next_field"
    
    # 6) Category not confirmed → show other categories
    if (state.get("category_confirmed") is False and 
        not state.get("other_categories_shown")):
        return "show_other_categories"
    
    # 7) Handle other category selection
    if (state.get("other_categories_shown") and 
        state.get("user_input") and 
        state.get("category_confirmed") is None):
        return "handle_category_selection"
    
    # 8) After selecting other category, ask for confirmation
    if (state.get("other_categories_shown") and 
        state.get("category_confirmed") is None and
        state.get("category") in CATEGORY_OPTIONS[1:]):
        if state.get("user_input"):
            return "handle_category_confirmation"
        else:
            return "ask_category_confirmation"
    
    # Default fallback - should not reach here in normal flow
    return "end_conversation"

def route_after_fill_field(state: TicketState) -> str:
    category = state.get("category")
    required = MANDATORY_FIELDS.get(category, []) if category else []
    filled = state.get("mandatory_fields", {}).keys()
    if all(field in filled for field in required):
        return "create_ticket"
    return "get_next_field"

# ===== LangGraph App =====
def create_langgraph_app():
    builder = StateGraph(TicketState)
    builder.add_node("main_router", main_router)
    builder.add_node("show_first_category_sop", show_first_category_sop)
    builder.add_node("handle_ticket_decision", handle_ticket_decision)
    builder.add_node("ask_category_confirmation", ask_category_confirmation)
    builder.add_node("handle_category_confirmation", handle_category_confirmation)
    builder.add_node("show_other_categories", show_other_categories)
    builder.add_node("handle_category_selection", handle_category_selection)
    builder.add_node("get_next_field", get_next_field)
    builder.add_node("fill_field", fill_field)
    builder.add_node("create_ticket", create_ticket)
    builder.add_node("end_conversation", end_conversation)

    builder.set_entry_point("main_router")
    builder.add_conditional_edges(
        "main_router", route_main,
        {
            "show_first_category_sop": "show_first_category_sop",
            "handle_ticket_decision": "handle_ticket_decision",
            "ask_category_confirmation": "ask_category_confirmation",
            "handle_category_confirmation": "handle_category_confirmation",
            "show_other_categories": "show_other_categories",
            "handle_category_selection": "handle_category_selection",
            "get_next_field": "get_next_field",
            "fill_field": "fill_field",
            "create_ticket": "create_ticket",
            "end_conversation": "end_conversation",
        }
    )
    builder.add_edge("show_first_category_sop", END)
    builder.add_edge("handle_ticket_decision", END)
    builder.add_edge("ask_category_confirmation", END)
    builder.add_edge("handle_category_confirmation", END)
    builder.add_edge("show_other_categories", END)
    builder.add_edge("handle_category_selection", END)
    builder.add_edge("get_next_field", END)
    builder.add_conditional_edges("fill_field", route_after_fill_field)
    builder.add_edge("create_ticket", END)
    builder.add_edge("end_conversation", END)

    memory = InMemorySaver()
    return builder.compile(checkpointer=memory)

app = FastAPI(title="Ticket Support API", version="1.0.0")
langgraph_app = create_langgraph_app()

# ===== Global error handler =====
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logging.error(f"Unhandled error: {exc}\n{tb}")
    return JSONResponse(status_code=500, content={"detail": str(exc), "traceback": tb})

# ===== Endpoints =====
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        cfg = {"configurable": {"thread_id": request.thread_id}}
        input_data = {}
        if request.user_input is not None:
            input_data["user_input"] = request.user_input

        response = langgraph_app.invoke(input_data, config=cfg)
        current_state = langgraph_app.get_state(cfg)
        vals = current_state.values if current_state else {}

        # Show buttons when listing other categories
        should_show_buttons = bool(
            vals.get("other_categories_shown", False) and 
            vals.get("category_confirmed") is None
        )

        return ChatResponse(
            message=response.get("last_message", ""),
            should_show_buttons=should_show_buttons,
            remaining_options=CATEGORY_OPTIONS[1:] if should_show_buttons else [],
            conversation_complete=vals.get("conversation_complete", False),
            state=vals
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/new_conversation", response_model=NewConversationResponse)
async def new_conversation():
    try:
        thread_id = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": thread_id}}
        try:
            response = langgraph_app.invoke({}, config=cfg)
            message = response.get("last_message", "").strip()
        except Exception as e:
            logging.error(f"Initial graph invoke failed: {e}")
            message = "Welcome to the support system! Please type yes or no to proceed."
        if not message:
            message = "Welcome to the support system! Please type yes or no to proceed."
        return NewConversationResponse(thread_id=thread_id, message=message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
