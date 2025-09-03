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

# ===== Node Functions =====
def show_first_category_sop(state: TicketState) -> TicketState:
    """Show initial payroll SOP"""
    new_state = dict(state)
    first_category = CATEGORY_OPTIONS[0]
    sop = SOP_DB.get(first_category, "No SOP available.")
    
    new_state.update({
        "category": first_category,
        "first_sop_shown": True,
        "last_message": (
            f"📘 {first_category.title()} SOP:\n\n{sop}\n\n"
            "Do you want to raise a ticket for this issue? (yes/no)"
        )
    })
    
    logging.info("Showing first SOP")
    return new_state

def handle_ticket_decision(state: TicketState) -> TicketState:
    """Handle yes/no for raising ticket"""
    new_state = dict(state)
    user_response = state.get("user_input", "").strip().lower()
    
    if user_response in ("yes", "y"):
        new_state["wants_ticket"] = True
        logging.info("User wants ticket - will ask for confirmation next")
    elif user_response in ("no", "n"):
        new_state.update({
            "wants_ticket": False,
            "conversation_complete": True,
            "last_message": "👍 Thank you! Have a great day."
        })
        logging.info("User doesn't want ticket - ending conversation")
    else:
        new_state["last_message"] = "Please respond with 'yes' or 'no'."
    
    return new_state

def ask_category_confirmation(state: TicketState) -> TicketState:
    """Ask for category confirmation"""
    new_state = dict(state)
    category = state.get("category", "Unknown")
    new_state["last_message"] = f"Do you want to confirm **{category.title()}** as your category? (yes/no)"
    logging.info(f"Asking confirmation for category: {category}")
    return new_state

def handle_category_confirmation(state: TicketState) -> TicketState:
    """Handle category confirmation response"""
    new_state = dict(state)
    user_response = state.get("user_input", "").strip().lower()
    
    if user_response in ("yes", "y"):
        new_state["category_confirmed"] = True
        logging.info("Category confirmed - will start field collection")
    elif user_response in ("no", "n"):
        new_state["category_confirmed"] = False
        logging.info("Category not confirmed - will show other options")
    else:
        new_state["last_message"] = "Please respond with 'yes' or 'no'."
    
    return new_state

def show_other_categories(state: TicketState) -> TicketState:
    """Show other category options"""
    new_state = dict(state)
    new_state.update({
        "other_categories_shown": True,
        "category_confirmed": None,
        "last_message": "Please select from the following categories:"
    })
    logging.info("Showing other categories")
    return new_state

def handle_category_selection(state: TicketState) -> TicketState:
    """Handle selection of alternative category"""
    new_state = dict(state)
    selected = state.get("user_input", "").strip().lower()
    remaining_categories = CATEGORY_OPTIONS[1:]
    
    if selected in remaining_categories:
        new_state.update({
            "category": selected,
            "other_categories_shown": False,
            "category_confirmed": None,
        })
        logging.info(f"Selected new category: {selected}")
    else:
        new_state["last_message"] = "Please select a valid option from the list."
    
    return new_state

def get_next_field(state: TicketState) -> TicketState:
    """Get the next required field to collect"""
    new_state = dict(state)
    category = state.get("category")
    
    if not category:
        new_state["last_message"] = "Error: No category found."
        return new_state
    
    required = MANDATORY_FIELDS.get(category, [])
    filled = state.get("mandatory_fields", {})
    
    for field in required:
        if field not in filled:
            new_state.update({
                "current_field": field,
                "last_message": f"Please provide your **{field.replace('_',' ').title()}**:"
            })
            logging.info(f"Requesting field: {field}")
            return new_state
    
    new_state["current_field"] = None
    logging.info("All mandatory fields collected")
    return new_state

def fill_field(state: TicketState) -> TicketState:
    """Fill the current field with user input"""
    new_state = dict(state)
    current_field = state.get("current_field")
    user_input = state.get("user_input", "").strip()
    
    if current_field and user_input:
        filled = dict(state.get("mandatory_fields", {}))
        filled[current_field] = user_input
        new_state.update({
            "mandatory_fields": filled,
            "current_field": None
        })
        logging.info(f"Filled field '{current_field}' with value: {user_input}")
    
    return new_state

def create_ticket(state: TicketState) -> TicketState:
    """Create the final ticket"""
    new_state = dict(state)
    ticket_id = f"TCKT-{uuid.uuid4().hex[:8].upper()}"
    category = state.get("category", "Unknown")
    
    new_state.update({
        "ticket_created": True,
        "conversation_complete": True,
        "last_message": (
            f"✅ **Ticket Created Successfully!**\n\n"
            f"**Ticket ID:** {ticket_id}\n"
            f"**Category:** {category.title()}\n"
            f"**Details:** {state.get('mandatory_fields', {})}\n\n"
            "You will receive updates via email. Thank you!"
        )
    })
    
    logging.info(f"Created ticket: {ticket_id} with details: {state.get('mandatory_fields', {})}")
    return new_state

def end_conversation(state: TicketState) -> TicketState:
    """End the conversation"""
    new_state = dict(state)
    new_state["conversation_complete"] = True
    return new_state

def main_router(state: TicketState) -> TicketState:
    """Main router - just passes state through"""
    return state

# ===== FIXED Main Routing Function =====
def route_main(state: TicketState) -> str:
    """Main routing function that ensures proper flow sequence"""
    
    logging.info(f"Routing with state: first_sop_shown={state.get('first_sop_shown')}, wants_ticket={state.get('wants_ticket')}, category_confirmed={state.get('category_confirmed')}, other_categories_shown={state.get('other_categories_shown')}")
    
    # Step 1: Show Payroll SOP first
    if not state.get('first_sop_shown'):
        return 'show_first_category_sop'

    # Step 2: Handle ticket decision after SOP is shown
    if state.get('first_sop_shown') and state.get('wants_ticket') is None and state.get('user_input'):
        return 'handle_ticket_decision'

    # Step 3: If no ticket wanted, end conversation
    if state.get('wants_ticket') is False:
        return 'end_conversation'

    # Step 4: CRITICAL - If yes to ticket, ALWAYS ask for confirmation first (not in other categories flow)
    if (state.get('wants_ticket') is True and 
        state.get('category_confirmed') is None and 
        not state.get('other_categories_shown', False)):
        if state.get('user_input'):
            return 'handle_category_confirmation'
        else:
            return 'ask_category_confirmation'

    # Step 5: If category confirmed, start field collection
    if state.get('category_confirmed') is True:
        if state.get('current_field') and state.get('user_input'):
            return 'fill_field'
        
        # Check if all required fields are collected
        category = state.get('category')
        required = MANDATORY_FIELDS.get(category, []) if category else []
        filled = state.get('mandatory_fields', {})
        
        if all(field in filled for field in required) and filled:
            if not state.get('ticket_created'):
                return 'create_ticket'
            else:
                return 'end_conversation'
        else:
            return 'get_next_field'

    # Step 6: If category not confirmed, show other categories
    if state.get('category_confirmed') is False and not state.get('other_categories_shown', False):
        return 'show_other_categories'

    # Step 7: Handle other category selection
    if state.get('other_categories_shown') and state.get('user_input') and state.get('category_confirmed') is None:
        return 'handle_category_selection'

    # Step 8: After selecting other category, ask for confirmation
    if (state.get('other_categories_shown') and 
        state.get('category_confirmed') is None and 
        state.get('category') in CATEGORY_OPTIONS[1:]):
        if state.get('user_input'):
            return 'handle_category_confirmation'
        else:
            return 'ask_category_confirmation'

    # Fallback
    return 'end_conversation'

# ===== LangGraph Setup =====
def create_langgraph_app():
    builder = StateGraph(TicketState)
    
    # Add nodes
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
    
    # Set entry point to main router
    builder.set_entry_point("main_router")
    
    # Use single main router for all routing decisions
    builder.add_conditional_edges(
        "main_router", 
        route_main,
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
    
    # All nodes return to main router for next decision (except terminal nodes)
    for node in ["show_first_category_sop", "handle_ticket_decision", "ask_category_confirmation",
                 "handle_category_confirmation", "show_other_categories", "handle_category_selection",
                 "get_next_field", "fill_field"]:
        builder.add_edge(node, "main_router")
    
    # Terminal nodes
    builder.add_edge("create_ticket", END)
    builder.add_edge("end_conversation", END)
    
    # Compile with memory
    memory = InMemorySaver()
    return builder.compile(checkpointer=memory)

# ===== FastAPI App =====
app = FastAPI(title="Ticket Support API", version="1.0.0")
langgraph_app = create_langgraph_app()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logging.error(f"Unhandled error: {exc}\n{tb}")
    return JSONResponse(status_code=500, content={"detail": str(exc), "traceback": tb})

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
        
        # Show buttons when displaying other categories
        should_show_buttons = bool(
            vals.get("other_categories_shown") and 
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
        logging.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/new_conversation", response_model=NewConversationResponse)
async def new_conversation():
    try:
        thread_id = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": thread_id}}
        
        response = langgraph_app.invoke({}, config=cfg)
        
        return NewConversationResponse(
            thread_id=thread_id,
            message=response.get("last_message", "Welcome to support!")
        )
    except Exception as e:
        logging.error(f"New conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
