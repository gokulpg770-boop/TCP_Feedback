import uuid
from typing import Optional, Dict, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# ===== State Definition =====
class TicketState(TypedDict, total=False):
    user_input: str
    last_message: str
    category: Optional[str]
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
def show_sop(state: TicketState) -> TicketState:
    new_state = dict(state)
    category = new_state.get("category", CATEGORY_OPTIONS[0])
    sop = SOP_DB.get(category, "No SOP available.")
    new_state["last_message"] = (
        f"📘 {category.title()} SOP:\n\n{sop}\n\n"
        "Do you want to raise a ticket for this issue? (yes/no)"
    )
    new_state["first_sop_shown"] = True
    new_state["awaiting_input"] = True  # Add this flag to indicate we're waiting for user input
    return new_state

def handle_ticket_decision(state: TicketState) -> TicketState:
    new_state = dict(state)
    response = new_state.get("user_input", "").strip().lower()
    if response in ("yes", "y"):
        new_state["wants_ticket"] = True
    elif response in ("no", "n"):
        new_state["wants_ticket"] = False
        new_state["conversation_complete"] = True
        new_state["last_message"] = "👍 Thank you! Have a great day."
    else:
        new_state["last_message"] = "Please respond with 'yes' or 'no'."
    return new_state

def ask_category_confirmation(state: TicketState) -> TicketState:
    new_state = dict(state)
    category = new_state.get("category", CATEGORY_OPTIONS[0])
    new_state["last_message"] = f"Do you want to confirm {category.title()} as your category? (yes/no)"
    return new_state

def handle_category_confirmation(state: TicketState) -> TicketState:
    new_state = dict(state)
    response = new_state.get("user_input", "").strip().lower()
    if response in ("yes", "y"):
        new_state["category_confirmed"] = True
    elif response in ("no", "n"):
        new_state["category_confirmed"] = False
    else:
        new_state["last_message"] = "Please respond with 'yes' or 'no'."
    return new_state

def show_other_categories(state: TicketState) -> TicketState:
    new_state = dict(state)
    new_state["other_categories_shown"] = True
    new_state["last_message"] = "Please select from the following categories:"
    return new_state

def handle_category_selection(state: TicketState) -> TicketState:
    new_state = dict(state)
    selected = new_state.get("user_input", "").strip().lower()
    remaining = CATEGORY_OPTIONS[1:]
    if selected in remaining:
        new_state["category"] = selected
        new_state["other_categories_shown"] = False
    else:
        new_state["last_message"] = "Please select a valid option from the list."
    return new_state

def get_next_field(state: TicketState) -> TicketState:
    new_state = dict(state)
    category = new_state.get("category")
    if not category:
        new_state["last_message"] = "Error: No category found."
        return new_state
    
    required = MANDATORY_FIELDS.get(category, [])
    filled = new_state.get("mandatory_fields", {})
    current_index = new_state.get("current_field_index", 0)
    
    if current_index < len(required):
        field = required[current_index]
        new_state["current_field"] = field
        new_state["last_message"] = f"Please provide your {field.replace('_', ' ').title()}:"
        new_state["current_field_index"] = current_index + 1
    else:
        new_state["current_field"] = None
        new_state["current_field_index"] = 0
    return new_state

def fill_field(state: TicketState) -> TicketState:
    new_state = dict(state)
    current_field = new_state.get("current_field")
    user_input = new_state.get("user_input", "")
    if current_field and user_input:
        filled = dict(new_state.get("mandatory_fields", {}))
        filled[current_field] = user_input
        new_state["mandatory_fields"] = filled
    new_state["user_input"] = None
    return new_state

def final_confirmation(state: TicketState) -> TicketState:
    new_state = dict(state)
    category = new_state["category"]
    details = new_state["mandatory_fields"]
    
    summary = ""
    for key, value in details.items():
        summary += f"- {key.replace('_', ' ').title()}: {value}\n"
    
    new_state["last_message"] = (
        f"**Ticket Summary**\n\n"
        f"Category: {category.title()}\n\n"
        f"Details:\n{summary}\n"
        "Do you want to submit this ticket? (yes/no)"
    )
    return new_state

def handle_final_confirmation(state: TicketState) -> TicketState:
    new_state = dict(state)
    response = new_state.get("user_input", "").strip().lower()
    if response in ("yes", "y"):
        new_state = create_ticket(new_state)
    elif response in ("no", "n"):
        new_state["conversation_complete"] = True
        new_state["last_message"] = "Ticket submission cancelled. Thank you!"
    else:
        new_state["last_message"] = "Please respond with 'yes' or 'no'."
    new_state["user_input"] = None
    return new_state

def create_ticket(state: TicketState) -> TicketState:
    new_state = dict(state)
    ticket_id = f"TCKT-{uuid.uuid4().hex[:8].upper()}"
    category = new_state["category"]
    details = new_state["mandatory_fields"]

    message = (
        f"✅ **Ticket Created Successfully!**\n\n"
        f"**Ticket ID:** {ticket_id}\n"
        f"**Category:** {category.title()}\n"
        f"**Details:** {details}\n\n"
        "You will receive updates via email. Thank you!"
    )
    new_state["last_message"] = message
    new_state["conversation_complete"] = True
    new_state["ticket_confirmed"] = True  # Ensure ticket_confirmed is set
    return new_state

def end_conversation(state: TicketState) -> TicketState:
    new_state = dict(state)
    new_state["conversation_complete"] = True
    new_state["last_message"] = "👍 Thank you! Have a great day."
    return new_state

# ===== Routing =====
def route(state: TicketState) -> str:
    # End graph execution if we're waiting for user input
    if state.get("awaiting_input"):
        return END
        
    if state.get("conversation_complete"):
        return END

    if not state.get("first_sop_shown"):
        return "show_sop"

    if state.get("wants_ticket") is None and state.get("user_input"):
        return "handle_ticket_decision"

    if state.get("wants_ticket") is False:
        return "end_conversation"

    if state.get("category_confirmed") is None:
        if state.get("other_categories_shown"):
            return "handle_category_selection"
        if state.get("user_input"):
            return "handle_category_confirmation"
        return "ask_category_confirmation"

    if state.get("category_confirmed") is False:
        return "show_other_categories"

    category = state.get("category")
    if not category:
        return "show_other_categories"

    required = MANDATORY_FIELDS.get(category, [])
    filled = state.get("mandatory_fields", {})

    if len(filled) < len(required):
        if state.get("user_input"):
            return "fill_field"
        return "get_next_field"

    if not state.get("ticket_confirmed"):
        if state.get("user_input"):
            return "handle_final_confirmation"
        return "final_confirmation"
    
    return END


# ===== LangGraph Setup (Exported Function) =====
def create_langgraph_app():
    builder = StateGraph(TicketState)

    builder.add_node("show_sop", show_sop)
    builder.add_node("handle_ticket_decision", handle_ticket_decision)
    builder.add_node("ask_category_confirmation", ask_category_confirmation)
    builder.add_node("handle_category_confirmation", handle_category_confirmation)
    builder.add_node("show_other_categories", show_other_categories)
    builder.add_node("handle_category_selection", handle_category_selection)
    builder.add_node("get_next_field", get_next_field)
    builder.add_node("fill_field", fill_field)
    builder.add_node("final_confirmation", final_confirmation)
    builder.add_node("handle_final_confirmation", handle_final_confirmation)
    builder.add_node("create_ticket", create_ticket)
    builder.add_node("end_conversation", end_conversation)

    builder.set_conditional_entry_point(route)

    for node in [
        "show_sop", "handle_ticket_decision", "ask_category_confirmation",
        "handle_category_confirmation", "show_other_categories", "handle_category_selection",
        "get_next_field", "fill_field", "final_confirmation", "handle_final_confirmation"
    ]:
        builder.add_conditional_edges(node, route)

    builder.add_edge("create_ticket", END)
    builder.add_edge("end_conversation", END)

    memory = InMemorySaver()
    return builder.compile(checkpointer=memory)



/////////////////////////


import streamlit as st
import uuid
from backend_v2 import create_langgraph_app , CATEGORY_OPTIONS  

# Create LangGraph app (imported from backend)
langgraph_app = create_langgraph_app()

def main():
    st.set_page_config(page_title="Support Ticket System", page_icon="🎫", layout="wide")
    st.title("🎫 AI Support Ticket System")
    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    with st.sidebar:
        st.header("🎛️ Controls")
        if st.button("🔄 New Conversation", type="primary"):
            start_new_conversation()

        st.markdown("---")
        st.markdown("### 📋 Flow")
        st.markdown("""
        1) **Payroll SOP** → "Raise ticket?" (yes/no)
        2) **If yes** → "Confirm Payroll?" (yes/no)
           - **Yes** → Collect fields → Final confirm → Create ticket
           - **No** → Show other categories → Confirm → Collect → Final confirm → Create
        3) **If no** → Exit
        """)

    chat_container = st.container()
    for message in st.session_state.messages:
        with chat_container:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if not st.session_state.conversation_started:
        start_new_conversation()

    if st.session_state.thread_id:
        handle_conversation_interface()

def start_new_conversation():
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.conversation_started = False
    st.session_state.last_response = None
    
    with st.spinner("Starting new conversation..."):
        cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
        response = langgraph_app.invoke({}, config=cfg)
        st.session_state.messages.append({"role": "assistant", "content": response.get("last_message", "Welcome!")})
        st.session_state.conversation_started = True
        st.session_state.last_response = response
        st.rerun()

def handle_conversation_interface():
    data = st.session_state.last_response
    
    if data and data.get("other_categories_shown") and data.get("category_confirmed") is None:
        display_category_buttons(CATEGORY_OPTIONS[1:])
    elif data and data.get("conversation_complete"):
        st.success("✅ Conversation completed!")
        if st.button("🔄 Start New Conversation", type="primary"):
            start_new_conversation()
    else:
        handle_chat_input()

def display_category_buttons(remaining_options):
    st.markdown("### 📝 Select a Category:")
    
    for i in range(0, len(remaining_options), 3):
        cols = st.columns(3)
        for j, option in enumerate(remaining_options[i:i+3]):
            with cols[j]:
                if st.button(f"🎯 {option.title()}", key=f"btn_{option}"):
                    process_user_input(f"Selected: {option.title()}", option)

def handle_chat_input():
    if prompt := st.chat_input("Type your message here..."):
        process_user_input(prompt, prompt)

def process_user_input(display_text: str, actual_input: str):
    st.session_state.messages.append({"role": "user", "content": display_text})

    with st.spinner("Processing..."):
        cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
        input_data = {"user_input": actual_input}
        response = langgraph_app.invoke(input_data, config=cfg)
        st.session_state.last_response = response
        st.session_state.messages.append({"role": "assistant", "content": response.get("last_message", "I'm sorry, I didn't understand that.")})
        
        if response.get("conversation_complete"):
            st.balloons()
        
        st.rerun()

if __name__ == "__main__":
    main()
///////////////////////////////////////////



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Ticket Support API")

# In-memory state storage (for simplicity; replace with database for production)
states: Dict[str, Dict] = {}

# ===== Models =====
class ChatRequest(BaseModel):
    thread_id: str
    user_input: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    should_show_buttons: bool = False
    remaining_options: List[str] = []
    conversation_complete: bool = False

class NewConversationResponse(BaseModel):
    thread_id: str
    message: str

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

# ===== Helper Functions =====
def get_state(thread_id: str) -> Dict:
    return states.get(thread_id, {})

def update_state(thread_id: str, new_state: Dict):
    states[thread_id] = new_state

def process_conversation(thread_id: str, user_input: Optional[str] = None) -> Dict:
    state = get_state(thread_id)
    
    # Initialize if new
    if not state:
        state = {
            "step": "show_sop",
            "category": "payroll",
            "wants_ticket": None,
            "category_confirmed": None,
            "mandatory_fields": {},
            "current_field_index": 0,
            "conversation_complete": False,
            "show_buttons": False,
            "last_message": ""
        }
    
    # Set user input if provided
    if user_input is not None:
        state["user_input"] = user_input

    step = state["step"]
    
    if step == "show_sop":
        sop = SOP_DB[state["category"]]
        message = (
            f"📘 {state['category'].title()} SOP:\n\n{sop}\n\n"
            "Do you want to raise a ticket for this issue? (yes/no)"
        )
        state["last_message"] = message
        state["step"] = "ask_ticket"

    elif step == "ask_ticket" and state.get("user_input"):
        response = state["user_input"].strip().lower()
        if response in ("yes", "y"):
            state["wants_ticket"] = True
            state["step"] = "ask_confirmation"
            state["last_message"] = f"Do you want to confirm **{state['category'].title()}** as your category? (yes/no)"
            state["user_input"] = None
        elif response in ("no", "n"):
            state["wants_ticket"] = False
            state["step"] = "end"
            state["conversation_complete"] = True
            state["last_message"] = "👍 Thank you! Have a great day."
            state["user_input"] = None
        else:
            state["last_message"] = "Please respond with 'yes' or 'no'."

    elif step == "ask_confirmation" and state.get("user_input"):
        response = state["user_input"].strip().lower()
        if response in ("yes", "y"):
            state["category_confirmed"] = True
            state["step"] = "collect_fields"
            state["current_field_index"] = 0
            state["mandatory_fields"] = {}
            state["user_input"] = None
            state = collect_fields(state)
        elif response in ("no", "n"):
            state["category_confirmed"] = False
            state["step"] = "show_other_categories"
            state["show_buttons"] = True
            state["last_message"] = "Please select from the following categories:"
            state["user_input"] = None
        else:
            state["last_message"] = "Please respond with 'yes' or 'no'."

    elif step == "show_other_categories" and state.get("user_input"):
        selected = state["user_input"].strip().lower()
        remaining = CATEGORY_OPTIONS[1:]
        if selected in remaining:
            state["category"] = selected
            state["step"] = "ask_confirmation"
            state["show_buttons"] = False
            state["last_message"] = f"Do you want to confirm **{selected.title()}** as your category? (yes/no)"
            state["user_input"] = None
        else:
            state["last_message"] = "Please select a valid option from the list."

    elif step == "collect_fields":
        required = MANDATORY_FIELDS[state["category"]]
        
        if state.get("user_input"):
            # Fill the last asked field
            current_field = required[state["current_field_index"] - 1]
            state["mandatory_fields"][current_field] = state["user_input"]
            state["user_input"] = None
        
        if state["current_field_index"] < len(required):
            field = required[state["current_field_index"]]
            state["last_message"] = f"Please provide your **{field.replace('_', ' ').title()}**:"
            state["current_field_index"] += 1
        else:
            # All fields collected - go to final confirmation
            state["step"] = "final_confirmation"
            summary = "\n".join([f"- **{k.replace('_', ' ').title()}:** {v}" for k, v in state["mandatory_fields"].items()])
            state["last_message"] = (
                f"**Ticket Summary:**\n\n"
                f"**Category:** {state['category'].title()}\n\n"
                f"**Details:**\n{summary}\n\n"
                "Confirm and submit ticket? (yes/no)"
            )

    elif step == "final_confirmation" and state.get("user_input"):
        response = state["user_input"].strip().lower()
        if response in ("yes", "y"):
            state = create_ticket(state)
            state["user_input"] = None
        elif response in ("no", "n"):
            state["step"] = "end"
            state["conversation_complete"] = True
            state["last_message"] = "Ticket submission cancelled. Thank you!"
            state["user_input"] = None
        else:
            state["last_message"] = "Please respond with 'yes' or 'no'."

    update_state(thread_id, state)
    return state

def collect_fields(state: Dict) -> Dict:
    """Helper to collect fields"""
    required = MANDATORY_FIELDS[state["category"]]
    
    if state["current_field_index"] < len(required):
        field = required[state["current_field_index"]]
        state["last_message"] = f"Please provide your **{field.replace('_', ' ').title()}**:"
        state["current_field_index"] += 1
    else:
        # All fields collected - go to final confirmation
        state["step"] = "final_confirmation"
        summary = "\n".join([f"- **{k.replace('_', ' ').title()}:** {v}" for k, v in state["mandatory_fields"].items()])
        state["last_message"] = (
            f"**Ticket Summary:**\n\n"
            f"**Category:** {state['category'].title()}\n\n"
            f"**Details:**\n{summary}\n\n"
            "Confirm and submit ticket? (yes/no)"
        )
    
    return state

def create_ticket(state: Dict) -> Dict:
    """Helper to create ticket"""
    ticket_id = f"TCKT-{uuid.uuid4().hex[:8].upper()}"
    category = state["category"]
    details = state["mandatory_fields"]
    
    message = (
        f"✅ **Ticket Created Successfully!**\n\n"
        f"**Ticket ID:** {ticket_id}\n"
        f"**Category:** {category.title()}\n"
        f"**Details:** {details}\n\n"
        "You will receive updates via email. Thank you!"
    )
    
    state["last_message"] = message
    state["conversation_complete"] = True
    return state

# ===== Endpoints =====
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    state = process_conversation(request.thread_id, request.user_input)
    
    should_show_buttons = bool(
        state.get("step") == "show_other_categories" and
        not state.get("user_input")
    )
    
    return ChatResponse(
        message=state["last_message"],
        should_show_buttons=should_show_buttons,
        remaining_options=CATEGORY_OPTIONS[1:] if should_show_buttons else [],
        conversation_complete=state["conversation_complete"]
    )

@app.post("/new_conversation", response_model=NewConversationResponse)
async def new_conversation():
    thread_id = str(uuid.uuid4())
    state = process_conversation(thread_id)
    
    return NewConversationResponse(
        thread_id=thread_id,
        message=state["last_message"]
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
///////
import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000"

def main():
    st.set_page_config(page_title="Support Ticket System", page_icon="🎫", layout="wide")
    st.title("🎫 AI Support Ticket System")
    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False
    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    with st.sidebar:
        st.header("🎛️ Controls")
        if st.button("🔄 New Conversation", type="primary"):
            start_new_conversation()

        st.markdown("---")
        st.markdown("### 🔌 API Status")
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ Connected")
            else:
                st.error("❌ API Error")
        except:
            st.error("❌ API Offline")

        st.markdown("---")
        st.markdown("### 📋 Flow")
        st.markdown("""
        1) **Payroll SOP** → "Raise ticket?" (yes/no)
        2) **If yes** → "Confirm Payroll?" (yes/no)
           - **Yes** → Collect fields → Final confirm → Create ticket
           - **No** → Show other categories → Confirm → Collect → Final confirm → Create
        3) **If no** → Exit
        """)

    chat_container = st.container()
    for message in st.session_state.messages:
        with chat_container:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if not st.session_state.conversation_started:
        start_new_conversation()

    if st.session_state.thread_id:
        handle_conversation_interface()

def start_new_conversation():
    st.session_state.messages = []
    st.session_state.thread_id = None
    st.session_state.conversation_started = False
    st.session_state.last_response = None
    
    with st.spinner("Starting new conversation..."):
        try:
            response = requests.post(f"{API_BASE_URL}/new_conversation")
            if response.status_code == 200:
                data = response.json()
                st.session_state.thread_id = data["thread_id"]
                st.session_state.messages.append({"role": "assistant", "content": data["message"]})
                st.session_state.conversation_started = True
                st.session_state.last_response = data
                st.rerun()
            else:
                st.error("Failed to start conversation")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")

def handle_conversation_interface():
    data = st.session_state.last_response
    
    if data and data.get("should_show_buttons"):
        display_category_buttons(data.get("remaining_options", []))
    elif data and data.get("conversation_complete"):
        st.success("✅ Conversation completed!")
        if st.button("🔄 Start New Conversation", type="primary"):
            start_new_conversation()
    else:
        handle_chat_input()

def display_category_buttons(remaining_options):
    st.markdown("### 📝 Select a Category:")
    
    for i in range(0, len(remaining_options), 3):
        cols = st.columns(3)
        for j, option in enumerate(remaining_options[i:i+3]):
            with cols[j]:
                if st.button(f"🎯 {option.title()}", key=f"btn_{option}"):
                    process_user_input(f"Selected: {option.title()}", option)

def handle_chat_input():
    if prompt := st.chat_input("Type your message here..."):
        process_user_input(prompt, prompt)

def process_user_input(display_text: str, actual_input: str):
    st.session_state.messages.append({"role": "user", "content": display_text})

    with st.spinner("Processing..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={"thread_id": st.session_state.thread_id, "user_input": actual_input},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.last_response = data
                st.session_state.messages.append({"role": "assistant", "content": data["message"]})
                
                if data["conversation_complete"]:
                    st.balloons()
                
                st.rerun()
            else:
                st.error("Failed to process message")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")

if __name__ == "__main__":
    main()
