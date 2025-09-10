import os
import uuid
import json
from typing import Dict, List, Annotated, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from fastapi.middleware.cors import CORSMiddleware
 
# Load environment variables
load_dotenv()
 
app = FastAPI(title="AI Support Ticket System", version="1.0.0")
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# --- Constants ---
TIME_SLOT_OPTIONS = [
    "9 AM - 12 PM",
    "12 PM - 3 PM",
    "3 PM - 6 PM",
    "6 PM - 9 PM",
    "2 PM - 10 PM",
    "Flexible/Any time"
]
 
SOP_DB = {
    "payroll": "If your payroll issue is about salary delay, please check ESS portal first. Contact payroll team if issue persists.",
    "hr": "For HR policy related issues, check the HR handbook on the portal. Escalate to HR for complex matters.",
    "it": "For IT issues, reboot and check the IT self-service portal first.",
    "facilities": "For AC/cleaning/maintenance, coordinate with facilities.",
    "finance": "Finance queries require proper documentation and approvals.",
    "security": "For security issues, follow the security protocol and report immediately.",
    "training": "For training requests, check available courses in the learning portal first.",
    "travel": "For travel requests, ensure you have manager approval before booking.",
    "procurement": "For procurement requests, follow the standard purchase approval process.",
    "general": "For general inquiries, check the employee handbook first."
}
 
# Required fields for each category
MANDATORY_FIELDS = {
    cat: ["employee_id", "issue_description", "preferred_time_slot"]
    for cat in SOP_DB.keys()
}
 
# --- Pydantic Models ---
class ConversationRequest(BaseModel):
    user_input: str
    session_id: str
    ordered_categories: Optional[List[str]] = None
 
class ConversationResponse(BaseModel):
    message: str
    conversation_complete: bool
    session_id: str
 
# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
 
# --- Tools ---
@tool
def get_sop_and_ask_to_create_ticket(category: str) -> str:
    sop_text = SOP_DB.get(category, f"No SOP found for {category}")
    return f"📘 {category.title()} SOP:\n\n{sop_text}\n\nDo you want to raise a ticket for this issue? (yes/no)"
 
@tool
def ask_for_category_confirmation(category: str) -> str:
    if not category:
        return "There seems to be an issue. Could you please specify the category again?"
    return f"Do you want to confirm {category.title()} as your category? (yes/no)"
 
@tool
def ask_user_to_choose_category(ordered_categories: List[str]) -> str:
    opts = ", ".join([cat.title() for cat in ordered_categories])
    return f"Please type one of the following categories:\n{opts}"
 
@tool
def ask_for_time_slot_selection() -> str:
    slots = "\n".join([f"{i+1}. {slot}" for i, slot in enumerate(TIME_SLOT_OPTIONS)])
    return (
        f"Please select your preferred time slot by typing the number or the full time slot:\n\n"
        f"{slots}\n\nType the number (1-{len(TIME_SLOT_OPTIONS)}) or the full time slot text:"
    )
 
@tool
def ask_for_next_required_field(category: str, collected_fields: Dict[str, str]) -> str:
    if not category or category not in MANDATORY_FIELDS:
        return "I seem to have lost the category context. Could you please state the category again?"
 
    required = MANDATORY_FIELDS[category]
    for field in required:
        if field not in collected_fields:
            if field == "preferred_time_slot":
                return "Now I need to ask for your preferred time slot."
            else:
                return f"Please provide your {field.replace('_', ' ').title()}:"
 
    return "All required fields have been collected. Now, please show the ticket summary."
 
@tool
def show_ticket_summary_and_ask_for_final_confirmation(category: str, details: dict) -> str:
    summary = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in details.items()])
    return (
        f"**Ticket Summary**\n\nCategory: {category.title()}\n\nDetails:\n{summary}\n\n"
        f"Do you want to submit this ticket? (yes/no)"
    )
 
@tool
def submit_ticket(category: str, details: dict) -> str:
    ticket_id = f"TCKT-{uuid.uuid4().hex[:8].upper()}"
    return (
        f"✅ Ticket Created!\n\n"
        f"Ticket ID: {ticket_id}\n"
        f"Category: {category.title()}\n"
        f"Details: {json.dumps(details)}\n\n"
        f"You will receive updates via email. Thank you!"
    )
 
@tool
def end_conversation(message: str) -> str:
    return message
 
# --- LangGraph App Creation ---
def create_langgraph_app():
    """Creates and configures the ReAct agent orchestrator using Ilmv2."""
 
    system_prompt_template = f"""
    You are an expert AI assistant for a corporate support ticketing system.
    Your goal is to guide the user through creating a support ticket by calling the correct tools in sequence.
 
    Conversation Flow:
    1. Start: Always begin by calling `get_sop_and_ask_to_create_ticket` with the current category.
    2. If user says 'yes', call `ask_for_category_confirmation`.
    3. If user confirms, collect required fields in this order:
       - employee_id
       - issue_description
       - preferred_time_slot
    4. Use `ask_for_next_required_field` to prompt for the next field.
    5. For time slot, call `ask_for_time_slot_selection`.
    6. After all fields are collected, call `show_ticket_summary_and_ask_for_final_confirmation`.
    7. Based on final confirmation, call `submit_ticket` or `end_conversation`.
    8. If user denies initial ticket creation, call `end_conversation`.
 
    Rules:
    - Always call a tool at each step.
    - Maintain conversation state: category and collected fields.
    - Available time slots: {', '.join(TIME_SLOT_OPTIONS)}.
    """
 
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_template),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    Ilmv2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    # Replace Gemini LLM with Ilmv2
    # NOTE: Ensure Ilmv2 is imported and initialized in your project.
    agent_model = Ilmv2.bind_tools(tools, parallel_tool_calls=False)
 
    checkpointer = InMemorySaver()
 
    # Create the orchestrator directly with Ilmv2
    orchestrator = create_react_agent(
        model=agent_model,
        tools=tools,
        name="Orchestrator",
        prompt=prompt,
        checkpointer=checkpointer,
        store=None  # Replace with a persistent store if required
    )
 
    return orchestrator
 
# Initialize
tools = [
    get_sop_and_ask_to_create_ticket,
    ask_for_category_confirmation,
    ask_user_to_choose_category,
    ask_for_next_required_field,
    ask_for_time_slot_selection,
    show_ticket_summary_and_ask_for_final_confirmation,
    submit_ticket,
    end_conversation,
]
langgraph_app = create_langgraph_app()
active_sessions = {}
 
# --- API Endpoint ---
@app.post("/chat")
async def chat(request: ConversationRequest):
    try:
        session_id = request.session_id
        config = {"configurable": {"thread_id": session_id}}
 
        # New session
        if session_id not in active_sessions:
            if not request.ordered_categories:
                raise HTTPException(status_code=400, detail="ordered_categories required for new session")
 
            active_sessions[session_id] = True
 
            first_category = request.ordered_categories[0]
            initial_prompt = (
                f"A user has started the support ticket process. "
                f"The initial suggested category is '{first_category}'. "
                f"The full list of possible categories is {request.ordered_categories}. "
                f"Please begin the process by calling the first appropriate tool."
            )
            messages = [HumanMessage(content=initial_prompt)]
        else:
            messages = [HumanMessage(content=request.user_input)]
 
        response_state = langgraph_app.invoke({"messages": messages}, config=config)
 
        content = ""
        conversation_complete = False
        last_tool_message = None
 
        # Find last ToolMessage
        for msg in reversed(response_state['messages']):
            if isinstance(msg, ToolMessage):
                last_tool_message = msg
                break
 
        if last_tool_message:
            content = last_tool_message.content
            if last_tool_message.name in ["submit_ticket", "end_conversation"]:
                conversation_complete = True
        else:
            last_message = response_state['messages'][-1]
            if isinstance(last_message, AIMessage):
                content = last_message.content
            else:
                content = "An unexpected error occurred."
 
        if conversation_complete and session_id in active_sessions:
            del active_sessions[session_id]
 
        return ConversationResponse(
            message=content,
            conversation_complete=conversation_complete,
            session_id=session_id
        )
 
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
 
