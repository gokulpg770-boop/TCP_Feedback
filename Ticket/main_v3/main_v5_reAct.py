#!/usr/bin/env python3
"""
AI Support Ticket System - With Time Slot Selection
Refactored to use a LangGraph ReAct Agent orchestrator.
"""
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
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
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
# Time slot options for all categories
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

# Updated MANDATORY_FIELDS - All categories have common attributes: employee_id, issue_description, preferred_time_slot
MANDATORY_FIELDS = {
    "payroll": ["employee_id", "issue_description", "preferred_time_slot"],
    "hr": ["employee_id", "issue_description", "preferred_time_slot"],
    "it": ["employee_id", "issue_description", "preferred_time_slot"],
    "facilities": ["employee_id", "issue_description", "preferred_time_slot"],
    "finance": ["employee_id", "issue_description", "preferred_time_slot"],
    "security": ["employee_id", "issue_description", "preferred_time_slot"],
    "training": ["employee_id", "issue_description", "preferred_time_slot"],
    "travel": ["employee_id", "issue_description", "preferred_time_slot"],
    "procurement": ["employee_id", "issue_description", "preferred_time_slot"],
    "general": ["employee_id", "issue_description", "preferred_time_slot"],
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
# Note: While the state is managed by the ReAct agent, this defines its structure for the checkpointer.
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# --- Tools (Unchanged) ---
@tool
def get_sop_and_ask_to_create_ticket(category: str) -> str:
    """Gets the Standard Operating Procedure (SOP) for a given category and asks the user if they want to proceed with creating a ticket."""
    sop_text = SOP_DB.get(category, f"No SOP found for {category}")
    return f"📘 {category.title()} SOP:\n\n{sop_text}\n\nDo you want to raise a ticket for this issue? (yes/no)"

@tool
def ask_for_category_confirmation(category: str) -> str:
    """Asks the user to confirm the selected support category."""
    if not category:
        return "There seems to be an issue. Could you please specify the category again?"
    return f"Do you want to confirm {category.title()} as your category? (yes/no)"

@tool
def ask_user_to_choose_category(ordered_categories: List[str]) -> str:
    """Asks the user to choose a category from the available options."""
    opts = ", ".join([cat.title() for cat in ordered_categories])
    return f"Please type one of the following categories:\n{opts}"

@tool
def ask_for_time_slot_selection() -> str:
    """Asks the user to select their preferred time slot from available options."""
    slots = "\n".join([f"{i+1}. {slot}" for i, slot in enumerate(TIME_SLOT_OPTIONS)])
    return f"Please select your preferred time slot by typing the number or the full time slot:\n\n{slots}\n\nType the number (1-{len(TIME_SLOT_OPTIONS)}) or the full time slot text:"

@tool
def ask_for_next_required_field(category: str, collected_fields: Dict[str, str]) -> str:
    """Asks the user for the next required field for the given category that has not been collected yet."""
    if not category or category not in MANDATORY_FIELDS:
        return "I seem to have lost the category context. Could you please state the category again?"
    
    required = MANDATORY_FIELDS[category]
    for field in required:
        if field not in collected_fields:
            if field == "preferred_time_slot":
                # With a ReAct agent, we rely on its reasoning to call the time slot tool next.
                return "Now I need to ask for your preferred time slot."
            else:
                return f"Please provide your {field.replace('_', ' ').title()}:"
    
    return "All required fields have been collected. Now, please show the ticket summary."

@tool
def show_ticket_summary_and_ask_for_final_confirmation(category: str, details: dict) -> str:
    """Displays a summary of the ticket and asks for final user confirmation."""
    summary = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in details.items()])
    return f"**Ticket Summary**\n\nCategory: {category.title()}\n\nDetails:\n{summary}\n\nDo you want to submit this ticket? (yes/no)"

@tool
def submit_ticket(category: str, details: dict) -> str:
    """Submits the ticket and provides a confirmation message to the user. This is a final step."""
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
    """Ends the conversation with a final message to the user. This is a final step."""
    return message

# --- LangGraph App Creation ---
def create_langgraph_app():
    """Creates and configures the ReAct agent orchestrator."""
    
    # This detailed prompt is crucial for guiding the ReAct agent.
    # It replaces the hard-coded logic of the previous StateGraph.
    system_prompt_template = f"""You are an expert AI assistant for a corporate support ticketing system. Your goal is to guide the user through creating a support ticket by calling the correct tools in sequence.

**Conversation Flow & Tool Usage:**
1.  **Start:** The conversation begins with a pre-selected category. Your first action is ALWAYS to call `get_sop_and_ask_to_create_ticket` with the current category.
2.  **User wants to create ticket (e.g., says "yes"):** Your next action MUST BE `ask_for_category_confirmation`.
3.  **User confirms category ("yes"):** Start collecting the required fields: `employee_id`, `issue_description`, and `preferred_time_slot`.
    - To ask for the next field, call `ask_for_next_required_field`. You MUST keep track of which fields you have already collected from the conversation history.
    - When the response from `ask_for_next_required_field` indicates it's time to collect the time slot, you MUST call `ask_for_time_slot_selection`.
4.  **User denies category ("no"):** Your next action is `ask_user_to_choose_category`.
5.  **Collecting Information:** You must remember the user's answers (their employee ID, issue description, etc.) from the conversation to use as arguments in later tool calls.
6.  **Final Confirmation:** Once all fields are collected, you MUST call `show_ticket_summary_and_ask_for_final_confirmation`. You MUST construct the `details` dictionary argument for this tool using the information you have collected from the user.
7.  **Submit or Cancel:** Based on the user's final confirmation, call either `submit_ticket` (again, passing the `category` and `details` dictionary) or `end_conversation`.
8.  **User denies initial ticket creation:** If the user says "no" at the very beginning, call `end_conversation`.

**IMPORTANT RULES:**
- You MUST call a tool at every step. Do not respond directly to the user.
- You are responsible for maintaining the state of the conversation, including the selected category and all collected fields.
- When calling a tool that requires arguments like `category` or `details`, you MUST provide them based on the conversation history.
- The available time slots are: {', '.join(TIME_SLOT_OPTIONS)}.
"""

    tools = [
        get_sop_and_ask_to_create_ticket, ask_for_category_confirmation,
        ask_user_to_choose_category, ask_for_next_required_field,
        ask_for_time_slot_selection,
        show_ticket_summary_and_ask_for_final_confirmation, submit_ticket,
        end_conversation,
    ]
    
    # Create the prompt template for the ReAct agent.
    # Note: We do NOT include an 'agent_scratchpad' placeholder.
    # The ReAct agent implementation in langgraph manages the scratchpad
    # as part of the main 'messages' list.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_template),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    # IMPORTANT: Replace ChatGoogleGenerativeAI with your actual Ilmv2 model.
    # This is a placeholder to make the code runnable.
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # Chain the prompt with the model, and bind the tools to the model.
    # This combined runnable will be the agent's model.
    agent_model = prompt | llm.bind_tools(tools, parallel_tool_calls=False)
    
    checkpointer = InMemorySaver()

    # Create the ReAct Agent orchestrator
    # Pass the chained model, not the raw LLM and a separate prompt.
    orchestrator = create_react_agent(
        model=agent_model,
        tools=tools,
        checkpointer=checkpointer,
    )
    return orchestrator

# Global app instance and session storage
langgraph_app = create_langgraph_app()
active_sessions = {}

# --- API Endpoints ---
@app.post("/chat")
async def chat(request: ConversationRequest):
    """Single endpoint for conversation handling with the ReAct agent."""
    try:
        session_id = request.session_id
        config = {"configurable": {"thread_id": session_id}}

        # Initialize session if new
        if session_id not in active_sessions:
            if not request.ordered_categories:
                raise HTTPException(status_code=400, detail="ordered_categories required for new session")
            
            active_sessions[session_id] = True  # Mark session as active
            
            # Prepare an initial message to kick-start the agent
            first_category = request.ordered_categories[0]
            initial_prompt = (
                f"A user has started the support ticket process. "
                f"The initial suggested category is '{first_category}'. "
                f"The full list of possible categories is {request.ordered_categories}. "
                f"Please begin the process by calling the first appropriate tool."
            )
            messages = [HumanMessage(content=initial_prompt)]
        else:
            # Continue conversation
            messages = [HumanMessage(content=request.user_input)]
        
        # Invoke the agent
        response_state = langgraph_app.invoke({"messages": messages}, config=config)
        
        # Extract the correct response to send back to the user.
        # The most reliable output is the content of the last tool call.
        content = ""
        conversation_complete = False
        last_tool_message = None

        # Find the last ToolMessage in the response state
        for msg in reversed(response_state['messages']):
            if isinstance(msg, ToolMessage):
                last_tool_message = msg
                break
        
        if last_tool_message:
            # This is the standard case: the agent called a tool.
            content = last_tool_message.content
            if last_tool_message.name in ["submit_ticket", "end_conversation"]:
                conversation_complete = True
        else:
            # Fallback: the agent responded directly without a tool.
            # We show the message but DO NOT end the conversation. This prevents premature endings.
            last_message = response_state['messages'][-1]
            if isinstance(last_message, AIMessage):
                content = last_message.content
            else:
                content = "An unexpected error occurred."

        # Clean up completed conversations
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


