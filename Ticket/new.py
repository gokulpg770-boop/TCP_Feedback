#!/usr/bin/env python3
"""
Terminal client for AI Support Ticket System - Correct Flow
"""
import requests
import uuid

class TicketSystemClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = str(uuid.uuid4())
    
    def start_conversation(self, ordered_categories):
        print(f"🚀 Starting conversation with ordered categories: {', '.join(ordered_categories)}")
        print("=" * 60)
        
        try:
            # Initialize conversation
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "user_input": "start",
                    "session_id": self.session_id,
                    "ordered_categories": ordered_categories
                }
            )
            
            if response.status_code != 200:
                print(f"Error: {response.text}")
                return
            
            data = response.json()
            print(f"🤖 AI Assistant: {data['message']}")
            
            # Continue conversation
            while not data.get('conversation_complete', False):
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                response = requests.post(
                    f"{self.base_url}/chat",
                    json={
                        "user_input": user_input,
                        "session_id": self.session_id
                    }
                )
                
                if response.status_code != 200:
                    print(f"Error: {response.text}")
                    break
                
                data = response.json()
                print(f"\n🤖 AI Assistant: {data['message']}")
            
            print("\n✅ Conversation completed!")
                
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to server. Make sure main.py is running.")
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    print("🎯 AI Support Ticket System - Correct SOP Flow")
    print("=" * 60)
    print("Enter categories in order (e.g., it,hr,payroll)")
    
    while True:
        categories_input = input("📝 Enter ordered categories (comma-separated): ").strip()
        
        if not categories_input:
            print("❌ Please enter at least one category.")
            continue
        
        categories = [cat.strip().lower() for cat in categories_input.split(',')]
        
        if not categories:
            print("❌ Please enter valid categories.")
            continue
        
        print(f"✅ Will process categories in order: {', '.join(categories)}")
        break
    
    client = TicketSystemClient()
    client.start_conversation(categories)

if __name__ == "__main__":
    main()
////////////////////////////
#!/usr/bin/env python3
"""
AI Support Ticket System - With Time Slot Selection
"""
import os
import uuid
import json
from typing import Dict, List, Annotated, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
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
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    category: str
    ordered_categories: List[str]
    collected_fields: Dict[str, str]
    final_output: dict

# --- Tools ---
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
                return "TIME_SLOT_REQUIRED"  # Signal that time slot selection is needed
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
        f"Details: {details}\n\n"
        f"You will receive updates via email. Thank you!"
    )

@tool
def end_conversation(message: str) -> str:
    """Ends the conversation with a final message to the user. This is a final step."""
    return message

# --- LangGraph App Creation ---
def create_langgraph_app():
    system_prompt = f"""You are an expert AI assistant for a corporate support ticketing system. Your goal is to guide the user through creating a support ticket by calling the correct tools in sequence.
    
    **Available Time Slots:** {', '.join(TIME_SLOT_OPTIONS)}
    
    **State Awareness:**
    At the beginning of each turn, you are given the CURRENT `category`, `ordered_categories`, and `collected_fields`.
    You MUST use these CURRENT values when calling tools. 
    
    **Conversation Flow & Tool Usage:**
    1.  **Start:** The conversation begins. Your first action is ALWAYS to call `get_sop_and_ask_to_create_ticket` with the current `category`.
    2.  **User wants to create ticket (e.g., says "yes" to the first prompt):** Your next action MUST BE `ask_for_category_confirmation`. Do NOT call any other tool at this stage.
    3.  **User confirms category ("yes"):** Start collecting fields by calling `ask_for_next_required_field`.
    4.  **User denies category ("no"):** Your next action is `ask_user_to_choose_category` with ordered_categories.
    5.  **User provides a new category:** Update the state and then go back to step 2 to confirm the new category.
    6.  **Collecting Fields:** After the user provides a value for a field, call `ask_for_next_required_field` again to get the next one.
    7.  **Time Slot Selection:** When you need to collect the time slot, call `ask_for_time_slot_selection`.
    8.  **Final Confirmation:** Once all fields are collected, call `show_ticket_summary_and_ask_for_final_confirmation`.
    9.  **Submit or Cancel:** Based on user input, call either `submit_ticket` or `end_conversation`.
    10. **User denies initial ticket creation:** If the user says "no" at step 1, call `end_conversation`.
    
    **IMPORTANT RULES:**
    - You MUST call a tool at every step. Do not respond directly.
    - For time slot selection, users can provide either the number (1-{len(TIME_SLOT_OPTIONS)}) or the full text.
    - `submit_ticket` and `end_conversation` are terminal steps.
    """
    
    system_message = SystemMessage(content=system_prompt)
    tools = [
        get_sop_and_ask_to_create_ticket, ask_for_category_confirmation,
        ask_user_to_choose_category, ask_for_next_required_field,
        ask_for_time_slot_selection,  # Added back time slot tool
        show_ticket_summary_and_ask_for_final_confirmation, submit_ticket,
        end_conversation,
    ]
    tool_node = ToolNode(tools)
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)
    
    def agent_node(state: AgentState):
        """The primary node that calls the LLM to decide the next action."""
        # Check if we need to trigger time slot selection
        last_tool_message = None
        for msg in reversed(state.get('messages', [])):
            if isinstance(msg, ToolMessage):
                last_tool_message = msg
                break
        
        # If the last tool message indicates time slot is required, call the time slot tool
        if (last_tool_message and 
            last_tool_message.name == 'ask_for_next_required_field' and 
            "TIME_SLOT_REQUIRED" in last_tool_message.content):
            
            # Directly call time slot selection tool
            response = AIMessage(
                content="",
                tool_calls=[{
                    'name': 'ask_for_time_slot_selection',
                    'args': {},
                    'id': f'call_{uuid.uuid4().hex[:8]}'
                }]
            )
            return {"messages": [response]}
        
        # Regular agent logic (similar to your working code)
        state_context = (
            f"This is your current memory of the conversation. Use it to decide your next action:\n"
            f"<state>\n"
            f"  <category>{state.get('category')}</category>\n"
            f"  <ordered_categories>{state.get('ordered_categories', [])}</ordered_categories>\n"
            f"  <collected_fields>{json.dumps(state.get('collected_fields', {}))}</collected_fields>\n"
            f"</state>"
        )
        
        messages_for_llm = [
            system_message,
            HumanMessage(content=state_context, name="StateContext"),
        ]
        messages_for_llm.extend(state["messages"])
        
        response = model.invoke(messages_for_llm)
        return {"messages": [response]}
    
    def state_updater_node(state: AgentState):
        """Updates the state based on the conversation."""
        messages = state['messages']
        
        last_human_message = None
        prompting_ai_call = None
        
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage) and messages[i].name != "StateContext":
                last_human_message = messages[i]
                for j in range(i - 1, -1, -1):
                    if isinstance(messages[j], AIMessage) and messages[j].tool_calls:
                        prompting_ai_call = messages[j]
                        break
                break
        
        if not last_human_message or not prompting_ai_call:
            return {}
        
        user_input = last_human_message.content.strip()
        prompting_tool_name = prompting_ai_call.tool_calls[0]['name']
        category = state.get("category")
        ordered_categories = state.get("ordered_categories", [])
        collected_fields = state.get("collected_fields", {}).copy()
        
        # Handle category selection from ask_user_to_choose_category
        if prompting_tool_name == 'ask_user_to_choose_category':
            for cat in ordered_categories:
                if user_input.lower() == cat.lower():
                    category = cat.lower()
                    collected_fields = {}  # Reset fields for new category
                    break
        
        # Handle time slot selection
        elif prompting_tool_name == 'ask_for_time_slot_selection':
            try:
                # Check if user provided a number
                if user_input.isdigit():
                    slot_index = int(user_input) - 1
                    if 0 <= slot_index < len(TIME_SLOT_OPTIONS):
                        collected_fields["preferred_time_slot"] = TIME_SLOT_OPTIONS[slot_index]
                else:
                    # Check if user provided the full text or partial match
                    user_input_lower = user_input.lower()
                    for slot in TIME_SLOT_OPTIONS:
                        if user_input_lower == slot.lower() or user_input in slot:
                            collected_fields["preferred_time_slot"] = slot
                            break
            except (ValueError, IndexError):
                pass  # Invalid input, don't update field
        
        # Handle field collection  
        elif prompting_tool_name == 'ask_for_next_required_field':
            required = MANDATORY_FIELDS.get(category, [])
            for field in required:
                if field not in collected_fields and field != "preferred_time_slot":
                    collected_fields[field] = user_input
                    break
        
        return {
            "category": category, 
            "ordered_categories": ordered_categories,
            "collected_fields": collected_fields
        }
    
    def output_formatter_node(state: AgentState):
        """Formats the final output."""
        last_message = state["messages"][-1] if state["messages"] else None
        content = ""
        is_complete = False
        
        last_tool_message = None
        for msg in reversed(state.get('messages', [])):
            if isinstance(msg, ToolMessage):
                last_tool_message = msg
                break
        
        if last_tool_message:
            content = last_tool_message.content
            if last_tool_message.name in ["submit_ticket", "end_conversation"]:
                is_complete = True
        elif isinstance(last_message, AIMessage) and not last_message.tool_calls:
            content = last_message.content
            is_complete = True
        
        return {"final_output": {"last_message": content, "conversation_complete": is_complete}}
    
    checkpointer = InMemorySaver()
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("state_updater", state_updater_node)
    builder.add_node("output_formatter", output_formatter_node)
    
    builder.set_entry_point("agent")
    
    def route_after_agent(state: AgentState):
        """Decide whether to execute tools or to end the conversation."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "output_formatter"
    
    builder.add_conditional_edges("agent", route_after_agent)
    builder.add_edge("tools", "state_updater")
    builder.add_edge("state_updater", "agent")
    builder.add_edge("output_formatter", END)
    
    graph = builder.compile(checkpointer=checkpointer)
    
    class App:
        """Wrapper class for handling state persistence."""
        def invoke(self, input_dict, config=None):
            if "phase" in input_dict and input_dict["phase"] == "init":
                ordered_categories = input_dict.get("ordered_categories", ["payroll"])
                first_category = ordered_categories[0] if ordered_categories else "payroll"
                initial_message = HumanMessage(content=f"The user has started the ticket creation process. The initial category is '{first_category}'. Please begin.")
                input_for_graph = {
                    "category": first_category, 
                    "ordered_categories": ordered_categories,
                    "collected_fields": {}, 
                    "messages": [initial_message]
                }
            else:
                input_for_graph = {"messages": [HumanMessage(content=input_dict['user_input'])]}
            
            final_state = graph.invoke(input_for_graph, config=config)
            output = output_formatter_node(final_state)
            return output.get("final_output")
    
    return App()

# Global app instance and session storage
langgraph_app = create_langgraph_app()
active_sessions = {}

# --- API Endpoints ---
@app.get("/time-slots")
async def get_time_slots():
    """Get available time slot options."""
    return {"time_slots": TIME_SLOT_OPTIONS}

@app.post("/chat")
async def chat(request: ConversationRequest):
    """Single endpoint for conversation handling"""
    try:
        session_id = request.session_id
        
        # Initialize session if new
        if session_id not in active_sessions:
            if not request.ordered_categories:
                raise HTTPException(status_code=400, detail="ordered_categories required for new session")
            
            active_sessions[session_id] = {"configurable": {"thread_id": session_id}}
            
            state = langgraph_app.invoke(
                {"ordered_categories": request.ordered_categories, "phase": "init"}, 
                active_sessions[session_id]
            )
        else:
            # Continue conversation
            config = active_sessions[session_id]
            state = langgraph_app.invoke(
                {"user_input": request.user_input}, 
                config
            )
        
        # Clean up completed conversations
        if state.get("conversation_complete", False):
            del active_sessions[session_id]
        
        return ConversationResponse(
            message=state.get("last_message", ""),
            conversation_complete=state.get("conversation_complete", False),
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

