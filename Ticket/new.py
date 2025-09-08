
#!/usr/bin/env python3
"""
AI Support Ticket System - Ordered Categories with No Validation
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

# --- Constants (No validation list needed) ---
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

MANDATORY_FIELDS = {
    "payroll": ["employee_id", "issue_description", "pay_period"],
    "hr": ["employee_id", "issue_description"],
    "it": ["employee_id", "issue_description"],
    "facilities": ["employee_id", "issue_description"],
    "finance": ["employee_id", "issue_description"],
    "security": ["employee_id", "issue_description"],
    "training": ["employee_id", "issue_description"],
    "travel": ["employee_id", "issue_description"],
    "procurement": ["employee_id", "issue_description"],
    "general": ["employee_id", "issue_description"],
}

# --- Pydantic Models ---
class ConversationRequest(BaseModel):
    user_input: str
    session_id: str
    ordered_categories: Optional[List[str]] = None  # Ordered list of categories

class ConversationResponse(BaseModel):
    message: str
    conversation_complete: bool
    session_id: str
    current_category: Optional[str] = None

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
    return f"Do you want to confirm {category.title()} as your category? (yes/no)"

@tool
def show_available_categories(ordered_categories: List[str]) -> str:
    """Shows the available categories from the ordered list for user to choose from."""
    categories_text = ", ".join([cat.title() for cat in ordered_categories])
    return f"Please type one of the following categories:\n{categories_text}"

@tool
def ask_for_next_required_field(category: str, collected_fields: Dict[str, str]) -> str:
    """Asks the user for the next required field for the given category that has not been collected yet."""
    required = MANDATORY_FIELDS.get(category, ["employee_id", "issue_description"])
    for field in required:
        if field not in collected_fields:
            return f"Please provide your {field.replace('_', ' ').title()}:"
    return "All required fields have been collected."

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
        f"✅ Ticket Created Successfully!\n\n"
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
    system_prompt = """You are an expert AI assistant for a corporate support ticketing system. Follow this EXACT conversation flow:

**FLOW STEPS:**
1. **Start**: Show SOP of FIRST category in ordered_categories list using `get_sop_and_ask_to_create_ticket`
2. **If user says "no" to SOP**: Show available categories using `show_available_categories`, then when user types one, go to step 3
3. **If user says "yes" to SOP OR selects a category**: Call `ask_for_category_confirmation`
4. **If user says "yes" to confirmation**: Collect fields using `ask_for_next_required_field`
5. **If user says "no" to confirmation**: Show available categories using `show_available_categories`
6. **Field Collection**: Keep calling `ask_for_next_required_field` until all fields collected
7. **All fields collected**: Call `show_ticket_summary_and_ask_for_final_confirmation`
8. **Final confirmation "yes"**: Call `submit_ticket` (TERMINAL)
9. **Final confirmation "no"**: Call `end_conversation` with "Ticket submission cancelled. Thank you!" (TERMINAL)

**IMPORTANT RULES:**
- You MUST call a tool at every step. Never respond directly.
- Use the current state values when calling tools.
- `submit_ticket` and `end_conversation` are terminal steps.
"""
    
    system_message = SystemMessage(content=system_prompt)
    tools = [
        get_sop_and_ask_to_create_ticket, ask_for_category_confirmation,
        show_available_categories, ask_for_next_required_field,
        show_ticket_summary_and_ask_for_final_confirmation, submit_ticket,
        end_conversation,
    ]
    tool_node = ToolNode(tools)
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)
    
    def agent_node(state: AgentState):
        """The primary node that calls the LLM to decide the next action."""
        state_context = (
            f"Current conversation state:\n"
            f"- Category: {state.get('category', 'None')}\n"
            f"- Ordered Categories: {state.get('ordered_categories', [])}\n"
            f"- Collected Fields: {json.dumps(state.get('collected_fields', {}))}\n"
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
        
        # Handle category selection from show_available_categories
        if prompting_tool_name == 'show_available_categories':
            # User typed a category name, set it as current category
            if user_input.lower() in [cat.lower() for cat in ordered_categories]:
                category = user_input.lower()
                collected_fields = {}  # Reset fields for new category
        
        # Handle field collection
        elif prompting_tool_name == 'ask_for_next_required_field':
            required = MANDATORY_FIELDS.get(category, ["employee_id", "issue_description"])
            for field in required:
                if field not in collected_fields:
                    collected_fields[field] = user_input
                    break
        
        return {
            "category": category,
            "ordered_categories": ordered_categories,
            "collected_fields": collected_fields
        }
    
    def output_formatter_node(state: AgentState):
        """Formats the final output."""
        last_tool_message = None
        for msg in reversed(state.get('messages', [])):
            if isinstance(msg, ToolMessage):
                last_tool_message = msg
                break
        
        content = ""
        is_complete = False
        
        if last_tool_message:
            content = last_tool_message.content
            if last_tool_message.name in ["submit_ticket", "end_conversation"]:
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
        def invoke(self, input_dict, config=None):
            if "phase" in input_dict and input_dict["phase"] == "init":
                ordered_categories = input_dict.get("ordered_categories", ["payroll"])
                first_category = ordered_categories[0] if ordered_categories else "payroll"
                
                initial_message = HumanMessage(content=f"Start with first category from list: {ordered_categories}")
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

# Global instances
langgraph_app = create_langgraph_app()
active_sessions = {}

# --- Single API Endpoint ---
@app.post("/chat")
async def chat(request: ConversationRequest):
    """Single endpoint for ordered category processing"""
    try:
        session_id = request.session_id
        
        # Initialize session if new
        if session_id not in active_sessions:
            if not request.ordered_categories:
                raise HTTPException(status_code=400, detail="ordered_categories required for new session")
            
            active_sessions[session_id] = {"configurable": {"thread_id": session_id}}
            
            # Start conversation with ordered categories
            config = active_sessions[session_id]
            state = langgraph_app.invoke(
                {
                    "phase": "init",
                    "ordered_categories": request.ordered_categories
                }, 
                config=config
            )
        else:
            # Continue conversation
            config = active_sessions[session_id]
            state = langgraph_app.invoke(
                {"user_input": request.user_input}, 
                config=config
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

///////////////////////////////////////////////////////

#!/usr/bin/env python3
"""
Terminal client for AI Support Ticket System - Ordered Categories
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
            # Initialize conversation with ordered categories
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
    print("🎯 AI Support Ticket System - Ordered Categories")
    print("=" * 60)
    print("Enter categories in order (e.g., payroll,it,hr,finance)")
    
    while True:
        categories_input = input("📝 Enter ordered categories (comma-separated): ").strip()
        
        if not categories_input:
            print("❌ Please enter at least one category.")
            continue
        
        # Parse categories - no validation, just split and clean
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
