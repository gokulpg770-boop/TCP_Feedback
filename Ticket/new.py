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
///////////////////////////////////////



#!/usr/bin/env python3
"""
AI Support Ticket System - Correct SOP Flow
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
    ordered_categories: Optional[List[str]] = None

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
    conversation_step: str  # track current step
    final_output: dict

# --- Tools ---
@tool
def show_sop_and_ask_create_ticket(category: str) -> str:
    """Shows SOP for the given category and asks if user wants to create ticket."""
    sop_text = SOP_DB.get(category, f"No SOP found for {category}")
    return f"📘 {category.title()} SOP:\n\n{sop_text}\n\nDo you want to raise a ticket for this issue? (yes/no)"

@tool
def ask_category_confirmation(category: str) -> str:
    """Asks user to confirm the category."""
    return f"Do you want to confirm {category.title()} as your category? (yes/no)"

@tool
def show_category_list(ordered_categories: List[str]) -> str:
    """Shows the list of available categories for user to choose from."""
    categories_text = ", ".join([cat.title() for cat in ordered_categories])
    return f"Please type one of the following categories:\n{categories_text}"

@tool
def ask_for_field(category: str, collected_fields: Dict[str, str]) -> str:
    """Asks for the next required field."""
    required = MANDATORY_FIELDS.get(category, ["employee_id", "issue_description"])
    for field in required:
        if field not in collected_fields:
            return f"Please provide your {field.replace('_', ' ').title()}:"
    return "All fields collected"

@tool
def show_summary_and_confirm(category: str, details: dict) -> str:
    """Shows ticket summary and asks for final confirmation."""
    summary = "\n".join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in details.items()])
    return f"**Ticket Summary**\n\nCategory: {category.title()}\n\nDetails:\n{summary}\n\nDo you want to submit this ticket? (yes/no)"

@tool
def submit_ticket_final(category: str, details: dict) -> str:
    """Submits the ticket - final step."""
    ticket_id = f"TCKT-{uuid.uuid4().hex[:8].upper()}"
    return (
        f"✅ Ticket Created Successfully!\n\n"
        f"Ticket ID: {ticket_id}\n"
        f"Category: {category.title()}\n"
        f"Details: {details}\n\n"
        f"You will receive updates via email. Thank you!"
    )

@tool
def end_conversation_final(message: str) -> str:
    """Ends conversation - final step."""
    return message

# --- LangGraph App Creation ---
def create_langgraph_app():
    system_prompt = """You are an AI assistant for support tickets. Follow this EXACT flow:

**CONVERSATION STEPS:**
1. **START**: Always call `show_sop_and_ask_create_ticket` with the FIRST category from ordered_categories
2. **If user says "yes" to SOP**: Call `ask_category_confirmation` 
3. **If user says "no" to SOP**: Call `show_category_list` with all ordered_categories
4. **After category list shown and user selects one**: Call `ask_category_confirmation` with selected category
5. **If user says "yes" to confirmation**: Start field collection with `ask_for_field`
6. **If user says "no" to confirmation**: Call `show_category_list` again
7. **Field collection**: Keep calling `ask_for_field` until all collected
8. **All fields done**: Call `show_summary_and_confirm`
9. **Final "yes"**: Call `submit_ticket_final` (TERMINAL)
10. **Final "no"**: Call `end_conversation_final` (TERMINAL)

**IMPORTANT:**
- ALWAYS start with SOP of first category
- You MUST call a tool at every step
- Use current state values when calling tools
- conversation_step tracks where we are in the flow
"""
    
    system_message = SystemMessage(content=system_prompt)
    tools = [
        show_sop_and_ask_create_ticket, ask_category_confirmation,
        show_category_list, ask_for_field,
        show_summary_and_confirm, submit_ticket_final, end_conversation_final,
    ]
    tool_node = ToolNode(tools)
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0).bind_tools(tools)
    
    def agent_node(state: AgentState):
        """LLM decides next action based on conversation step and state."""
        conversation_step = state.get('conversation_step', 'start')
        category = state.get('category', '')
        ordered_categories = state.get('ordered_categories', [])
        collected_fields = state.get('collected_fields', {})
        
        state_context = (
            f"Current state:\n"
            f"- Conversation Step: {conversation_step}\n"
            f"- Category: {category}\n" 
            f"- Ordered Categories: {ordered_categories}\n"
            f"- Collected Fields: {json.dumps(collected_fields)}\n"
        )
        
        messages_for_llm = [
            system_message,
            HumanMessage(content=state_context, name="StateContext"),
        ]
        messages_for_llm.extend(state["messages"])
        
        response = model.invoke(messages_for_llm)
        return {"messages": [response]}
    
    def state_updater_node(state: AgentState):
        """Updates state based on user input and current conversation step."""
        messages = state['messages']
        
        # Find last human message and prompting AI call
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
        
        user_input = last_human_message.content.strip().lower()
        prompting_tool_name = prompting_ai_call.tool_calls[0]['name']
        
        # Get current state
        category = state.get("category")
        ordered_categories = state.get("ordered_categories", [])
        collected_fields = state.get("collected_fields", {}).copy()
        conversation_step = state.get("conversation_step", "start")
        
        # Update state based on tool and user response
        if prompting_tool_name == 'show_sop_and_ask_create_ticket':
            if user_input == "yes":
                conversation_step = "confirm_category"
            elif user_input == "no":
                conversation_step = "show_list"
        
        elif prompting_tool_name == 'show_category_list':
            # User selected a category from the list
            for cat in ordered_categories:
                if user_input == cat.lower():
                    category = cat.lower()
                    collected_fields = {}  # Reset fields
                    conversation_step = "confirm_category"
                    break
        
        elif prompting_tool_name == 'ask_category_confirmation':
            if user_input == "yes":
                conversation_step = "collect_fields"
            elif user_input == "no":
                conversation_step = "show_list"
        
        elif prompting_tool_name == 'ask_for_field':
            # User provided field value
            required = MANDATORY_FIELDS.get(category, ["employee_id", "issue_description"])
            for field in required:
                if field not in collected_fields:
                    collected_fields[field] = last_human_message.content.strip()  # Keep original case
                    break
            
            # Check if all fields collected
            all_collected = all(field in collected_fields for field in required)
            if all_collected:
                conversation_step = "show_summary"
        
        elif prompting_tool_name == 'show_summary_and_confirm':
            if user_input == "yes":
                conversation_step = "submit"
            elif user_input == "no":
                conversation_step = "cancel"
        
        return {
            "category": category,
            "ordered_categories": ordered_categories,
            "collected_fields": collected_fields,
            "conversation_step": conversation_step
        }
    
    def output_formatter_node(state: AgentState):
        """Formats final output."""
        last_tool_message = None
        for msg in reversed(state.get('messages', [])):
            if isinstance(msg, ToolMessage):
                last_tool_message = msg
                break
        
        content = ""
        is_complete = False
        
        if last_tool_message:
            content = last_tool_message.content
            if last_tool_message.name in ["submit_ticket_final", "end_conversation_final"]:
                is_complete = True
        
        return {"final_output": {"last_message": content, "conversation_complete": is_complete}}
    
    # Build the graph
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
                
                initial_message = HumanMessage(content="Initialize conversation")
                input_for_graph = {
                    "category": first_category,
                    "ordered_categories": ordered_categories,
                    "collected_fields": {},
                    "conversation_step": "start",
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

# --- API Endpoint ---
@app.post("/chat")
async def chat(request: ConversationRequest):
    """Single endpoint with correct SOP flow"""
    try:
        session_id = request.session_id
        
        if session_id not in active_sessions:
            if not request.ordered_categories:
                raise HTTPException(status_code=400, detail="ordered_categories required")
            
            active_sessions[session_id] = {"configurable": {"thread_id": session_id}}
            
            config = active_sessions[session_id]
            state = langgraph_app.invoke(
                {
                    "phase": "init",
                    "ordered_categories": request.ordered_categories
                }, 
                config=config
            )
        else:
            config = active_sessions[session_id]
            state = langgraph_app.invoke(
                {"user_input": request.user_input}, 
                config=config
            )
        
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
