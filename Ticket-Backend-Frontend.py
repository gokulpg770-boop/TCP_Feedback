#frontend.py
import uuid
import streamlit as st
from backend import create_langgraph_app, CATEGORY_OPTIONS
config = {"configurable": {"thread_id": 1}}
app = create_langgraph_app()
 
def main():
    st.set_page_config(page_title="🎫 AI Support Ticket System", page_icon="🎫")
    st.title("🎫 AI Ticket System")
    st.markdown("---")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "booted" not in st.session_state:
        st.session_state.booted = False
    if "done" not in st.session_state:
        st.session_state.done = False
    with st.sidebar:
        st.subheader("Start Demo")
        category = st.selectbox("Initial Category", CATEGORY_OPTIONS, index=0)
        if st.button("Start", type="primary"):
            start_conversation(category)
        st.markdown("---")
        if st.button("New Conversation"):
            reset_conversation()
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if not st.session_state.booted:
        st.info("Use the sidebar to start the demo.")
        return
    if st.session_state.done:
        st.success("✅ Conversation completed.")
        return
    text = st.chat_input("Type here...")
    if text:
        append_user(text)
        state = app.invoke({"user_input": text}, config=config)
        append_assistant(state.get("last_message", ""))
        if state.get("conversation_complete"):
            st.session_state.done = True
            st.balloons()
        st.rerun()
 
def start_conversation(category: str):
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.done = False
    st.session_state.booted = True
    state = app.invoke({"category": category, "phase": "init"}, config=config)
    append_assistant(state.get("last_message", ""))
    st.rerun()
 
def reset_conversation():
    st.session_state.messages = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.booted = False
    st.session_state.done = False
    st.rerun()
 
def append_user(text: str):
    st.session_state.messages.append({"role": "user", "content": text})
 
def append_assistant(text: str):
    if not text:
        return
    # Avoid duplicate assistant messages
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        if st.session_state.messages[-1]["content"].strip() == text.strip():
            return
    st.session_state.messages.append({"role": "assistant", "content": text})
 
if __name__ == "__main__":
    main()

/////////////////////////////////////////////////////////////

#backend.py
import os
import uuid
from typing import Dict, List, Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# --- Load API Key from .env file ---
load_dotenv()

# --- Constants ---
CATEGORY_OPTIONS = [
    "payroll", "hr", "it", "facilities", "finance",
]
SOP_DB = {
    "payroll": "If your payroll issue is about salary delay, please check the ESS portal first. Contact payroll team if issue persists.",
    "hr": "For HR policy related issues, check the HR handbook on the portal. Escalate to HR for complex matters.",
    "it": "For IT issues, reboot and check the IT self-service portal first.",
    "facilities": "For AC/cleaning/maintenance, coordinate with facilities.",
    "finance": "Finance queries require proper documentation and approvals.",
}
MANDATORY_FIELDS = {
    "payroll": ["employee_id", "issue_description"],
    "hr": ["employee_id", "issue_description"],
    "it": ["employee_id", "issue_description"],
    "facilities": ["employee_id", "issue_description"],
    "finance": ["employee_id", "issue_description"],
}

# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    category: str
    collected_fields: Dict[str, str]
    final_output: dict

# --- Tools ---
@tool
def get_sop_and_ask_to_create_ticket(category: str) -> str:
    """Gets the Standard Operating Procedure (SOP) for a given category and asks the user if they want to proceed with creating a ticket."""
    return f"📘 {category.title()} SOP:\n\n{SOP_DB[category]}\n\nDo you want to raise a ticket for this issue? (yes/no)"

@tool
def ask_for_category_confirmation(category: str) -> str:
    """Asks the user to confirm the selected support category."""
    if not category:
        return "There seems to be an issue. Could you please specify the category again?"
    return f"Do you want to confirm {category.title()} as your category? (yes/no)"

@tool
def ask_user_to_choose_category() -> str:
    """Asks the user to choose a category from the available options."""
    opts = ", ".join(CATEGORY_OPTIONS)
    return f"Please type a category from the following options: {opts}"

@tool
def ask_for_next_required_field(category: str, collected_fields: Dict[str, str]) -> str:
    """Asks the user for the next required field for the given category that has not been collected yet."""
    if not category or category not in MANDATORY_FIELDS:
         return "I seem to have lost the category context. Could you please state the category again?"
    required = MANDATORY_FIELDS[category]
    for field in required:
        if field not in collected_fields:
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

# --- Graph Definition ---
def create_langgraph_app():
    system_prompt = f"""You are an expert AI assistant for a corporate support ticketing system. Your goal is to guide the user through creating a support ticket by calling the correct tools in sequence.

    **State Awareness:**
    You have access to the current `category` and the `collected_fields` from the state. You MUST use them when calling tools.

    **Conversation Flow & Tool Usage:**
    1.  **Start:** The conversation begins. Your first action is ALWAYS to call `get_sop_and_ask_to_create_ticket` with the current `category`.
    2.  **User wants to create ticket (e.g., says "yes" to the first prompt):** Your next action MUST BE `ask_for_category_confirmation`. Do NOT call any other tool at this stage.
    3.  **User confirms category ("yes"):** Start collecting fields by calling `ask_for_next_required_field`.
    4.  **User denies category ("no"):** Your next action is `ask_user_to_choose_category`.
    5.  **User provides a new category:** Update the state and then go back to step 2 to confirm the new category.
    6.  **Collecting Fields:** After the user provides a value for a field, call `ask_for_next_required_field` again to get the next one.
    7.  **Final Confirmation:** Once all fields are collected, call `show_ticket_summary_and_ask_for_final_confirmation`.
    8.  **Submit or Cancel:** Based on user input, call either `submit_ticket` or `end_conversation`.
    9.  **User denies initial ticket creation:** If the user says "no" at step 1, call `end_conversation`.

    **IMPORTANT RULES:**
    - You MUST call a tool at every step. Do not respond directly.
    - `submit_ticket` and `end_conversation` are terminal steps.
    """
    
    system_message = SystemMessage(content=system_prompt)

    tools = [
        get_sop_and_ask_to_create_ticket, ask_for_category_confirmation,
        ask_user_to_choose_category, ask_for_next_required_field,
        show_ticket_summary_and_ask_for_final_confirmation, submit_ticket,
        end_conversation,
    ]
    tool_node = ToolNode(tools)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0).bind_tools(tools)

    def agent_node(state: AgentState):
        """The primary node that calls the LLM to decide the next action."""
        # ** THE FIX IS HERE **
        # We create a new list of messages for the LLM call that includes the explicit state.
        # This makes the agent aware of the current category and collected fields.
        messages_for_llm = [system_message]
        
        # Add a message that explicitly states the current context for the LLM
        state_context = (
            f"Current State Context:\n"
            f"- Category: {state.get('category')}\n"
            f"- Collected Fields: {state.get('collected_fields')}"
        )
        messages_for_llm.append(HumanMessage(content=state_context))
        
        # Add the rest of the conversation history
        messages_for_llm.extend(state["messages"])
        
        response = model.invoke(messages_for_llm)
        return {"messages": [response]}

    def state_updater_node(state: AgentState):
        """This node updates the state based on the conversation, which is crucial for the agent's memory."""
        last_user_message = state['messages'][-2] if len(state['messages']) > 1 else None
        last_ai_message = state['messages'][-1]
        
        user_input_content = ""
        if last_user_message and isinstance(last_user_message, HumanMessage):
            user_input_content = last_user_message.content.lower().strip()

        category = state.get("category")
        collected_fields = state.get("collected_fields", {}).copy()

        if isinstance(last_ai_message, AIMessage) and last_ai_message.tool_calls:
            tool_name = last_ai_message.tool_calls[0].get('name')
            
            if tool_name == 'ask_user_to_choose_category' and user_input_content in CATEGORY_OPTIONS:
                category = user_input_content
                collected_fields = {}
            
            elif tool_name == 'ask_for_next_required_field':
                required = MANDATORY_FIELDS.get(category, [])
                for field in required:
                    if field not in collected_fields:
                        if user_input_content and last_user_message:
                            collected_fields[field] = last_user_message.content.strip()
                        break
        
        return {"category": category, "collected_fields": collected_fields}

    def output_formatter_node(state: AgentState):
        """This node formats the final output to be compatible with the Streamlit frontend."""
        last_message = state["messages"][-1] if state["messages"] else None
        content = ""
        is_complete = False
        
        if last_message:
            if isinstance(last_message, ToolMessage):
                content = last_message.content
                if last_message.name in ["submit_ticket", "end_conversation"]:
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

    def should_execute_tools(state: AgentState):
        if state["messages"] and state["messages"][-1].tool_calls:
            return "tools"
        return "output_formatter"

    builder.add_conditional_edges("agent", should_execute_tools)
    builder.add_edge("tools", "state_updater")
    builder.add_edge("state_updater", "output_formatter") 
    builder.add_edge("output_formatter", END)

    graph = builder.compile(checkpointer=checkpointer)
    
    class App:
        """The wrapper class that correctly handles state persistence."""
        def invoke(self, input_dict, config=None):
            final_state_chunk = None
            
            if "phase" in input_dict and input_dict["phase"] == "init":
                category = input_dict.get("category", "payroll")
                initial_message = HumanMessage(content=f"The user has started the ticket creation process. The initial category is '{category}'. Please begin.")
                input_for_stream = {
                    "category": category, 
                    "collected_fields": {}, 
                    "messages": [initial_message]
                }
                for s in graph.stream(input_for_stream, config=config):
                    final_state_chunk = s
            else:
                if "user_input" in input_dict:
                    for s in graph.stream({"messages": [HumanMessage(content=input_dict['user_input'])]}, config=config):
                        final_state_chunk = s

            if final_state_chunk and "output_formatter" in final_state_chunk:
                return final_state_chunk["output_formatter"].get("final_output")
                
            return {"last_message": "Sorry, something went wrong. The agent did not return a final response.", "conversation_complete": True}

    return App()

__all__ = ["create_langgraph_app", "CATEGORY_OPTIONS"]

