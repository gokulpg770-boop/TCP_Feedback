#!/usr/bin/env python3
"""
AI Support Ticket System - CLI Version
A terminal-based conversational AI for creating support tickets.
"""

import os
import uuid
import sys
import json
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
    At the beginning of each turn, you are given the CURRENT `category` and `collected_fields`.
    You MUST use these CURRENT values when calling tools. For example, if `collected_fields` is `{{"employee_id": "123"}}`, and you call `ask_for_next_required_field`, you MUST pass `{{"employee_id": "123"}}` as the `collected_fields` argument. Do not use an empty dictionary if the state indicates fields have already been collected.
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
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0).bind_tools(tools)

    def agent_node(state: AgentState):
        """The primary node that calls the LLM to decide the next action."""
        # FIX: Create a clear, structured context message for the LLM to ensure it uses the most current state.
        state_context = (
            f"This is your current memory of the conversation. Use it to decide your next action:\n"
            f"<state>\n"
            f"  <category>{state.get('category')}</category>\n"
            f"  <collected_fields>{json.dumps(state.get('collected_fields', {}))}</collected_fields>\n"
            f"</state>"
        )
        
        # The message list for the LLM is now: System Prompt, Current State, then the full Conversation History.
        messages_for_llm = [
            system_message,
            HumanMessage(content=state_context, name="StateContext"),
        ]
        messages_for_llm.extend(state["messages"])
        
        response = model.invoke(messages_for_llm)
        return {"messages": [response]}

    def state_updater_node(state: AgentState):
        """
        This node updates the state based on the conversation. It correctly identifies the user's
        input and the question it was in response to, ensuring accurate data collection.
        """
        messages = state['messages']
        
        last_human_message = None
        prompting_ai_call = None
        
        # Find the last human message
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage) and messages[i].name != "StateContext":
                last_human_message = messages[i]
                # Now find the AI tool call that came before it
                for j in range(i - 1, -1, -1):
                    if isinstance(messages[j], AIMessage) and messages[j].tool_calls:
                        prompting_ai_call = messages[j]
                        break
                break
        
        if not last_human_message or not prompting_ai_call:
            return {} # Pattern not found, do not update state

        user_input = last_human_message.content.strip()
        prompting_tool_name = prompting_ai_call.tool_calls[0]['name']

        category = state.get("category")
        collected_fields = state.get("collected_fields", {}).copy()

        if prompting_tool_name == 'ask_user_to_choose_category' and user_input.lower() in CATEGORY_OPTIONS:
            category = user_input.lower()
            collected_fields = {}

        elif prompting_tool_name == 'ask_for_next_required_field':
            required = MANDATORY_FIELDS.get(category, [])
            for field in required:
                if field not in collected_fields:
                    collected_fields[field] = user_input
                    break
        
        return {"category": category, "collected_fields": collected_fields}

    def output_formatter_node(state: AgentState):
        """This node formats the final output to be compatible with the CLI."""
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
    builder.add_edge("state_updater", "agent") # Loop back to the agent
    builder.add_edge("output_formatter", END)
    
    graph = builder.compile(checkpointer=checkpointer)
    
    class App:
        """The wrapper class that correctly handles state persistence."""
        def invoke(self, input_dict, config=None):
            if "phase" in input_dict and input_dict["phase"] == "init":
                category = input_dict.get("category", "payroll")
                initial_message = HumanMessage(content=f"The user has started the ticket creation process. The initial category is '{category}'. Please begin.")
                input_for_graph = {
                    "category": category, 
                    "collected_fields": {}, 
                    "messages": [initial_message]
                }
            else:
                input_for_graph = {"messages": [HumanMessage(content=input_dict['user_input'])]}
            
            final_state = graph.invoke(input_for_graph, config=config)
            output = output_formatter_node(final_state)
            return output.get("final_output")

    return App()

# --- CLI Application ---
class TicketSystemCLI:
    def __init__(self):
        self.app = create_langgraph_app()
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        self.conversation_active = False
    
    def print_header(self):
        """Print the application header."""
        print("=" * 60)
        print("🎫 AI SUPPORT TICKET SYSTEM")
        print("=" * 60)
        print()
    
    def print_separator(self):
        """Print a separator line."""
        print("-" * 60)
    
    def show_categories(self):
        """Display available categories."""
        print("Available Categories:")
        for i, category in enumerate(CATEGORY_OPTIONS, 1):
            print(f"  {i}. {category.title()}")
        print()
    
    def get_category_choice(self):
        """Get category choice from user."""
        while True:
            self.show_categories()
            try:
                choice = input("Select a category (1-5) or type category name: ").strip().lower()
                
                if choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(CATEGORY_OPTIONS):
                        return CATEGORY_OPTIONS[choice_num - 1]
                    else:
                        print("❌ Invalid number. Please choose 1-5.")
                        continue
                
                if choice in CATEGORY_OPTIONS:
                    return choice
                else:
                    print(f"❌ Invalid category. Please choose from: {', '.join(CATEGORY_OPTIONS)}")
                    continue
                    
            except ValueError:
                print("❌ Invalid input. Please try again.")
                continue
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                sys.exit(0)
    
    def start_conversation(self, category: str):
        """Start a new conversation with the given category."""
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        self.conversation_active = True
        
        print(f"\n🚀 Starting ticket creation for: {category.title()}")
        self.print_separator()
        
        state = self.app.invoke({"category": category, "phase": "init"}, config=self.config)
        
        if state and state.get("last_message"):
            print(f"🤖 Assistant: {state['last_message']}")
        
        return state.get("conversation_complete", False)
    
    def handle_user_input(self):
        """Handle user input and get AI response."""
        try:
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                print("❌ Please provide some input.")
                return False
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye!")
                return True
            
            if user_input.lower() in ['restart', 'new']:
                return self.restart_conversation()
            
            state = self.app.invoke({"user_input": user_input}, config=self.config)
            
            if state and state.get("last_message"):
                print(f"\n🤖 Assistant: {state['last_message']}")
                
                if state.get("conversation_complete"):
                    print("\n" + "=" * 60)
                    print("✅ Conversation completed!")
                    print("=" * 60)
                    return True
            
            return False
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return True
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            return False
    
    def restart_conversation(self):
        """Restart the conversation with a new category."""
        print("\n🔄 Starting new conversation...")
        self.print_separator()
        category = self.get_category_choice()
        self.start_conversation(category)
        return False
    
    def show_help(self):
        """Show help information."""
        print("\n📖 Help:")
        print("  - Answer the assistant's questions to create a ticket")
        print("  - Type 'restart' or 'new' to start over")
        print("  - Type 'quit', 'exit', or 'bye' to leave")
        print("  - Press Ctrl+C to exit anytime")
        print()
    
    def run(self):
        """Main application loop."""
        self.print_header()
        
        print("Welcome to the AI Support Ticket System!")
        print("This system will help you create support tickets efficiently.")
        self.show_help()
        
        # Main application loop - continues as long as the user wants to create tickets.
        while True:
            # Initial category selection for a new ticket
            category = self.get_category_choice()
            
            # Start the conversation for the current ticket
            is_complete = self.start_conversation(category)
            
            # Main conversation loop for the current ticket
            while not is_complete:
                is_complete = self.handle_user_input()
            
            # Ask if user wants to create another ticket
            try:
                another = input("\nWould you like to create another ticket? (yes/no): ").strip().lower()
                if another in ['no', 'n']:
                    print("\n👋 Thank you for using the AI Support Ticket System!")
                    break  # Exit the main loop
                elif another not in ['yes', 'y']:
                    print("Invalid input. Exiting.")
                    break
                # If 'yes', the loop continues to the next iteration, starting a new ticket.
                print()
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break

def main():
    """Entry point for the CLI application."""
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            print("❌ Error: GOOGLE_API_KEY not found in environment variables.")
            print("Please set up your .env file with your Google API key.")
            sys.exit(1)
        
        cli = TicketSystemCLI()
        cli.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()


