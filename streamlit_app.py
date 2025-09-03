import streamlit as st
import uuid

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

# ===== Main App =====
def main():
    st.set_page_config(page_title="Support Ticket System", page_icon="🎫", layout="wide")
    st.title("🎫 AI Support Ticket System")
    st.markdown("---")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "step" not in st.session_state:
        st.session_state.step = "show_sop"
    if "category" not in st.session_state:
        st.session_state.category = "payroll"
    if "wants_ticket" not in st.session_state:
        st.session_state.wants_ticket = None
    if "category_confirmed" not in st.session_state:
        st.session_state.category_confirmed = None
    if "mandatory_fields" not in st.session_state:
        st.session_state.mandatory_fields = {}
    if "current_field_index" not in st.session_state:
        st.session_state.current_field_index = 0
    if "conversation_complete" not in st.session_state:
        st.session_state.conversation_complete = False
    if "show_buttons" not in st.session_state:
        st.session_state.show_buttons = False

    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        if st.button("🔄 New Conversation", type="primary"):
            reset_conversation()

        st.markdown("---")
        st.markdown("### 📋 Flow")
        st.markdown("""
        1) **Payroll SOP** → "Raise ticket?" (yes/no)
        2) **If yes** → "Confirm Payroll?" (yes/no)
           - **Yes** → Collect fields → Create ticket
           - **No** → Show other categories → Confirm → Collect → Create
        3) **If no** → Exit
        """)

        # Show current state for debugging
        if st.checkbox("🔍 Show Debug Info"):
            st.markdown("### Debug Info")
            st.json({k: v for k, v in st.session_state.items() if k not in ['messages']})

    # Chat interface
    chat_container = st.container()
    for message in st.session_state.messages:
        with chat_container:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Process current step
    process_step()

    # Handle input or buttons based on state
    if st.session_state.show_buttons:
        display_category_buttons()
    elif not st.session_state.conversation_complete:
        handle_chat_input()

def reset_conversation():
    """Reset all session state for new conversation"""
    st.session_state.messages = []
    st.session_state.step = "show_sop"
    st.session_state.category = "payroll"
    st.session_state.wants_ticket = None
    st.session_state.category_confirmed = None
    st.session_state.mandatory_fields = {}
    st.session_state.current_field_index = 0
    st.session_state.conversation_complete = False
    st.session_state.show_buttons = False
    process_step()
    st.rerun()

def process_step():
    """Process the current step in the conversation"""
    step = st.session_state.step
    
    if step == "show_sop":
        show_sop()
    elif step == "ask_ticket":
        # Already asked in SOP - waiting for input
        pass
    elif step == "ask_confirmation":
        # Already asked - waiting for input
        pass
    elif step == "collect_fields":
        collect_fields()
    elif step == "create_ticket":
        create_ticket()
        st.session_state.conversation_complete = True
    elif step == "end":
        st.session_state.conversation_complete = True

def show_sop():
    """Show SOP and ask for ticket decision"""
    sop = SOP_DB[st.session_state.category]
    message = (
        f"📘 {st.session_state.category.title()} SOP:\n\n{sop}\n\n"
        "Do you want to raise a ticket for this issue? (yes/no)"
    )
    st.session_state.messages.append({"role": "assistant", "content": message})
    st.session_state.step = "ask_ticket"

def handle_chat_input():
    """Handle text input from user"""
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        process_user_response(prompt)
        st.rerun()

def process_user_response(response: str):
    """Process user response based on current step"""
    response = response.strip().lower()
    step = st.session_state.step
    
    if step == "ask_ticket":
        if response in ("yes", "y"):
            st.session_state.wants_ticket = True
            st.session_state.step = "ask_confirmation"
            ask_confirmation()
        elif response in ("no", "n"):
            st.session_state.wants_ticket = False
            st.session_state.step = "end"
            end_conversation()
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Please respond with 'yes' or 'no'."})
    
    elif step == "ask_confirmation":
        if response in ("yes", "y"):
            st.session_state.category_confirmed = True
            st.session_state.step = "collect_fields"
            st.session_state.current_field_index = 0
            st.session_state.mandatory_fields = {}
            collect_fields()
        elif response in ("no", "n"):
            st.session_state.category_confirmed = False
            st.session_state.step = "show_other_categories"
            st.session_state.show_buttons = True
            show_other_categories()
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Please respond with 'yes' or 'no'."})
    
    elif step == "collect_fields":
        required = MANDATORY_FIELDS[st.session_state.category]
        current_field = required[st.session_state.current_field_index - 1]  # Last asked field
        st.session_state.mandatory_fields[current_field] = response
        collect_fields()  # Proceed to next field or create ticket

def ask_confirmation():
    """Ask for category confirmation"""
    category = st.session_state.category
    message = f"Do you want to confirm **{category.title()}** as your category? (yes/no)"
    st.session_state.messages.append({"role": "assistant", "content": message})

def end_conversation():
    """End the conversation"""
    message = "👍 Thank you! Have a great day."
    st.session_state.messages.append({"role": "assistant", "content": message})
    st.session_state.conversation_complete = True

def show_other_categories():
    """Show other category options"""
    message = "Please select from the following categories:"
    st.session_state.messages.append({"role": "assistant", "content": message})
    st.session_state.show_buttons = True

def display_category_buttons():
    """Display category selection buttons"""
    remaining_options = [opt for opt in CATEGORY_OPTIONS if opt != st.session_state.category]
    
    for i in range(0, len(remaining_options), 3):
        cols = st.columns(3)
        for j, option in enumerate(remaining_options[i:i+3]):
            with cols[j]:
                if st.button(f"🎯 {option.title()}", key=f"btn_{option}"):
                    st.session_state.category = option
                    st.session_state.show_buttons = False
                    st.session_state.step = "ask_confirmation"
                    st.session_state.messages.append({"role": "user", "content": f"Selected: {option.title()}"})
                    ask_confirmation()
                    st.rerun()

def collect_fields():
    """Collect mandatory fields one by one"""
    category = st.session_state.category
    required = MANDATORY_FIELDS[category]
    
    if st.session_state.current_field_index < len(required):
        field = required[st.session_state.current_field_index]
        message = f"Please provide your **{field.replace('_', ' ').title()}**:"
        st.session_state.messages.append({"role": "assistant", "content": message})
        st.session_state.current_field_index += 1
    else:
        # All fields collected - create ticket
        st.session_state.step = "create_ticket"
        create_ticket()

def create_ticket():
    """Create the final ticket"""
    ticket_id = f"TCKT-{uuid.uuid4().hex[:8].upper()}"
    category = st.session_state.category
    details = st.session_state.mandatory_fields
    
    message = (
        f"✅ **Ticket Created Successfully!**\n\n"
        f"**Ticket ID:** {ticket_id}\n"
        f"**Category:** {category.title()}\n"
        f"**Details:** {details}\n\n"
        "You will receive updates via email. Thank you!"
    )
    st.session_state.messages.append({"role": "assistant", "content": message})
    st.session_state.conversation_complete = True

if __name__ == "__main__":
    main()
