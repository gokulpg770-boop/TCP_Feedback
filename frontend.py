import streamlit as st
import requests
import uuid
from typing import Dict, Any

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

    render_sidebar()

    chat_container = st.container()
    display_chat_history(chat_container)

    if not st.session_state.conversation_started:
        start_new_conversation()

    if st.session_state.thread_id:
        handle_conversation_interface()

def render_sidebar():
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
        1) Show Payroll SOP → "Raise ticket?" (yes/no)  
        2) If yes → "Confirm Payroll?" (yes/no)  
           - Yes → Collect attributes → Create ticket  
           - No → Show other 9 categories → Confirm → Collect → Create  
        3) If no → Exit
        """)

def display_chat_history(chat_container):
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

def start_new_conversation():
    with st.spinner("Starting new conversation..."):
        try:
            response = requests.post(f"{API_BASE_URL}/new_conversation")
            if response.status_code == 200:
                data = response.json()
                st.session_state.thread_id = data["thread_id"]
                st.session_state.messages = [{"role": "assistant", "content": data["message"]}]
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
        display_option_buttons(data.get("remaining_options", []))
    elif data and data.get("conversation_complete"):
        st.success("✅ Conversation completed! Click 'New Conversation' to start again.")
        if st.button("🔄 Start New Conversation", type="primary"):
            start_new_conversation()
    else:
        handle_chat_input()

def display_option_buttons(remaining_options):
    st.markdown("### 📝 Select a Category:")
    for i in range(0, len(remaining_options), 3):
        cols = st.columns(3)
        for j, option in enumerate(remaining_options[i:i+3]):
            with cols[j]:
                if st.button(f"🎯 {option.title()}", key=f"btn_{option}_{st.session_state.thread_id}", use_container_width=True):
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
                bot_message = data.get("message", "I'm sorry, I didn't understand that.")
                st.session_state.messages.append({"role": "assistant", "content": bot_message})
                if data.get("conversation_complete"):
                    st.balloons()
                st.rerun()
            else:
                try:
                    err = response.json()
                    st.error(f"Failed to process message: {err.get('detail', response.text)}")
                except Exception:
                    st.error("Failed to process message")
        except Exception as e:
            error_msg = f"❌ Connection error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
