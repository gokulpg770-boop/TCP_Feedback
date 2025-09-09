#!/usr/bin/env python3
"""
Streamlit UI for AI Support Ticket System - Minimal Clean Version
"""
import streamlit as st
import requests
import uuid

class TicketSystemUI:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
        # Hardcoded query-to-category mapping for demo
        self.query_mappings = {
            "salary not credited": ["payroll", "finance"],
            "salary not got": ["payroll", "finance"],
            "salary delay": ["payroll", "hr"],
            "salary problem": ["payroll", "finance"],
            "pay not received": ["payroll", "finance"],
            "salary": ["payroll", "finance"],
            "laptop not working": ["it", "facilities"],
            "computer problem": ["it"],
            "internet issue": ["it"],
            "password reset": ["it", "security"],
            "laptop": ["it"],
            "computer": ["it"],
            "leave approval": ["hr"],
            "leave": ["hr"],
            "ac not working": ["facilities", "it"],
            "ac": ["facilities"],
            "expense": ["finance"],
            "training": ["training"],
            "travel": ["travel"],
            "security": ["security"],
        }
    
    def detect_categories_from_query(self, user_query):
        """Detect categories from user query"""
        user_query_lower = user_query.lower()
        
        for query_pattern, categories in self.query_mappings.items():
            if query_pattern in user_query_lower:
                return categories
        
        # Fallback to partial matching
        for query_pattern, categories in self.query_mappings.items():
            if any(word in user_query_lower for word in query_pattern.split()):
                return categories
        
        return ["general"]
    
    def initialize_session(self):
        """Initialize session state"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if 'conversation_started' not in st.session_state:
            st.session_state.conversation_started = False
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'conversation_complete' not in st.session_state:
            st.session_state.conversation_complete = False
    
    def start_conversation(self, user_query, categories):
        """Start conversation with API"""
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "user_input": user_query,
                    "session_id": st.session_state.session_id,
                    "ordered_categories": categories
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Add to conversation history
                st.session_state.conversation_history.append({"role": "user", "message": user_query})
                st.session_state.conversation_history.append({"role": "assistant", "message": data['message']})
                
                st.session_state.conversation_started = True
                st.session_state.conversation_complete = data.get('conversation_complete', False)
                return True
            else:
                st.error(f"Error: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to server")
            return False
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return False
    
    def send_message(self, user_input):
        """Send follow-up message"""
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    "user_input": user_input,
                    "session_id": st.session_state.session_id
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Add to conversation history
                st.session_state.conversation_history.append({"role": "user", "message": user_input})
                st.session_state.conversation_history.append({"role": "assistant", "message": data['message']})
                
                st.session_state.conversation_complete = data.get('conversation_complete', False)
                return True
            else:
                st.error(f"Error: {response.status_code}")
                return False
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return False

def main():
    st.set_page_config(
        page_title="Support Ticket System",
        page_icon="🎫",
        layout="centered"
    )
    
    # Custom CSS for clean styling
    st.markdown("""
    <style>
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 12px 20px;
        font-size: 16px;
    }
    
    .stButton > button {
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
    }
    
    .main > div {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize UI
    ui = TicketSystemUI()
    ui.initialize_session()
    
    # Header
    st.title("🎫 Support Ticket System")
    st.markdown("---")
    
    if not st.session_state.conversation_started:
        # Input phase
        st.subheader("What do you need help with?")
        
        # Input field
        user_query = st.text_input(
            "Describe your issue",
            placeholder="e.g., salary not received this month",
            label_visibility="collapsed"
        )
        
        # Submit button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Submit", type="primary", use_container_width=True):
                if user_query:
                    categories = ui.detect_categories_from_query(user_query)
                    
                    with st.spinner("Processing..."):
                        success = ui.start_conversation(user_query, categories)
                    
                    if success:
                        st.rerun()
                else:
                    st.error("Please enter your issue")
    
    else:
        # Conversation phase
        st.subheader("💬 Chat")
        
        # Display conversation
        for entry in st.session_state.conversation_history:
            with st.chat_message(entry["role"]):
                st.write(entry["message"])
        
        # Check if complete
        if st.session_state.conversation_complete:
            st.success("✅ Ticket Created Successfully!")
            
            if st.button("Create New Ticket", type="primary"):
                # Reset session
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        else:
            # Continue conversation
            user_input = st.chat_input("Type your response...")
            
            if user_input:
                with st.spinner("Processing..."):
                    success = ui.send_message(user_input)
                
                if success:
                    st.rerun()

if __name__ == "__main__":
    main()
