import streamlit as st
import requests
from datetime import datetime
import json
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")
TOKEN_ENDPOINT = f"{API_URL}/token"
CHAT_ENDPOINT = f"{API_URL}/chat"
USER_ENDPOINT = f"{API_URL}/user/me"
HEALTH_ENDPOINT = f"{API_URL}/health"

# Initialize session state
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def check_backend_health() -> bool:
    """Check if the backend API is healthy."""
    try:
        response = requests.get(HEALTH_ENDPOINT)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def login(username: str, password: str) -> bool:
    """Attempt to login with username and password."""
    try:
        response = requests.post(
            TOKEN_ENDPOINT,
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            st.session_state.access_token = response.json()["access_token"]
            # Get and store user info immediately after login
            user_info = get_user_info()
            if user_info:
                st.session_state.user_info = user_info
                st.success(f"Welcome, {user_info.get('name', 'User')}!")
                return True
            else:
                st.error("Login successful but could not retrieve user information.")
                return False
        else:
            st.error("Invalid username or password.")
            return False
    except requests.exceptions.RequestException:
        st.error("Could not connect to the server. Please try again later.")
        return False

def get_user_info() -> Optional[dict]:
    """Get current user information."""
    if not st.session_state.access_token:
        return None
    
    try:
        response = requests.get(
            USER_ENDPOINT,
            headers={"Authorization": f"Bearer {st.session_state.access_token}"}
        )
        if response.status_code == 200:
            user_info = response.json()
            # Ensure we have all required fields
            if not all(k in user_info for k in ['username', 'name', 'role', 'employee_id']):
                st.error("Incomplete user information received from server")
                return None
            return user_info
        elif response.status_code == 401:
            # Token expired or invalid
            st.session_state.access_token = None
            st.session_state.user_info = None
            st.error("Your session has expired. Please log in again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to server: {str(e)}")
        return None

def send_message(message: str) -> Optional[str]:
    """Send a message to the HR Agent and get response."""
    if not st.session_state.access_token or not st.session_state.user_info:
        st.error("Please log in to continue.")
        return None
    
    # Debug logging for user context
    st.write("✅ [DEBUG] user_context being sent to backend:", {
        "username": st.session_state.user_info.get('username'),
        "name": st.session_state.user_info.get('name'),
        "role": st.session_state.user_info.get('role'),
        "employee_id": st.session_state.user_info.get('employee_id')
    })
    
    try:
        # Include user context in the request
        response = requests.post(
            CHAT_ENDPOINT,
            headers={"Authorization": f"Bearer {st.session_state.access_token}"},
            json={
                "message": message,
                "user_context": {
                    "username": st.session_state.user_info.get('username'),
                    "name": st.session_state.user_info.get('name'),
                    "role": st.session_state.user_info.get('role'),
                    "employee_id": st.session_state.user_info.get('employee_id')
                }
            }
        )
        if response.status_code == 200:
            return response.json()["response"]
        elif response.status_code == 403:
            return "Sorry, you don't have permission to access this information."
        else:
            return f"Error: {response.json().get('detail', 'Unknown error occurred')}"
    except requests.exceptions.RequestException as e:
        return f"Error connecting to the server: {str(e)}"

def main():
    st.set_page_config(
        page_title="HR Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stTextInput>div>div>input {
            font-size: 16px;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
            color: #2c3e50;  /* Dark blue-gray text color */
        }
        .chat-message.user {
            background-color: #e3f2fd;  /* Light blue background */
            border: 1px solid #bbdefb;
        }
        .chat-message.assistant {
            background-color: #f5f5f5;  /* Light gray background */
            border: 1px solid #e0e0e0;
        }
        .chat-message .content {
            display: flex;
            margin-top: 0.5rem;
            color: #2c3e50;  /* Dark blue-gray text color */
        }
        .chat-message strong {
            color: #1a237e;  /* Darker blue for headers */
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar for login and user info
    with st.sidebar:
        st.title("🤖 HR Assistant")
        
        if not check_backend_health():
            st.error("⚠️ Backend service is not available. Please check if the server is running.")
            return

        if not st.session_state.access_token:
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if login(username, password):
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        else:
            if st.session_state.user_info:
                st.subheader("User Information")
                st.write(f"👤 Username: {st.session_state.user_info['username']}")
                st.write(f"👥 Role: {st.session_state.user_info['role']}")
                if st.session_state.user_info.get('department'):
                    st.write(f"🏢 Department: {st.session_state.user_info['department']}")
                if st.session_state.user_info.get('employee_id'):
                    st.write(f"🆔 Employee ID: {st.session_state.user_info['employee_id']}")
            
            if st.button("Logout"):
                st.session_state.access_token = None
                st.session_state.user_info = None
                st.session_state.chat_history = []
                st.rerun()

    # Main chat interface
    if st.session_state.access_token:
        st.title("💬 HR Assistant Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.container():
                if message["role"] == "user":
                    st.markdown(f"""
                        <div class="chat-message user">
                            <div><strong>You:</strong></div>
                            <div class="content">{message["content"]}</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="chat-message assistant">
                            <div><strong>HR Assistant:</strong></div>
                            <div class="content">{message["content"]}</div>
                        </div>
                    """, unsafe_allow_html=True)

        # Chat input
        user_input = st.text_input(
            "Type your message here...",
            key="chat_input",
            on_change=lambda: handle_send()
        )

def handle_send():
    """Handle sending a message and updating chat history."""
    if st.session_state.chat_input:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": st.session_state.chat_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get response from HR Agent
        response = send_message(st.session_state.chat_input)
        
        # Add assistant response to chat history
        if response:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat()
            })
        
        # Clear input
        st.session_state.chat_input = ""
        st.rerun()

if __name__ == "__main__":
    main()
