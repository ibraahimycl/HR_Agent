import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
TOKEN_ENDPOINT = f"{API_URL}/token"
CHAT_ENDPOINT = f"{API_URL}/chat"
USER_ENDPOINT = f"{API_URL}/user/me"
HEALTH_ENDPOINT = f"{API_URL}/health"
LOGOUT_ENDPOINT = f"{API_URL}/logout"

if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def check_backend_health() -> bool:
    try:
        return requests.get(HEALTH_ENDPOINT, timeout=3).status_code == 200
    except requests.exceptions.RequestException:
        return False


def login(username: str, password: str) -> bool:
    try:
        resp = requests.post(TOKEN_ENDPOINT, data={"username": username, "password": password})
        if resp.status_code == 200:
            st.session_state.access_token = resp.json()["access_token"]
            user_info = get_user_info()
            if user_info:
                st.session_state.user_info = user_info
                st.success(f"Welcome, {user_info.get('name', 'User')}!")
                return True
            st.error("Login successful but could not retrieve user information.")
            return False
        st.error("Invalid username or password.")
        return False
    except requests.exceptions.RequestException:
        st.error("Could not connect to the server. Please try again later.")
        return False


def get_user_info() -> Optional[dict]:
    if not st.session_state.access_token:
        return None
    try:
        resp = requests.get(
            USER_ENDPOINT,
            headers={"Authorization": f"Bearer {st.session_state.access_token}"},
        )
        if resp.status_code == 200:
            info = resp.json()
            if all(k in info for k in ["username", "role"]):
                return info
            st.error("Incomplete user information received.")
            return None
        if resp.status_code == 401:
            st.session_state.access_token = None
            st.session_state.user_info = None
            st.error("Session expired. Please log in again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to server: {e}")
        return None


def send_message(message: str) -> Optional[dict]:
    if not st.session_state.access_token or not st.session_state.user_info:
        st.error("Please log in to continue.")
        return None
    try:
        resp = requests.post(
            CHAT_ENDPOINT,
            headers={"Authorization": f"Bearer {st.session_state.access_token}"},
            json={
                "message": message,
                "user_context": {
                    "username": st.session_state.user_info.get("username"),
                    "name": st.session_state.user_info.get("name"),
                    "role": st.session_state.user_info.get("role"),
                    "employee_id": st.session_state.user_info.get("employee_id"),
                },
            },
        )
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 403:
            return {"response": "You do not have permission to access this information.", "steps": [], "citations": []}
        return {"response": f"Error: {resp.json().get('detail', 'Unknown error')}", "steps": [], "citations": []}
    except requests.exceptions.RequestException as e:
        return {"response": f"Connection error: {e}", "steps": [], "citations": []}


def do_logout():
    if st.session_state.access_token:
        try:
            requests.post(
                LOGOUT_ENDPOINT,
                headers={"Authorization": f"Bearer {st.session_state.access_token}"},
            )
        except Exception:
            pass
    st.session_state.access_token = None
    st.session_state.user_info = None
    st.session_state.chat_history = []


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_agent_reasoning(steps: list):
    """Render agent tool calls in an expandable panel."""
    if not steps:
        return
    with st.expander(f"🧠 Agent Reasoning  ({len(steps)} step{'s' if len(steps) > 1 else ''})", expanded=False):
        for i, step in enumerate(steps):
            tool_name = step.get("tool", "unknown")
            tool_input = step.get("input", {})
            tool_output = step.get("output", "")

            # Tool label with color coding
            color_map = {
                "get_employee_info": "#1976D2",
                "calculate_hr_metrics": "#388E3C",
                "query_policy": "#7B1FA2",
            }
            color = color_map.get(tool_name, "#616161")

            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding-left: 10px; margin-bottom: 8px;'>"
                f"<strong style='color:{color}'>🔧 Step {i + 1}: {tool_name}</strong>"
                "</div>",
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Input:**")
                if isinstance(tool_input, dict):
                    st.json(tool_input)
                else:
                    st.code(str(tool_input), language="json")

            with col2:
                st.markdown("**Output:**")
                try:
                    parsed = json.loads(tool_output)
                    # Truncate large outputs for display
                    if "policy_context" in parsed:
                        parsed["policy_context"] = parsed["policy_context"][:400] + "  [truncated]"
                    st.json(parsed)
                except Exception:
                    output_display = tool_output[:600] + "..." if len(tool_output) > 600 else tool_output
                    st.code(output_display)

            if i < len(steps) - 1:
                st.divider()


def render_citations(citations: list):
    """Render policy source citations."""
    if not citations:
        return
    with st.expander(f"📎 Policy Sources  ({len(citations)} source{'s' if len(citations) > 1 else ''})", expanded=False):
        for c in citations:
            section = c.get("section_title", "HR Policy")
            policy_type = c.get("policy_type", "")
            excerpt = c.get("excerpt", "")
            label = "📄 Leave Policy" if policy_type == "leave_policy" else "📋 Attendance Policy"
            st.markdown(f"**{section}** &nbsp; <small style='color:#888'>{label}</small>", unsafe_allow_html=True)
            if excerpt:
                st.markdown(f"> {excerpt}")
            st.divider()


def render_chat_message(message: dict):
    """Render a single chat message with optional reasoning and citations."""
    role = message["role"]
    content = message["content"]

    if role == "user":
        st.markdown(
            f"""<div class="chat-message user">
                <div><strong>You</strong></div>
                <div class="content">{content}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div class="chat-message assistant">
                <div><strong>HR Assistant</strong></div>
                <div class="content">{content}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        render_agent_reasoning(message.get("steps", []))
        render_citations(message.get("citations", []))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def handle_send():
    if not st.session_state.chat_input:
        return

    user_msg = st.session_state.chat_input
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_msg,
        "timestamp": datetime.now().isoformat(),
    })

    result = send_message(user_msg)
    if result:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result.get("response", "No response."),
            "steps": result.get("steps", []),
            "citations": result.get("citations", []),
            "timestamp": datetime.now().isoformat(),
        })

    st.session_state.chat_input = ""
    st.rerun()


def main():
    st.set_page_config(
        page_title="HR Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
        <style>
        .chat-message {
            padding: 1.2rem 1.5rem;
            border-radius: 8px;
            margin-bottom: 0.8rem;
            color: #1a1a2e;
        }
        .chat-message.user {
            background-color: #e8f0fe;
            border-left: 4px solid #1976D2;
        }
        .chat-message.assistant {
            background-color: #f8f9fa;
            border-left: 4px solid #43a047;
        }
        .chat-message .content {
            margin-top: 6px;
            line-height: 1.6;
        }
        .chat-message strong {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #555;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("🤖 HR Assistant")

        if not check_backend_health():
            st.error("⚠️ Backend is not available. Please check if the server is running.")
            return

        if not st.session_state.access_token:
            st.subheader("Login")
            username = st.text_input("Employee ID / Username")
            password = st.text_input("Password", type="password")
            if st.button("Login", use_container_width=True):
                if login(username, password):
                    st.rerun()
        else:
            if st.session_state.user_info:
                st.subheader("Session")
                info = st.session_state.user_info
                st.markdown(f"👤 **{info.get('name', info.get('username', ''))}")
                role_badge = "🔑 HR" if info.get("role") == "hr" else "👷 Employee"
                st.markdown(f"{role_badge}")
                if info.get("department"):
                    st.markdown(f"🏢 {info.get('department')}")
                if info.get("employee_id"):
                    st.markdown(f"🆔 {info.get('employee_id')}")

            st.divider()
            if st.button("🚪 Logout", use_container_width=True):
                do_logout()
                st.rerun()

            st.divider()
            st.caption("**Legend**")
            st.caption("🔵 `get_employee_info`")
            st.caption("🟢 `calculate_hr_metrics`")
            st.caption("🟣 `query_policy` + RAG")

    # Main chat area
    if st.session_state.access_token:
        st.title("💬 HR Assistant")

        for message in st.session_state.chat_history:
            render_chat_message(message)

        st.text_input(
            "Ask anything about HR policies, your leave balance, salary, or calculations…",
            key="chat_input",
            on_change=handle_send,
        )

        if st.session_state.chat_history:
            if st.button("🗑️ Clear chat", use_container_width=False):
                st.session_state.chat_history = []
                st.rerun()
    else:
        st.info("👈 Please log in from the sidebar to start chatting.")


if __name__ == "__main__":
    main()
