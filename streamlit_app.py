# streamlit_app.py
import streamlit as st
import asyncio
import json
import time
from dotenv import load_dotenv
from pathlib import Path
from fastmcp import Client
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from fastmcp.client.transports import StreamableHttpTransport

load_dotenv()

# Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# SERVER_PATH = str(Path(__file__).parent / "ChatbotServer.py") //stdio transport

#Http server URL for FastMCP
SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")


def query_gemini_llm(prompt):
    try:
        result = llm.invoke(prompt)
        return getattr(result, "content", str(result))
    except Exception as e:
        return f"Error calling Gemini: {e}"


def _parse_tool_result(res):
    """
    Normalize tool return into a plain dict:
      - If already dict -> return it
      - If wrapper object with .data or .structured_content dict -> return that
      - If wrapper object with content list of TextContent -> try to parse JSON from text
      - Else return fallback dict with error/representation
    """
    if res is None:
        return None
    if isinstance(res, dict):
        return res

    # prefer .data or .structured_content if it's a dict
    for attr in ("data", "structured_content"):
        v = getattr(res, attr, None)
        if isinstance(v, dict):
            return v

    # some wrappers give "content" as list of TextContent objects
    content = getattr(res, "content", None)
    if isinstance(content, list) and content:
        first = content[0]
        # TextContent-like: try .text or dict['text']
        text = getattr(first, "text", None) if not isinstance(first, dict) else first.get("text")
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {"status": "success", "content_text": text}

    # fallback: try to get __dict__
    if hasattr(res, "__dict__"):
        try:
            d = vars(res)
            return {k: (v if not hasattr(v, "text") else getattr(v, "text")) for k, v in d.items()}
        except Exception:
            pass

    return {"status": "failure", "error": f"Unrecognized tool result: {str(res)}"}


async def _call_tool_with_fresh_client(tool_name: str, payload: dict, max_retries: int = 3, retry_delay: float = 0.6):
    """
    Create a fresh Client, connect in an async context, call the tool, return parsed dict.
    Retries a few times to account for startup lag.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        transport = StreamableHttpTransport(url=SERVER_URL)
        client = Client(transport)        
        try:
            async with client:
                res = await client.call_tool(tool_name, payload)
                return _parse_tool_result(res)
        except Exception as e:
            last_exc = e
            # small backoff then retry
            await asyncio.sleep(retry_delay)
        finally:
            # ensure client closed by context manager; continue to next attempt if failed
            pass
    return {"status": "failure", "error": f"call_tool failed after {max_retries} attempts: {last_exc}"}


async def store_message_and_get_id(conversation_id, sender, content):
    parsed = await _call_tool_with_fresh_client(
        "add_message", {"conversation_id": conversation_id, "sender": sender, "content": content}
    )
    if parsed and parsed.get("status") == "success":
        return parsed.get("message_id")
    return parsed  # return dict for debugging on failure


async def add_feedback(message_id, action, description):
    parsed = await _call_tool_with_fresh_client(
        "add_message_feedback", {"message_id": message_id, "action": action, "description": description, "rating": None}
    )
    return parsed


def run_async(coro):
    """Run an async coroutine from sync code (Streamlit)."""
    return asyncio.run(coro)


def main():
    st.title("💬Chatbot with Feedback")

    # session state defaults
    st.session_state.setdefault("user_message_id", None)
    st.session_state.setdefault("ai_message_id", None)
    st.session_state.setdefault("response", "")
    st.session_state.setdefault("last_query", "")

    user_query = st.text_input("Enter your query:", key="query_input")

    if st.button("Send") and user_query:
        st.session_state["last_query"] = user_query

        # 1) store user's message
        user_store_res = run_async(store_message_and_get_id(conversation_id=1, sender="user", content=user_query))
        if isinstance(user_store_res, dict):
            st.warning(f"Could not store user message: {user_store_res}")
            st.session_state["user_message_id"] = None
        else:
            st.session_state["user_message_id"] = user_store_res

        # 2) query Gemini
        response = query_gemini_llm(user_query)
        st.session_state["response"] = response

        # 3) store AI message
        ai_store_res = run_async(store_message_and_get_id(conversation_id=1, sender="ai", content=response))
        if isinstance(ai_store_res, dict):
            st.warning(f"Could not store AI message: {ai_store_res}")
            st.session_state["ai_message_id"] = None
        else:
            st.session_state["ai_message_id"] = ai_store_res

    if st.session_state["response"]:
        st.text_area("Gemini Response:", value=st.session_state["response"], height=200)

        st.markdown("### Provide Feedback on AI's Response")
        feedback_choice = st.radio("Your Feedback:", ["None", "Thumbs Up", "Thumbs Down"], index=0)
        feedback_comment = st.text_area("Add feedback details (optional):", key="feedback_comment_area")

        action = None
        if feedback_choice == "Thumbs Up":
            action = "positive"
        elif feedback_choice == "Thumbs Down":
            action = "negative"

        if st.button("Submit Feedback") and action:
            if not st.session_state.get("ai_message_id"):
                st.error("No AI message_id available — cannot submit feedback. Make sure you 'Send' a query first.")
            else:
                with st.spinner("Saving feedback..."):
                    res = run_async(add_feedback(st.session_state["ai_message_id"], action, feedback_comment))
                    if isinstance(res, dict) and res.get("status") == "success":
                        st.success("✅ Feedback saved!")
                    else:
                        st.error(f"❌ Failed to save feedback: {res}")

    st.markdown("---")
    st.markdown("**Debug info (useful for troubleshooting)**")
    st.json(
        {
            "user_message_id": st.session_state.get("user_message_id"),
            "ai_message_id": st.session_state.get("ai_message_id"),
            "last_query": st.session_state.get("last_query"),
            "server_url": SERVER_URL,
        }
    )


if __name__ == "__main__":
    main()
