# feedback_client.py
import asyncio
import sys
from pathlib import Path
import httpx
from fastmcp import Client

# Path to your MCP server
SERVER_FILENAME = "feedback_server.py"

# Your API endpoint
API_URL = "https://example.com/api/feedback"  # Replace with real endpoint


async def fetch_feedback_from_api(api_url):
    """
    Fetch feedback from an external API.
    Expected API response: list of { "action": "...", "description": "..." }
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(api_url)
            resp.raise_for_status()  # Raise exception for HTTP errors
            data = resp.json()

            # Ensure data is in expected format
            if isinstance(data, dict) and "feedback" in data:
                data = data["feedback"]  # Example if API wraps feedback in a key

            if not isinstance(data, list):
                print("API did not return a list, ignoring data.")
                return []

            # Validate and normalize entries
            valid_data = []
            for item in data:
                action = str(item.get("action", "")).lower()
                description = str(item.get("description", "")).strip()
                if action in ("positive", "negative") and description:
                    valid_data.append({"action": action, "description": description})

            return valid_data

    except Exception as e:
        print(f"❌ Failed to fetch API data: {e}")
        return []


async def run():
    # Resolve server file path
    server_path = str(Path(__file__).parent / SERVER_FILENAME)

    # Start client and spawn server via stdio
    client = Client(server_path)

    async with client:
        # List available server tools
        tools = await client.list_tools()
        print("Available tools:", tools)

        # Step 1: Try getting feedback from API
        api_feedback = await fetch_feedback_from_api(API_URL)

        # Step 2: Fallback to dummy feedback if API provided nothing
        if not api_feedback:
            print("⚠ No API feedback found. Using dummy feedback instead.")
            api_feedback = [
                {"action": "positive", "description": "Great support experience!"},
                {"action": "negative", "description": "Response time was too slow."},
                {"action": "positive", "description": "Loved how the issue was handled!"},
            ]

        # Step 3: Insert each feedback row into DB
        for i, feedback in enumerate(api_feedback, start=1):
            print(f"\nClient: Sending feedback #{i} to server...")
            res = await client.call_tool("add_feedback", feedback)
            print(f"Server response: {res}")

        # Step 4: List recent rows from DB
        print("\nClient: Listing recent feedback from DB...")
        lst = await client.call_tool("list_feedback", {"limit": 10, "offset": 0})
        print("Recent rows:", lst)


if __name__ == "__main__":
    asyncio.run(run())





# feedback_server.py
import os
import sys
import psycopg2
from dotenv import load_dotenv

# FastMCP v2
from fastmcp import FastMCP

load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

mcp = FastMCP(name="FeedbackServer")

def get_conn():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
        print("Server: Connected to PostgreSQL")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Server: DB connection failed: {e}", file=sys.stderr)
        return None

# Helper - allow dict or positional arguments
def _arg(payload_or_val, key, default=None):
    if isinstance(payload_or_val, dict):
        return payload_or_val.get(key, default)
    return payload_or_val if payload_or_val is not None else default
@mcp.tool
def list_feedback(limit=None, offset=None) -> dict:
    """List recent feedback rows via MCP tool."""
    # Accept dict or positional args
    limit = _arg(limit, "limit", limit)
    offset = _arg(offset, "offset", offset)

    # Validate and default pagination
    try:
        limit = int(limit) if limit is not None else 10
        offset = int(offset) if offset is not None else 0
    except ValueError:
        return {"status": "failure", "error": "limit and offset must be integers"}

    if limit < 1 or limit > 1000:
        return {"status": "failure", "error": "limit must be between 1 and 1000"}
    if offset < 0:
        return {"status": "failure", "error": "offset must be >= 0"}

    conn = get_conn()
    if not conn:
        return {"status": "failure", "error": "database unavailable"}

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, action, description, COALESCE(created_at, NOW()) AS created_at
                FROM feedback
                ORDER BY id DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            rows = cur.fetchall()
        conn.commit()
        # Convert to list of dicts
        result = [
            {
                "id": r[0],
                "action": r[1],
                "description": r[2],
                "created_at": r[3].isoformat(),
            }
            for r in rows
        ]
        return {"status": "success", "rows": result, "limit": limit, "offset": offset}
    except Exception as e:
        conn.rollback()
        print(f"Server: select error: {e}", file=sys.stderr)
        return {"status": "failure", "error": str(e)}
    finally:
        conn.close()

@mcp.tool
def add_feedback(action=None, description=None) -> dict:
    """Insert a feedback row via MCP tool."""
    action = _arg(action, "action", action)
    description = _arg(description, "description", description)

    if not isinstance(action, str):
        return {"status": "failure", "error": "action is required and must be a string"}
    action_norm = action.strip().lower()
    if action_norm not in ("positive", "negative"):
        return {"status": "failure", "error": "action must be 'positive' or 'negative'"}
    if not isinstance(description, str) or not description.strip():
        return {"status": "failure", "error": "description cannot be empty"}

    conn = get_conn()
    if not conn:
        return {"status": "failure", "error": "database unavailable"}
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (action, description) VALUES (%s, %s) RETURNING id",
                (action_norm, description.strip()),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
        return {"status": "success", "id": new_id}
    except Exception as e:
        conn.rollback()
        print(f"Server: insert error: {e}", file=sys.stderr)
        return {"status": "failure", "error": str(e)}
    finally:
        conn.close()

# -------------------
# TEST: Insert row on server startup
# -------------------
def insert_test_row():
    """Inserts a test feedback row at server startup."""
    conn = get_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (action, description) VALUES (%s, %s) RETURNING id",
                ("positive", "Test feedback inserted by server on startup")
            )
            new_id = cur.fetchone()[0]
        conn.commit()
        print(f"Test insert successful! Row ID = {new_id}")
    except Exception as e:
        conn.rollback()
        print(f"Test insert failed: {e}", file=sys.stderr)
    finally:
        conn.close()

if __name__ == "__main__":
    # Run a test DB insert when server starts
    # insert_test_row()

    # Start MCP server
    mcp.run(transport="stdio")
