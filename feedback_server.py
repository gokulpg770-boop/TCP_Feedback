
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
