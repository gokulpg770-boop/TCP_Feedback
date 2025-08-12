# mcp_server.py
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
        return psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
        print("Server: Connected to PostgreSQL database")   
    except psycopg2.OperationalError as e:
        print(f"Server: DB connection failed: {e}", file=sys.stderr)
        return None

# Helper to support dict payloads or positional args
def _arg(payload_or_val, key, default=None):
    if isinstance(payload_or_val, dict):
        return payload_or_val.get(key, default)
    return payload_or_val if payload_or_val is not None else default

@mcp.tool
def add_feedback(action=None, description=None) -> dict:
    """
    Insert a feedback row. action must be 'positive' or 'negative'.
    Accepts either:
      - add_feedback(action="positive", description="text")
      - add_feedback({"action":"positive","description":"text"})
    Returns: {status: "success"|"failure", id?: int, error?: str}
    """
    action = _arg(action, "action", action)
    description = _arg(description, "description", description)

    if not isinstance(action, str):
        return {"status": "failure", "error": "action is required and must be a string"}

get_conn()