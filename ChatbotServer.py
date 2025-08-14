# ChatbotServer.py
import os
import sys
import psycopg2
from dotenv import load_dotenv
from fastmcp import FastMCP
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO   = os.getenv("EMAIL_TO")  # default recipient

def send_email(subject, body, recipient=None):
    """Send an email notification."""
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_USER
        msg["To"] = recipient or EMAIL_TO
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, recipient or EMAIL_TO, msg.as_string())

        print(f"✅ Email sent to {recipient or EMAIL_TO}")
    except Exception as e:
        print(f"❌ Email sending failed: {e}", file=sys.stderr)

# Load DB credentials
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

mcp = FastMCP(name="ChatbotServer")

def get_conn():
    try:
        return psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
        )
    except psycopg2.OperationalError as e:
        print(f"DB connection failed: {e}", file=sys.stderr)
        return None

def _arg(payload_or_val, key, default=None):
    if isinstance(payload_or_val, dict):
        return payload_or_val.get(key, default)
    return payload_or_val if payload_or_val is not None else default

@mcp.tool
def list_feedback(limit=None, offset=None) -> dict:
    """List recent feedback entries."""
    try:
        limit = int(limit or 10)
        offset = int(offset or 0)
    except ValueError:
        return {"status": "failure", "error": "Invalid limit/offset"}

    conn = get_conn()
    if not conn:
        return {"status": "failure", "error": "Database unavailable"}

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT f.id, f.action, f.description, f.rating, f.created_at,
                      m.id as message_id, m.content
                FROM feedback f
                LEFT JOIN messages m ON m.id = f.message_id
                ORDER BY f.created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            rows = cur.fetchall()
        return {
            "status": "success",
            "feedback": [
                {
                    "id": r[0],
                    "action": r[1],
                    "description": r[2],
                    "rating": r[3],
                    "created_at": r[4].isoformat(),
                    "message_id": r[5],
                    "message_content": r[6],
                }
                for r in rows
            ],
        }
    except Exception as e:
        return {"status": "failure", "error": str(e)}
    finally:
        conn.close()

@mcp.tool
def add_message_feedback(message_id=None, action=None, description=None, rating=None) -> dict:
    """Store feedback for a message."""
    message_id = _arg(message_id, "message_id")
    action = _arg(action, "action")
    description = _arg(description, "description")
    rating = _arg(rating, "rating")

    if not message_id or action not in ("positive", "negative"):
        return {"status": "failure", "error": "message_id and valid action required"}

    conn = get_conn()
    if not conn:
        return {"status": "failure", "error": "Database unavailable"}

    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO feedback (message_id, action, description, rating) VALUES (%s,%s,%s,%s) RETURNING id",
                (message_id, action, description, rating)
            )
            fb_id = cur.fetchone()[0]

        # Trigger email notification
        subject = f"New Feedback Received - {action.title()}"
        body = f"""
        A new feedback entry has been submitted:

        Message ID: {message_id}
        Action: {action}
        Rating: {rating}
        Description: {description}
        Feedback ID: {fb_id}
        """
        send_email(subject, body)

        return {"status": "success", "feedback_id": fb_id}
    except Exception as e:
        return {"status": "failure", "error": str(e)}
    finally:
        conn.close()

@mcp.tool
def add_message(conversation_id=None, sender=None, content=None) -> dict:
    """Store a message and return its ID."""
    conversation_id = _arg(conversation_id, "conversation_id")
    sender = _arg(sender, "sender")
    content = _arg(content, "content")

    if not conversation_id or not sender or not content:
        return {"status": "failure", "error": "conversation_id, sender, and content required"}

    conn = get_conn()
    if not conn:
        return {"status": "failure", "error": "Database unavailable"}

    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (conversation_id, sender, content) VALUES (%s, %s, %s) RETURNING id",
                (conversation_id, sender, content)
            )
            msg_id = cur.fetchone()[0]
        return {"status": "success", "message_id": msg_id}
    except Exception as e:
        return {"status": "failure", "error": str(e)}
    finally:
        conn.close()

# if __name__ == "__main__":
#     mcp.run(transport="stdio")

if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8000,
        path="/mcp"
    )

