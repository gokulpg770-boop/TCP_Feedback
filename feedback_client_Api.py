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
