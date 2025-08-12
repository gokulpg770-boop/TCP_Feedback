import asyncio
import re
import sys
from fastmcp import Client

async def main():
    try:
        # If your server is a local script, pass the path to Client
        # e.g., Client("./mcp_server.py") or use config; here we use stdio inference:
        client = Client("./feedback_server.py")  # adjust to your server script path
        async with client:
            print("Client: inserting feedback...")
            result = await client.call_tool(
                "add_feedback",
                {"action": "positive", "description": "Great response quality and speed."}
            )
            print(f"Client: server -> {result}")

            m = re.search(r"id=(\d+)", str(result))
            new_id = int(m.group(1)) if m else None

            if new_id is not None:
                print(f"Client: fetching feedback id={new_id}...")
                row = await client.call_tool("get_feedback_by_id", {"id": new_id})
                print(f"Client: row -> {row}")

            print("Client: listing recent feedback...")
            rows = await client.call_tool("list_feedback", {"limit": 10, "offset": 0})
            print(f"Client: {len(rows)} rows")
            for r in rows:
                print(r)
    except Exception as e:
        print(f"Client error: {e}", file=sys.stderr)

if __name__ == "__main__":
    asyncio.run(main())
