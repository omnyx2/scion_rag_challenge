"""Example usage with GeminiClient (instruct + image)"""

import asyncio
from llm_client.gemini_client import GeminiClient
from llm_client.types import ImageInput


async def run():
    client = GeminiClient(model="gemini-2.5-flash", max_retries=2)
    client.set_mode("instruct")
    client.add_system("You are a vision-capable assistant.")
    client.add_user(
        "Describe the object in the image.",
        images=[ImageInput(url="https://example.com/cat.jpg")],
    )
    res = await client.agenerate()
    print(res.text)
    print(client.get_token_usage())


if __name__ == "__main__":
    asyncio.run(run())
