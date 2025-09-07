"""Example usage with OpenAIClient (uninstruct)"""

from llm_client.openai_client import OpenAIClient
from llm_client.types import ImageInput


def run():
    client = OpenAIClient(model="gpt-4o-mini", max_retries=3)
    client.set_mode("uninstruct")
    client.add_system("You are a helpful assistant.")
    client.add_user(
        "Summarize this paragraph: The quick brown fox jumps over the lazy dog."
    )
    res = client.generate(max_tokens=200)
    print("Text:\n", res.text)
    print("Tokens:\n", client.get_token_usage())


if __name__ == "__main__":
    run()
