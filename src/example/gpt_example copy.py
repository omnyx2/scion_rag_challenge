from src.llm.openai_client import OpenAIClient


client = OpenAIClient(model="gpt-4o-mini", max_retries=5)
client.set_mode("uninstruct")
client.add_system("You are a helpful assistant.")
client.add_user(
    "Summarize this paragraph: Artificial intelligence is transforming industries by..."
)


res = client.generate(max_tokens=200)
print("[GPT Response]", res.text)
print("Tokens used:", res.total_tokens)
