from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import math
import random
from typing import Dict, Any, List, Optional

try:
    import google.generativeai as genai
except Exception as e:  # pragma: no cover
    genai = None

DEFAULT_PROMPT = {
    "decompose": (
        """
hi!
"""
    ).strip(),
    "chain": (
        """
You are an expert question planner.
Decompose the given multi-hop question into an **ordered** chain of single-hop questions such that each step leads naturally to the next and ultimately to the final answer.

Rules:
- Each question should be solvable on its own given prior steps and provided context.
- Keep the chain concise (3-6 steps typical). No answers, only questions.
- Output strict JSON only.

Schema:
{
  "steps": [
    {"index": 1, "question": "..."},
    {"index": 2, "question": "..."}
  ]
}
"""
    ).strip(),
    "rewrite": (
        """
You are an expert question rewriter.
Rewrite the given multi-hop question into a single, simpler **single-hop** question that preserves the original target and remains answerable (ideally with the provided context).

Rules:
- Keep core semantics and target unchanged.
- Remove multi-hop dependencies by substituting intermediate facts directly if they are present in the context. If not, reformulate to a single relation query.
- Output strict JSON only.

Schema:
{
  "single_hop_question": "..."
}
"""
    ).strip(),
}


def _extract_json(text: str) -> Any:
    """Extract a JSON object/array from a model response.

    Handles fenced blocks and stray prose. Raises ValueError on failure.
    """
    if text is None:
        raise ValueError("Empty response from model.")
    # Try code fences first
    fence = re.search(r"```(?:json)?\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
    candidate = fence.group(1) if fence else text
    # Trim leading/trailing non-json
    start = candidate.find("{")
    start_arr = candidate.find("[")
    if start == -1 or (start_arr != -1 and start_arr < start):
        start = start_arr
    if start == -1:
        raise ValueError("No JSON object/array found in response.")
    # Find last closing brace/bracket
    end = max(candidate.rfind("}"), candidate.rfind("]")) + 1
    payload = candidate[start:end]
    return json.loads(payload)


def _retry_sleep(base: float, attempt: int, jitter: bool = True) -> None:
    # Exponential backoff with jitter
    delay = base * (2 ** (attempt - 1))
    if jitter:
        delay = delay * (0.5 + random.random())
    time.sleep(min(delay, 30.0))


def init_gemini(
    model: str, api_key: Optional[str], temperature: float, seed: Optional[int]
) -> Any:
    if genai is None:
        raise RuntimeError(
            "google-generativeai is not installed.\nInstall: pip install google-generativeai"
        )
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Missing API key. Set --api_key or GOOGLE_API_KEY env var.")
    genai.configure(api_key=key)
    generation_config = {
        "temperature": temperature,
        "candidate_count": 1,
        "max_output_tokens": 2048,
        "response_mime_type": "application/json",
    }
    safety_settings = None  # use defaults
    model_obj = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
    # Set seed if supported
    if seed is not None:
        try:
            model_obj._generation_config["seed"] = seed  # type: ignore[attr-defined]
        except Exception:
            pass
    return model_obj


def build_messages(
    mode: str, question: str, context: Optional[str]
) -> List[Dict[str, str]]:
    system = DEFAULT_PROMPT[mode]
    user_parts = []
    user_parts.append(f"Question:\n{question}".strip())
    if context:
        user_parts.append(f"\nContext:\n{context}".strip())
    # For Gemini, we pass as a single string with system-style instruction included in the prompt
    prompt = system + "\n\n" + "\n\n".join(user_parts)
    return [{"role": "user", "parts": [prompt]}]


def call_gemini(
    model_obj: Any, messages: List[Dict[str, str]], max_retries: int = 5
) -> Any:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = model_obj.generate_content(messages[0]["parts"][0])
            print(resp)
            text = resp.text if hasattr(resp, "text") else str(resp)
            return _extract_json(text)
        except Exception as e:  # includes parsing and API errors
            last_err = e
            if attempt == max_retries:
                break
            _retry_sleep(0.7, attempt)
    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert multi-hop questions to single-hop with Gemini"
    )
    parser.add_argument(
        "--model",
        default="gemini-1.5-pro",
        help="Gemini model name, e.g., gemini-1.5-pro or gemini-1.5-flash.",
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("GOOGLE_API_KEY"),
        help="Google API key; otherwise uses GOOGLE_API_KEY env var.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    model_obj = init_gemini(args.model, args.api_key, args.temperature, args.seed)
    messages = build_messages("decompose", "what is your name", "You are kind person")
    parsed = call_gemini(model_obj, messages)
    print(messages, parsed)
