#!/usr/bin/env python3
"""
multi_hopt_to_single_hop.py

Convert multi-hop questions into single-hop questions using Google's Gemini models.

Features
- Reads JSONL input with fields for question/context.
- Modes: 
    * decompose: produce a minimal set of single-hop Qs that collectively solve the original.
    * chain: ordered decomposition where each Q depends on prior answers.
    * rewrite: rewrite into a single simpler (single-hop) question that preserves answerability.
- Strict JSON outputs, robust parsing, retries with exponential backoff, optional rate limiting.
- Works with environment variable GOOGLE_API_KEY or --api_key.

Example
-------
python multi_hopt_to_single_hop.py \
  --input data/multihop.jsonl \
  --output data/singlehop.jsonl \
  --question_field question --context_field context \
  --mode decompose --model gemini-1.5-pro

Input JSONL example (one JSON per line):
{"id":"hotpot_0001", "question":"Which author wrote the novel adapted into the film directed by...", "context":"<optional extra context or passages>", "answer":"<optional>"}

Output JSONL example (decompose):
{"id":"hotpot_0001", "mode":"decompose", "single_hop_questions":["Who directed ...?", "Which novel was ...?", "Who wrote that novel?"], "meta":{"model":"gemini-1.5-pro","temperature":0.2}}

Notes
-----
- If your dataset lacks context, the model will rely on world knowledge. You can pass retrieved passages in `context` to improve faithfulness.
- To throttle requests: use --qps 1.5 (max queries per second).
- To ensure deterministic output: set --temperature 0.0 and --seed.
"""

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
You are an expert question decomposer.
Given a potentially multi-hop question (and optional context), produce the **minimal** set of independent, single-hop questions that, when answered, enable a solver to answer the original question.

Requirements:
- Each item must be answerable without external tools other than the given context and common knowledge.
- Avoid redundancy; keep the set as small as possible while sufficient.
- Prefer entity- and relation-focused questions.
- Do NOT answer anything. Only output the questions.
- Output strict JSON only, no extra text.

Return JSON with this schema:
{
  "single_hop_questions": ["q1", "q2", ...]
}
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
            text = resp.text if hasattr(resp, "text") else str(resp)
            return _extract_json(text)
        except Exception as e:  # includes parsing and API errors
            last_err = e
            if attempt == max_retries:
                break
            _retry_sleep(0.7, attempt)
    raise RuntimeError(f"Gemini call failed after {max_retries} attempts: {last_err}")


def process_record(
    record: Dict[str, Any],
    mode: str,
    model_obj: Any,
    qfield: str,
    cfield: Optional[str],
) -> Dict[str, Any]:
    question = record.get(qfield)
    if not question:
        raise ValueError(f"Record missing '{qfield}' field: {record}")
    context = record.get(cfield) if cfield else None
    messages = build_messages(mode, question, context)
    parsed = call_gemini(model_obj, messages)

    out: Dict[str, Any] = {
        "id": record.get("id"),
        "mode": mode,
        **(
            {"original_question": question} if "original_question" not in record else {}
        ),
    }

    if mode == "decompose":
        if not isinstance(parsed, dict) or "single_hop_questions" not in parsed:
            raise ValueError(f"Unexpected model output for decompose: {parsed}")
        out.update({"single_hop_questions": parsed["single_hop_questions"]})
    elif mode == "chain":
        if not isinstance(parsed, dict) or "steps" not in parsed:
            raise ValueError(f"Unexpected model output for chain: {parsed}")
        out.update({"steps": parsed["steps"]})
    elif mode == "rewrite":
        key = "single_hop_question"
        if not isinstance(parsed, dict) or key not in parsed:
            raise ValueError(f"Unexpected model output for rewrite: {parsed}")
        out.update({key: parsed[key]})
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Carry through optional fields
    for k in ("context", "answer"):
        if k in record:
            out[k] = record[k]
    return out


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Robust JSONL reader.
    - utf-8-sig 로 열어 BOM 제거.
    - 한 줄에 여분 문자열이 있어도 JSON 부분만 추출 시도.
    - 실패 시 몇 번째 줄인지 함께 에러 표시.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig", newline="\n") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # 모델 응답 등에서 코드펜스/문구가 섞인 경우를 대비해 JSON 부분만 추출
                try:
                    rows.append(_extract_json(line))
                except Exception as e:
                    raise ValueError(
                        f"Invalid JSON on line {i}: {e}\n"
                        f"Line content (truncated): {line[:200]}..."
                    )
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert multi-hop questions to single-hop with Gemini"
    )
    parser.add_argument(
        "--input",
        required=False,
        help="Path to input JSONL. If omitted, runs a single prompt from --question.",
    )
    parser.add_argument("--output", required=False, help="Path to output JSONL.")
    parser.add_argument(
        "--question", help="Direct question string when --input is omitted."
    )
    parser.add_argument(
        "--context",
        default=None,
        help="Optional context string when --input is omitted.",
    )
    parser.add_argument(
        "--question_field",
        default="question",
        help="Field name for the question in input JSONL.",
    )
    parser.add_argument(
        "--context_field",
        default="context",
        help="Field name for context in input JSONL.",
    )
    parser.add_argument(
        "--mode", choices=["decompose", "chain", "rewrite"], default="decompose"
    )
    parser.add_argument(
        "--model",
        default="gemini-1.5-pro",
        help="Gemini model name, e.g., gemini-1.5-pro or gemini-1.5-flash.",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="Google API key; otherwise uses GOOGLE_API_KEY env var.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument(
        "--qps", type=float, default=None, help="Queries per second limit (approx)."
    )

    args = parser.parse_args()

    if not args.input and not args.question:
        parser.error("Provide --input JSONL or --question string.")

    model_obj = init_gemini(args.model, args.api_key, args.temperature, args.seed)

    results: List[Dict[str, Any]] = []
    last_time = 0.0
    min_interval = 1.0 / args.qps if args.qps and args.qps > 0 else 0.0

    if args.input:
        records = read_jsonl(args.input)
        for idx, rec in enumerate(records, start=1):
            now = time.time()
            if min_interval > 0 and now - last_time < min_interval:
                time.sleep(min_interval - (now - last_time))
            try:
                out = process_record(
                    rec, args.mode, model_obj, args.question_field, args.context_field
                )
                out.setdefault("meta", {})
                out["meta"].update(
                    {"model": args.model, "temperature": args.temperature}
                )
                results.append(out)
            except Exception as e:
                # Log failure inline and continue
                results.append(
                    {
                        "id": rec.get("id"),
                        "mode": args.mode,
                        "error": str(e),
                        "original": rec,
                    }
                )
            last_time = time.time()
        if not args.output:
            parser.error("--output is required when --input is provided.")
        write_jsonl(args.output, results)
        print(f"Wrote {len(results)} rows to {args.output}")
    else:
        # Single-run mode
        rec = {"id": None, "question": args.question}
        if args.context:
            rec["context"] = args.context
        out = process_record(rec, args.mode, model_obj, "question", "context")
        print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
