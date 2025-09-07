from __future__ import annotations
import argparse
import os
import subprocess
import sys


"""

# 실행 예시
python /workspace/src/extract_questions.py \
  --input /workspace/src/data/scion/test.csv \
  --output /workspace/src/data/scion/questions/questions.jsonl
"""

# =============================
# File: run_extract_and_convert.py
# =============================
#!/usr/bin/env python3
"""
run_extract_and_convert.py

단일 컬럼에서 질문을 뽑아(JSONL) 곧바로 Gemini 변환 스크립트(multi_hop_to_single_hop.py)를 실행합니다.

예시
----
python run_extract_and_convert.py \
  --source /workspace/src/data/scion/test.csv \
  --outdir /workspace/src/data/output \
  --question-col 'Question' \
  --mode decompose \
  --model gemini-2.5-flash

환경변수 GOOGLE_API_KEY 또는 --api-key 필요.
"""


essential = [
    "/workspace/src/extract_questions.py",
    "/workspace/src/multi_hop_to_single_hop.py",
]


def _check_files():
    missing = [p for p in essential if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            f"필수 파일이 없습니다: {missing}. 현재 작업 디렉토리: {os.getcwd()}"
        )


def main():
    _check_files()

    ap = argparse.ArgumentParser(description="질문 추출 후 Gemini 단일홉 변환 실행")
    ap.add_argument("--source", required=True, help="원본 데이터 파일 경로")
    ap.add_argument("--outdir", required=True, help="중간/결과 파일 출력 디렉토리")
    ap.add_argument(
        "--question-col", default=None, help="질문 컬럼명(없으면 자동 탐지)"
    )
    ap.add_argument(
        "--mode", choices=["decompose", "chain", "rewrite"], default="decompose"
    )
    ap.add_argument("--model", default="gemini-1.5-pro")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--qps", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    qjsonl = os.path.join(args.outdir, "questions.jsonl")
    outjsonl = os.path.join(args.outdir, f"singlehop_{args.mode}.jsonl")

    # 1) 질문만 추출
    cmd1 = [
        sys.executable,
        "extract_questions.py",
        "--input",
        args.source,
        "--output",
        qjsonl,
    ]
    if args.question_col:
        cmd1 += ["--question-col", args.question_col]
    print("[run]", " ".join(cmd1))
    subprocess.run(cmd1, check=True)

    # 2) Gemini 변환
    cmd2 = [
        sys.executable,
        "multi_hop_to_single_hop.py",
        "--input",
        qjsonl,
        "--output",
        outjsonl,
        "--mode",
        args.mode,
        "--model",
        args.model,
        "--temperature",
        str(args.temperature),
    ]
    if args.api_key:
        cmd2 += ["--api_key", args.api_key]
    if args.seed is not None:
        cmd2 += ["--seed", str(args.seed)]
    if args.qps is not None:
        cmd2 += ["--qps", str(args.qps)]

    print("[run]", " ".join(cmd2))
    subprocess.run(cmd2, check=True)

    print(f"Done. Output → {outjsonl}")


if __name__ == "__main__":
    main()
