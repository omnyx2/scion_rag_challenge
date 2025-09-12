# rerank_inplace_by_config.py  (progress-logging 강화판)
import re
import os
import json
import argparse
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# --------------------------
# Config & Columns
# --------------------------
def log(msg: str):
    print(msg, flush=True)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_article_cols(
    df: pd.DataFrame, pattern=r"^prediction_retrieved_article_name_(\d+)$"
) -> List[str]:
    rx = re.compile(pattern)
    cols = [c for c in df.columns if rx.match(c)]
    if not cols:
        raise ValueError(
            "prediction_retrieved_article_name_{n} 형태의 컬럼을 찾지 못했습니다."
        )
    cols = sorted(cols, key=lambda c: int(rx.match(c).group(1)))
    return cols


# --------------------------
# Parsing & Embedding Text
# --------------------------
TITLE_RX = re.compile(r"^\s*Title\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
ABSTRACT_RX = re.compile(
    r"^\s*Abstract\s*:\s*(.+?)(?=^\s*[A-Za-z][A-Za-z0-9_\- ]{0,20}\s*:|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def extract_title_abstract(text: str) -> Tuple[str, str]:
    if not isinstance(text, str):
        return "", ""
    title_match = TITLE_RX.search(text)
    abstract_match = ABSTRACT_RX.search(text)
    title = title_match.group(1).strip() if title_match else ""
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    return title, abstract


def build_embedding_text(
    candidate_text: str, embedding_mode: str, fallback_fields: List[str] = None
) -> str:
    """
    embedding_mode 예: '3*title+abstract'
    - Title/Abstract 파싱 실패 시: fallback_fields 라벨 블록을 순서대로 붙임.
    - 모두 실패하면 원문 전체 사용.
    """
    title, abstract = extract_title_abstract(candidate_text)

    if embedding_mode.lower() == "3*title+abstract":
        if title or abstract:
            return f"{title} {title} {title} {abstract}".strip()

    if fallback_fields:
        parts = []
        for label in fallback_fields:
            rx = re.compile(
                rf"^\s*{re.escape(label)}\s*:\s*(.+?)(?=^\s*[A-Za-z].*?:|\Z)",
                re.IGNORECASE | re.MULTILINE | re.DOTALL,
            )
            m = rx.search(candidate_text or "")
            if m:
                parts.append(m.group(1).strip())
        if parts:
            return " ".join(parts)

    return candidate_text or ""


# --------------------------
# Embedding & Similarity
# --------------------------
def encode_texts(
    model: SentenceTransformer, texts: List[str], batch_size: int = 64, desc: str = ""
) -> np.ndarray:
    if desc:
        log(f"🧮 Encoding {len(texts)} items [{desc}] (batch={batch_size})…")
    return model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=True,  # 진행바 ON
        normalize_embeddings=True,  # 정규화 → dot == cosine
    )


def cosine_sim_vector_to_matrix(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    # vec: (d,), mat: (N, d)  -> (N,)
    return mat @ vec  # 이미 정규화됨


# --------------------------
# Main Rerank
# --------------------------
def rerank_inplace(
    df: pd.DataFrame,
    question_col: str,
    article_cols: List[str],
    model_name: str,
    embedding_mode: str,
    batch_size: int = 64,
    text_fields_from_config: List[str] = None,
) -> pd.DataFrame:
    start = time.time()

    if question_col not in df.columns:
        raise ValueError(f"'{question_col}' 컬럼이 없습니다.")

    # 모델 로드
    log(f"🧠 Loading model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # 질문 임베딩
    questions = df[question_col].fillna("").astype(str).tolist()
    q_start = time.time()
    q_embs = encode_texts(
        model, questions, batch_size=batch_size, desc="questions"
    )  # (R, d)
    log(f"✅ Questions encoded in {time.time() - q_start:.2f}s")

    # 각 행마다 후보 50개 파싱 → 임베딩 텍스트 생성
    log("🧵 Building candidate embedding texts per row (parsing Title/Abstract)…")
    build_start = time.time()
    cand_texts_per_row: List[List[str]] = []
    for r_i, (_, row) in enumerate(df.iterrows(), 1):
        row_texts = []
        for c in article_cols:
            raw_text = row.get(c, "")
            emb_text = build_embedding_text(
                str(raw_text),
                embedding_mode=embedding_mode,
                fallback_fields=(text_fields_from_config or []),
            )
            row_texts.append(emb_text)
        cand_texts_per_row.append(row_texts)
        if r_i % 200 == 0 or r_i == len(df):
            log(f"  • parsed {r_i}/{len(df)} rows")
    log(f"✅ Candidate texts built in {time.time() - build_start:.2f}s")

    # 후보 임베딩 + 재정렬
    log("🔁 Reranking within row (placing best at _1 … worst at _50)…")
    R = len(df)
    for i in range(R):
        # 후보 임베딩 (행 단위)
        c_start = time.time()
        cand_texts = cand_texts_per_row[i]
        cand_embs = encode_texts(
            model, cand_texts, batch_size=batch_size, desc=f"row {i + 1}/{R}"
        )  # (C, d)

        # 유사도 계산 → 내림차순 인덱스
        sims = cosine_sim_vector_to_matrix(q_embs[i], cand_embs)  # (C,)
        order = np.argsort(-sims)

        # 같은 행 내부에서 article_cols 순서만 바꿔 끼우기
        original_values = [df.at[df.index[i], c] for c in article_cols]
        reordered_values = [original_values[idx] for idx in order]
        for j, col in enumerate(article_cols):
            df.at[df.index[i], col] = reordered_values[j]

        # 진행 로그
        took = time.time() - c_start
        if (i + 1) % 50 == 0 or (i + 1) == R:
            log(f"  • processed {i + 1}/{R} rows (last row took {took:.2f}s)")

    log(f"✅ Reranking done in {time.time() - start:.2f}s total")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="config 기반 임베딩 방식으로 행 내부 후보(1~50) 재정렬(in-place)"
    )
    parser.add_argument("--input", required=True, help="입력 파일 경로 (CSV)")
    parser.add_argument(
        "--output",
        default=None,
        help="출력 파일 경로 (지정 없으면 _reranked_inplace.csv)",
    )
    parser.add_argument(
        "--config",
        default="/workspace/configs/query_encoder/config_gte-multilingual-base.json",
        help="임베딩 설정 JSON 경로",
    )
    parser.add_argument("--question_col", default="Question", help="질문 컬럼명")
    parser.add_argument(
        "--pattern",
        default=r"^prediction_retrieved_article_name_(\d+)$",
        help="후보 컬럼 정규식",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="임베딩 배치 사이즈")
    args = parser.parse_args()

    # 로드
    log(f"📂 Loading CSV: {args.input}")
    df = pd.read_csv(args.input)
    log(f"📏 Rows: {len(df)}, Columns: {len(df.columns)}")

    cfg = load_config(args.config)
    model_name = cfg.get("model_name", "Alibaba-NLP/gte-multilingual-base")
    embedding_mode = cfg.get("embedding_mode", "3*title+abstract")
    text_fields = cfg.get("text_fields", ["context"])

    article_cols = find_article_cols(df, args.pattern)
    log(
        f"🔎 Candidate columns: {len(article_cols)} ({article_cols[0]}..{article_cols[-1]})"
    )
    log(f"⚙️ Embedding mode: {embedding_mode}")

    # 재정렬
    df_out = rerank_inplace(
        df=df,
        question_col=args.question_col,
        article_cols=article_cols,
        model_name=model_name,
        embedding_mode=embedding_mode,
        batch_size=args.batch_size,
        text_fields_from_config=text_fields,
    )

    # 저장
    if args.output:
        out_path = args.output
    else:
        stem, ext = os.path.splitext(args.input)
        out_path = f"{stem}_reranked_inplace.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"✅ Saved: {out_path}")
    log(f"• Model: {model_name}")
    log(
        f"• Reordered columns: {len(article_cols)} ({article_cols[0]}..{article_cols[-1]})"
    )


if __name__ == "__main__":
    main()
