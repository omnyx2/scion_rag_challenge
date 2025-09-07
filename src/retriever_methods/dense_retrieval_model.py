#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense Retrieval with Snowflake Arctic Embed on a CSV VectorDB and JSONL Questions.

- Loads document embeddings from a CSV (columns: doc_id, text, embedding as JSON list).
- Loads query encoder model name from a config JSON (e.g., {"model_name": "Snowflake/snowflake-arctic-embed-l-v2.0", ...}).
- Lets you select questions by ID, index, or range from a JSONL file.
- Runs retrieval for original_question and each single_hop_questions.
- Saves one JSON per question to:
    /workspace/results/retrival_docs/{yymmdd_hhmmss}/{question_id}_{model_name}_{trial_id}.json

Example:
    python src/retriever_methods/dense_retrieval_snowflake_arctic.py \
      --vectordb_csv /workspace/results/vectordb/arctic_embed_test_3title_abstract_Snowflake_snowflake-arctic-embed-l-v2.0/arctic_embed_test_3title_abstract_Snowflake_snowflake-arctic-embed-l-v2.0.csv \
      --config_json /workspace/configs/query_encoder/config_snowflake.json \
      --questions_jsonl /workspace/data/expr/singlehop_decompose.jsonl \
      --ids row_000018,row_000042 \
      --top_k 10 \
      --device auto
Example:
Example:
    python retriever_methods/dense_retrieval_model.py \
      --vectordb_csv /workspace/results/vectordb/250907_085140/bge_m3_test_1title_abstract_BAAI_bge-m3.csv \
      --config_json /workspace/configs/query_encoder/config_bge_m3.json \
      --questions_jsonl /workspace/data/expr/singlehop_decompose.jsonl \
      --range 1-50 \
      --top_k 10 \
      --device auto
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime as dt
import json
import os
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Try FAISS (faster). If unavailable, fall back to NumPy.
try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

# --- Utilities ---


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize."""
    norms = np.sqrt((x * x).sum(axis=1, keepdims=True)) + eps
    return x / norms


def _now_kst() -> dt.datetime:
    """Return current time in Asia/Seoul."""
    try:
        import pytz  # optional

        tz = pytz.timezone("Asia/Seoul")
        return dt.datetime.now(tz)
    except Exception:
        # Fallback without pytz
        return dt.datetime.utcnow() + dt.timedelta(hours=9)


def _timestamp_folder_kst() -> str:
    t = _now_kst()
    return t.strftime("%y%m%d_%H%M%S")


def _safe_model_name_for_path(model_name: str) -> str:
    return model_name.replace("/", "-").replace(" ", "_")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --- VectorDB loader ---


@dataclasses.dataclass
class VectorDB:
    doc_ids: List[str]
    texts: List[str]
    embeddings: np.ndarray  # (N, D), float32, L2-normalized


def load_vectordb_from_csv(csv_path: str) -> VectorDB:
    """
    Expects columns: 'doc_id', 'text', 'embedding' (stringified list of floats).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"VectorDB CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"doc_id", "text", "embedding"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}, got {set(df.columns)}"
        )

    # Parse embeddings
    def parse_emb(x: str) -> List[float]:
        return ast.literal_eval(x)

    embs = df["embedding"].apply(parse_emb).tolist()
    emb_arr = np.array(embs, dtype=np.float32)
    emb_arr = _l2_normalize(emb_arr.astype(np.float32))

    doc_ids = df["doc_id"].astype(str).tolist()
    texts = df["text"].astype(str).tolist()

    return VectorDB(doc_ids=doc_ids, texts=texts, embeddings=emb_arr)


# --- Index builder ---


class RetrieverIndex:
    def __init__(self, embeddings: np.ndarray):
        """
        embeddings: (N, D) L2-normalized float32 vectors.
        """
        self.N, self.D = embeddings.shape
        self.embeddings = embeddings
        self.use_faiss = _HAS_FAISS
        if self.use_faiss:
            # cosine similarity = inner product on normalized vectors
            self.index = faiss.IndexFlatIP(self.D)  # type: ignore
            self.index.add(embeddings)  # type: ignore
        else:
            self.index = None

    def search(
        self, query_vecs: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        query_vecs: (Q, D), normalized.
        returns (scores, indices)
            scores: (Q, top_k) cosine similarities
            indices: (Q, top_k) int64 indices into the corpus
        """
        if self.use_faiss:
            scores, idx = self.index.search(query_vecs.astype(np.float32), top_k)  # type: ignore
            return scores, idx
        else:
            # NumPy fallback: cosine = dot for normalized vectors
            sims = query_vecs @ self.embeddings.T  # (Q, N)
            # Argpartition for efficiency
            top_idx = np.argpartition(-sims, kth=min(top_k, self.N - 1), axis=1)[
                :, :top_k
            ]
            # Re-sort each row by similarity
            sorted_idx = np.take_along_axis(
                top_idx, np.argsort(-np.take_along_axis(sims, top_idx, axis=1)), axis=1
            )
            sorted_scores = np.take_along_axis(sims, sorted_idx, axis=1)
            return sorted_scores, sorted_idx


# --- Query Encoder ---


class QueryEncoder:
    """
    Wraps Snowflake Arctic Embed encoder using SentenceTransformers.
    """

    def __init__(
        self, model_name: str, device: str = "auto", max_length: Optional[int] = None
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.model = None

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(
                model_name, trust_remote_code=True,device=None if device == "auto" else device
            )
            print(f"[INFO] Loaded Snowflake Arctic Embed model: {model_name}", file=sys.stderr)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{model_name}'. "
                f"Install 'sentence-transformers' and ensure the model is available. Original error: {e}"
            )

    def encode_queries(
        self, queries: List[str], instruction: Optional[str] = None
    ) -> np.ndarray:
        """
        Returns L2-normalized (Q, D) float32 array.
        
        For Snowflake Arctic Embed, we use a specific query prefix.
        """
        if instruction is None:
            # Snowflake Arctic Embed uses "Represent this sentence for searching relevant passages: "
            # as the default instruction for queries
            instruction = "Represent this sentence for searching relevant passages: "

        # Apply instruction prefix to queries
        q_texts = [instruction + q.strip() for q in queries]

        # Encode using SentenceTransformer
        q_vecs = self.model.encode(
            q_texts,
            batch_size=32,
            normalize_embeddings=False,  # we'll normalize explicitly
            convert_to_numpy=True,
            show_progress_bar=False,
            device=None if self.device == "auto" else self.device,
        ).astype(np.float32)

        # L2 normalize the embeddings
        q_vecs = _l2_normalize(q_vecs)
        return q_vecs


# --- Questions loader & selector ---


@dataclasses.dataclass
class QuestionItem:
    qid: str
    original_question: str
    single_hop_questions: List[str]
    meta: Dict[str, Any]


def load_questions_jsonl(path: str) -> List[QuestionItem]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Questions JSONL not found: {path}")
    items: List[QuestionItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj.get("id"))
            oq = str(obj.get("original_question", ""))
            sh = obj.get("single_hop_questions", []) or []
            meta = obj.get("meta", {}) or {}
            items.append(
                QuestionItem(
                    qid=qid,
                    original_question=oq,
                    single_hop_questions=list(sh),
                    meta=meta,
                )
            )
    return items


def select_questions(
    all_items: List[QuestionItem],
    ids: Optional[List[str]] = None,
    idxs: Optional[List[int]] = None,
    range_spec: Optional[str] = None,
) -> List[QuestionItem]:
    sel: List[QuestionItem] = []
    # Build map for id lookup
    id_map = {it.qid: it for it in all_items}

    if ids:
        for qid in ids:
            if qid not in id_map:
                raise KeyError(f"Question ID not found: {qid}")
            sel.append(id_map[qid])

    if idxs:
        # 1-based indexing
        for i in idxs:
            if i < 1 or i > len(all_items):
                raise IndexError(f"Index out of range (1..{len(all_items)}): {i}")
            sel.append(all_items[i - 1])

    if range_spec:
        # format "start-end" inclusive, 1-based
        if "-" not in range_spec:
            raise ValueError("range must be like '1-10'")
        s, e = range_spec.split("-", 1)
        s_i = int(s.strip())
        e_i = int(e.strip())
        if s_i < 1 or e_i < 1 or s_i > e_i or e_i > len(all_items):
            raise ValueError(f"Bad range {range_spec}; valid is 1..{len(all_items)}")
        for i in range(s_i, e_i + 1):
            sel.append(all_items[i - 1])

    # If nothing specified, default to the first item
    if not sel:
        sel = [all_items[0]] if all_items else []

    return sel


# --- Retrieval runner ---


def run_retrieval_for_question(
    q_item: QuestionItem,
    encoder: QueryEncoder,
    index: RetrieverIndex,
    vectordb: VectorDB,
    top_k: int,
    model_name: str,
    query_instruction: Optional[str] = None,
) -> Dict[str, Any]:
    queries: List[Tuple[str, Dict[str, Any]]] = []

    # original question
    queries.append((q_item.original_question, {"type": "original"}))

    # single-hop questions
    for i, q in enumerate(q_item.single_hop_questions):
        queries.append((q, {"type": "single_hop", "index": i}))

    q_texts = [q for q, _ in queries]
    q_vecs = encoder.encode_queries(q_texts, instruction=query_instruction)

    scores, idx = index.search(q_vecs, top_k=top_k)

    results: List[Dict[str, Any]] = []
    for qi, (qtext, qmeta) in enumerate(queries):
        hits = []
        for k in range(scores.shape[1]):
            doc_idx = int(idx[qi, k])
            hits.append(
                {
                    "rank": k + 1,
                    "score": float(scores[qi, k]),  # cosine sim in [-1, 1]
                    "doc_id": vectordb.doc_ids[doc_idx],
                    "text": vectordb.texts[doc_idx],
                }
            )
        results.append(
            {
                "query": qtext,
                "query_meta": qmeta,
                "top_k": top_k,
                "hits": hits,
            }
        )

    out = {
        "id": q_item.qid,
        "model_name": model_name,
        "timestamp_kst": _now_kst().isoformat(),
        "trial_id": uuid.uuid4().hex[:8],
        "queries": results,
        "meta": q_item.meta,
    }
    return out


def save_question_result(
    out_root: str,
    timestamp_folder: str,
    qid: str,
    model_name: str,
    trial_id: str,
    payload: Dict[str, Any],
) -> str:
    safe_model = _safe_model_name_for_path(model_name)
    folder = os.path.join(out_root, timestamp_folder)
    _ensure_dir(folder)
    fname = f"{qid}_{safe_model}_{trial_id}.json"
    path = os.path.join(folder, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# --- Main ---


def main():
    p = argparse.ArgumentParser(description="Dense Retrieval with configured model")
    p.add_argument(
        "--vectordb_csv", type=str, required=True, help="CSV with doc_id,text,embedding"
    )
    p.add_argument(
        "--config_json",
        type=str,
        required=True,
        help="JSON with at least {'model_name': model name}",
    )
    p.add_argument(
        "--questions_jsonl", type=str, required=True, help="JSONL of questions"
    )
    p.add_argument("--ids", type=str, default="", help="Comma-separated question IDs")
    p.add_argument(
        "--idx",
        type=str,
        default="",
        help="Comma-separated 1-based indices (e.g., '1,3,5')",
    )
    p.add_argument(
        "--range",
        dest="range_spec",
        type=str,
        default="",
        help="1-based inclusive range like '1-10'",
    )
    p.add_argument(
        "--top_k", type=int, default=10, help="Number of docs to retrieve per query"
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Encoding device: 'auto', 'cpu', 'cuda:0', etc.",
    )
    p.add_argument(
        "--max_length", type=int, default=512, help="Max token length for queries"
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="/workspace/results/retrival_docs",
        help="Output root directory",
    )
    p.add_argument(
        "--query_instruction",
        type=str,
        default="",
        help="Override query instruction string",
    )
    args = p.parse_args()

    # Load config
    if not os.path.exists(args.config_json):
        raise FileNotFoundError(f"Config JSON not found: {args.config_json}")
    with open(args.config_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_name = cfg.get("model_name") or cfg.get("model") or "Snowflake/snowflake-arctic-embed-l-v2.0"
    if not isinstance(model_name, str):
        raise ValueError("Config must contain 'model_name' as a string.")

    # Load vectordb
    print(f"[INFO] Loading VectorDB from {args.vectordb_csv} ...", file=sys.stderr)
    vectordb = load_vectordb_from_csv(args.vectordb_csv)
    print(
        f"[INFO] Loaded {len(vectordb.doc_ids)} documents, dim={vectordb.embeddings.shape[1]}.",
        file=sys.stderr,
    )

    # Build index
    print(
        f"[INFO] Building index (FAISS={'yes' if _HAS_FAISS else 'no'}) ...",
        file=sys.stderr,
    )
    index = RetrieverIndex(vectordb.embeddings)

    # Load questions
    print(f"[INFO] Loading questions from {args.questions_jsonl} ...", file=sys.stderr)
    all_items = load_questions_jsonl(args.questions_jsonl)
    if not all_items:
        print("[WARN] No questions found.", file=sys.stderr)
        return

    ids = [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids else []
    idxs = (
        [int(s.strip()) for s in args.idx.split(",") if s.strip()] if args.idx else []
    )
    range_spec = args.range_spec.strip() or None

    selected = select_questions(
        all_items, ids=ids or None, idxs=idxs or None, range_spec=range_spec
    )
    print(f"[INFO] Selected {len(selected)} question(s).", file=sys.stderr)

    # Load encoder
    print(
        f"[INFO] Loading query encoder '{model_name}' (device={args.device}) ...",
        file=sys.stderr,
    )
    encoder = QueryEncoder(
        model_name=model_name, device=args.device, max_length=args.max_length
    )

    # Instruction for Snowflake Arctic Embed
    instruction = (
        args.query_instruction.strip()
        or "Represent this sentence for searching relevant passages: "
    )

    # Output timestamp folder (KST yymmdd_hhmmss)
    tstamp_folder = _timestamp_folder_kst()
    out_paths = []

    # Run per question
    for q_item in selected:
        payload = run_retrieval_for_question(
            q_item=q_item,
            encoder=encoder,
            index=index,
            vectordb=vectordb,
            top_k=args.top_k,
            model_name=model_name,
            query_instruction=instruction,
        )
        path = save_question_result(
            out_root=args.output_root,
            timestamp_folder=tstamp_folder,
            qid=payload["id"],
            model_name=payload["model_name"],
            trial_id=payload["trial_id"],
            payload=payload,
        )
        out_paths.append(path)
        print(f"[OK] Saved: {path}", file=sys.stderr)

    print(
        json.dumps(
            {
                "saved_files": out_paths,
                "output_folder": os.path.join(args.output_root, tstamp_folder),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()