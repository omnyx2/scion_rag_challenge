import os
import json
import csv
import jsonlines
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_config(config_path="../configs/query_encoder/config_1.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vdb(output_file):
    """CSV 또는 JSONL에서 VectorDB 불러오기"""
    if output_file.endswith(".csv"):
        df = pd.read_csv(output_file)
        df["embedding"] = df["embedding"].apply(lambda x: np.array(eval(x)))
    elif output_file.endswith(".jsonl"):
        records = []
        with jsonlines.open(output_file, "r") as reader:
            for row in reader:
                row["embedding"] = np.array(row["embedding"])
                records.append(row)
        df = pd.DataFrame(records)
    else:
        raise ValueError("Unsupported format. Use CSV or JSONL.")
    return df


def retrieve(query, config_path="../configs/query_encoder/config_1.json", top_k=3):
    # 1. 설정 불러오기
    config = load_config(config_path)
    output_file = config.get("output_file", None)
    if not output_file or not os.path.exists(output_file):
        raise FileNotFoundError("❌ VectorDB not found. 먼저 main.py로 생성하세요.")

    # 2. DB 로드
    df = load_vdb(output_file)

    # 3. 모델 로드
    model_name = config["model_name"]
    model = SentenceTransformer(model_name)

    # 4. 쿼리 임베딩
    query_emb = model.encode([query], convert_to_numpy=True)

    # 5. 유사도 계산
    embeddings = np.stack(df["embedding"].to_numpy())
    sims = cosine_similarity(query_emb, embeddings)[0]

    # 6. Top-k 추출
    top_idx = np.argsort(sims)[::-1][:top_k]
    results = df.iloc[top_idx].copy()
    results["similarity"] = sims[top_idx]
    return results[["id", "text", "similarity"]]


if __name__ == "__main__":
    query = "What is the role of AI in healthcare?"
    results = retrieve(quzery, top_k=5)
    print("🔎 Query:", query)
    print(results.to_string(index=False))
