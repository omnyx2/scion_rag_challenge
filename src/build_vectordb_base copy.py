import os
import json
import csv
import jsonlines
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_texts_from_csv(meta, text_fields, sample=True):
    csv_path = (
        meta["sample_csv"] if sample and "sample_csv" in meta else meta["saved_csv"]
    )
    df = pd.read_csv(os.path.join("/workspace/src", csv_path))  # base 경로 맞춰줌

    # 지정한 컬럼들을 합쳐서 하나의 string으로 만들기
    texts = []
    for _, row in df.iterrows():
        parts = [
            str(row[col]) for col in text_fields if col in row and pd.notna(row[col])
        ]
        if parts:
            texts.append(" ".join(parts))
    return texts


def save_as_csv(docs, embeddings, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "embedding"])
        for i, (doc, emb) in enumerate(zip(docs, embeddings)):
            writer.writerow([i, doc, emb.tolist()])


def save_as_jsonl(docs, embeddings, output_file):
    with jsonlines.open(output_file, mode="w") as writer:
        for i, (doc, emb) in enumerate(zip(docs, embeddings)):
            writer.write({"id": i, "text": doc, "embedding": emb.tolist()})


def main(config_path="../configs/query_encoder/config_1.json"):
    # 1. 설정 로드
    config = load_config(config_path)
    meta = load_meta(config["input_meta"])
    text_fields = config["text_fields"]

    # 2. 데이터 로드
    docs = build_texts_from_csv(meta, text_fields, sample=True)

    # 3. 모델 로드
    model_name = config["model_name"]
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # 4. 임베딩 생성
    print("Encoding documents...")
    embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=True)

    # 5. 저장 경로 생성
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    output_dir = os.path.join(config["output_dir"], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"{config['nickname']}_{safe_model_name}.{config['output_format']}"
    )

    # 6. 저장
    if config["output_format"] == "csv":
        save_as_csv(docs, embeddings, output_file)
    elif config["output_format"] == "jsonl":
        save_as_jsonl(docs, embeddings, output_file)
    else:
        raise ValueError("Unsupported output format: use 'csv' or 'jsonl'")

    # 7. config 업데이트 (output_file 기록)
    config["output_file"] = output_file
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"✅ VectorDB saved to {output_file}")


if __name__ == "__main__":
    main()
