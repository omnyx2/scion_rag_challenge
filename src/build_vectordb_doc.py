import os
import json
import csv
import jsonlines
from datetime import datetime
from sentence_transformers import SentenceTransformer



def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_docs(jsonl_path, embedding_mode="title+abstract"):
    """
    JSONL 파일에서 문서를 로드하고 지정된 모드에 따라 텍스트 생성
    
    Args:
        jsonl_path: JSONL 파일 경로
        embedding_mode: "title", "abstract", "title+abstract" 중 선택
    
    Returns:
        texts: 임베딩할 텍스트 리스트
        doc_ids: 문서 ID 리스트
    """
    texts = []
    doc_ids = []
    
    with jsonlines.open(jsonl_path) as reader:
        for doc in reader:
            doc_id = doc.get("id", "")
            title = doc.get("title", "")
            abstract = doc.get("abstract", "")
            
            if embedding_mode == "title":
                text = title
            elif embedding_mode == "abstract":
                text = abstract
            elif embedding_mode == "title+abstract":
                text = f"{title} {title} {title} {abstract}"
            else:
                raise ValueError(f"Invalid embedding_mode: {embedding_mode}")
            
            if text.strip():  # 빈 텍스트 제외
                texts.append(text)
                doc_ids.append(doc_id)
    
    return texts, doc_ids


def save_as_csv(doc_ids, texts, embeddings, output_file):
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "text", "embedding"])
        for doc_id, text, emb in zip(doc_ids, texts, embeddings):
            writer.writerow([doc_id, text, emb.tolist()])


def save_as_jsonl(doc_ids, texts, embeddings, output_file):
    with jsonlines.open(output_file, mode="w") as writer:
        for doc_id, text, emb in zip(doc_ids, texts, embeddings):
            writer.write({
                "doc_id": doc_id, 
                "text": text, 
                "embedding": emb.tolist()
            })


def main(config_path="../configs/query_encoder/config_jina.json"):
    # 1. 설정 로드
    config = load_config(config_path)
    
    # 2. JSONL 경로 및 임베딩 모드 설정
    jsonl_path = config.get("jsonl_path", "../data/expr/docs.jsonl")
    embedding_mode = config.get("embedding_mode", "title+abstract")  # 기본값: title+abstract
    
    # 3. 데이터 로드
    print(f"Loading documents from {jsonl_path}")
    print(f"Embedding mode: {embedding_mode}")
    texts, doc_ids = load_jsonl_docs(jsonl_path, embedding_mode)
    print(f"Loaded {len(texts)} documents")
    
    # 4. 모델 로드
    model_name = config["model_name"]
    print(f"Loading model: {model_name}")
   
    model = SentenceTransformer(model_name)
    
    # 5. 임베딩 생성
    print("Encoding documents...")
    task = "retrieval.query"
    embeddings = model.encode(texts, task=task, convert_to_numpy=True, show_progress_bar=True)

    # 6. 저장 경로 생성
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    output_dir = os.path.join(config["output_dir"], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일명에 임베딩 모드 포함
    safe_mode = embedding_mode.replace("+", "_")
    output_file = os.path.join(
        output_dir, 
        f"{config['nickname']}_{safe_mode}_{safe_model_name}.{config['output_format']}"
    )
    
    # 7. 저장
    if config["output_format"] == "csv":
        save_as_csv(doc_ids, texts, embeddings, output_file)
    elif config["output_format"] == "jsonl":
        save_as_jsonl(doc_ids, texts, embeddings, output_file)
    else:
        raise ValueError("Unsupported output format: use 'csv' or 'jsonl'")
    
    # 8. config 업데이트
    config["output_file"] = output_file
    config["jsonl_path"] = jsonl_path
    config["embedding_mode"] = embedding_mode
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"✅ VectorDB saved to {output_file}")
    print(f"✅ Embedding mode: {embedding_mode}")
    print(f"✅ Total documents embedded: {len(texts)}")


if __name__ == "__main__":
    main()