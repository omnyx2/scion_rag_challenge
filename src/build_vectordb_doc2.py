#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VectorDB Builder with Document Embeddings

Builds vector database from JSONL documents using various embedding models.
Supports multiple embedding modes and saves results as CSV or JSONL.

Example:
    python build_vectordb_doc2.py \
      --config ../configs/query_encoder/config_bge_m3.json \
      --embedding_mode title+abstract 
      
"""

import argparse
import csv
import json
import jsonlines
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_docs(jsonl_path: str, embedding_mode: str = "title+abstract") -> tuple[list[str], list[str]]:
    """
    Load documents from JSONL file and generate text based on embedding mode.
    
    Args:
        jsonl_path: Path to JSONL file
        embedding_mode: "title", "abstract", or "title+abstract"
    
    Returns:
        texts: List of texts for embedding
        doc_ids: List of document IDs
    """
    texts, doc_ids = [], []
    
    with jsonlines.open(jsonl_path) as reader:
        for doc in reader:
            doc_id = doc.get("id", "")
            title = doc.get("title", "").strip()
            abstract = doc.get("abstract", "").strip()
            
            if embedding_mode == "title":
                text = title
            elif embedding_mode == "abstract":
                text = abstract
            elif embedding_mode == "title+abstract":
                text = f"{title} {abstract}".strip()
            else:
                raise ValueError(f"Invalid embedding_mode: {embedding_mode}. Use 'title', 'abstract', or 'title+abstract'")
            
            if text:  # Skip empty texts
                texts.append(text)
                doc_ids.append(doc_id)
    
    return texts, doc_ids


def save_vectordb(doc_ids: list[str], texts: list[str], embeddings, output_file: str, format_type: str) -> None:
    """Save vector database to file."""
    if format_type == "csv":
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["doc_id", "text", "embedding"])
            for doc_id, text, emb in zip(doc_ids, texts, embeddings):
                writer.writerow([doc_id, text, emb.tolist()])
    elif format_type == "jsonl":
        with jsonlines.open(output_file, mode="w") as writer:
            for doc_id, text, emb in zip(doc_ids, texts, embeddings):
                writer.write({
                    "doc_id": doc_id,
                    "text": text,
                    "embedding": emb.tolist()
                })
    else:
        raise ValueError("Unsupported output format: use 'csv' or 'jsonl'")


def create_output_path(config: dict, embedding_mode: str) -> str:
    """Generate output file path with timestamp and model info."""
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    safe_model_name = config["model_name"].replace("/", "_")
    safe_mode = embedding_mode.replace("+", "_")
    
    output_dir = os.path.join(config["output_dir"], timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{config['nickname']}_{safe_mode}_{safe_model_name}.{config['output_format']}"
    return os.path.join(output_dir, filename)


def main():
    parser = argparse.ArgumentParser(description="Build VectorDB from JSONL documents")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to config JSON file")
    parser.add_argument("--embedding_mode", type=str, 
                       choices=["title", "abstract", "title+abstract"],
                       help="Embedding mode (overrides config)")
    parser.add_argument("--jsonl_path", type=str,
                       help="Path to JSONL file (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine paths and settings (CLI args override config)
    jsonl_path = args.jsonl_path or config.get("jsonl_path", "data/expr/docs.jsonl")
    embedding_mode = args.embedding_mode or config.get("embedding_mode", "title+abstract")
    
    # Load documents
    print(f"Loading documents from {jsonl_path}")
    print(f"Embedding mode: {embedding_mode}")
    texts, doc_ids = load_jsonl_docs(jsonl_path, embedding_mode)
    print(f"Loaded {len(texts)} documents")
    
    # Load model
    model_name = config["model_name"]
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    
    # Generate embeddings
    print("Encoding documents...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    # Save results
    output_file = create_output_path(config, embedding_mode)
    save_vectordb(doc_ids, texts, embeddings, output_file, config["output_format"])
    
    # Update config with results
    config.update({
        "output_file": output_file,
        "jsonl_path": jsonl_path,
        "embedding_mode": embedding_mode
    })
    
    with open(args.config, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"✅ VectorDB saved to {output_file}")
    print(f"✅ Embedding mode: {embedding_mode}")
    print(f"✅ Total documents embedded: {len(texts)}")


if __name__ == "__main__":
    main()