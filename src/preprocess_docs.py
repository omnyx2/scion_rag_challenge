# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
from pathlib import Path

# 1) Load the uploaded file
csv_path = "/workspace/src/data/scion/test.csv"
df = pd.read_csv(
    csv_path, dtype=str, keep_default_na=False
)  # read everything as string to avoid NaNs

# 2) Drop any columns that look like "prediction" signals
prediction_like_cols = [
    c
    for c in df.columns
    if re.search(r"pred(ic)?t(ion|ed|ions)?|^pred$|^y_pred$|prediction", c, re.I)
]
df_no_pred_cols = df.drop(columns=prediction_like_cols, errors="ignore")

# 3) Identify candidate text columns (object dtype)
text_cols = [c for c in df_no_pred_cols.columns if df_no_pred_cols[c].dtype == object]


# Heuristic: choose the column with the highest average string length as the main text column
def avg_len(col):
    return df_no_pred_cols[col].apply(lambda x: len(str(x))).mean()


print(text_cols)

if text_cols:
    main_text_col = max(text_cols, key=avg_len)
else:
    # If no text columns are found, create an empty DataFrame
    main_text_col = None

# 4) Build a cleaned text column by concatenating all remaining text columns (for broader recall)
if text_cols:
    # Clean whitespace & join all non-empty text fields per row
    df_no_pred_cols["_joined_text"] = df_no_pred_cols[text_cols].apply(
        lambda row: " ".join(
            [str(v).strip() for v in row.values if str(v).strip() != ""]
        ),
        axis=1,
    )
else:
    df_no_pred_cols["_joined_text"] = ""

# 5) Define retrieval-related keyword patterns (covering common spellings and terms)
retrieval_keywords = [
    r"\bretrieval\b",
    r"\bretriver\b",
    r"\bretriever\b",
    r"\bretrival\b",
    r"\bretrival\b",
    r"\bretrival\b",
    r"\bretrived\b",
    r"\bRAG\b",
    r"\bretrieval-augmented\b",
    r"\bBM25\b",
    r"\bvector\s*search\b",
    r"\bsemantic\s*search\b",
    r"\bindex(er|ing)?\b",
    r"\bembedding(s)?\b",
    r"\bANN\b",
    r"\bFAISS\b",
    r"\bMilvus\b",
    r"\bPinecone\b",
    r"\bWeaviate\b",
    r"\bchroma(db)?\b",
    r"\bdocument\s*store\b",
    r"\bdense\s*retriev(al|er)\b",
    r"\bsparse\s*retriev(al|er)\b",
    r"\btop[- ]?k\b",
    r"\brecall@?\d*\b",
    r"\bmrr\b",
    r"\bcolbert\b",
    r"\bcontriever\b",
    r"\bquery\s*(embedding|expansion)\b",
    r"\brelevance\s*feedback\b",
    r"\bBM25L?\b",
    r"\bhybrid\s*search\b",
    r"\bkeyword\s*search\b",
]
retrieval_pattern = re.compile("|".join(retrieval_keywords), re.IGNORECASE)

# 6) Filter rows whose aggregated text mentions retrieval-ish concepts
df_retrieval = df_no_pred_cols[
    df_no_pred_cols["_joined_text"].apply(
        lambda x: bool(retrieval_pattern.search(str(x)))
    )
].copy()

# 7) Build a canonical "document text" per row
# Prefer the main_text_col if it exists, otherwise use the _joined_text
print(main_text_col)
if main_text_col:
    doc_texts = df_retrieval[main_text_col].fillna("").astype(str).tolist()
else:
    doc_texts = df_retrieval["_joined_text"].fillna("").astype(str).tolist()


# 8) Normalize text for deduplication (lowercase, collapse whitespace)
def normalize_text(t):
    t = str(t)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


normalized = [normalize_text(t) for t in doc_texts]

# 9) Deduplicate using 99% similarity threshold
unique_indices = []
kept_norm_texts = []

for i, text in enumerate(normalized):
    is_dup = False
    for kept in kept_norm_texts:
        ratio = SequenceMatcher(a=kept, b=text).ratio()
        if ratio >= 0.99:  # 99% or higher considered duplicate
            is_dup = True
            break
    if not is_dup:
        unique_indices.append(i)
        kept_norm_texts.append(text)

# 10) Collect the unique documents (original, non-normalized)
unique_docs = [doc_texts[i] for i in unique_indices]

# 11) Save to a single TXT file and also a CSV for convenience
output_txt = "./data/retrieval_docs.txt"
output_csv = "./data/retrieval_docs.csv"

# Write TXT with separators
with open(output_txt, "w", encoding="utf-8") as f:
    for idx, doc in enumerate(unique_docs, 1):
        f.write(f"### Document {idx}\n")
        f.write(doc)
        f.write("\n\n---\n\n")

# Write CSV with a single column
pd.DataFrame({"document": unique_docs}).to_csv(
    output_csv, index=False, encoding="utf-8"
)
