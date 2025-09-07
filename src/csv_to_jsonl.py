import pandas as pd
import json
import re


def csv_to_jsonl(csv_path, jsonl_path, keyword):
    # CSV 읽기
    df = pd.read_csv(csv_path)

    # 키워드가 포함된 컬럼 선택
    target_cols = [col for col in df.columns if keyword in col]

    if not target_cols:
        print(f"'{keyword}'가 포함된 컬럼이 없습니다.")
        return

    # 해당 컬럼들의 모든 데이터를 하나의 배열에 모으기
    values = []
    for col in target_cols:
        values.extend(df[col].dropna().astype(str).tolist())

    # 중복 제거
    unique_values = list(set(values))

    # JSONL 형식으로 저장
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, val in enumerate(unique_values):
            # 정규식으로 Title, Abstract, Source 추출
            title_match = re.search(r"Title:\s*(.*?)(?:, Abstract:|$)", val)
            abstract_match = re.search(r"Abstract:\s*(.*?)(?:, Source:|$)", val)
            source_match = re.search(r"Source:\s*(.*?)(?:,|$)", val)
            record = {
                "id": idx,
                "title": title_match.group(1).strip() if title_match else "",
                "abstract": abstract_match.group(1).strip() if abstract_match else "",
                "source": source_match.group(1).strip() if source_match else "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"저장 완료: {jsonl_path}")


# 사용 예시
csv_to_jsonl(
    "/workspace/src/data/scion/test.csv",
    "./docs.jsonl",
    keyword="retrieved_article_name_",
)
