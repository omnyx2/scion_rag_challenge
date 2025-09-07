import os
import json
import glob
import argparse
import copy
from typing import List, Dict, Optional
import google.generativeai as genai

# --- Gemini API 호출을 위한 기본 설정 및 함수 ---
"""

python preprocess_and_generate_answer.py     --input_dir "/home/hyunseok/Projects/PythonPlayground/results/retrival_docs/250907_164655_PwC_3title_abstact/"     --max_rank 3
"""

def init_gemini(
    model_name: str, api_key: str, temperature: float, seed: Optional[int]
) -> genai.GenerativeModel:
    """
    Gemini 모델을 초기화합니다.
    사용자 제공 코드 기반.
    """
    genai.configure(api_key=api_key)
    generation_config = genai.GenerationConfig(
        temperature=temperature,
        candidate_count=1,
    )
    # 결정론적 출력을 위해 시드 설정
    if seed is not None:
        generation_config.seed = seed

    return genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )


def call_gemini(model: genai.GenerativeModel, messages: List[Dict[str, str]]) -> str:
    """
    Gemini 모델을 호출하고 응답 텍스트를 반환합니다.
    사용자 제공 코드 기반, 예외 처리 추가.
    """
    try:
        # messages는 [{'role': 'user', 'parts': [prompt]}] 형태의 리스트
        prompt = messages[0]["parts"][0]
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return f"API 호출 중 오류 발생: {e}"


# --- 사용자 요청 기능 구현 ---


def build_prompt(original_query: str, context_json_string: str) -> str:
    """
    최종 질문과 JSON 컨텍스트 문자열을 받아 API 프롬프트를 생성합니다.
    """
    prompt_template = """You will be given a JSON object as a string which contains a series of related search queries and their retrieved documents ('hits'). Do not make answer from external knowledge. You must make answer inside of Context.
Your main task is to answer the specific 'Question' provided below. Use the entire JSON data as context to formulate your answer, paying close attention to the 'text' fields within the 'hits' lists.

The JSON data has a list of queries. The 'original' query is the one you need to answer. The other queries are supplementary and provide additional context.

If the 'Question' is in Korean, format your answer in Korean as follows:
##제목##

##서론##

##본론##

##결론##

If the 'Question' is in English, format your answer in English as follows, If English then Just write title inside of ##{{Title}}##:
##{{Title}}##

##Introduction##

##Main Body##

##Conclusion##

--- Context (JSON Data) ---
{context}

--- Question ---
{query}
"""
    return prompt_template.format(context=context_json_string, query=original_query)


def process_file(
    filepath: str, max_rank: int, model_obj: genai.GenerativeModel
) -> List[Dict]:
    """
    단일 JSON 파일을 처리하는 주 함수입니다.
    1. 파일을 읽고 'original' 타입의 질문을 찾습니다.
    2. 전체 데이터의 'hits'를 max_rank 기준으로 필터링합니다.
    3. 필터링된 전체 JSON 객체를 문자열로 변환하여 컨텍스트로 사용합니다.
    4. 'original' 질문과 컨텍스트로 Gemini API를 호출합니다.
    5. 최종 결과를 지정된 형식으로 반환합니다.
    """
    print(f"--- Processing file: {os.path.basename(filepath)} ---")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading or parsing {filepath}: {e}")
        return []

    # 1. 'original' 타입의 쿼리를 찾습니다.
    original_query_obj = next(
        (
            q
            for q in data.get("queries", [])
            if q.get("query_meta", {}).get("type") == "original"
        ),
        None,
    )

    if not original_query_obj:
        print(f"Warning: No 'original' query found in {filepath}. Skipping file.")
        return []

    original_query_text = original_query_obj.get("query")
    if not original_query_text:
        print(f"Warning: 'original' query in {filepath} has no text. Skipping file.")
        return []

    # 2. 전체 데이터를 복사하고 모든 query의 'hits'를 필터링합니다.
    filtered_data = copy.deepcopy(data)
    for query_item in filtered_data.get("queries", []):
        hits = query_item.get("hits", [])
        # rank가 max_rank 이하인 것들만 남깁니다.
        query_item["hits"] = [
            hit for hit in hits if hit.get("rank", float("inf")) <= max_rank
        ]

    # 3. 필터링된 전체 데이터를 컨텍스트용 JSON 문자열로 변환합니다.
    context_str = json.dumps(filtered_data, ensure_ascii=False, indent=2)

    # 4. 프롬프트를 생성하고 API를 호출합니다.
    prompt = build_prompt(original_query_text, context_str)
    messages = [{"role": "user", "parts": [prompt]}]
    api_result = call_gemini(model_obj, messages)

    # 5. 최종 결과 데이터를 구성합니다.
    file_id = data.get("id", os.path.basename(filepath).split("_")[1])
    model_name = model_obj.model_name
    result_data = {
        "id": file_id,
        "result": api_result,
        "prompt": prompt,
        "model": model_name,
        "retrival": filtered_data,
    }

    print(f"Successfully processed file {os.path.basename(filepath)}.")
    return [result_data]  # 파일 당 하나의 결과가 있으므로 리스트에 담아 반환


def main():
    """
    스크립트의 메인 실행 함수.
    """
    parser = argparse.ArgumentParser(
        description="JSON 파일을 전처리하고 Gemini API를 사용하여 답변을 생성합니다."
    )
    # 필수 인자
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="처리할 JSON 파일이 있는 입력 디렉토리 경로.",
    )
    parser.add_argument(
        "--max_rank",
        type=int,
        required=True,
        help="포함할 최대 'hit' 순위. 이 순위보다 높은 'hit'은 제외됩니다.",
    )
    # API 관련 인자 (사용자 제공 코드 기반)
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="사용할 Gemini 모델 이름 (예: gemini-1.5-pro-latest, gemini-1.5-flash-latest).",
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("GOOGLE_API_KEY"),
        help="Google API 키. 설정하지 않으면 GOOGLE_API_KEY 환경 변수를 사용합니다.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    # 출력 디렉토리 인자
    parser.add_argument(
        "--output_dir",
        default="/workspace/data/expr/final_result",
        help="각 결과를 개별 JSON 파일로 저장할 디렉토리 경로.",
    )

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError(
            "API 키가 필요합니다. --api_key 인자를 사용하거나 GOOGLE_API_KEY 환경 변수를 설정해주세요."
        )

    # Gemini 모델 초기화
    model_obj = init_gemini(args.model, args.api_key, args.temperature, args.seed)

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 입력 디렉토리에서 모든 관련 JSON 파일 찾기
    search_pattern = os.path.join(args.input_dir, "row_*.json")
    file_paths = glob.glob(search_pattern)

    if not file_paths:
        print(
            f"경고: '{args.input_dir}'에서 'row_*.json' 패턴과 일치하는 파일을 찾을 수 없습니다."
        )
        return

    processed_count = 0
    for path in file_paths:
        processed_data_list = process_file(path, args.max_rank, model_obj)

        if processed_data_list:
            result_data = processed_data_list[0]
            result_id = result_data["id"]
            output_filename = os.path.join(args.output_dir, f"{result_id}.json")

            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=4)

            processed_count += 1

    print(
        f"\n✅ 처리가 완료되었습니다. 총 {processed_count}개의 결과 파일이 '{args.output_dir}'에 저장되었습니다."
    )


if __name__ == "__main__":
    main()
