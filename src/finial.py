import os
import json
import pandas as pd
import re


def create_sorted_csv_from_json(folder_path, output_csv_filename):
    """
    지정된 폴더에서 row_{n}.json 패턴의 파일들을 읽어 'id'와 'result'를 추출하고,
    'id'의 숫자 값을 기준으로 내림차순 정렬하여 CSV 파일로 저장합니다.

    Args:
        folder_path (str): JSON 파일들이 있는 폴더의 경로.
        output_csv_filename (str): 저장할 CSV 파일의 이름.
    """
    all_data = []

    # 1. 폴더 존재 여부 확인
    if not os.path.isdir(folder_path):
        print(f"오류: '{folder_path}' 폴더를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # 2. 폴더 내의 모든 파일을 순회
    for filename in os.listdir(folder_path):
        # 'row_'로 시작하고 '.json'으로 끝나는 파일만 처리
        if filename.startswith("row_") and filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 'id'와 'result' 키가 있는지 확인 후 리스트에 추가
                    if "id" in data and "result" in data:
                        all_data.append({"id": data["id"], "result": data["result"]})
                    else:
                        print(
                            f"경고: '{filename}' 파일에 'id' 또는 'result' 키가 없습니다."
                        )
            except json.JSONDecodeError:
                print(f"경고: '{filename}' 파일은 유효한 JSON 형식이 아닙니다.")
            except Exception as e:
                print(f"'{filename}' 처리 중 오류 발생: {e}")

    # 3. 추출된 데이터가 없으면 함수 종료
    if not all_data:
        print("처리할 JSON 데이터가 없습니다.")
        return

    # 4. Pandas DataFrame으로 변환
    df = pd.DataFrame(all_data)

    # 5. 'id'에서 숫자 부분만 추출하여 정렬 기준으로 사용
    # 예: "row_000045" -> 45
    # 정규 표현식 '\d+'는 하나 이상의 연속된 숫자를 찾습니다.
    df["sort_key"] = df["id"].str.extract(r"(\d+)", expand=False).astype(int)

    # 'sort_key'를 기준으로 내림차순 정렬
    df_sorted = df.sort_values(by="sort_key", ascending=True)

    # 정렬에 사용된 'sort_key' 열은 제거
    df_sorted = df_sorted.drop(columns=["sort_key"])

    # 6. 결과를 CSV 파일로 저장
    try:
        df_sorted.to_csv(output_csv_filename, index=False, encoding="utf-8-sig")
        print(
            f"성공! 총 {len(df_sorted)}개의 데이터를 '{output_csv_filename}' 파일로 저장했습니다."
        )
        print("\nCSV 파일 상위 5개 행:")
        print(df_sorted.head())
    except Exception as e:
        print(f"CSV 파일 저장 중 오류 발생: {e}")


if __name__ == "__main__":
    # ⚠️ 여기에 실제 JSON 파일이 있는 폴더 경로를 입력하세요.
    target_folder_path = "/workspace/data/expr/final_result"

    # 저장될 CSV 파일의 이름을 지정합니다.
    output_filename = "final_result_sorted.csv"

    create_sorted_csv_from_json(target_folder_path, output_filename)
