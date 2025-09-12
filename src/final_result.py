import json
import os
import glob
import pandas as pd
import datetime


def process_json_files(directory_path):
    """
    Processes all 'row_*.json' files in a directory to extract, flatten, and save data to a CSV file.

    Args:
        directory_path (str): The path to the directory containing the JSON files.
    """
    # Find all JSON files matching the pattern 'row_*.json'
    file_paths = glob.glob(os.path.join(directory_path, "row_*.json"))

    # Sort files numerically based on the ID in the filename (e.g., row_000003)
    file_paths.sort(key=lambda f: int(os.path.basename(f).split("_")[-1].split(".")[0]))

    all_rows_data = []

    # Process each file
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # --- Data Extraction and Processing ---
            row_id = data.get("id")
            result_prediction = data.get("result")
            retrieval_data = data.get("retrival", {})

            # Collect all 'hits' from all 'queries'
            all_hits = []
            for query in retrieval_data.get("queries", []):
                all_hits.extend(query.get("hits", []))

            # Sort all hits by rank to prioritize higher-ranked documents
            all_hits.sort(key=lambda x: x.get("rank", float("inf")))

            # Extract unique texts, preserving the order based on the first appearance (by rank)
            unique_texts = []
            seen_texts = set()
            for hit in all_hits:
                title = hit.get("title")
                abstract = hit.get("abstract")
                source = hit.get("source")
                text = (
                    "Title: "
                    + title
                    + "\nAbstract: "
                    + abstract
                    + "\nSource: "
                    + source
                )
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_texts.append(text)

            # --- Row Creation ---
            # Create a dictionary for the current row
            row_data = {"id": row_id}

            # Add retrieved article texts, up to 50 columns
            for i in range(50):
                column_name = f"Prediction_retrieved_article_name_{i + 1}"
                if i < len(unique_texts):
                    row_data[column_name] = unique_texts[i]
                else:
                    # Fill remaining columns with "없음" if there are less than 50 texts
                    row_data[column_name] = "없음"

            # Add the final columns
            row_data["Prediction"] = result_prediction
            row_data["elapsed_times"] = 0.6758

            all_rows_data.append(row_data)

        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Could not process file {file_path}: {e}")
            continue

    if not all_rows_data:
        print("No data was processed. The CSV file will not be created.")
        return

    # Create a pandas DataFrame from the collected data
    df = pd.DataFrame(all_rows_data)

    # Define the column order
    column_order = (
        ["id"]
        + [f"Prediction_retrieved_article_name_{i}" for i in range(1, 51)]
        + ["Prediction", "elapsed_times"]
    )
    df = df[column_order]
    return df


def append_new_data_frame_to_base_csv(
    base_csv_uri, output_folder_path, output_file_name, new_dataframe
):
    # 3. 지정된 경로의 CSV 파일을 읽어와 데이터프레임으로 변환
    try:
        my_df = pd.read_csv(base_csv_uri)
        new_dataframe.drop(columns="elapsed_times", inplace=False)
        print(f"--- '{base_csv_uri}' 파일에서 읽어온 데이터프레임 ---")
        print(new_dataframe)
        print("\n" + "=" * 30 + "\n")
        new_dataframe.drop(columns="id", inplace=False)
        # 4. 기존 데이터프레임과 새로 읽어온 데이터프레임 연결(병합)
        # ignore_index=True 옵션은 두 데이터프레임의 인덱스를 새로 정렬합니다.
        combined_df = pd.concat([my_df, new_dataframe], axis=1)

        print("--- 두 데이터프레임을 병합한 결과 ---")
        print(combined_df)
        print("\n" + "=" * 30 + "\n")

        # 5. 저장할 폴더가 없으면 생성
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
            print(f"'{output_folder_path}' 폴더가 없어 새로 생성했습니다.")

        # 6. 최종 데이터프레임을 새로운 CSV 파일로 저장
        output_path = os.path.join(output_folder_path, output_file_name)
        print(output_path)
        # index=False 옵션은 데이터프레임의 인덱스를 파일에 쓰지 않도록 합니다.
        combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"성공: 병합된 데이터가 '{output_path}' 경로에 저장되었습니다.")

    except FileNotFoundError:
        print(
            f"오류: '{output_file_name}' 파일을 찾을 수 없습니다. 경로를 확인해주세요."
        )
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        print(f"Successfully created CSV file at: {output_file_name}")


# --- Execution Example ---
if __name__ == "__main__":
    # Define the path to your JSON files.
    # IMPORTANT: Replace this with the actual path to your files.
    # For example: '/workspace/results/final_result/'
    # 현재 날짜와 시간을 가져옵니다.
    now = datetime.datetime.now()

    # 원하는 형식(yymmdd_hhmmss)으로 문자열을 만듭니다.
    timestamp_str = now.strftime("%y%m%d_%H%M%S")
    retrival_docs_folder = "/workspace/data/expr/final_result"
    output_directory = (
        "/workspace/data/expr/competition_submission/" + timestamp_str + "/"
    )  # Assuming the files are in the same directory as the script for this example

    # Define the desired path for the output CSV file.
    output_file = "final_predictions_" + timestamp_str + ".csv"
    base_csv_uri = "/workspace/data/rag_test_data/scion.csv"
    # Create dummy files for demonstration purposes
    # In your real case, you would already have these files.
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    new_df = process_json_files(retrival_docs_folder)
    print(new_df)
    append_new_data_frame_to_base_csv(
        base_csv_uri, output_directory, output_file, new_df
    )
