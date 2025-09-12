# scion_rag_challenge

Basically to excute code, get inside of src and run it.
this use docker compose and devcontainer for ~~~
here is how to execute

0. open vscode and use dev container to set environments
1. Prepare the question docs and go to /workspace/search_science_on_chellenge
2. put text.csv under /workspace
3. execute below

```bash
TARGET_DOCUMENTS=50 python main.py batch test.csv  # 각 질문당 100개 문서
```

4. you needs api key for gemini and science_on_api
5. take /workspace/search_science_on_chellenge/outputs/search_documents_20250912_013206.jsonl to /workspace/data/expr
6. Move to src/ and execute line by line in main.ipynb
7. if you want to execute fast you and you the command in the ipynb shell to bash directly ipynb require to setup to use gpu

> Other Option

```
python multi_hop_to_single_hop.py \
  --input /workspace/data/rag_test_data/questions.jsonl \
  --output /workspace/data/expr/singlehop_decompose.jsonl \
  --question_field question --context_field context \
  --mode decompose --model gemini-2.5-flash


python -m retrieval_system.main --config_json ../configs/query_encoder/config_gte-multilingual-base.json --questions_jsonl /workspace/data/expr singlehop_decompose.jsonl  --range  1-50 \ --top_k  50 \ --device  auto \ --schema_json  /workspace/configs/csv_schema/

python preprocess_and_generate_answer.py \
  --input_dir /workspace/results/retrival_docs/250908_235532 --max_rank 1 --parallel True

python final_result.py

```

8. results are in /workspace/data/expr/final_result, once you execute preprocess_and_generate_answer.py, it will overwrite so make sure execute once or and the name of folder
