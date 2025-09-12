# scion_rag_challenge

Basically to excute code, get inside of src and run it.
this use docker compose and devcontainer for ~~~
here is how to execute

1. prepare Query

   1. put test.csv to /workspace/data/rag_test_data
   2. extract column "Question" from text.csv and make query list
   3. You can optionally seprate question as Multi-hop and Single-hop with multi_hop_to_single_hop.py

2. prepare the query_encoding model for multilingual

   1. make a hugginface id and get token for desire model
   2. we already put model in the configs, in this work we mainly use Alibaba-NLP/gte-multilingual-base

3. prepare base vectorDataBase

   1. using Query we extract Search the docs from base system(scion)
   2. searched file will generate at
   3. choose a model for query_encoding with path of config file
   4.

4.

python3 extract_questions.py \
 --input /workspace/data/rag_test_data/scion.csv \
 --output /workspace/data/rag_test_data/questions.jsonl \
 --question-col Question

python3 build_vectordb_search.py

python3 multi_hop_to_single_hop.py \
 --input /workspace/data/rag_test_data/questions.jsonl \
 --output /workspace/data/expr/singlehop_decompose.jsonl \
 --question_field question --context_field context \
 --mode decompose --model gemini-2.5-flash

python retriever_methods/dense_retrieval_model_2.py \
 --vectordb_csv /workspace/results/vectordb/250908_140123/vector_db_gte_test_Alibaba-NLP_gte-multilingual-base.csv \
 --config_json /workspace/configs/query_encoder/config_gte-multilingual-base.json \
 --questions_jsonl /workspace/data/expr/singlehop_decompose.jsonl \
 --range 1-50 \
 --top_k 50 \
 --device auto
