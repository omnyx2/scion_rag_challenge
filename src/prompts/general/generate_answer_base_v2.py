def build_prompt(original_query: str, context_json_string: str) -> str:
    """
    최종 질문과 JSON 컨텍스트 문자열을 받아 API 프롬프트를 생성합니다.
    """
    prompt_template = """You will be given a JSON object as a string which contains a series of related search queries and their retrieved documents ('hits'). Do not make answer from external knowledge. You must make answer inside of Context.
Your main task is to answer the specific 'Question' provided below. Use the entire JSON data as context to formulate your answer, paying close attention to the 'text' fields within the 'hits' lists.

The JSON data has a list of queries. The 'original' query is the one you need to answer. The other queries are supplementary and provide additional context.
Do not use "*", "-" symbols. Please don't use "**" to emphasize words. Don't answer in markdown format just write it simple.

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

--- Context ---
{context}

--- Question ---
{query}

"""
    return prompt_template.format(context=context_json_string, query=original_query)
