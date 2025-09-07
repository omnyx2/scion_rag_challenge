import os
from llm.llm_client import GeminiClient

# 경고: 이 코드를 실행하기 전에 'GOOGLE_API_KEY' 환경 변수를 설정해야 합니다.
# 예시: os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY_HERE'
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("오류: GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
    exit()

try:
    # 1. GeminiClient 객체 생성
    # 모델 이름을 'gemini-pro'로 설정하고, 최대 재시도 횟수는 3회로 지정합니다.
    client = GeminiClient(api_key=api_key, model_name="gemini-pro", max_retries=3)
    print("Gemini 클라이언트가 성공적으로 초기화되었습니다.")

    # 2. 일반(Uninstruct) 프롬프트 실행
    # 자유로운 대화나 일반적인 질문에 사용됩니다.
    print("\n--- Uninstruct 프롬프트 실행 ---")
    uninstruct_prompt = "대한민국에서 가장 높은 산은 어디인가요?"
    uninstruct_response = client.generate_uninstruct(uninstruct_prompt)
    print(f"프롬프트: {uninstruct_prompt}")
    print(f"응답: {uninstruct_response}")

    # 3. 지시(Instruct) 프롬프트 실행
    # 특정 작업을 수행하도록 명확한 지시를 내릴 때 사용됩니다.
    print("\n--- Instruct 프롬프트 실행 ---")
    instruct_prompt = "다음 문장을 15자 이내로 요약해 주세요: '이것은 인공지능이 제공하는 객체 지향적 LLM 클라이언트의 파이썬 코드 사용 예시입니다.'"
    system_prompt = "당신은 요약 전문가입니다."
    instruct_response = client.generate_instruct(instruct_prompt, system_prompt=system_prompt)
    print(f"프롬프트: {instruct_prompt}")
    print(f"시스템 프롬프트: {system_prompt}")
    print(f"응답: {instruct_response}")

    # 4. 이미지 프롬프트 실행 (로컬 이미지 파일 필요)
    # 이미지 파일 경로를 올바르게 지정해야 합니다.
    # 예: 'cat_image.jpg'라는 파일이 스크립트와 같은 디렉토리에 있다고 가정합니다.
    image_path = "cat_image.jpg"
    if os.path.exists(image_path):
        print("\n--- 이미지 프롬프트 실행 ---")
        image_prompt = "이 이미지에 대해 묘사해 주세요."
        # 'gemini-pro-vision' 모델은 이미지 처리 전용이므로,
        # GeminiClient의 모델 이름을 이 모델로 변경하여 사용해야 합니다.
        client_vision = GeminiClient(api_key=api_key, model_name="gemini-pro-vision")
        image_response = client_vision.generate_with_image(image_prompt, image_path)
        print(f"이미지 경로: {image_path}")
        print(f"프롬프트: {image_prompt}")
        print(f"응답: {image_response}")
    else:
        print(f"\n경고: '{image_path}' 경로에 파일이 존재하지 않아 이미지 테스트를 건너뜁니다.")

    # 5. 토큰 사용량 및 프롬프트 히스토리 확인
    # 모든 요청이 종료된 후, 누적된 정보를 조회합니다.
    print("\n--- 최종 결과 확인 ---")
    print(f"총 사용 토큰 수: {client.get_token_usage()} 토큰")
    print("프롬프트 히스토리:")
    for i, item in enumerate(client.get_prompt_history(), 1):
        print(f"  {i}. 타입: {item['response_type']}, 프롬프트: '{item['prompt']}'")

except ValueError as e:
    print(f"오류: 클라이언트 초기화 실패 - {e}")
except ConnectionError as e:
    print(f"네트워크 오류: API 호출 실패 - {e}")
