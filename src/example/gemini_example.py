# examples/gemini_example.py

import asyncio
import sys
import os
import httpx
import base64

# 'src' 경로를 sys.path에 추가하여 llm_client 패키지를 찾을 수 있도록 설정
# 이 스크립트가 'examples' 폴더 안에 있다고 가정합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_client.gemini_client import GeminiClient


# --- Helper Function ---
async def image_url_to_base64(url: str) -> str:
    """
    주어진 URL에서 이미지를 비동기적으로 다운로드하고 Base64로 인코딩합니다.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()  # HTTP 오류가 발생하면 예외를 일으킴
        return base64.b64encode(response.content).decode("utf-8")


# --- Main Logic ---
async def main():
    """
    수정된 GeminiClient의 주요 기능(텍스트, 채팅, 이미지)을 시연합니다.
    """
    # 중요: GOOGLE_API_KEY 환경 변수가 설정되어 있어야 합니다.
    # 예: export GOOGLE_API_KEY='your_api_key_here'
    if not os.getenv("GOOGLE_API_KEY"):
        print("🔴 GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("실행 전 'export GOOGLE_API_KEY=...' 명령어를 실행해주세요.")
        return

    # 이미지(멀티모달)를 지원하는 모델로 클라이언트 초기화
    client = GeminiClient(model="gemini-2.5-flash", history_enabled=True)
    print(f"✅ GeminiClient 초기화 완료 (Model: {client.model})")
    print("-" * 50)

    # 1. 텍스트 생성 예시 ✍️
    print("\n1️⃣  간단한 텍스트 생성 요청")
    client.add_user("파이썬의 데코레이터(decorator)에 대해 간결하게 설명해줘.")

    res_text = await client.agenerate(temperature=0.3)

    print("\n[🤖 Gemini 응답 - 텍스트]")
    print(res_text.text)
    print(
        f"(Prompt Tokens: {res_text.prompt_tokens}, Completion Tokens: {res_text.completion_tokens})"
    )
    print("-" * 50)

    # 2. 대화(채팅) 예시 💬
    print("\n2️⃣  대화 연속성을 활용한 추가 요청")
    client.add_user("방금 설명한 데코레이터를 사용하는 간단한 코드 예시를 보여줘.")

    res_chat = await client.agenerate(temperature=0.3)

    print("\n[🤖 Gemini 응답 - 채팅]")
    print(res_chat.text)
    print(
        f"(Prompt Tokens: {res_chat.prompt_tokens}, Completion Tokens: {res_chat.completion_tokens})"
    )
    print("-" * 50)

    # 3. 이미지(멀티모달) 생성 예시 🖼️
    print("\n3️⃣  이미지와 텍스트를 함께 사용한 멀티모달 요청")
    # 대화 기록을 초기화하여 새로운 주제로 시작
    client.clear_history()

    # 공개된 이미지 URL (고양이 사진)
    image_url = (
        "https://storage.googleapis.com/static.aifor.fun/docs/cat-with-sunglasses.jpg"
    )
    print(f"이미지를 다음 URL에서 가져옵니다: {image_url}")

    try:
        # 이미지를 Base64로 인코딩
        base64_image = await image_url_to_base64(image_url)

        # 텍스트 프롬프트와 인코딩된 이미지 데이터를 함께 전달
        client.add_user(
            "이 이미지에 대해 재미있는 한 문장으로 묘사해줘.",
            images=[base64_image],  # Base64 인코딩된 문자열을 리스트에 담아 전달
        )

        res_image = await client.agenerate(max_tokens=100)

        print("\n[🤖 Gemini 응답 - 이미지]")
        print(res_image.text)
        print(
            f"(Prompt Tokens: {res_image.prompt_tokens}, Completion Tokens: {res_image.completion_tokens})"
        )

    except httpx.HTTPStatusError as e:
        print(f"🔴 이미지 다운로드 실패: {e}")
    except Exception as e:
        print(f"🔴 멀티모달 요청 중 오류 발생: {e}")
    finally:
        print("-" * 50)


if __name__ == "__main__":
    # 비동기 main 함수 실행
    asyncio.run(main())
