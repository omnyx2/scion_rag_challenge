# embedding_processor.py
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


def generate_batch_embeddings(
    documents_data: List[Dict],
    model_name: str,
    truncate_dimension: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    SentenceTransformer 모델을 로드하고 주어진 문서들의 임베딩을 생성합니다.

    Args:
        documents_data (List[Dict]): 각 딕셔너리가 문서를 나타내는 리스트.
        model_name (str): 사용할 SentenceTransformer 모델의 이름 또는 경로.
        truncate_dimension (Optional[int]): 임베딩을 잘라낼 차원.
                                            None일 경우 모델의 기본 출력 차원을 사용합니다.

    Returns:
        Optional[np.ndarray]: 문서 임베딩의 numpy 배열. 실패 시 None을 반환합니다.
    """
    if not documents_data:
        print("⚠️ 임베딩을 생성할 문서가 없습니다.")
        return None

    # 트러케이션(truncation) 적용 여부 안내
    if truncate_dimension:
        print(
            f"모델 로딩: {model_name} (임베딩 차원을 {truncate_dimension}으로 조절합니다)"
        )
    else:
        print(f"모델 로딩: {model_name}")

    try:
        # 모델 로드 시 truncate_dim 파라미터 전달
        model = SentenceTransformer(
            model_name, trust_remote_code=True, truncate_dim=truncate_dimension
        )
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생 ({model_name}): {e}")
        return None

    # 임베딩할 텍스트 추출
    texts_to_embed = [doc.get("embedding_text", "") for doc in documents_data]
    if not any(texts_to_embed):
        print(
            "⚠️ 경고: 모든 문서에서 'embedding_text' 키를 찾을 수 없거나 값이 비어있습니다."
        )
        return None

    print("문서 인코딩 중...")
    embeddings = model.encode(
        texts_to_embed, convert_to_numpy=True, show_progress_bar=True
    )

    if embeddings is not None:
        print(f"✅ 임베딩 생성 완료! (최종 차원: {embeddings.shape[1]})")

    return embeddings
