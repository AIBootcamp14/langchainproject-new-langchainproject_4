# src/modules/llm.py

"""
LLM 및 임베딩 모델 설정
Upstage Solar API를 사용한 한국어 최적화 모델 정의
"""

import os
from typing import Optional

# 써드파티 라이브러리
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_upstage import ChatUpstage, UpstageEmbeddings

# 프로젝트에서는 SOLAR_API_KEY를 환경 변수로 사용함.
SOLAR_API_KEY_ENV_NAME = "SOLAR_API_KEY"


def _check_api_key(api_key: Optional[str]) -> None:
    """
    API 키 존재 여부를 확인하고, 없으면 에러 발생. (내부 헬퍼 함수)
    """
    if not api_key or api_key.startswith("YOUR_"):
        raise ValueError(
            f"{SOLAR_API_KEY_ENV_NAME}가 환경 변수에 설정되지 않았습니다. "
            "'.env' 파일을 확인하거나 Upstage 콘솔에서 API 키를 발급받으세요."
        )


def get_llm(
    model: str = "solar-pro",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    streaming: bool = False,
) -> BaseChatModel:
    """
    Upstage Solar LLM 인스턴스를 반환합니다.

    Args:
        model: 사용할 모델명 ("solar-pro" 또는 "solar-1-mini-chat").
        temperature: 생성 온도 값 (0.0 ~ 1.0).
        max_tokens: 최대 토큰 수.
        streaming: 스트리밍 모드 여부.

    Returns:
        ChatUpstage 인스턴스.

    Raises:
        ValueError: API 키가 설정되지 않은 경우.
    """
    api_key = os.getenv(SOLAR_API_KEY_ENV_NAME)
    _check_api_key(api_key)

    return ChatUpstage(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )


def get_embeddings(
    model: str = "solar-embedding-1-large",
) -> Embeddings:
    """
    Upstage Solar Embeddings 인스턴스를 반환합니다.

    Args:
        model: 사용할 임베딩 모델명.
               - "solar-embedding-1-large": 일반용 (기본값)

    Returns:
        UpstageEmbeddings 인스턴스.

    Raises:
        ValueError: API 키가 설정되지 않은 경우.
    """
    api_key = os.getenv(SOLAR_API_KEY_ENV_NAME)
    _check_api_key(api_key)

    return UpstageEmbeddings(
        api_key=api_key,
        model=model,
    )


def get_sql_llm() -> BaseChatModel:
    """
    Text-to-SQL 전용 LLM 인스턴스를 반환합니다.
    정확한 SQL 생성을 위해 온도를 0으로 설정합니다.

    Returns:
        SQL 생성용 ChatUpstage 인스턴스.
    """
    return get_llm(
        model="solar-pro",
        temperature=0.0,
        max_tokens=1000,
        streaming=False,
    )


def test_connection() -> bool:
    """
    LLM 및 임베딩 연결을 테스트합니다.
    """
    try:
        llm = get_llm()
        # 간단한 호출로 연결 테스트
        llm.invoke("안녕", max_tokens=1)
        print("✓ LLM 연결 성공 (solar-pro)")

        embeddings = get_embeddings()
        # 임베딩 생성으로 연결 테스트
        test_embedding = embeddings.embed_query("테스트")
        print(f"✓ 임베딩 연결 성공 (차원: {len(test_embedding)})")

        return True
    except ValueError:
        # API 키가 없는 경우 (의도된 에러)
        print(f"✗ 연결 실패: {SOLAR_API_KEY_ENV_NAME}가 설정되지 않았습니다.")
        return False
    except Exception as e:
        # 기타 연결 오류
        print(f"✗ 연결 실패: {e.__class__.__name__} - {e}")
        return False


if __name__ == "__main__":
    # 개발 환경에서 .env 로드가 필요할 수 있으므로 여기서 로드 (메인 스크립트가 아니기 때문)
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== LLM 모듈 테스트 시작 ===\n")
    test_connection()

    # 테스트 성공 시 추가 응답 테스트
    if test_connection():
        llm = get_llm()
        response = llm.invoke("LangChain이 무엇인지 한 문장으로 설명해주세요.")
        print(f"\n=== LLM 응답 테스트:")
        print(f"응답: {response.content}")