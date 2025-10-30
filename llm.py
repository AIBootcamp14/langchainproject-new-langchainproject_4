"""LLM 및 임베딩 모델 설정
Upstage Solar API를 사용한 한국어 최적화 모델"""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

# 환경변수 로드
load_dotenv()


def get_llm(
    model: str = "solar-pro",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    streaming: bool = False
) -> BaseChatModel:
    """
    Upstage Solar LLM 인스턴스를 반환합니다.

    Args:
        model: 사용할 모델명 ("solar-pro" 또는 "solar-1-mini-chat")
        temperature: 생성 온도 값 (0.0 ~ 1.0)
        max_tokens: 최대 토큰 수
        streaming: 스트리밍 모드 여부

    Returns:
        ChatUpstage 인스턴스

    Raises:
        ValueError: API 키가 설정되지 않은 경우
    """
    api_key = os.getenv("UPSTAGE_API_KEY")

    if not api_key or api_key == "your-actual-api-key-here":
        raise ValueError(
            "UPSTAGE_API_KEY가 설정되지 않았습니다. "
            ".env 파일을 확인하거나 https://console.upstage.ai에서 API 키를 발급받으세요."
        )

    return ChatUpstage(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming
    )


def get_embeddings(
    model: str = "solar-embedding-1-large"
) -> Embeddings:
    """
    Upstage Solar Embeddings 인스턴스를 반환합니다.

    Args:
        model: 사용할 임베딩 모델명
            - "solar-embedding-1-large": 일반용
            - "solar-embedding-1-large-query": 쿼리 최적화
            - "solar-embedding-1-large-passage": 문서 최적화

    Returns:
        UpstageEmbeddings 인스턴스

    Raises:
        ValueError: API 키가 설정되지 않은 경우
    """
    api_key = os.getenv("UPSTAGE_API_KEY")

    if not api_key or api_key == "your-actual-api-key-here":
        raise ValueError(
            "UPSTAGE_API_KEY가 설정되지 않았습니다. "
            ".env 파일을 확인하거나 https://console.upstage.ai에서 API 키를 발급받으세요."
        )

    return UpstageEmbeddings(
        api_key=api_key,
        model=model
    )


def get_sql_llm() -> BaseChatModel:
    """
    Text-to-SQL 전용 LLM 인스턴스를 반환합니다.
    SQL 생성에 최적화된 설정을 사용합니다.

    Returns:
        SQL 생성용 ChatUpstage 인스턴스
    """
    return get_llm(
        model="solar-pro",  # SQL 생성에 적합한 모델명 사용
        temperature=0.0,    # 정확한 SQL 생성을 위해 온도를 0으로 설정
        max_tokens=1000,    # SQL 쿼리 길이를 고려한 제한
        streaming=False     # SQL은 스트리밍 불필요
    )


def test_connection() -> bool:
    """
    LLM 연결을 테스트합니다.

    Returns:
        연결 성공 여부
    """
    try:
        llm = get_llm()
        response = llm.invoke("안녕하세요")
        print(f"✓ LLM 연결 성공: {response.content[:50]}...")

        embeddings = get_embeddings()
        test_embedding = embeddings.embed_query("테스트")
        print(f"✓ 임베딩 연결 성공: 차원 {len(test_embedding)}")

        return True
    except Exception as e:
        print(f"✗ 연결 실패: {e}")
        return False


if __name__ == "__main__":
    # 모듈 테스트
    print("=== LLM 모듈 테스트 시작 ===\n")

    # 연결 테스트
    if test_connection():
        print("\n=== LLM 응답 테스트:")
        llm = get_llm()
        response = llm.invoke("LangChain이 무엇인지 간단히 설명해주세요.")
        print(f"응답: {response.content}")

        print("\n=== 임베딩 테스트:")
        embeddings = get_embeddings()
        docs = ["LangChain은 LLM 애플리케이션 개발 프레임워크입니다."]
        vectors = embeddings.embed_documents(docs)
        print(f"벡터 차원: {len(vectors[0])}")
        print(f"벡터 샘플: {vectors[0][:5]}...")
    else:
        print("\n✗ API 키를 설정해야 합니다.")
