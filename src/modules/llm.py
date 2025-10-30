import os
from typing import Optional

from dotenv import load_dotenv
from langchain_upstage import ChatUpstage

# 환경 변수(.env) 파일 로드
load_dotenv()

def get_solar_api_key() -> Optional[str]:
    """
    환경 변수에서 SOLAR_API_KEY를 안전하게 불러오기
    키가 없으면 None을 반환
    """
    return os.getenv("SOLAR_API_KEY")

def get_solar_llm() -> ChatUpstage:
    """
    Solar API 키를 사용하여 ChatUpstage 모델 인스턴스를 초기화하고 반환

    Raises:
        ValueError: SOLAR_API_KEY가 환경 변수에 설정되지 않았을 경우

    Returns:
        ChatUpstage: LangChain과 통합된 Solar LLM 객체
    """
    api_key: Optional[str] = get_solar_api_key()

    if not api_key:
        # 키가 없으면 에러를 발생시켜서 환경 설정을 확인하게 함
        raise ValueError("SOLAR_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요")

    # RAG의 정확도를 위해 temperature를 낮게 설정
    llm = ChatUpstage(
        api_key=api_key,
        model="solar-1-mini-chat",  # 프로젝트에 확정된 모델 사용
        temperature=0.1,
    )
    return llm


# 실제로 API 키를 불러와서 LLM 객체를 생성하는지 확인하는 
# 간단한 테스트 스크립트
if __name__ == "__main__":
    print("Solar LLM 초기화 테스트 시작...")
    try:
        solar_llm = get_solar_llm()
        print(f"Solar LLM 객체가 성공적으로 생성되었어: {type(solar_llm)}")
        # 간단한 호출 테스트 (API 키가 유효한지 확인)
        test_response = solar_llm.invoke("안녕하세요. 당신은 누구인가요?")
        print("\nLLM 응답 테스트:")
        print(test_response.content)
        print("\n테스트 성공!")
    except ValueError as e:
        print(f"\n[에러 발생] 환경 변수 문제: {e}")
        print("SOLAR_API_KEY를 .env 파일에 설정했는지 확인해주세요.")
    except Exception as e:
        print(f"\n[에러 발생] LLM 호출 문제: {e}")
        print("API 키가 유효한지, 네트워크 연결이 정상인지 확인해주세요.")







