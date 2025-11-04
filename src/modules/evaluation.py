# src/modules/evaluation.py

import os
import json
from typing import Final, Dict, Any, List, Optional # Optional 추가

# 써드파티 라이브러리
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy

# 프로젝트 모듈
from src.modules.llm import get_solar_llm # <-- 클래스 대신 함수를 임포트 (수정)
from src.utils.utils import ensure_directory # <-- 테스트 결과 저장을 위해 추가

# --- 설정 및 초기화 (PEP 8: 모듈 수준 상수는 대문자로) ---
TEST_SET_PATH: Final[str] = os.path.join("data", "tests", "test_questions.json")
EVALUATION_OUTPUT_PATH: Final[str] = "evaluation_results.csv" # <-- CSV 파일 경로 상수화

# 1. Ragas 평가용 LLM 설정: Solar API 사용 확정
# Ragas는 평가를 위해 LLM이 필요해. Solar LLM 인스턴스를 Ragas Wrapper로 감싼다.
EVALUATOR_LLM: Optional[LangchainLLMWrapper] = None
try:
    # 평가의 일관성을 위해 온도를 0.0으로 설정 (수정)
    solar_llm = get_solar_llm(temperature=0.0) 
    EVALUATOR_LLM = LangchainLLMWrapper(solar_llm)
    print("평가용 LLM (Solar) 연결 완료.")
except Exception as e:
    print(f"경고: 평가용 LLM 초기화 오류. 환경 변수 확인 필요: {e}")

# 평가 지표 정의
METRICS_TO_EVALUATE: Final[List[Any]] = [
    faithfulness,
    answer_relevancy,
    # context_recall, # Context Recall은 'ground_truth' 필드가 필요해서 일단 제외
]
# --------------------


def load_test_set(file_path: str) -> List[Dict[str, Any]]:
    """JSON 파일에서 테스트 셋 데이터를 로드한다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"오류: 테스트 파일 경로를 찾을 수 없습니다: {file_path}")
        return []


def prepare_ragas_dataset(raw_data: List[Dict[str, Any]], rag_outputs: List[Dict[str, Any]]) -> Dataset:
    """
    Ragas 평가에 필요한 형식으로 데이터를 변환한다.
    
    Args:
        raw_data: load_test_set에서 로드한 원래 테스트 셋 (질문, 정답)
        rag_outputs: RAG 파이프라인(API)을 실행해서 얻은 결과 (답변, contexts)
    """
    
    # 데이터 개수가 맞는지 확인
    if len(raw_data) != len(rag_outputs):
        raise ValueError(
            f"테스트 데이터({len(raw_data)}개)와 RAG 출력 데이터({len(rag_outputs)}개)의 개수가 일치하지 않습니다."
        )

    # Ragas Dataset 구조에 맞게 데이터 준비
    data_dict: Dict[str, List] = {
        'question': [],
        'answer': [],  # RAG 챗봇이 생성한 답변
        'contexts': [], # RAG 챗봇이 답변 시 검색한 문서 조각 (청크)
        'ground_truth': [], # 우리가 정의한 모범 정답 (정확도 평가용)
    }

    # raw_data와 rag_outputs를 매핑하여 Ragas에 필요한 키를 채운다.
    for raw_item, output_item in zip(raw_data, rag_outputs):
        data_dict['question'].append(raw_item['question'])
        data_dict['answer'].append(output_item['answer'])
        data_dict['contexts'].append(output_item['contexts'])
        data_dict['ground_truth'].append(raw_item['expected_answer'])

    return Dataset.from_dict(data_dict)


def run_evaluation(ragas_dataset: Dataset):
    """Ragas를 사용하여 RAG 파이프라인 성능을 평가하고 결과를 출력한다."""
    if EVALUATOR_LLM is None:
        print("평가용 LLM이 설정되지 않아 평가를 건너뜁니다.")
        return

    print("--- RAG 성능 평가 시작 (Solar API 사용) ---")
    
    score = evaluate(
        dataset=ragas_dataset,
        metrics=METRICS_TO_EVALUATE,
        llm=EVALUATOR_LLM  # Solar LLM 래퍼 연결
    )
    
    print("\n--- RAG 최종 평가 결과 ---")
    results_df = score.to_pandas()
    print(results_df)
    
    # 최종 결과를 CSV 파일로 저장
    # ensure_directory 함수를 사용하여 data 폴더가 존재하는지 확인 후 저장
    ensure_directory("data")
    results_df.to_csv(os.path.join("data", EVALUATION_OUTPUT_PATH), index=False)
    print(f"\ndata/{EVALUATION_OUTPUT_PATH} 파일에 결과가 저장되었습니다.")


if __name__ == "__main__":
    # 이 부분은 팀장이 API 배포 후에 실행해야 하는 시뮬레이션
    
    # 1. 테스트 셋 로드
    raw_test_data = load_test_set(TEST_SET_PATH)
    
    if not raw_test_data:
        print("\n테스트 데이터를 로드할 수 없어 평가를 진행하지 않습니다.")
    else:
        print(f"로드된 질문 개수: {len(raw_test_data)}")
        
        # 2. 🚨 RAG 파이프라인 실행 시뮬레이션 🚨
        # 이 'dummy_rag_outputs'는 팀장의 API(POST /ask)가 실제로 출력해야 할 형식
        # 실제 API 호출 결과 (답변, 검색된 context 조각, 출처 URL 등)를 저장
        dummy_rag_outputs: List[Dict[str, Any]] = [
            # 더미 데이터는 raw_test_data 개수와 맞춰야 함 (10개)
            {"answer": "LCEL은 LangChain 구성 요소를 파이프라인처럼 연결하는 방식이며, 이를 통해 모듈성과 성능을 향상시킵니다.", "contexts": ["LCEL은 체인을 쉽게 만들 수 있게 하는 기본 구성 요소입니다.", "LCEL은 지연 실행 및 스트리밍 같은 고급 기능을 지원합니다."], "source_urls": ["url1"]},
            {"answer": "ChromaDB에 문서를 저장하려면 먼저 문서를 청크로 나누고, 임베딩 모델을 정의한 후, Chroma.from_documents를 호출해야 합니다.", "contexts": ["Chroma.from_documents는 임베딩 함수와 청크된 문서를 받아 컬렉션을 생성합니다."], "source_urls": ["url2"]},
            {"answer": "ReAct 프롬프트는 Agent가 추론(Thought)하고 행동(Action)을 결정하는 구조를 가지며, 이는 복잡한 작업 수행에 필수적입니다.", "contexts": ["ReAct 프롬프트는 (Thought, Action, Action Input) 튜플을 반복하는 방식으로 구성됩니다."], "source_urls": ["url3"]},
            {"answer": "코드 블록 보존을 위해서는 RecursiveCharacterTextSplitter를 사용하여 청크 크기(chunk_size)를 충분히 크게 설정하고, 청크 오버랩(chunk_overlap)을 두어 코드 문맥을 유지해야 합니다.", "contexts": ["HTML 문서 로드 시 코드 블록이 깨지지 않게 청킹을 조정하는 것이 중요합니다."], "source_urls": ["url4"]},
            {"answer": "WebBaseLoader로 문서를 로드한 후, RecursiveCharacterTextSplitter를 생성하여 split_documents 메서드를 호출하면 됩니다. 예시 코드는 다음과 같습니다...", "contexts": ["WebBaseLoader는 URL을 받아 HTML 내용을 Document 객체로 로드합니다.", "RecursiveCharacterTextSplitter는 지정된 구분자로 문서를 청크로 나눕니다."], "source_urls": ["url5"]},
            {"answer": "콜백 시스템은 체인, LLM, Retriever 호출 전후에 실행되는 후크를 제공하여, 로깅, 모니터링, 디버깅 및 스트리밍 기능을 구현하는 데 사용됩니다.", "contexts": ["LangChain은 다양한 이벤트 후크를 지원하는 중앙 집중식 콜백 시스템을 제공합니다."], "source_urls": ["url6"]},
            {"answer": "검색 유형을 'mmr'(Maximal Marginal Relevance)로 설정하면 검색 결과의 관련성과 다양성을 동시에 최대화하여, 검색된 문서들이 서로 다른 정보를 포함하도록 합니다.", "contexts": ["MMR은 유사도 점수와 벡터 공간에서의 거리를 모두 고려합니다."], "source_urls": ["url7"]},
            {"answer": "PydanticOutputParser는 LLM의 자유 형식 텍스트 출력을 구조화된 Pydantic 모델 객체로 파싱하여, 출력을 안정적으로 다룰 수 있게 합니다. 전체 예제 코드는 문서에서 확인할 수 있습니다.", "contexts": ["PydanticOutputParser는 JSON 스키마를 프롬프트에 삽입하여 LLM이 구조화된 응답을 생성하도록 유도합니다."], "source_urls": ["url8"]},
            {"answer": "ConversationBufferMemory는 대화 기록을 저장하는 메모리 모듈이며, RunnableWithMessageHistory를 사용하여 RAG 체인에 연결하여 세션별 대화 기록을 관리할 수 있습니다.", "contexts": ["메모리는 주로 채팅 애플리케이션에서 대화의 연속성을 유지하는 데 사용됩니다."], "source_urls": ["url9"]},
            {"answer": "RunnablePassthrough는 입력 값을 다음 단계로 그대로 전달하는 역할을 하며, 이를 사용해 Retriever의 결과를 Context 키에 할당하고 Question을 그대로 유지하여 LLM에 전달할 수 있습니다.", "contexts": ["RunnablePassthrough.assign은 기존 입력에 새로운 키를 추가할 때 유용합니다."], "source_urls": ["url10"]},
        ]
        
        # 3. Ragas Dataset 준비
        ragas_dataset = prepare_ragas_dataset(raw_test_data, dummy_rag_outputs)
        
        # 4. 평가 실행
        run_evaluation(ragas_dataset)
        
    print("\n**남은 작업 (상기시키기):**")
    print("1. 팀장(너)이 API 배포 후, 10개 질문을 API에 던져 'answer'와 'contexts'를 받아와야 이 스크립트를 최종 실행할 수 있음")
    print("2. `src/modules/llm.py`에서 `get_solar_llm`의 `temperature`가 평가용으로 0.0으로 설정되었음.")