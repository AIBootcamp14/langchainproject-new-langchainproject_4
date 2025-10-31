import os
import json
from typing import Final, Dict, Any, List

# RAGAS 평가 도구 임포트
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper

# 팀원 2가 구현할 Solar LLM 모듈 임포트 (가정)
# 팀원 2에게 src/modules/llm.py에 SolarChatModel을 구현해달라고 요청
from src.modules.llm import SolarChatModel 
# from dotenv import load_dotenv # 환경 변수 로드는 팀원 5 담당

# --- 설정 및 초기화 ---
TEST_SET_PATH: Final[str] = os.path.join("data", "tests", "test_questions.json")

# 1. Ragas 평가용 LLM 설정: Solar API 사용 확정
# Ragas는 평가를 위해 LLM이 필요해. SolarChatModel 인스턴스를 Ragas Wrapper로 감싼다.
try:
    # SolarChatModel 초기화 시 환경 변수(SOLAR_API_KEY)를 사용한다고 가정
    solar_llm = SolarChatModel() 
    EVALUATOR_LLM = LangchainLLMWrapper(solar_llm)
except Exception as e:
    print(f"경고: SolarChatModel 초기화 오류. 환경 변수 확인 필요: {e}")
    EVALUATOR_LLM = None

# 평가 지표 정의
METRICS_TO_EVALUATE = [
    faithfulness,
    answer_relevancy,
    # context_recall, # Context Recall은 'ground_truth' 필드가 필요해서 일단 제외
]
# --------------------


def load_test_set(file_path: str) -> List[Dict[str, Any]]:
    """JSON 파일에서 테스트 셋 데이터를 로드한다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_ragas_dataset(raw_data: List[Dict[str, Any]], rag_outputs: List[Dict[str, Any]]) -> Dataset:
    """
    Ragas 평가에 필요한 형식으로 데이터를 변환한다.
    
    Args:
        raw_data: load_test_set에서 로드한 원래 테스트 셋 (질문, 정답)
        rag_outputs: RAG 파이프라인(API)을 실행해서 얻은 결과 (답변, contexts, source_urls)
    """
    
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
    results_df.to_csv('evaluation_results.csv', index=False)
    print("\nevaluation_results.csv 파일에 결과가 저장되었습니다.")


if __name__ == "__main__":
    # 이 부분은 팀장이 API 배포 후에 실행해야 하는 시뮬레이션
    
    # 1. 테스트 셋 로드
    raw_test_data = load_test_set(TEST_SET_PATH)
    
    print(f"로드된 질문 개수: {len(raw_test_data)}")
    
    # 2. 🚨 RAG 파이프라인 실행 시뮬레이션 🚨
    # 이 'dummy_rag_outputs'는 팀장의 API(POST /ask)가 실제로 출력해야 할 형식
    # 팀장(너)은 API 완성 후, 이 10개 질문을 API에 던져서 실제 결과를 여기에 채워 넣어야 함
    
    dummy_rag_outputs: List[Dict[str, Any]] = [
        # 실제 API 호출 결과 (답변, 검색된 context 조각, 출처 URL 등)를 저장
        # contexts는 검색된 문서의 '텍스트' 내용 리스트여야 함
        {"answer": "...", "contexts": ["... 검색된 청크 내용 1 ...", "... 검색된 청크 내용 2 ..."], "source_urls": ["url1", "url2"]},
        # ... 9개 항목 더 ...
    ]
    
    # 3. Ragas Dataset 준비
    # ragas_dataset = prepare_ragas_dataset(raw_test_data, dummy_rag_outputs)
    
    # 4. 평가 실행
    # run_evaluation(ragas_dataset)
    
    print("\n**남은 작업:**")
    print("1. 팀원 2에게 src/modules/llm.py에 SolarChatModel 클래스를 구현하도록 요청")
    print("2. 팀장(너)이 API 배포 후, 10개 질문을 API에 던져 'answer'와 'contexts'를 받아와야 이 스크립트를 최종 실행할 수 있음")