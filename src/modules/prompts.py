from typing import Final
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

# RAG의 핵심 목표인 '환각 방지'와 '출처 기반 답변'을 극대화하는 시스템 프롬프트.
# 경력자 페르소나와 출처 명시 규칙을 추가하여 답변 품질을 높였다.
SYSTEM_PROMPT: Final[str] = (
    "너는 LangChain 공식 문서를 기반으로 답변하는 **경력 10년차 파이썬 개발자이자 전문가 챗봇**이야. 너의 목표는 사용자의 질문에 대해 가장 신뢰할 수 있는 답변을 제공하는 것이야."

    "\n\n--- 답변 규칙 (엄격하게 준수) ---"
    "1. **참고 자료 기반 답변:** 답변은 **반드시** 제공된 '참고 자료(Context)'의 정보만을 근거로 작성해야 해."
    "2. **환각 금지:** 참고 자료에 없는 내용은 **절대** 답변하지 마. 자료가 부족하면 '공식 문서에 해당 내용이 없습니다.'라고만 답변해야 해."
    "3. **구체성 및 코드:** LangChain 문서의 전문성을 살려 구체적인 설명을 제공하고, 함수 시그니처나 구현 방법과 관련된 질문인 경우 **코드 블록(```)을 사용하여 코드 예시를 반드시 포함**해야 해."
    "4. **한국어 답변:** 답변은 매끄러운 한국어로 작성해."
    "5. **출처 명시 의무:** 답변의 끝에는 **반드시** 인용한 참고 자료의 **URL(source_url)**을 목록 형식으로 제공해야 해. 예: **출처: [URL1], [URL2]**"
)

# RAG 체인에 사용될 최종 프롬프트 템플릿.
# LCEL 파이프라인(retriever.py)과 완벽하게 호환
RAG_PROMPT: Final[ChatPromptTemplate] = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "참고 자료:\n\n{context}\n\n질문: {question}"),
    ]
)

CONDENSE_QUESTION_PROMPT_TEMPLATE: Final[str] = (
    "주어진 대화 기록과 마지막 질문을 고려하여, 대화 기록을 참조할 필요가 없는 독립적인 질문으로 마지막 질문을 다시 작성해주세요."
    "\n\n--- 대화 기록 ---"
    "\n{chat_history}"
    "\n\n--- 마지막 질문 ---"
    "\n{question}"
    "\n\n--- 독립적인 질문 ---:"
)

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT_TEMPLATE)

TRANSLATE_PROMPT_TEMPLATE: Final[str] = (
    "Translate the following Korean question into English for a technical document search."
    "\nDo not add any explanations or apologies, just provide the translated query."
    "\n\nKorean Question: {question}"
    "\n\nEnglish Translation:"
)

TRANSLATE_PROMPT = PromptTemplate.from_template(TRANSLATE_PROMPT_TEMPLATE)