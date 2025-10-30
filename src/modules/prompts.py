from typing import Final
from langchain_core.prompts import ChatPromptTemplate

# 팀원 3이 정의할 환각 방지 시스템 프롬프트(예시)
SYSTEM_PROMPT: Final[str] = (
    "너는 LangChain 공식 문서 전문가 챗봇이야. "
    "다음 참고 자료(Context)를 바탕으로 사용자의 질문(Question)에 대해 가장 정확하고 도움이 되는 답변을 한국어로 제공해. "
    "답변은 반드시 참고 자료 내의 정보만을 사용해야 하며, 참고 자료에 없는 내용은 절대 답변하지 마. "
    "특히, 코드 예시가 있다면 해당 코드를 그대로 제공해."
)

# RAG 체인에 사용될 최종 프롬프트 템플릿 (입력: context, question)
RAG_PROMPT: Final[ChatPromptTemplate] = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "참고 자료:\n\n{context}\n\n질문: {question}"),
    ]
)