# src/utils/chunking_strategy.py

"""
문서 분할(Chunking) 전략 모듈
코드 블록을 보호하는 커스텀 텍스트 분할기를 구현
"""

import re
from typing import List, Dict, Any, Optional, Final

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# PEP 8: 모듈 수준 상수는 대문자로
# 커스텀 구분자 정의
CODE_BLOCK_START_DELIMITER: Final[str] = "```"
CODE_BLOCK_END_DELIMITER: Final[str] = "```"
# 코드 블록을 일시적으로 대체할 마커 (문서 내용과 겹치지 않도록 고유하게 만듦)
CODE_BLOCK_PLACEHOLDER: Final[str] = "<CODE_BLOCK_PROTECTED_{}>"


class CodeBlockPreservingSplitter(RecursiveCharacterTextSplitter):
    """
    코드 블록을 하나의 덩어리로 간주하여 분할되지 않도록 보호하는 텍스트 분할기.
    """

    def __init__(
        self,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        length_function: Any = len,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """초기화: 기본 분할기는 마크다운용으로 설정"""
        
        # Markdown에 최적화된 기본 구분자 사용
        if separators is None:
            # RecursiveCharacterTextSplitter 기본값 (Markdown 최적화)
            separators = [
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "\u200b",  # 제로 폭 공백
                "",
            ]

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=length_function,
            is_separator_regex=is_separator_regex,
            **kwargs,
        )
        
        # 코드 블록 카운터를 인스턴스 변수로 초기화
        self._code_block_counter: int = 0

    def _extract_code_blocks(self, text: str) -> str:
        """
        코드 블록을 찾아 PLACEHOLDER로 대체하고, 카운터를 업데이트한다.
        
        Args:
            text: 처리할 원본 텍스트
            
        Returns:
            코드 블록이 PLACEHOLDER로 대체된 텍스트
        """
        self._code_block_counter = 0 # 처리 시작 시 카운터 초기화
        
        # 💡 [핵심 수정]: nonlocal 대신 self._code_block_counter 사용
        def replace_match(match: re.Match) -> str:
            """정규식 매치 객체를 PLACEHOLDER로 치환"""
            placeholder = CODE_BLOCK_PLACEHOLDER.format(self._code_block_counter)
            self._code_block_counter += 1
            return placeholder

        # 코드 블록(```...```)을 정규식으로 찾아서 replace_match 함수로 치환
        # re.DOTALL은 .이 \n까지 포함하도록 설정
        pattern = re.compile(
            rf"{CODE_BLOCK_START_DELIMITER}.*?{CODE_BLOCK_END_DELIMITER}", 
            re.DOTALL
        )
        processed_text = pattern.sub(replace_match, text)
        
        return processed_text

    def _restore_code_blocks(self, splits: List[str], original_text: str) -> List[str]:
        """
        PLACEHOLDER를 원래 코드 블록으로 복원한다.
        """
        # 원본 텍스트에서 모든 코드 블록을 추출
        code_blocks: List[str] = re.findall(
            rf"({CODE_BLOCK_START_DELIMITER}.*?{CODE_BLOCK_END_DELIMITER})", 
            original_text, 
            re.DOTALL
        )
        
        restored_splits: List[str] = []
        
        # 분할된 청크를 순회하며 PLACEHOLDER를 복원
        for split in splits:
            current_split = split
            
            # 각 청크 내의 모든 PLACEHOLDER를 순회하며 복원
            for i in range(self._code_block_counter):
                placeholder = CODE_BLOCK_PLACEHOLDER.format(i)
                
                if placeholder in current_split:
                    if i < len(code_blocks):
                        # PLACEHOLDER를 해당 인덱스의 실제 코드 블록으로 치환
                        current_split = current_split.replace(placeholder, code_blocks[i])
                    else:
                        # 오류 방지를 위한 예외 처리 (발생하면 안 됨)
                        current_split = current_split.replace(placeholder, "")
                        
            restored_splits.append(current_split)
            
        return restored_splits

    # RecursiveCharacterTextSplitter의 핵심 메서드를 오버라이드
    def split_text(self, text: str) -> List[str]:
        """
        텍스트를 분할하기 전에 코드 블록을 보호하고, 분할 후 복원한다.
        """
        if self._code_block_counter > 0:
             # 카운터가 남아있다면 초기화 (안전을 위한 추가 체크)
             self._code_block_counter = 0

        # 1. 코드 블록을 PLACEHOLDER로 대체하여 분할기가 코드를 쪼개지 않도록 보호
        text_with_placeholders: str = self._extract_code_blocks(text)

        # 2. 부모 클래스의 split_text를 호출하여 텍스트를 분할
        # 이 분할 과정에서 코드 블록 PLACEHOLDER는 하나의 긴 단어처럼 취급되어 분할되지 않음
        splits_with_placeholders: List[str] = super().split_text(text_with_placeholders)

        # 3. 분할된 청크에서 PLACEHOLDER를 원래 코드 블록으로 복원
        final_splits: List[str] = self._restore_code_blocks(splits_with_placeholders, text)

        return final_splits


if __name__ == "__main__":
   


# 테스트용 코드
    
    # 1. 분할되지 않아야 할 코드 블록을 포함한 문서
    test_document = """
# LangChain LCEL 가이드

LangChain Expression Language (LCEL)은 체인을 구성하는 가장 좋은 방법입니다.

## 1. 간단한 체인 구성

다음은 Prompt, Model, OutputParser를 연결하는 간단한 예시입니다.
이 코드 블록은 분할되면 안 됩니다.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

# 체인 호출
chain.invoke({"topic": "python"})
"""