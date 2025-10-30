# src/utils/chunking_strategy.py

"""
구조 기반 텍스트 분할기 (StructuredTextSplitter)
코드 블록, 함수 시그니처, Markdown/HTML 구조를 보존하는 지능형 텍스트 분할을 제공합니다.
"""

import re
import ast
from typing import List, Dict, Any, Optional, Tuple

# 써드파티 라이브러리
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from bs4 import BeautifulSoup


class StructuredTextSplitter:
    """
    구조를 인식하는 고급 텍스트 분할기
    Markdown 섹션과 코드 블록의 보존을 최우선으로 하여 문서를 청킹합니다.
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        code_block_max_size: int = 3000,  # 코드 블록 최대 크기 (이보다 크면 분할)
        preserve_code_blocks: bool = True,
        preserve_functions: bool = True,
        preserve_markdown_structure: bool = True,
    ) -> None:
        """
        Args:
            chunk_size: 기본 청크 크기.
            chunk_overlap: 청크 간 중복 크기.
            code_block_max_size: 코드 블록의 최대 크기.
            preserve_code_blocks: 코드 블록 보존 여부.
            preserve_functions: 함수/클래스 정의 보존 여부 (Python에 한함).
            preserve_markdown_structure: Markdown 구조 (헤더) 보존 여부.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.code_block_max_size = code_block_max_size
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_functions = preserve_functions
        self.preserve_markdown_structure = preserve_markdown_structure

        # 기본 텍스트 분할기 (fallback 및 일반 텍스트 재분할용)
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서 리스트를 구조 기반으로 분할하는 메인 메서드.

        Args:
            documents: 원본 Document 리스트.

        Returns:
            구조 기반으로 분할된 Document 리스트.
        """
        all_chunks: List[Document] = []

        for doc in documents:
            chunks: List[Document] = self.split_text(
                doc.page_content,
                metadata=doc.metadata,
            )
            all_chunks.extend(chunks)

        return all_chunks

    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        텍스트를 구조 기반으로 분할.

        Args:
            text: 분할할 텍스트.
            metadata: 원본 문서 메타데이터.

        Returns:
            분할된 Document 리스트.
        """
        if metadata is None:
            metadata = {}

        # 1. 코드 블록 추출 및 보호
        text_with_placeholders, code_blocks = self._extract_code_blocks(text)

        # 2. Markdown 구조 분석
        sections = self._parse_markdown_structure(text_with_placeholders)

        # 3. 섹션별로 청킹
        chunks: List[Document] = []
        for section in sections:
            section_chunks: List[Document] = self._chunk_section(
                section,
                code_blocks,
                parent_metadata=metadata,
            )
            chunks.extend(section_chunks)

        # 4. 코드 블록 복원 및 최종 정리
        final_chunks: List[Document] = self._restore_code_blocks(chunks, code_blocks)

        return final_chunks

    def _extract_code_blocks(self, text: str) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """코드 블록을 추출하고 플레이스홀더로 대체"""
        code_blocks: Dict[str, Dict[str, Any]] = {}
        code_block_counter: int = 0

        # 펜스드(fenced) 코드 블록 패턴 (```)
        code_block_pattern = r'```(\w+)?\n(.*?)```'

        def replace_code_block(match: re.Match) -> str:
            nonlocal code_block_counter
            language: str = match.group(1) or 'plain'
            code_content: str = match.group(2)

            block_id: str = f"__CODE_BLOCK_{code_block_counter}__"
            code_block_counter += 1

            # 함수/클래스 이름 추출 (Python만)
            functions: List[str] = self._extract_function_names(code_content, language) if self.preserve_functions else []
            classes: List[str] = self._extract_class_names(code_content, language) if self.preserve_functions else []

            code_blocks[block_id] = {
                'language': language,
                'content': code_content,
                'type': 'fenced',
                'functions': functions,
                'classes': classes,
            }

            return block_id

        # 펜스드 코드 블록 대체
        text_with_placeholders: str = re.sub(
            code_block_pattern,
            replace_code_block,
            text,
            flags=re.DOTALL,
        )
        
        # 인라인 코드는 분할에 영향을 미치지 않으므로 복잡한 로직 없이 놔둠

        return text_with_placeholders, code_blocks

    def _parse_markdown_structure(self, text: str) -> List[Dict[str, Any]]:
        """Markdown 구조를 파싱하여 섹션으로 분리 (헤더 기반)"""
        sections: List[Dict[str, Any]] = []
        
        # 임시 섹션 (헤더가 없는 도입부나 끝부분 처리)
        current_section: Dict[str, Any] = {
            'level': 0,
            'title': 'Introduction',
            'content': [],
        }

        lines: List[str] = text.split('\n')

        for line in lines:
            header_match: Optional[re.Match] = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match and self.preserve_markdown_structure:
                level: int = len(header_match.group(1))
                title: str = header_match.group(2)

                # 현재 섹션 저장 (빈 섹션이 아니면)
                if current_section['content']:
                    sections.append(current_section)

                # 새 섹션 시작
                current_section = {
                    'level': level,
                    'title': title,
                    'content': [line],  # 헤더 라인 자체도 내용에 포함
                }
            else:
                current_section['content'].append(line)

        # 마지막 섹션 저장
        if current_section['content']:
            sections.append(current_section)

        # 섹션이 없으면 전체를 하나의 섹션으로
        if not sections and lines:
            sections.append({
                'level': 0,
                'title': 'Content',
                'content': lines,
            })

        return sections

    def _chunk_section(
        self,
        section: Dict[str, Any],
        code_blocks: Dict[str, Dict[str, Any]],
        parent_metadata: Dict[str, Any],
    ) -> List[Document]:
        """개별 섹션을 청킹"""
        section_text: str = '\n'.join(section['content'])
        section_title: str = section['title']
        section_level: int = section['level']
        chunks: List[Document] = []
        
        # 섹션 텍스트에서 코드 블록 플레이스홀더를 분리하여 처리
        parts: List[str] = re.split(r'(__CODE_BLOCK_\d+__)', section_text)
        
        current_text_chunk: List[str] = []
        for part in parts:
            if part.startswith('__CODE_BLOCK_') and part in code_blocks:
                # 일반 텍스트가 모여 있으면 먼저 청크로 분할하여 저장
                if current_text_chunk:
                    text_to_split: str = '\n'.join(current_text_chunk).strip()
                    if text_to_split:
                        # 일반 텍스트 분할 (recursive splitter 사용)
                        text_docs: List[Document] = self.base_splitter.split_documents([
                            Document(page_content=text_to_split, metadata=parent_metadata)
                        ])
                        
                        for text_doc in text_docs:
                            text_doc.metadata.update({
                                'section_title': section_title,
                                'section_level': section_level,
                                'chunk_type': 'text',
                                'has_code': False,
                            })
                            chunks.append(text_doc)

                    current_text_chunk = []
                
                # 코드 블록 처리
                code_info: Dict[str, Any] = code_blocks[part]
                
                # 코드 블록이 너무 크면 분할, 아니면 단일 청크로 유지
                if len(code_info['content']) > self.code_block_max_size and self.preserve_code_blocks:
                    code_chunks: List[Document] = self._split_large_code_block(
                        code_info,
                        section,
                        parent_metadata,
                    )
                    chunks.extend(code_chunks)
                else:
                    # 작은 코드 블록은 플레이스홀더 상태로 청크에 저장 (나중에 복원)
                    chunk_metadata: Dict[str, Any] = {
                        **parent_metadata,
                        'section_title': section_title,
                        'section_level': section_level,
                        'chunk_type': 'code',
                        'language': code_info['language'],
                        'has_code': True,
                        'functions': code_info.get('functions', []),
                        'classes': code_info.get('classes', []),
                    }
                    chunks.append(Document(page_content=part, metadata=chunk_metadata))

            elif part.strip():
                # 일반 텍스트 부분 축적
                current_text_chunk.append(part)

        # 마지막으로 남아있는 일반 텍스트 처리
        if current_text_chunk:
            text_to_split: str = '\n'.join(current_text_chunk).strip()
            if text_to_split:
                text_docs: List[Document] = self.base_splitter.split_documents([
                    Document(page_content=text_to_split, metadata=parent_metadata)
                ])
                
                for text_doc in text_docs:
                    text_doc.metadata.update({
                        'section_title': section_title,
                        'section_level': section_level,
                        'chunk_type': 'text',
                        'has_code': False,
                    })
                    chunks.append(text_doc)
            
        return chunks

    def _split_large_code_block(
        self,
        code_info: Dict[str, Any],
        section: Dict[str, Any],
        parent_metadata: Dict[str, Any],
    ) -> List[Document]:
        """큰 코드 블록을 함수/클래스 단위 또는 라인 수 기반으로 분할"""
        chunks: List[Document] = []
        code_content: str = code_info['content']
        language: str = code_info['language']
        section_title: str = section['title']
        section_level: int = section['level']

        if language == 'python' and self.preserve_functions:
            # Python 코드를 함수/클래스 단위로 분할 시도
            try:
                tree = ast.parse(code_content)
                code_lines: List[str] = code_content.split('\n')
                
                # 함수/클래스 정의만 추출하여 청크 생성
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        start_line: int = node.lineno - 1
                        end_line: int = getattr(node, 'end_lineno', start_line + 1)
                        
                        function_code: str = '\n'.join(code_lines[start_line:end_line])

                        chunk_type: str = 'code_function' if isinstance(node, ast.FunctionDef) else 'code_class'
                        
                        # 코드 블록 플레이스홀더로 다시 감싸서 저장 (나중에 복원)
                        placeholder_code: str = f"```python\n{function_code}\n```"

                        chunks.append(Document(
                            page_content=placeholder_code,
                            metadata={
                                **parent_metadata,
                                'section_title': section_title,
                                'section_level': section_level,
                                'chunk_type': chunk_type,
                                'language': language,
                                'has_code': True,
                                'function_name': node.name if isinstance(node, ast.FunctionDef) else None,
                                'class_name': node.name if isinstance(node, ast.ClassDef) else None,
                            }
                        ))
                return chunks # 함수/클래스 단위 분할 성공 시 반환
                
            except SyntaxError:
                # 유효하지 않은 Python 구문이 포함된 경우 (예: 일부 코드만 포함)
                print(f"\n[경고] Python 코드 블록 구문 분석 실패. 라인 기반으로 분할: {section_title}")
            except Exception as e:
                # 기타 파싱 오류
                print(f"\n[경고] 코드 블록 파싱 중 오류 발생: {e}. 라인 기반으로 분할: {section_title}")
        
        # 함수/클래스 분할 실패 또는 Python이 아닌 경우: 라인 수 기반 분할
        lines: List[str] = code_content.split('\n')
        
        # chunk_size를 기준으로 대략적인 라인 수 계산
        # (문자열 길이를 기반으로 청크를 분할하는 것보다 코드에 적합)
        chunk_lines: int = max(10, self.chunk_size // 40) 

        for i in range(0, len(lines), chunk_lines):
            chunk_code: str = '\n'.join(lines[i:i + chunk_lines])
            
            # 코드 블록 플레이스홀더로 다시 감싸서 저장 (나중에 복원)
            placeholder_code: str = f"```{language}\n{chunk_code}\n```"
            
            chunks.append(Document(
                page_content=placeholder_code,
                metadata={
                    **parent_metadata,
                    'section_title': section_title,
                    'section_level': section_level,
                    'chunk_type': 'code_partial',
                    'language': language,
                    'has_code': True,
                }
            ))
            
        return chunks

    def _restore_code_blocks(
        self,
        chunks: List[Document],
        code_blocks: Dict[str, Dict[str, Any]],
    ) -> List[Document]:
        """플레이스홀더를 실제 코드 블록으로 복원"""
        restored_chunks: List[Document] = []
        
        # 펜스드 코드 블록 복원 패턴
        fenced_code_pattern = r'```(\w+)?\n(.*?)```'
        
        for chunk in chunks:
            content: str = chunk.page_content

            # 1. 원본 플레이스홀더 복원 (작은 코드 블록)
            for block_id, code_info in code_blocks.items():
                if block_id in content:
                    if code_info['type'] == 'fenced':
                        replacement: str = f"```{code_info['language']}\n{code_info['content']}\n```"
                    else:  # inline code는 보통 그대로 유지되지만 만약을 위해 처리
                        replacement: str = f"`{code_info['content']}`"

                    content = content.replace(block_id, replacement)
            
            # 2. 큰 코드 블록 분할에서 생성된 임시 플레이스홀더 복원
            # 이 경우는 content 자체가 이미 코드 블록 형태일 수 있음.

            restored_chunks.append(Document(
                page_content=content,
                metadata=chunk.metadata,
            ))

        return restored_chunks

    def _extract_function_names(self, code: str, language: str) -> List[str]:
        """코드에서 함수 이름 추출 (Python만)"""
        if language == 'python':
            # Python 함수 패턴
            pattern = r'def\s+(\w+)\s*\('
            return re.findall(pattern, code)
        return []

    def _extract_class_names(self, code: str, language: str) -> List[str]:
        """코드에서 클래스 이름 추출 (Python만)"""
        if language == 'python':
            # Python 클래스 패턴
            pattern = r'class\s+(\w+)\s*(?:\(|:)'
            return re.findall(pattern, code)
        return []

class HTMLStructuredSplitter(StructuredTextSplitter):
    """
    HTML 문서를 위한 구조 기반 분할기 (StructuredTextSplitter를 상속하여 기본 로직 재활용 가능)
    """
    
    # 이 클래스는 현재 사용하지 않지만, 확장성을 위해 구조는 유지함.
    pass