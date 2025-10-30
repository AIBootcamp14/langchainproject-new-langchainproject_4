"""
고급 텍스트 분할기 - 구조 기반 청킹 전략
코드 블록, 함수 시그니처, Markdown/HTML 구조를 보존하는 지능형 텍스트 분할
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from bs4 import BeautifulSoup
import ast


@dataclass
class ChunkMetadata:
    """청크 메타데이터"""
    chunk_type: str  # "code", "text", "header", "list", "table"
    language: Optional[str] = None  # 코드 블록의 언어
    header_level: Optional[int] = None  # 헤더 레벨 (1-6)
    parent_section: Optional[str] = None  # 부모 섹션 제목
    has_code: bool = False  # 코드 포함 여부
    function_names: List[str] = None  # 포함된 함수 이름들
    class_names: List[str] = None  # 포함된 클래스 이름들


class StructuredTextSplitter:
    """
    구조를 인식하는 고급 텍스트 분할기
    코드 블록과 함수 시그니처를 보존하면서 문서를 청킹
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        code_block_max_size: int = 3000,  # 코드 블록 최대 크기
        preserve_code_blocks: bool = True,
        preserve_functions: bool = True,
        preserve_markdown_structure: bool = True
    ):
        """
        Args:
            chunk_size: 기본 청크 크기
            chunk_overlap: 청크 간 중복 크기
            code_block_max_size: 코드 블록의 최대 크기 (이보다 크면 분할)
            preserve_code_blocks: 코드 블록 보존 여부
            preserve_functions: 함수/클래스 정의 보존 여부
            preserve_markdown_structure: Markdown 구조 보존 여부
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.code_block_max_size = code_block_max_size
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_functions = preserve_functions
        self.preserve_markdown_structure = preserve_markdown_structure

        # 기본 텍스트 분할기 (fallback용)
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서 리스트를 구조 기반으로 분할

        Args:
            documents: 원본 문서 리스트

        Returns:
            구조 기반으로 분할된 문서 리스트
        """
        all_chunks = []

        for doc in documents:
            chunks = self.split_text(
                doc.page_content,
                metadata=doc.metadata
            )
            all_chunks.extend(chunks)

        return all_chunks

    def split_text(self, text: str, metadata: Dict = None) -> List[Document]:
        """
        텍스트를 구조 기반으로 분할

        Args:
            text: 분할할 텍스트
            metadata: 원본 문서 메타데이터

        Returns:
            분할된 Document 리스트
        """
        if metadata is None:
            metadata = {}

        # 1. 코드 블록 추출 및 보호
        text_with_placeholders, code_blocks = self._extract_code_blocks(text)

        # 2. Markdown 구조 분석
        sections = self._parse_markdown_structure(text_with_placeholders)

        # 3. 섹션별로 청킹
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(
                section,
                code_blocks,
                parent_metadata=metadata
            )
            chunks.extend(section_chunks)

        # 4. 코드 블록 복원 및 최종 처리
        final_chunks = self._restore_code_blocks(chunks, code_blocks)

        return final_chunks

    def _extract_code_blocks(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        코드 블록을 추출하고 플레이스홀더로 대체

        Args:
            text: 원본 텍스트

        Returns:
            (플레이스홀더가 포함된 텍스트, 코드 블록 딕셔너리)
        """
        code_blocks = {}
        code_block_counter = 0

        # 백틱 코드 블록 패턴 (```)
        code_block_pattern = r'```(\w+)?\n(.*?)```'

        def replace_code_block(match):
            nonlocal code_block_counter
            language = match.group(1) or 'plain'
            code_content = match.group(2)

            # 코드 블록 ID 생성
            block_id = f"__CODE_BLOCK_{code_block_counter}__"
            code_block_counter += 1

            # 코드 블록 저장
            code_blocks[block_id] = {
                'language': language,
                'content': code_content,
                'type': 'fenced',
                'functions': self._extract_function_names(code_content, language),
                'classes': self._extract_class_names(code_content, language)
            }

            return block_id

        # 코드 블록 대체
        text_with_placeholders = re.sub(
            code_block_pattern,
            replace_code_block,
            text,
            flags=re.DOTALL
        )

        # 인라인 코드도 보호 (선택적)
        inline_code_pattern = r'`([^`]+)`'

        def replace_inline_code(match):
            nonlocal code_block_counter
            code_content = match.group(1)

            # 짧은 인라인 코드는 그대로 유지
            if len(code_content) < 50:
                return match.group(0)

            block_id = f"__INLINE_CODE_{code_block_counter}__"
            code_block_counter += 1

            code_blocks[block_id] = {
                'language': 'inline',
                'content': code_content,
                'type': 'inline'
            }

            return block_id

        text_with_placeholders = re.sub(
            inline_code_pattern,
            replace_inline_code,
            text_with_placeholders
        )

        return text_with_placeholders, code_blocks

    def _parse_markdown_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Markdown 구조를 파싱하여 섹션으로 분리

        Args:
            text: Markdown 텍스트

        Returns:
            섹션 리스트
        """
        sections = []
        current_section = {
            'level': 0,
            'title': 'Introduction',
            'content': [],
            'subsections': []
        }

        lines = text.split('\n')

        for line in lines:
            # 헤더 감지
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2)

                # 현재 섹션 저장
                if current_section['content']:
                    sections.append(current_section)

                # 새 섹션 시작
                current_section = {
                    'level': level,
                    'title': title,
                    'content': [],
                    'subsections': []
                }
            else:
                current_section['content'].append(line)

        # 마지막 섹션 저장
        if current_section['content']:
            sections.append(current_section)

        # 섹션이 없으면 전체를 하나의 섹션으로
        if not sections:
            sections.append({
                'level': 0,
                'title': 'Content',
                'content': lines,
                'subsections': []
            })

        return sections

    def _chunk_section(
        self,
        section: Dict[str, Any],
        code_blocks: Dict[str, str],
        parent_metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        개별 섹션을 청킹

        Args:
            section: 섹션 정보
            code_blocks: 코드 블록 딕셔너리
            parent_metadata: 부모 메타데이터

        Returns:
            청킹된 Document 리스트
        """
        chunks = []
        section_text = '\n'.join(section['content'])

        # 섹션이 충분히 작으면 그대로 유지
        if len(section_text) <= self.chunk_size:
            chunk_metadata = {
                **parent_metadata,
                'section_title': section['title'],
                'section_level': section['level'],
                'chunk_type': 'section',
                'has_code': any(block_id in section_text for block_id in code_blocks)
            }

            chunks.append(Document(
                page_content=section_text,
                metadata=chunk_metadata
            ))
        else:
            # 섹션이 크면 추가 분할
            sub_chunks = self._smart_split_section(
                section_text,
                code_blocks,
                section,
                parent_metadata
            )
            chunks.extend(sub_chunks)

        return chunks

    def _smart_split_section(
        self,
        text: str,
        code_blocks: Dict[str, str],
        section: Dict[str, Any],
        parent_metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        섹션을 스마트하게 분할 (코드 블록 보존)

        Args:
            text: 섹션 텍스트
            code_blocks: 코드 블록 딕셔너리
            section: 섹션 정보
            parent_metadata: 부모 메타데이터

        Returns:
            분할된 Document 리스트
        """
        chunks = []
        current_chunk = []
        current_size = 0

        lines = text.split('\n')

        for line in lines:
            # 코드 블록 플레이스홀더 확인
            if any(block_id in line for block_id in code_blocks):
                # 현재 청크 저장
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata={
                            **parent_metadata,
                            'section_title': section['title'],
                            'section_level': section['level'],
                            'chunk_type': 'text',
                            'has_code': False
                        }
                    ))
                    current_chunk = []
                    current_size = 0

                # 코드 블록을 별도 청크로
                for block_id in code_blocks:
                    if block_id in line:
                        code_info = code_blocks[block_id]

                        # 코드 블록이 너무 크면 분할
                        if len(code_info['content']) > self.code_block_max_size:
                            code_chunks = self._split_large_code_block(
                                code_info,
                                section,
                                parent_metadata
                            )
                            chunks.extend(code_chunks)
                        else:
                            chunks.append(Document(
                                page_content=line,  # 플레이스홀더 유지
                                metadata={
                                    **parent_metadata,
                                    'section_title': section['title'],
                                    'section_level': section['level'],
                                    'chunk_type': 'code',
                                    'language': code_info['language'],
                                    'has_code': True,
                                    'functions': code_info.get('functions', []),
                                    'classes': code_info.get('classes', [])
                                }
                            ))
            else:
                # 일반 텍스트 라인
                line_size = len(line)

                if current_size + line_size > self.chunk_size and current_chunk:
                    # 현재 청크 저장
                    chunk_text = '\n'.join(current_chunk)
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata={
                            **parent_metadata,
                            'section_title': section['title'],
                            'section_level': section['level'],
                            'chunk_type': 'text',
                            'has_code': False
                        }
                    ))

                    # 오버랩 처리
                    overlap_lines = current_chunk[-5:] if len(current_chunk) > 5 else current_chunk
                    current_chunk = overlap_lines
                    current_size = sum(len(l) for l in overlap_lines)

                current_chunk.append(line)
                current_size += line_size

        # 마지막 청크 저장
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    **parent_metadata,
                    'section_title': section['title'],
                    'section_level': section['level'],
                    'chunk_type': 'text',
                    'has_code': False
                }
            ))

        return chunks

    def _split_large_code_block(
        self,
        code_info: Dict[str, Any],
        section: Dict[str, Any],
        parent_metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        큰 코드 블록을 함수/클래스 단위로 분할

        Args:
            code_info: 코드 블록 정보
            section: 섹션 정보
            parent_metadata: 부모 메타데이터

        Returns:
            분할된 코드 청크 리스트
        """
        chunks = []
        code_content = code_info['content']
        language = code_info['language']

        if language == 'python':
            # Python 코드를 함수/클래스 단위로 분할
            try:
                tree = ast.parse(code_content)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        # 함수/클래스 코드 추출
                        start_line = node.lineno - 1
                        end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10

                        code_lines = code_content.split('\n')[start_line:end_line]
                        function_code = '\n'.join(code_lines)

                        chunks.append(Document(
                            page_content=f"```{language}\n{function_code}\n```",
                            metadata={
                                **parent_metadata,
                                'section_title': section['title'],
                                'section_level': section['level'],
                                'chunk_type': 'code_function',
                                'language': language,
                                'has_code': True,
                                'function_name': node.name if isinstance(node, ast.FunctionDef) else None,
                                'class_name': node.name if isinstance(node, ast.ClassDef) else None
                            }
                        ))
            except:
                # 파싱 실패 시 라인 수 기반으로 분할
                lines = code_content.split('\n')
                chunk_lines = self.chunk_size // 40  # 평균 라인 길이 추정

                for i in range(0, len(lines), chunk_lines):
                    chunk_code = '\n'.join(lines[i:i + chunk_lines])
                    chunks.append(Document(
                        page_content=f"```{language}\n{chunk_code}\n```",
                        metadata={
                            **parent_metadata,
                            'section_title': section['title'],
                            'section_level': section['level'],
                            'chunk_type': 'code_partial',
                            'language': language,
                            'has_code': True
                        }
                    ))
        else:
            # 다른 언어는 라인 수 기반 분할
            lines = code_content.split('\n')
            chunk_lines = self.chunk_size // 40

            for i in range(0, len(lines), chunk_lines):
                chunk_code = '\n'.join(lines[i:i + chunk_lines])
                chunks.append(Document(
                    page_content=f"```{language}\n{chunk_code}\n```",
                    metadata={
                        **parent_metadata,
                        'section_title': section['title'],
                        'section_level': section['level'],
                        'chunk_type': 'code',
                        'language': language,
                        'has_code': True
                    }
                ))

        return chunks

    def _restore_code_blocks(
        self,
        chunks: List[Document],
        code_blocks: Dict[str, Dict[str, Any]]
    ) -> List[Document]:
        """
        플레이스홀더를 실제 코드 블록으로 복원

        Args:
            chunks: 플레이스홀더가 포함된 청크 리스트
            code_blocks: 코드 블록 딕셔너리

        Returns:
            코드 블록이 복원된 청크 리스트
        """
        restored_chunks = []

        for chunk in chunks:
            content = chunk.page_content

            # 플레이스홀더 복원
            for block_id, code_info in code_blocks.items():
                if block_id in content:
                    if code_info['type'] == 'fenced':
                        replacement = f"```{code_info['language']}\n{code_info['content']}\n```"
                    else:  # inline
                        replacement = f"`{code_info['content']}`"

                    content = content.replace(block_id, replacement)

            # 복원된 청크 생성
            restored_chunk = Document(
                page_content=content,
                metadata=chunk.metadata
            )
            restored_chunks.append(restored_chunk)

        return restored_chunks

    def _extract_function_names(self, code: str, language: str) -> List[str]:
        """
        코드에서 함수 이름 추출

        Args:
            code: 코드 문자열
            language: 프로그래밍 언어

        Returns:
            함수 이름 리스트
        """
        function_names = []

        if language == 'python':
            # Python 함수 패턴
            pattern = r'def\s+(\w+)\s*\('
            function_names = re.findall(pattern, code)
        elif language in ['javascript', 'typescript']:
            # JavaScript/TypeScript 함수 패턴
            patterns = [
                r'function\s+(\w+)\s*\(',  # function declaration
                r'const\s+(\w+)\s*=\s*(?:async\s+)?\(',  # arrow function
                r'(\w+)\s*:\s*(?:async\s+)?\(',  # method
            ]
            for pattern in patterns:
                function_names.extend(re.findall(pattern, code))

        return function_names

    def _extract_class_names(self, code: str, language: str) -> List[str]:
        """
        코드에서 클래스 이름 추출

        Args:
            code: 코드 문자열
            language: 프로그래밍 언어

        Returns:
            클래스 이름 리스트
        """
        class_names = []

        if language == 'python':
            # Python 클래스 패턴
            pattern = r'class\s+(\w+)\s*(?:\(|:)'
            class_names = re.findall(pattern, code)
        elif language in ['javascript', 'typescript']:
            # JavaScript/TypeScript 클래스 패턴
            pattern = r'class\s+(\w+)\s*(?:extends\s+\w+)?\s*\{'
            class_names = re.findall(pattern, code)

        return class_names


class HTMLStructuredSplitter(StructuredTextSplitter):
    """
    HTML 문서를 위한 구조 기반 분할기
    """

    def split_html(self, html_content: str, metadata: Dict = None) -> List[Document]:
        """
        HTML 콘텐츠를 구조 기반으로 분할

        Args:
            html_content: HTML 문자열
            metadata: 메타데이터

        Returns:
            분할된 Document 리스트
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        # 코드 블록 보호
        code_elements = soup.find_all(['pre', 'code'])
        code_blocks = {}

        for i, elem in enumerate(code_elements):
            placeholder = f"__HTML_CODE_{i}__"
            code_blocks[placeholder] = {
                'content': elem.get_text(),
                'language': elem.get('class', [''])[0] if elem.get('class') else 'plain',
                'type': 'html_code'
            }
            elem.string = placeholder

        # 섹션별로 분할
        sections = []
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(header.name[1])
            title = header.get_text().strip()

            # 헤더 다음 콘텐츠 수집
            content = []
            for sibling in header.find_next_siblings():
                if sibling.name and sibling.name.startswith('h') and sibling.name[1].isdigit():
                    if int(sibling.name[1]) <= level:
                        break
                content.append(str(sibling))

            sections.append({
                'level': level,
                'title': title,
                'content': '\n'.join(content)
            })

        # 섹션을 청크로 변환
        chunks = []
        for section in sections:
            section_text = BeautifulSoup(section['content'], 'html.parser').get_text()

            # 코드 블록 복원
            for placeholder, code_info in code_blocks.items():
                if placeholder in section_text:
                    section_text = section_text.replace(
                        placeholder,
                        f"```{code_info['language']}\n{code_info['content']}\n```"
                    )

            if len(section_text) <= self.chunk_size:
                chunks.append(Document(
                    page_content=section_text,
                    metadata={
                        **(metadata or {}),
                        'section_title': section['title'],
                        'section_level': section['level'],
                        'chunk_type': 'html_section'
                    }
                ))
            else:
                # 큰 섹션은 추가 분할
                sub_chunks = self.base_splitter.split_documents([
                    Document(page_content=section_text, metadata=metadata or {})
                ])
                for chunk in sub_chunks:
                    chunk.metadata.update({
                        'section_title': section['title'],
                        'section_level': section['level'],
                        'chunk_type': 'html_section_part'
                    })
                chunks.extend(sub_chunks)

        return chunks


# 편의 함수들
def create_smart_splitter(
    chunk_size: int = 1500,
    preserve_code: bool = True,
    preserve_structure: bool = True
) -> StructuredTextSplitter:
    """
    스마트 텍스트 분할기 생성

    Args:
        chunk_size: 청크 크기
        preserve_code: 코드 블록 보존 여부
        preserve_structure: 구조 보존 여부

    Returns:
        구성된 StructuredTextSplitter 인스턴스
    """
    return StructuredTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(200, chunk_size // 5),
        preserve_code_blocks=preserve_code,
        preserve_functions=preserve_code,
        preserve_markdown_structure=preserve_structure
    )