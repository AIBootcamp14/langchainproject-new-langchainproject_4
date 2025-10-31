# 📁 Team 폴더 파일 목록

## 📊 전체 구조

```
Team/
├── README.md                      # 팀 협업 메인 가이드
├── COLLABORATION_CHECKLIST.md     # 미션 체크리스트
├── FILE_LIST.md                   # 이 파일 (파일 설명)
├── requirements.txt               # Python 패키지 목록
├── .env.example                   # 환경 변수 템플릿
│
├── 📦 미션 1: 크롤링 + 청킹
│   ├── data_collector.py          # 크롤러 + chunk_documents() 함수
│   ├── advanced_text_splitter.py  # 구조 기반 청킹 (코드 블록 보존)
│   └── utils.py                   # 유틸리티 함수
│
└── 📦 미션 2: 임베딩 + Vector DB
    ├── llm.py                     # Upstage 임베딩 모델 설정
    ├── vector_database_docker.py  # Docker ChromaDB 클라이언트
    ├── initialize_vector_db.py    # Vector DB 적재 스크립트
    ├── docker-compose.yml         # Docker 컨테이너 설정
    ├── Dockerfile                 # 앱 컨테이너 이미지
    └── docker_run.sh              # Docker 관리 스크립트
```

---

## 📋 파일 상세 설명

### 📖 문서 파일

#### `README.md` (12KB)
- **용도**: 팀 협업 메인 가이드
- **내용**:
  - 빠른 시작 가이드
  - 미션 1, 2 사용법
  - 코드 예제
  - 팀원 협업 방법
  - 트러블슈팅
- **대상**: 모든 팀원

#### `COLLABORATION_CHECKLIST.md` (19KB)
- **용도**: 미션 완료 검증 체크리스트
- **내용**:
  - 미션 1 체크리스트 (크롤링 + 청킹)
  - 미션 2 체크리스트 (임베딩 + Vector DB)
  - 통합 검증 절차
  - TODO 리스트
  - GitHub 업데이트 가이드
- **대상**: 미션 담당자 (본인)

#### `FILE_LIST.md` (이 파일)
- **용도**: 전체 파일 설명서
- **내용**: 각 파일의 용도와 핵심 기능

---

### ⚙️ 설정 파일

#### `requirements.txt` (846B)
- **용도**: Python 패키지 의존성 목록
- **설치 방법**: `pip install -r requirements.txt`
- **주요 패키지**:
  - `langchain`: LangChain 프레임워크
  - `langchain-upstage`: Upstage API 연동
  - `chromadb`: Vector Database
  - `beautifulsoup4`: HTML 파싱
  - `selenium`: 동적 페이지 크롤링

#### `.env.example` (961B)
- **용도**: 환경 변수 템플릿
- **사용법**:
  ```bash
  cp .env.example .env
  # .env 파일을 열어 UPSTAGE_API_KEY 입력
  ```
- **필수 변수**:
  - `UPSTAGE_API_KEY`: Upstage API 키

---

## 🎯 미션 1: 크롤링 + 청킹

### `data_collector.py` (17KB, 527줄)
- **용도**: LangChain 문서 크롤러 + 청킹 함수
- **핵심 기능**:
  - ✅ `collect_documents()`: 웹 크롤링
  - ✅ `chunk_documents()`: 구조 기반 청킹 (⭐ 핵심 함수)
  - ✅ `save_to_sqlite()`: SQLite DB 저장
- **주요 클래스**: `DataCollector`
- **실행 방법**:
  ```python
  from data_collector import DataCollector
  collector = DataCollector()
  docs = collector.collect_documents(max_pages=30)
  chunks = collector.chunk_documents(docs, use_structured_splitter=True)
  ```

### `advanced_text_splitter.py` (24KB, 703줄)
- **용도**: 구조 기반 텍스트 분할기
- **핵심 기능**:
  - ✅ 코드 블록(````) 보존
  - ✅ Python 함수/클래스 정의 보존
  - ✅ Markdown 헤더 기반 분할
  - ✅ 청크 크기 최적화
- **주요 클래스**: `StructuredTextSplitter`
- **테스트 방법**:
  ```bash
  python advanced_text_splitter.py
  ```

### `utils.py` (8.9KB, 295줄)
- **용도**: 공통 유틸리티 함수
- **주요 함수**:
  - `extract_text_from_html()`: HTML → 텍스트
  - `clean_text()`: 텍스트 정제
  - `validate_url()`: URL 검증
  - `retry_on_failure()`: 재시도 데코레이터

---

## 🎯 미션 2: 임베딩 + Vector DB

### `llm.py` (4.1KB, 165줄)
- **용도**: Upstage 임베딩 모델 설정
- **핵심 기능**:
  - ✅ `get_embeddings()`: Solar 임베딩 모델 로드 (⭐ 핵심 함수)
  - ✅ `get_llm()`: Solar LLM 로드
- **모델**:
  - 임베딩: `solar-embedding-1-large` (4096차원)
  - LLM: `solar-pro`
- **테스트 방법**:
  ```bash
  python llm.py
  ```

### `vector_database_docker.py` (12KB, 389줄)
- **용도**: Docker ChromaDB 클라이언트
- **핵심 기능**:
  - ✅ HTTP 기반 ChromaDB 연결
  - ✅ 문서 벡터화 및 저장
  - ✅ 유사도 검색
  - ✅ 통계 조회
- **주요 클래스**: `DockerVectorDatabase`
- **연결 정보**:
  - 호스트: `localhost`
  - 포트: `8000`
  - 컬렉션: `langchain_docs`

### `initialize_vector_db.py` (13KB, 333줄)
- **용도**: Vector DB 적재 스크립트
- **핵심 기능**:
  - ✅ `load_to_vector_db()`: 청크 데이터 적재 (⭐ 핵심 함수)
  - ✅ `run_full_pipeline()`: 전체 파이프라인 실행
  - ✅ 배치 처리 (100개씩)
  - ✅ 진행률 표시
- **주요 클래스**: `VectorDBInitializer`
- **실행 방법**:
  ```bash
  # 전체 파이프라인
  python initialize_vector_db.py --docker --reset --max-pages 30

  # 테스트 모드
  python initialize_vector_db.py --docker --test-only
  ```

---

## 🐳 Docker 관련 파일

### `docker-compose.yml` (1.7KB)
- **용도**: Docker 컨테이너 오케스트레이션
- **서비스**:
  - `chromadb`: ChromaDB 서버 (포트 8000)
  - `app`: LangChain 앱 (선택)
- **볼륨**: `./data/chroma_docker` (데이터 영구 저장)
- **실행 방법**:
  ```bash
  docker-compose up -d chromadb
  ```

### `Dockerfile` (1.0KB)
- **용도**: 앱 컨테이너 이미지 빌드
- **베이스 이미지**: `python:3.10-slim`
- **포함 내용**:
  - Python 패키지
  - 애플리케이션 코드
- **빌드 방법**:
  ```bash
  docker build -t langchain-rag-app .
  ```

### `docker_run.sh` (4.6KB, 실행 가능)
- **용도**: Docker 관리 스크립트
- **명령어**:
  - `start`: 컨테이너 시작
  - `stop`: 컨테이너 중지
  - `restart`: 재시작
  - `status`: 상태 확인
  - `logs`: 로그 확인
  - `clean`: 전체 삭제
  - `init-db`: Vector DB 초기화
- **실행 방법**:
  ```bash
  ./docker_run.sh start
  ./docker_run.sh status
  ```

---

## 🎓 파일 크기 및 복잡도

| 파일 | 크기 | 줄 수 | 복잡도 | 중요도 |
|------|------|-------|--------|--------|
| `README.md` | 12KB | - | 낮음 | ⭐⭐⭐⭐⭐ |
| `COLLABORATION_CHECKLIST.md` | 19KB | - | 낮음 | ⭐⭐⭐⭐ |
| `data_collector.py` | 17KB | 527 | 높음 | ⭐⭐⭐⭐⭐ |
| `advanced_text_splitter.py` | 24KB | 703 | 높음 | ⭐⭐⭐⭐⭐ |
| `llm.py` | 4.1KB | 165 | 중간 | ⭐⭐⭐⭐⭐ |
| `vector_database_docker.py` | 12KB | 389 | 중간 | ⭐⭐⭐⭐⭐ |
| `initialize_vector_db.py` | 13KB | 333 | 중간 | ⭐⭐⭐⭐⭐ |
| `utils.py` | 8.9KB | 295 | 낮음 | ⭐⭐⭐ |
| `docker-compose.yml` | 1.7KB | - | 낮음 | ⭐⭐⭐⭐ |
| `docker_run.sh` | 4.6KB | - | 낮음 | ⭐⭐⭐ |

---

## 🔑 핵심 함수 위치

### 미션 1 핵심 함수

#### `DataCollector.chunk_documents()`
📍 **위치**: `data_collector.py:194-249`
```python
def chunk_documents(
    self,
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_structured_splitter: bool = True
) -> List[Document]:
```

#### `StructuredTextSplitter.split_documents()`
📍 **위치**: `advanced_text_splitter.py:122-176`
```python
def split_documents(
    self,
    documents: List[Document]
) -> List[Document]:
```

### 미션 2 핵심 함수

#### `get_embeddings()`
📍 **위치**: `llm.py:53-82`
```python
def get_embeddings(
    model: str = "solar-embedding-1-large"
) -> Embeddings:
```

#### `VectorDBInitializer.load_to_vector_db()`
📍 **위치**: `initialize_vector_db.py:129-205`
```python
def load_to_vector_db(
    self,
    documents: List[Document],
    batch_size: int = 100,
    show_progress: bool = True
) -> int:
```

---

## 📦 의존성 관계

```
initialize_vector_db.py
    ├── data_collector.py
    │   ├── advanced_text_splitter.py
    │   └── utils.py
    ├── vector_database_docker.py
    │   └── llm.py
    └── llm.py

docker_run.sh
    └── docker-compose.yml
```

---

## 🚀 GitHub 업데이트 시 포함할 파일

### ✅ 필수 파일 (13개)
- [x] `README.md`
- [x] `COLLABORATION_CHECKLIST.md`
- [x] `FILE_LIST.md`
- [x] `requirements.txt`
- [x] `.env.example`
- [x] `data_collector.py`
- [x] `advanced_text_splitter.py`
- [x] `utils.py`
- [x] `llm.py`
- [x] `vector_database_docker.py`
- [x] `initialize_vector_db.py`
- [x] `docker-compose.yml`
- [x] `Dockerfile`
- [x] `docker_run.sh`

### ❌ 제외할 파일
- `.env` (API 키 포함, 보안 위험)
- `*.pyc` (컴파일된 Python 파일)
- `__pycache__/` (캐시 디렉토리)
- `data/` (로컬 데이터베이스)

---

## 📞 참고 사항

### 팀원이 실행하려면?

1. **환경 설정**:
   ```bash
   cd Team
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   # .env 파일에 API 키 입력
   ```

2. **미션 1 테스트**:
   ```bash
   python advanced_text_splitter.py
   python -c "from data_collector import DataCollector; ..."
   ```

3. **미션 2 테스트**:
   ```bash
   ./docker_run.sh start
   python llm.py
   python initialize_vector_db.py --docker --test-only
   ```

---

**작성일**: 2025-10-30
**버전**: 1.0
**총 파일 수**: 15개 (문서 3개 + 코드 10개 + Docker 2개)
