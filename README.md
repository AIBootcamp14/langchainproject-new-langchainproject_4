# 🤝 LangChain RAG 팀 협업 - 미션 1 & 2

## 📋 프로젝트 개요

이 폴더는 LangChain 문서 기반 RAG 시스템 구축을 위한 **팀 협업 전용** 코드입니다.

### 담당 미션
1. **미션 1**: LangChain 문서 크롤링 + 코드 블록 보존 청킹 전략
2. **미션 2**: Upstage 임베딩 모델 + ChromaDB 컨테이너 Vector DB 구축

---

## 📂 파일 구조

```
Team/
├── README.md                      # 이 파일
├── COLLABORATION_CHECKLIST.md     # 미션 체크리스트 (선택)
│
├── 미션 1: 크롤링 + 청킹
│   ├── data_collector.py          # ⭐ 크롤러 + chunk_documents() 함수
│   ├── advanced_text_splitter.py  # ⭐ 구조 기반 청킹 (코드 블록 보존)
│   └── utils.py                   # 유틸리티 함수
│
├── 미션 2: 임베딩 + Vector DB
│   ├── llm.py                     # ⭐ Upstage 임베딩 모델 설정
│   ├── vector_database_docker.py  # ⭐ Docker ChromaDB 클라이언트
│   ├── initialize_vector_db.py    # ⭐ Vector DB 적재 스크립트
│   ├── docker-compose.yml         # Docker 설정
│   ├── Dockerfile                 # 앱 컨테이너 이미지
│   ├── docker_run.sh              # Docker 관리 스크립트
│   └── .env.example               # 환경 변수 템플릿
│
└── requirements.txt               # Python 패키지 목록
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 1. 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 UPSTAGE_API_KEY 입력
```

### 2. Docker 설치 (미션 2 필수)

```bash
# Docker Desktop 설치 확인
docker --version
docker-compose --version

# ChromaDB 컨테이너 시작
./docker_run.sh start

# 상태 확인
./docker_run.sh status
```

---

## ✅ 미션 1: LangChain 문서 크롤링 + 청킹

### 핵심 기능

**구조 기반 청킹 전략** - 코드 블록을 보존하면서 문서를 분할합니다.

#### 특징
- ✅ 코드 블록(````) 완전 보존
- ✅ Python 함수/클래스 정의 유지
- ✅ Markdown 헤더 기반 논리적 분할
- ✅ 청크 크기 최적화 (기본 1500자, 코드 블록 3000자)

### 사용 방법

```python
from data_collector import DataCollector

# 1. 크롤러 초기화
collector = DataCollector()

# 2. LangChain 문서 크롤링 (최대 30개)
documents = collector.collect_documents(max_pages=30)
print(f"수집된 문서: {len(documents)}개")

# 3. 구조 기반 청킹 (코드 블록 보존)
chunked_docs = collector.chunk_documents(
    documents,
    chunk_size=1500,          # 기본 청크 크기
    chunk_overlap=200,        # 오버랩
    use_structured_splitter=True  # ⭐ 구조 기반 분할 활성화
)
print(f"청크 결과: {len(chunked_docs)}개")

# 4. 반환값 확인
for i, chunk in enumerate(chunked_docs[:3]):
    print(f"\n[청크 {i+1}]")
    print(f"내용: {chunk.page_content[:100]}...")
    print(f"메타데이터: {chunk.metadata}")
```

### 청킹 전략 검증

```bash
# 구조 기반 분할기 테스트
python advanced_text_splitter.py

# 코드 블록 보존 확인
python -c "
from data_collector import DataCollector

collector = DataCollector()
docs = collector.collect_documents(max_pages=3)
chunks = collector.chunk_documents(docs, use_structured_splitter=True)

# 코드 블록 완전성 검사
broken = sum(1 for c in chunks if c.page_content.count('\`\`\`') % 2 != 0)
print(f'총 청크: {len(chunks)}개')
print(f'깨진 코드 블록: {broken}개')
print(f'결과: {\"✅ 통과\" if broken == 0 else \"⚠️ 확인 필요\"}')
"
```

---

## ✅ 미션 2: Upstage 임베딩 + Vector DB

### 핵심 기능

**Upstage Solar 임베딩** + **ChromaDB 컨테이너**를 활용한 Vector DB 구축

#### 특징
- ✅ `solar-embedding-1-large` 모델 (4096차원)
- ✅ Docker 기반 ChromaDB 서버
- ✅ 팀원 청크 데이터 수신 인터페이스
- ✅ 배치 처리 (100개씩) + 진행률 표시

### 1단계: 임베딩 모델 설정

```python
from llm import get_embeddings

# Upstage Solar 임베딩 모델 로드
embeddings = get_embeddings(model="solar-embedding-1-large")

# 테스트
test_vector = embeddings.embed_query("LangChain이란 무엇인가?")
print(f"임베딩 차원: {len(test_vector)}차원")  # 4096
```

```bash
# 테스트 실행
python llm.py
```

### 2단계: ChromaDB 컨테이너 시작

```bash
# Docker ChromaDB 시작
./docker_run.sh start

# 헬스체크
curl http://localhost:8000/api/v1/heartbeat

# 로그 확인
./docker_run.sh logs chromadb
```

### 3단계: 팀원 청크 데이터 받기

**팀원이 제공할 청크 데이터 형식:**

```python
from langchain.schema import Document

# 팀원이 전달하는 청크 리스트
team_chunks = [
    Document(
        page_content="LangChain은 LLM 기반 애플리케이션 개발 프레임워크입니다...",
        metadata={
            "source": "https://python.langchain.com/docs/introduction",
            "title": "LangChain 소개",
            "chunk_index": 0
        }
    ),
    Document(
        page_content="```python\nfrom langchain.llms import OpenAI\n```",
        metadata={
            "source": "https://python.langchain.com/docs/modules/llms",
            "title": "LLM 모듈",
            "chunk_index": 1
        }
    ),
    # ... 더 많은 청크
]
```

### 4단계: Vector DB에 적재

```python
from initialize_vector_db import VectorDBInitializer

# 1. 초기화 (Docker 모드)
initializer = VectorDBInitializer(use_docker=True)

# 2. 팀원 청크 데이터 적재
loaded_count = initializer.load_to_vector_db(
    documents=team_chunks,  # 팀원이 제공한 청크
    batch_size=100,         # 배치 크기
    show_progress=True      # 진행률 표시
)

print(f"✅ {loaded_count}개 청크 적재 완료")
```

### 5단계: 검증

```python
from vector_database_docker import DockerVectorDatabase

# Vector DB 연결
vdb = DockerVectorDatabase()
vdb.init_vectorstore()

# 통계 확인
stats = vdb.get_statistics()
print(f"총 문서: {stats['total_documents']}개")

# 검색 테스트
results = vdb.search("LangChain 메모리 관리", k=3)
print(f"검색 결과: {len(results)}개")
for i, doc in enumerate(results):
    print(f"\n[결과 {i+1}]")
    print(f"내용: {doc.page_content[:100]}...")
    print(f"출처: {doc.metadata['source']}")
```

---

## 🔄 통합 파이프라인

**전체 과정을 한번에 실행** (크롤링 → 청킹 → 임베딩 → Vector DB)

```bash
# 방법 1: 전체 파이프라인 실행
python initialize_vector_db.py --docker --reset --max-pages 30

# 방법 2: Docker 스크립트 사용
./docker_run.sh init-db

# 방법 3: 테스트 모드 (5개 문서만)
python initialize_vector_db.py --docker --test-only
```

### 옵션 설명
- `--docker`: Docker ChromaDB 사용 (필수)
- `--reset`: 기존 데이터 삭제 후 재로딩
- `--max-pages N`: 최대 N개 문서 크롤링
- `--test-only`: 테스트 모드 (5개 문서)

---

## 🎯 팀원과 협업하기

### 팀원 4에게 청크 데이터 받기

**1. 팀원에게 요청할 형식:**

```python
# 팀원 4님께: 아래 형식으로 청크 데이터를 전달해주세요

from langchain.schema import Document

chunks = [
    Document(
        page_content="청크 텍스트...",
        metadata={
            "source": "문서 URL",
            "title": "문서 제목",
            "chunk_index": 0
        }
    ),
    # ... 더 많은 청크
]

# 파일로 저장 (선택)
import pickle
with open("team4_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
```

**2. 청크 데이터 적재:**

```python
from initialize_vector_db import VectorDBInitializer
import pickle

# 팀원 데이터 로드 (파일로 받은 경우)
with open("team4_chunks.pkl", "rb") as f:
    team_chunks = pickle.load(f)

# Vector DB에 적재
initializer = VectorDBInitializer(use_docker=True)
loaded_count = initializer.load_to_vector_db(team_chunks)
print(f"✅ {loaded_count}개 청크 적재 완료")
```

---

## 🧪 테스트 및 검증

### 1. 미션 1 검증

```bash
# 청킹 전략 테스트
python advanced_text_splitter.py

# 크롤링 + 청킹 통합 테스트
python -c "
from data_collector import DataCollector
collector = DataCollector()
docs = collector.collect_documents(max_pages=5)
chunks = collector.chunk_documents(docs, use_structured_splitter=True)
print(f'✅ 크롤링: {len(docs)}개')
print(f'✅ 청킹: {len(chunks)}개')
"
```

### 2. 미션 2 검증

```bash
# 임베딩 모델 테스트
python llm.py

# ChromaDB 연결 테스트
curl http://localhost:8000/api/v1/heartbeat

# Vector DB 적재 테스트
python initialize_vector_db.py --docker --test-only
```

### 3. 통합 검증

```bash
# 전체 파이프라인 실행
python initialize_vector_db.py --docker --reset --max-pages 10

# 결과 확인
python -c "
from vector_database_docker import DockerVectorDatabase
vdb = DockerVectorDatabase()
vdb.init_vectorstore()
stats = vdb.get_statistics()
print(f'✅ Vector DB 문서: {stats[\"total_documents\"]}개')
"
```

---

## 📊 Docker 관리

```bash
# ChromaDB 시작
./docker_run.sh start

# 상태 확인
./docker_run.sh status

# 로그 확인
./docker_run.sh logs chromadb

# 재시작
./docker_run.sh restart

# 중지
./docker_run.sh stop

# 전체 삭제 (데이터 포함)
./docker_run.sh clean

# 도움말
./docker_run.sh help
```

---

## 🔧 트러블슈팅

### 문제 1: "UPSTAGE_API_KEY가 설정되지 않았습니다"

```bash
# .env 파일 확인
cat .env

# API 키 추가
echo "UPSTAGE_API_KEY=your-actual-key" > .env
```

### 문제 2: Docker ChromaDB 연결 실패

```bash
# 컨테이너 상태 확인
docker ps | grep chroma

# ChromaDB 재시작
./docker_run.sh restart

# 포트 확인
lsof -i :8000
```

### 문제 3: 코드 블록이 깨짐

```python
# advanced_text_splitter.py 설정 확인
splitter = StructuredTextSplitter(
    preserve_code_blocks=True,  # ⭐ 반드시 True
    code_block_max_size=3000    # 충분히 큰 값
)
```

---

## 📚 주요 함수 레퍼런스

### 미션 1: chunk_documents()

```python
def chunk_documents(
    self,
    documents: List[Document],      # 입력: 크롤링된 문서 리스트
    chunk_size: int = 1000,         # 기본 청크 크기
    chunk_overlap: int = 200,       # 오버랩 크기
    use_structured_splitter: bool = True  # 구조 기반 분할 사용
) -> List[Document]:                # 출력: 청크된 문서 리스트
    """
    문서를 구조 기반으로 청킹합니다.
    코드 블록과 함수 정의를 보존합니다.
    """
```

### 미션 2: load_to_vector_db()

```python
def load_to_vector_db(
    self,
    documents: List[Document],      # 입력: 청크 리스트
    batch_size: int = 100,          # 배치 크기
    show_progress: bool = True      # 진행률 표시
) -> int:                           # 출력: 적재된 문서 수
    """
    청크 데이터를 Vector DB에 적재합니다.
    Upstage 임베딩을 사용하여 벡터화합니다.
    """
```

---

## 🎓 추가 자료

- **LangChain 공식 문서**: https://python.langchain.com
- **Upstage AI 콘솔**: https://console.upstage.ai
- **ChromaDB 문서**: https://docs.trychroma.com
- **Docker 문서**: https://docs.docker.com

---

## 📞 팀 협업 체크리스트

미션 완료 전 확인사항은 `COLLABORATION_CHECKLIST.md` 파일을 참고하세요.

### 주요 체크포인트
- [ ] 미션 1: 코드 블록 보존 청킹 동작 확인
- [ ] 미션 2: Upstage 임베딩 모델 테스트 통과
- [ ] Docker ChromaDB 정상 실행
- [ ] 팀원 데이터 수신 인터페이스 준비
- [ ] 통합 테스트 성공

---

**작성일**: 2025-10-30
**버전**: 1.0
**담당**: 미션 1 & 2 (크롤링, 청킹, 임베딩, Vector DB)
