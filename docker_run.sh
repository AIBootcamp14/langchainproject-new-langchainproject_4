#!/bin/bash

# Docker 환경 실행 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🐳 LangChain RAG Docker 환경 관리${NC}"
echo "========================================"

# 명령어 파싱
CMD=${1:-help}

case $CMD in
    start)
        echo -e "${YELLOW}시작: Docker Compose 환경${NC}"
        docker-compose up -d

        echo -e "\n${GREEN}✅ 서비스 시작 완료!${NC}"
        echo "  - ChromaDB: http://localhost:8000"
        echo "  - Streamlit: http://localhost:8501"

        echo -e "\n${YELLOW}서비스 상태 확인 중...${NC}"
        sleep 5
        docker-compose ps
        ;;

    stop)
        echo -e "${YELLOW}중지: Docker Compose 환경${NC}"
        docker-compose down
        echo -e "${GREEN}✅ 서비스 중지 완료!${NC}"
        ;;

    restart)
        echo -e "${YELLOW}재시작: Docker Compose 환경${NC}"
        docker-compose restart
        echo -e "${GREEN}✅ 서비스 재시작 완료!${NC}"
        ;;

    build)
        echo -e "${YELLOW}빌드: Docker 이미지${NC}"
        docker-compose build --no-cache
        echo -e "${GREEN}✅ 이미지 빌드 완료!${NC}"
        ;;

    logs)
        echo -e "${YELLOW}로그 확인${NC}"
        SERVICE=${2:-}
        if [ -z "$SERVICE" ]; then
            docker-compose logs -f --tail=100
        else
            docker-compose logs -f --tail=100 $SERVICE
        fi
        ;;

    status)
        echo -e "${YELLOW}서비스 상태${NC}"
        docker-compose ps

        echo -e "\n${YELLOW}헬스체크${NC}"

        # ChromaDB 헬스체크
        if curl -f http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then
            echo -e "  ChromaDB: ${GREEN}✅ 정상${NC}"
        else
            echo -e "  ChromaDB: ${RED}❌ 오류${NC}"
        fi

        # Streamlit 헬스체크
        if curl -f http://localhost:8501/ > /dev/null 2>&1; then
            echo -e "  Streamlit: ${GREEN}✅ 정상${NC}"
        else
            echo -e "  Streamlit: ${RED}❌ 오류${NC}"
        fi
        ;;

    init-db)
        echo -e "${YELLOW}Vector DB 초기화${NC}"

        # ChromaDB가 실행 중인지 확인
        if ! curl -f http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then
            echo -e "${RED}❌ ChromaDB가 실행되지 않았습니다.${NC}"
            echo "먼저 './docker_run.sh start'를 실행하세요."
            exit 1
        fi

        # 초기화 스크립트 실행
        echo "데이터 로딩 중..."
        python initialize_vector_db.py --docker --reset --max-pages 30

        echo -e "${GREEN}✅ Vector DB 초기화 완료!${NC}"
        ;;

    clean)
        echo -e "${YELLOW}클린업: 모든 컨테이너와 볼륨 제거${NC}"
        read -p "정말로 모든 데이터를 삭제하시겠습니까? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down -v
            rm -rf ./data/chroma_docker
            echo -e "${GREEN}✅ 클린업 완료!${NC}"
        else
            echo "취소되었습니다."
        fi
        ;;

    shell)
        echo -e "${YELLOW}컨테이너 쉘 접속${NC}"
        SERVICE=${2:-langchain-app}
        docker-compose exec $SERVICE /bin/bash
        ;;

    test)
        echo -e "${YELLOW}테스트 실행${NC}"

        # 테스트 모드로 초기화
        echo "테스트 데이터 로딩..."
        python initialize_vector_db.py --docker --test-only

        echo -e "${GREEN}✅ 테스트 완료!${NC}"
        ;;

    help|*)
        echo "사용법: $0 {start|stop|restart|build|logs|status|init-db|clean|shell|test} [옵션]"
        echo ""
        echo "명령어:"
        echo "  start    - Docker Compose 서비스 시작"
        echo "  stop     - Docker Compose 서비스 중지"
        echo "  restart  - Docker Compose 서비스 재시작"
        echo "  build    - Docker 이미지 빌드"
        echo "  logs     - 로그 확인 (옵션: 서비스명)"
        echo "  status   - 서비스 상태 확인"
        echo "  init-db  - Vector DB 초기화 및 데이터 로딩"
        echo "  clean    - 모든 컨테이너와 데이터 삭제"
        echo "  shell    - 컨테이너 쉘 접속 (옵션: 서비스명)"
        echo "  test     - 테스트 모드 실행 (5개 문서만)"
        echo "  help     - 도움말 표시"
        echo ""
        echo "예제:"
        echo "  $0 start           # 서비스 시작"
        echo "  $0 logs chromadb   # ChromaDB 로그 확인"
        echo "  $0 init-db         # Vector DB 초기화"
        ;;
esac