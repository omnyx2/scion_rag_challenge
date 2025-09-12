#!/usr/bin/env python3
"""
검색 메타데이터 시스템 - 메인 실행 파일
- 사용자 친화적 명령행 인터페이스
- 단일/배치/변환 모드 지원
- 상세한 사용법 안내
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional

# 상위 디렉토리의 모듈 import를 위한 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from search_meta_system import SearchMetaSystem
from utils.file_manager import FileManager

def print_usage():
    """사용법 출력"""
    print("""
🔍 검색 메타데이터 시스템 v2.0
=====================================

사용법: python main.py [명령] [옵션]

📋 명령어:
  single [질문]           - 단일 질문 처리
  batch [CSV파일]         - CSV 파일에서 배치 처리
  convert                 - 최신 JSON 결과를 CSV/JSONL로 변환
  info                    - 시스템 정보 출력
  help                    - 이 도움말 출력

📝 예시:
  python main.py single "인공지능의 미래는 어떻게 될까요?"
  python main.py batch test.csv
  python main.py convert
  python main.py info

⚙️  환경 변수:
  GEMINI_API_KEY         - Gemini API 키 (필수)
  TARGET_DOCUMENTS       - 목표 문서 수 (기본: 50)
  OUTPUT_DIRECTORY       - 출력 디렉토리 (기본: ./outputs)
  LOG_LEVEL              - 로그 레벨 (기본: INFO)

📁 출력 파일:
  - search_meta_results_YYYYMMDD_HHMMSS.json  (원본 결과)
  - search_results_YYYYMMDD_HHMMSS.csv        (CSV 형식)
  - search_documents_YYYYMMDD_HHMMSS.jsonl    (JSONL 형식)
  - elapsed_times.json                         (실행 시간)

🔧 설정 파일:
  - ./configs/scienceon_api_credentials.json  (ScienceON API 자격증명)
""")

def get_gemini_api_key() -> str:
    """Gemini API 키 가져오기"""
    # 환경 변수에서 먼저 확인
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key
    
    # 설정 파일에서 확인
    config_path = Path("./configs/gemini_api_credentials.json")
    if config_path.exists():
        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get("api_key", "")
        except Exception as e:
            logging.warning(f"설정 파일 읽기 실패: {e}")
    
    # 사용자 입력 요청
    print("⚠️  GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
    api_key = input("Gemini API 키를 입력하세요: ").strip()
    
    if not api_key:
        print("❌ API 키가 필요합니다.")
        sys.exit(1)
    
    return api_key

def run_single_mode(system: SearchMetaSystem, query: str):
    """단일 모드 실행"""
    print(f"🔍 단일 질문 처리 시작")
    print("=" * 50)
    print(f"질문: {query}")
    print("=" * 50)
    
    try:
        result = system.process_single_query(query)
        
        if result["status"] == "success":
            print(f"✅ 처리 완료!")
            print(f"   찾은 문서: {result['document_count']}개")
            print(f"   처리 시간: {result['processing_time_seconds']:.2f}초")
            print(f"   검색어: {len(result['search_terms'])}개")
        else:
            print(f"❌ 처리 실패: {result.get('error_message', '알 수 없는 오류')}")
            
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        logging.error(f"단일 모드 실행 실패: {e}")

def run_batch_mode(system: SearchMetaSystem, csv_path: str, max_queries: int = None):
    """배치 모드 실행"""
    print(f"📊 배치 처리 시작")
    print("=" * 50)
    print(f"CSV 파일: {csv_path}")
    if max_queries:
        print(f"최대 처리 질문 수: {max_queries}개")
    print("=" * 50)
    
    try:
        # CSV 파일 존재 확인
        if not Path(csv_path).exists():
            print(f"❌ CSV 파일을 찾을 수 없습니다: {csv_path}")
            return
        
        result = system.process_batch_from_csv(csv_path, max_queries=max_queries)
        
        batch_info = result.get("batch_statistics", {})
        if batch_info.get("total_queries", 0) > 0:
            print(f"✅ 배치 처리 완료!")
            print(f"   총 질문: {batch_info['total_queries']}개")
            print(f"   성공: {batch_info['successful_queries']}개")
            print(f"   실패: {batch_info['failed_queries']}개")
            print(f"   성공률: {batch_info['success_rate']:.1f}%")
            print(f"   총 문서: {batch_info['total_documents_found']}개")
            print(f"   평균 문서/질문: {batch_info['avg_documents_per_query']:.1f}개")
        else:
            print(f"❌ 배치 처리 실패: {batch_info.get('error', '알 수 없는 오류')}")
            
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        logging.error(f"배치 모드 실행 실패: {e}")

def run_convert_mode(system: SearchMetaSystem):
    """변환 모드 실행"""
    print(f"🔄 결과 변환 시작")
    print("=" * 50)
    
    try:
        files = system.convert_latest_results()
        
        print(f"✅ 변환 완료!")
        print(f"   CSV 파일: {files['csv']}")
        print(f"   JSONL 파일: {files['jsonl']}")
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        logging.error(f"변환 모드 실행 실패: {e}")

def run_info_mode(system: SearchMetaSystem):
    """정보 모드 실행"""
    system.print_system_info()

def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "help":
        print_usage()
        return
    
    # API 키 가져오기
    try:
        api_key = get_gemini_api_key()
    except KeyboardInterrupt:
        print("\n❌ 사용자가 취소했습니다.")
        return
    
    # 시스템 초기화
    try:
        system = SearchMetaSystem(api_key)
    except Exception as e:
        print(f"❌ 시스템 초기화 실패: {e}")
        logging.error(f"시스템 초기화 실패: {e}")
        return
    
    try:
        if command == "single":
            if len(sys.argv) < 3:
                print("❌ 질문을 입력해주세요.")
                print("사용법: python main.py single \"질문내용\"")
                return
            
            query = sys.argv[2]
            run_single_mode(system, query)
            
        elif command == "batch":
            if len(sys.argv) < 3:
                print("❌ CSV 파일 경로를 입력해주세요.")
                print("사용법: python main.py batch test.csv [최대질문수]")
                return
            
            csv_path = sys.argv[2]
            max_queries = None
            if len(sys.argv) > 3:
                try:
                    max_queries = int(sys.argv[3])
                except ValueError:
                    print("❌ 최대 질문 수는 숫자여야 합니다.")
                    return
            
            run_batch_mode(system, csv_path, max_queries)
            
        elif command == "convert":
            run_convert_mode(system)
            
        elif command == "info":
            run_info_mode(system)
            
        else:
            print(f"❌ 알 수 없는 명령어: {command}")
            print_usage()
            return
            
    except KeyboardInterrupt:
        print("\n❌ 사용자가 취소했습니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")
        logging.error(f"메인 실행 실패: {e}")
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()
