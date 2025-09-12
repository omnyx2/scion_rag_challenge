"""
배치 쿼리 처리기 모듈
- 여러 질문에 대한 배치 처리
- 진행률 표시 및 통계
- 결과 저장 및 변환
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from processors.single_query_processor import SingleQueryProcessor
from utils.file_manager import FileManager
from utils.result_converter import ResultConverter

class BatchQueryProcessor:
    """배치 쿼리 처리기"""
    
    def __init__(self, single_processor: SingleQueryProcessor, file_manager: FileManager, 
                 result_converter: ResultConverter):
        """
        배치 쿼리 처리기 초기화
        
        Args:
            single_processor: 단일 쿼리 처리기
            file_manager: 파일 관리자
            result_converter: 결과 변환기
        """
        self.single_processor = single_processor
        self.file_manager = file_manager
        self.result_converter = result_converter
        
    def process_queries_from_csv(self, csv_path: str, target_documents: int = 50, 
                                max_queries: Optional[int] = None) -> Dict[str, Any]:
        """
        CSV 파일에서 쿼리를 읽어 배치 처리
        
        Args:
            csv_path: CSV 파일 경로
            target_documents: 질문당 목표 문서 수
            max_queries: 최대 처리할 질문 수 (None이면 전체)
            
        Returns:
            배치 처리 결과
        """
        try:
            # CSV 파일 읽기
            queries = self._load_queries_from_csv(csv_path, max_queries)
            
            if not queries:
                logging.error("처리할 질문이 없습니다")
                return self._create_batch_error_result("처리할 질문이 없습니다")
            
            logging.info(f"CSV에서 {len(queries)}개의 질문을 로드했습니다: {csv_path}")
            
            # 배치 처리 실행
            return self.process_queries(queries, target_documents)
            
        except Exception as e:
            logging.error(f"CSV 배치 처리 실패: {e}")
            return self._create_batch_error_result(str(e))
    
    def process_queries(self, queries: List[str], target_documents: int = 50) -> Dict[str, Any]:
        """
        질문 리스트에 대한 배치 처리
        
        Args:
            queries: 처리할 질문 리스트
            target_documents: 질문당 목표 문서 수
            
        Returns:
            배치 처리 결과
        """
        start_time = datetime.now()
        results = []
        successful_queries = 0
        failed_queries = 0
        total_documents = 0
        
        logging.info(f"배치 실행 모드: {len(queries)}개 쿼리")
        logging.info(f"배치 처리 시작: {len(queries)}개 쿼리")
        
        for i, query in enumerate(queries, 1):
            try:
                logging.info(f"  진행률: {i}/{len(queries)} - {query[:30]}...")
                logging.info(f"진행률: {i}/{len(queries)}")
                
                # 단일 쿼리 처리
                result = self.single_processor.process_query(query, target_documents)
                results.append(result)
                
                if result.get("status") == "success":
                    successful_queries += 1
                    total_documents += result.get("document_count", 0)
                else:
                    failed_queries += 1
                    
            except Exception as e:
                logging.error(f"질문 {i} 처리 실패: {e}")
                failed_queries += 1
                results.append({
                    "query": query,
                    "status": "error",
                    "error_message": str(e),
                    "documents": [],
                    "document_count": 0,
                    "timestamp": datetime.now().isoformat()
                })
        
        # 배치 결과 생성
        batch_result = self._create_batch_result(
            results=results,
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            total_documents=total_documents,
            processing_time=datetime.now() - start_time
        )
        
        # 결과 저장
        self._save_batch_results(batch_result)
        
        # 자동 변환
        self._auto_convert_results(batch_result)
        
        logging.info(f"배치 처리 완료: {successful_queries}/{len(queries)} 성공")
        
        return batch_result
    
    def _load_queries_from_csv(self, csv_path: str, max_queries: Optional[int] = None) -> List[str]:
        """CSV 파일에서 질문 로드"""
        try:
            df = pd.read_csv(csv_path)
            
            # 첫 번째 컬럼을 질문으로 사용
            queries = df.iloc[:, 0].dropna().tolist()
            
            if max_queries:
                queries = queries[:max_queries]
            
            return queries
            
        except Exception as e:
            logging.error(f"CSV 파일 읽기 실패: {e}")
            return []
    
    def _create_batch_result(self, results: List[Dict[str, Any]], successful_queries: int, 
                           failed_queries: int, total_documents: int, processing_time) -> Dict[str, Any]:
        """배치 결과 생성 (기존 형식에 맞춤)"""
        # 키워드 통계 계산
        total_korean_keywords = sum(len(r.get("keywords", {}).get("korean", [])) for r in results if r.get("status") == "success")
        total_english_keywords = sum(len(r.get("keywords", {}).get("english", [])) for r in results if r.get("status") == "success")
        total_search_queries = sum(len(r.get("search_terms", [])) for r in results if r.get("status") == "success")
        
        return {
            "batch_statistics": {
                "total_queries": len(results),
                "successful_queries": successful_queries,
                "failed_queries": failed_queries,
                "success_rate": (successful_queries / len(results)) * 100 if results else 0,
                "total_documents_found": total_documents,
                "avg_documents_per_query": total_documents / successful_queries if successful_queries > 0 else 0,
                "total_korean_keywords": total_korean_keywords,
                "total_english_keywords": total_english_keywords,
                "avg_korean_keywords_per_query": total_korean_keywords / successful_queries if successful_queries > 0 else 0,
                "avg_english_keywords_per_query": total_english_keywords / successful_queries if successful_queries > 0 else 0,
                "total_search_queries_generated": total_search_queries,
                "avg_search_queries_per_query": total_search_queries / successful_queries if successful_queries > 0 else 0,
                "processing_timestamp": datetime.now().isoformat()
            },
            "results": results,
            "execution_mode": "batch",
            "batch_timestamp": datetime.now().isoformat()
        }
    
    def _create_batch_error_result(self, error_message: str) -> Dict[str, Any]:
        """배치 에러 결과 생성"""
        return {
            "batch_info": {
                "total_queries": 0,
                "successful_queries": 0,
                "failed_queries": 0,
                "success_rate": 0,
                "total_documents": 0,
                "average_documents_per_query": 0,
                "processing_time_seconds": 0,
                "timestamp": datetime.now().isoformat(),
                "error": error_message
            },
            "results": []
        }
    
    def _save_batch_results(self, batch_result: Dict[str, Any]):
        """배치 결과 저장"""
        try:
            output_file = self.file_manager.save_json_results(batch_result)
            logging.info(f"결과가 {output_file}에 저장되었습니다.")
            print(f"   결과 파일: {output_file}")
        except Exception as e:
            logging.error(f"결과 저장 실패: {e}")
    
    def _auto_convert_results(self, batch_result: Dict[str, Any]):
        """결과 자동 변환"""
        try:
            # CSV 변환
            print("\n🔄 자동으로 CSV 변환을 시작합니다...")
            csv_file = self.result_converter.convert_to_csv(batch_result)
            print(f"✅ CSV 변환 완료: {csv_file}")
            
            # JSONL 변환
            print("🔄 자동으로 JSONL 변환을 시작합니다...")
            jsonl_file = self.result_converter.convert_to_jsonl(batch_result)
            print(f"✅ JSONL 변환 완료: {jsonl_file}")
            
        except Exception as e:
            logging.error(f"자동 변환 실패: {e}")
    
    def print_batch_summary(self, batch_result: Dict[str, Any]):
        """배치 결과 요약 출력"""
        batch_info = batch_result.get("batch_statistics", {})
        
        print(f"\n📊 배치 실행 결과:")
        print(f"   총 처리된 질문: {batch_info.get('total_queries', 0)}개")
        print(f"   성공한 질문: {batch_info.get('successful_queries', 0)}개")
        print(f"   실패한 질문: {batch_info.get('failed_queries', 0)}개")
        print(f"   성공률: {batch_info.get('success_rate', 0):.1f}%")
        print(f"   총 찾은 문서: {batch_info.get('total_documents_found', 0)}개")
        print(f"   평균 문서/질문: {batch_info.get('avg_documents_per_query', 0):.1f}개")
