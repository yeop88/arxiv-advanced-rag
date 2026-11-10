"""Advanced RAG — arXiv 학술논문 어시스턴트.

4단계 파이프라인:
  stage1_query_transform : 질의 변환 (multi-query / HyDE / decompose)
  stage2_rerank          : 리랭킹 (cross-encoder)
  stage3_generate        : 생성 (인용 기반 답변)
  stage4_verify          : 검증 (groundedness 체크 + 자기수정)
"""
