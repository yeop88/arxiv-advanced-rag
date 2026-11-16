"""
tests/test_units.py — 순수 함수 단위 테스트 (네트워크/LLM 불필요)

LLM 호출 없이 검증 가능한 핵심 로직들을 빠르게 점검합니다:
  - RRF 융합 순위 계산
  - LLM 출력 파싱 헬퍼(_split_lines, _parse_json)
  - 멀티턴 문맥화의 '히스토리 없음' 단축 경로

실행: python tests/test_units.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document


def test_rrf() -> None:
    """RRF: 두 랭킹에서 모두 상위인 문서가 1등이 되어야 한다."""
    from rag.stages.stage2_rerank import _reciprocal_rank_fusion

    A = Document(page_content="doc A")
    B = Document(page_content="doc B")
    C = Document(page_content="doc C")
    ranking1 = [A, B, C]  # A가 1위
    ranking2 = [A, C, B]  # A가 1위
    fused = _reciprocal_rank_fusion([ranking1, ranking2], k=60)
    top_doc = fused[0][0]
    assert top_doc.page_content == "doc A", "두 랭킹 모두 1위인 A가 종합 1위여야 함"
    # 점수는 내림차순 정렬
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True), "RRF 점수가 내림차순이어야 함"
    print("  test_rrf ✅")


def test_split_lines() -> None:
    """_split_lines: 번호/불릿 접두 제거, 빈 줄 제거."""
    from rag.stages.stage1_query_transform import _split_lines

    out = _split_lines("1. first\n- second\n\n  3) third  ")
    assert out == ["first", "second", "third"], out
    assert _split_lines("3D Gaussian Splatting\n4-bit quantization\n2. 8-bit") == [  # 숫자로 시작하는 검색어 보존
        "3D Gaussian Splatting", "4-bit quantization", "8-bit"]
    print("  test_split_lines ✅")


def test_parse_json() -> None:
    """_parse_json: 코드펜스/주변 텍스트가 있어도 JSON 추출."""
    from rag.stages.stage4_verify import _parse_json

    assert _parse_json('{"grounded": true}')["grounded"] is True
    fenced = '```json\n{"faithfulness": 0.8}\n```'
    assert _parse_json(fenced)["faithfulness"] == 0.8
    noisy = '설명 텍스트 {"a": 1} 뒤에도 말이 있음'
    assert _parse_json(noisy)["a"] == 1
    print("  test_parse_json ✅")


def test_contextualize_no_history() -> None:
    """히스토리가 없으면 LLM 호출 없이 원문 그대로 반환해야 한다."""
    from rag.stages.contextualize import contextualize

    q = "리랭킹은 왜 필요한가요?"
    assert contextualize(q, None) == q
    assert contextualize(q, []) == q
    print("  test_contextualize_no_history ✅")


class _FakeEmbeddings:
    """테스트용 가짜 임베딩: 각 텍스트를 서로 직교하는 one-hot 벡터로(유사도 0)."""

    def embed_documents(self, texts):
        n = len(texts)
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def test_mmr_per_source_cap() -> None:
    """MMR: 논문당 상한으로 한 논문의 쏠림을 막고 커버리지를 넓혀야 한다."""
    import rag.core.llm as llm
    from rag.core.schemas import RetrievedChunk
    from rag.stages.stage2_rerank import _mmr_select

    orig = llm.get_embeddings
    llm.get_embeddings = lambda: _FakeEmbeddings()
    try:
        # P1이 관련도 상위를 독점하는 상황 (점수 내림차순)
        srcs = ["P1", "P1", "P1", "P2", "P3"]
        cands = [
            RetrievedChunk(content=f"c{i}", source=s, title=s, rerank_score=5 - i)
            for i, s in enumerate(srcs)
        ]
        picked = _mmr_select(cands, k=4, lambda_=0.6, same_source_penalty=0.9, max_per_source=2)
        got = [c.source for c in picked]
        assert len(picked) == 4, got
        assert got.count("P1") <= 2, f"P1 상한 2 초과: {got}"
        assert {"P2", "P3"} <= set(got), f"다른 논문 커버 실패: {got}"

        # 후보 논문이 부족하면 상한을 완화해 슬롯을 채운다
        only_p1 = [RetrievedChunk(content=f"c{i}", source="P1", title="P1", rerank_score=3 - i)
                   for i in range(3)]
        picked2 = _mmr_select(only_p1, k=3, lambda_=0.6, same_source_penalty=0.9, max_per_source=2)
        assert len(picked2) == 3, "후보가 한 논문뿐이면 상한을 완화해 k를 채워야 함"
    finally:
        llm.get_embeddings = orig
    print("  test_mmr_per_source_cap ✅")


class _OneHotEmbeddings:
    """질의 'a'·'b'·'c' 를 각 축 방향으로 보내는 가짜 임베딩(검색 순위를 예측 가능하게)."""

    _axis = {"a": 0, "b": 1, "c": 2}

    def embed_query(self, text):
        v = [0.0, 0.0, 0.0]
        v[self._axis[text]] = 1.0
        return v

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


def test_pinecone_store_adapter() -> None:
    """PineconeStore: 검색 본문은 docstore 기준, get/delete 가 Chroma 와 같은 모양으로 동작."""
    import tempfile
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_pinecone import FakePineconeIndex

    from rag.store.docstore import DocStore
    from rag.store.vectorstore import PineconeStore, get_all_documents

    path = Path(tempfile.mkdtemp()) / "docstore.jsonl"
    store = PineconeStore(FakePineconeIndex(), "t", _OneHotEmbeddings(), DocStore(path))
    store.upsert_vectors(
        ["id-a", "id-b"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ["본문 A", "본문 B"], [{"source": "P1", "section": None}, {"source": "P2"}],
    )
    hits = store.similarity_search_with_relevance_scores("a", k=2)
    assert hits[0][0].page_content == "본문 A" and hits[0][0].id == "id-a", hits
    assert hits[0][1] > hits[1][1], "더 가까운 문서의 관련도가 높아야 함"
    assert hits[0][0].metadata == {"source": "P1", "section": None}, "메타데이터는 docstore 원본 그대로"
    assert len(get_all_documents(store)) == 2 and store.count() == 2
    assert DocStore(path).get("id-b")["text"] == "본문 B", "docstore 가 디스크에 저장돼야 함"
    assert store.get(include=["metadatas"])["metadatas"][1] == {"source": "P2"}, "Chroma 식 get(include=...) 호출"

    store.delete(ids=["id-a"])
    assert store.count() == 1 and store.get()["ids"] == ["id-b"], "삭제가 인덱스·docstore 양쪽에 반영"
    ids = store.add_documents([Document(page_content="c", metadata={"source": "P3"})])
    assert store.similarity_search("c", k=1)[0].id == ids[0]
    print("  test_pinecone_store_adapter ✅")


def _db_login_error() -> str | None:
    """설정된 PostgreSQL 에 로그인할 수 없으면 이유(테스트의 DB 부분을 건너뛰려고), 되면 None."""
    from config import settings

    if not settings.database_url:
        return "PostgreSQL 로그인 정보 없음"
    try:
        import psycopg

        psycopg.connect(settings.database_url, connect_timeout=3).close()
        return None
    except Exception as e:  # noqa: BLE001
        return f"DB 로그인 실패: {str(e).splitlines()[-1][:60]}"


def test_postgres_url() -> None:
    """.env 의 POSTGRES_* 항목 → 접속 URL(특수문자는 인코딩), 사용자·DB 가 없으면 빈 값(저장소 끔)."""
    import os

    from config import _postgres_url

    keys = ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        assert _postgres_url() == ""
        os.environ.update(POSTGRES_DB="arxiv_rag", POSTGRES_USER="app", POSTGRES_PASSWORD="p@ss:w/rd", POSTGRES_PORT="5433")
        assert _postgres_url() == "postgresql://app:p%40ss%3Aw%2Frd@localhost:5433/arxiv_rag"
        os.environ.pop("POSTGRES_PASSWORD")
        assert _postgres_url() == "postgresql://app@localhost:5433/arxiv_rag"
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    print("  test_postgres_url ✅")


def test_pgvector_store() -> None:
    """PgVectorStore: Pinecone 어댑터와 같은 검사를 실제 PostgreSQL 임시 테이블로(pgvector 확장이 있을 때만)."""
    from config import settings
    from rag.store.vectorstore import PgVectorStore, _vec_literal, get_all_documents, get_vectors

    assert _vec_literal([1.0, 0.5, -0.25]) == "[1,0.5,-0.25]"
    why = _db_login_error()
    if why:
        print(f"  test_pgvector_store ✅ ({why} — DB 왕복은 건너뜀)")
        return
    import psycopg

    with psycopg.connect(settings.database_url) as c:
        has = c.execute("SELECT count(*) FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
    if not has:
        print("  test_pgvector_store ✅ (pgvector 확장 없음 — DB 왕복은 건너뜀)")
        return
    store = PgVectorStore(settings.database_url, "test_pgvector_units", _OneHotEmbeddings())
    try:
        store.delete_all()
        store.upsert_vectors(["id-a", "id-b"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                             ["본문 A", "본문 B"], [{"source": "P1", "section": None}, {"source": "P2"}])
        hits = store.similarity_search_with_relevance_scores("a", k=2)
        assert hits[0][0].page_content == "본문 A" and hits[0][0].id == "id-a", hits
        assert abs(hits[0][1] - 1.0) < 1e-6 and abs(hits[1][1] - 0.5) < 1e-6, "점수 (cos+1)/2"
        assert hits[0][0].metadata == {"source": "P1", "section": None}
        assert len(get_all_documents(store)) == 2 and store.count() == 2
        assert get_vectors(store, ["id-b"]) == {"id-b": [0.0, 1.0, 0.0]}
        only_meta = store.get(include=["metadatas"])   # Chroma 식 호출(enrich·auto_fetch 가 씀)
        assert only_meta["ids"] == ["id-a", "id-b"] and only_meta["documents"] is None
        assert only_meta["metadatas"][1] == {"source": "P2"}
        store.delete(ids=["id-a"])
        assert store.count() == 1 and store.get()["ids"] == ["id-b"]
        ids = store.add_documents([Document(page_content="c", metadata={"source": "P3"})])
        assert store.similarity_search("c", k=1)[0].id == ids[0]
        store.upsert_vectors(["id-b"], [[0.0, 0.0, 1.0]], ["본문 B2"], [{"source": "P2"}])
        assert store.count() == 2 and "본문 B2" in store.get()["documents"], "같은 ID 는 덮어쓰기"
    finally:
        store.delete_all()
    print("  test_pgvector_store ✅ (DB 왕복 포함)")


def test_exclude_auto_view() -> None:
    """기준 코퍼스 보기: 자동 수집 청크를 숨긴 결과가 '처음부터 그 청크가 없던 스토어'와 같아야 한다."""
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_pinecone import FakePineconeIndex

    from rag.store.docstore import DocStore
    from rag.store.vectorstore import ExcludeOriginStore, PineconeStore, get_all_documents, get_vectors

    def make(with_auto: bool):
        st = PineconeStore(FakePineconeIndex(), "t", _OneHotEmbeddings(),
                           DocStore(Path(tempfile.mkdtemp()) / "d.jsonl"))
        ids, vecs, texts, metas = [], [], [], []
        for i in range(6):  # 수동 수집 청크: 질의 'a' 와 점점 멀어지게
            ids.append(f"m{i}"); vecs.append([1.0, 0.1 * (i + 1), 0.0])
            texts.append(f"수동 {i}"); metas.append({"source": f"P{i}"})
        if with_auto:  # 자동 수집 청크가 질의 'a' 에 가장 가깝다 → 숨기지 않으면 상위를 차지
            for i in range(4):
                ids.append(f"a{i}"); vecs.append([1.0, 0.01 * i, 0.0])
                texts.append(f"자동 {i}"); metas.append({"source": f"A{i}", "origin": "auto"})
        st.upsert_vectors(ids, vecs, texts, metas)
        return st

    full, clean = make(True), make(False)
    view = ExcludeOriginStore(full)
    view.MARGIN = 0  # 여유 없이 → 첫 조회가 자동 청크로 가득 차 '더 가져오기' 경로를 타게 함
    got = [d.id for d in view.similarity_search("a", k=3)]
    want = [d.id for d in clean.similarity_search("a", k=3)]
    assert got == want == ["m0", "m1", "m2"], (got, want)
    assert [d.id for d in full.similarity_search("a", k=1)] == ["a0"], "원본 스토어에는 자동 청크가 그대로 있어야 함"
    assert sorted(d.id for d in get_all_documents(view)) == [f"m{i}" for i in range(6)], "BM25 용 목록에서도 숨김"
    assert set(get_vectors(view, ["m0", "a0"])) == {"m0", "a0"}, "저장 벡터 조회는 실제 스토어로 넘겨야 함"
    assert view.count() == 10, "그 밖의 속성은 감싼 스토어로 넘김"
    print("  test_exclude_auto_view ✅")


def test_auto_rerank_weight() -> None:
    """가중치: 자동 수집 청크 점수만 낮추고, 1.0 이면 순위가 그대로여야 한다."""
    import rag.stages.stage2_rerank as s2
    from config import settings
    from rag.core.schemas import RetrievedChunk

    class _Fake:
        def predict(self, pairs):
            return [0.9, 0.8]

    def cands():
        return [RetrievedChunk(content="자동", source="A", origin="auto"), RetrievedChunk(content="기존", source="B")]

    orig = (s2.get_reranker, settings.rerank_diversity, settings.auto_rerank_weight)
    s2.get_reranker, settings.rerank_diversity = (lambda: _Fake()), False
    try:
        settings.auto_rerank_weight = 1.0
        assert [c.source for c in s2.rerank("q", cands())] == ["A", "B"]
        settings.auto_rerank_weight = 0.5
        got = s2.rerank("q", cands())
        assert [c.source for c in got] == ["B", "A"] and abs(got[1].rerank_score - 0.45) < 1e-9, got
    finally:
        s2.get_reranker, settings.rerank_diversity, settings.auto_rerank_weight = orig
    from rag.ingest.auto_fetch import is_back_matter

    assert is_back_matter("References") and is_back_matter("Appendix A") and is_back_matter("BIBLIOGRAPHY")
    assert not is_back_matter("3.2 Method") and not is_back_matter(None)
    print("  test_auto_rerank_weight ✅")


def test_relation_claim() -> None:
    """관계·비교 단정 문장: 두 근거 이상 인용 또는 관계 표현, "(추론)" 표시면 제외(§5.54)."""
    from rag.stages.citation_check import is_relation_claim

    assert is_relation_claim("두 접근 방식은 역할을 나누어 맡을 수 있습니다 [1]", [1])
    assert is_relation_claim("A는 쿼리를, B는 색인을 다룹니다 [1][2]", [1, 2])
    assert not is_relation_claim("(추론) 두 접근 방식은 결합할 수 있습니다 [1][2]", [1, 2])
    assert not is_relation_claim("LoRA는 사전학습 가중치를 고정합니다 [1]", [1])


def test_original_signal() -> None:
    """3차 신호: 주어인 기법의 원 논문 제목이 근거에 없을 때만(파생 논문 제목에 이름이 있어도)."""
    import rag.ingest.auto_fetch as af
    from rag.core.schemas import RetrievedChunk

    assert af.subject_name("CLIP은 어떻게 학습하나요?") == "CLIP"
    assert af.subject_name("Vision Transformer는 이미지를 어떻게 나누나요?") == "Vision Transformer"
    assert af.subject_name("RAG는 무엇인가요?") is None and af.subject_name("트랜스포머는 왜?") is None
    orig = "Learning Transferable Visual Models From Natural Language Supervision"
    deriv = [RetrievedChunk(content="CLIP CLIP", source="1", title="Long-CLIP: Unlocking the Long-Text Capability of CLIP")]
    q = "CLIP은 어떻게 학습하나요?"
    assert af.original_signal(q, deriv, guess=lambda _: orig, more=lambda _: [])
    assert not af.original_signal(q, deriv + [RetrievedChunk(content="x", source="2", title=orig)],
                                  guess=lambda _: orig, more=lambda _: [])
    # 첫 추정이 틀려도 후보 중 하나가 근거에 있으면 켜지 않음
    assert not af.original_signal(q, [RetrievedChunk(content="x", source="2", title=orig)],
                                  guess=lambda _: "Contrastive Image Text Pretraining Wrong", more=lambda _: [orig])
    assert not af.original_signal(q, deriv, guess=lambda _: None, more=lambda _: [orig])  # 모른다고 하면 끔
    # 이름만 돌려준 추정은 제목이 아님 → 후보로 판단(파생 "Vicinity Vision Transformer" 와 맞다고 보지 않음)
    vq = "Vision Transformer는 어떻게 분류하나요?"
    vit = [RetrievedChunk(content="x", source="3", title="Vicinity Vision Transformer")]
    assert af.original_signal(vq, vit, guess=lambda _: "Vision Transformer",
                              more=lambda _: ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"])
    assert not af.original_signal("트랜스포머는 왜?", deriv, guess=lambda _: orig)
    # 원 논문이 DB 에 있으면(근거 5개 밖일 뿐) 켜지 않음
    assert not af.original_signal(q, deriv, guess=lambda _: orig, more=lambda _: [], known_titles={orig})
    # 같은 주어로 최근 수집했으면 켜지 않음
    import tempfile
    from config import settings
    keep = settings.auto_usage_path
    with tempfile.TemporaryDirectory() as d:
        settings.auto_usage_path = str(Path(d) / "usage.json")
        try:
            assert af.original_signal(q, deriv, guess=lambda _: orig, more=lambda _: [], check_tried=True)
            af.mark_original_tried("CLIP")
            assert not af.original_signal(q, deriv, guess=lambda _: orig, more=lambda _: [], check_tried=True)
        finally:
            settings.auto_usage_path = keep


def test_auto_fetch_decisions() -> None:
    """답변 신호는 대상이 근거에 없을 때만, 선정은 인용이 가장 많은 관련 논문을 먼저."""
    from types import SimpleNamespace as NS

    import rag.ingest.auto_fetch as af
    from rag.core.schemas import RetrievedChunk

    assert af.named_terms("SimCLR와 BM25, RAG 및 Neural 모델") == ["SimCLR", "BM25"]
    no = "제공된 논문에서는 찾을 수 없습니다."
    ev = [RetrievedChunk(content="SPLADE and BM25 compared", source="1", title="SPLADE paper")]
    assert af.answer_signal("SimCLR는 무엇인가요?", no, ev), "명칭이 제목에 없으면 수집"
    assert not af.answer_signal("SPLADE는 무엇인가요?", no, ev), "제목에 있으면 수집 안 함"
    assert af.answer_signal("SPLADE와 BM25 차이는?", no, ev), "제목에 없고 한 청크에서만 언급되면 수집(이슈 18)"
    ev2 = ev + [RetrievedChunk(content="BM25 baseline", source="2", title="Other paper")]
    assert not af.answer_signal("SPLADE와 BM25 차이는?", no, ev2), "두 청크 이상에서 언급되면 수집 안 함"
    assert af.answer_signal("SPLADE와 RETRO 차이는?", no, ev), "명칭 하나라도 본문에 없으면 수집"
    ev3 = [RetrievedChunk(content="x", source="3", title="From Local to Global: A Graph RAG Approach"),
           RetrievedChunk(content="x", source="4", title="Tips and Tricks for Visual Question Answering")]
    assert not af.answer_signal("GraphRAG의 장점은?", no, ev3), "띄어쓰기가 달라도 제목에 있으면 수집 안 함"
    assert not af.answer_signal("VQA 기법은?", no, ev3), "약어가 제목 머리글자와 맞으면 수집 안 함"
    assert af.answer_signal("SAM 구조는?", no, ev3), "머리글자가 맞지 않으면 수집"
    hyde = [RetrievedChunk(content="We propose HyDE. HyDE generates a hypothetical document.", source="5",
                           title="Precise Zero-Shot Dense Retrieval without Relevance Labels")]
    assert not af.answer_signal("HyDE는 무엇인가요?", no, hyde), "제목에 없어도 한 청크에 여러 번 나오면 원 논문으로 봄"
    assert af._word_count("RETRO", "retrocausal RETRO, Retro") == 2 and af._word_count("FiD", "fidelity") == 0
    assert not af._is_acronym("RL") and af._is_acronym("VQA") and not af._is_acronym("XLNet"), "약어는 대문자 3~6자"
    assert not af.answer_signal("SimCLR는 무엇인가요?", "답입니다 [1].", ev), "답변했으면 수집 안 함"
    assert not af.answer_signal("맛집을 추천해 주세요", no, ev), "명칭이 없으면 수집 안 함"

    a, b, c, d = (NS(cited_by=n) for n in (5, 900, 50, 99999))
    ranked = [(a, 0.99), (c, 0.9), (b, 0.6), (d, 0.4)]  # d 는 인용이 많아도 관련도 미달이라 우선하지 않음
    assert [x[0] for x in af.select_papers(ranked)] == [b, a, c]
    t = NS(cited_by=1, title_match=True)
    ranked2 = [(a, 0.99), (c, 0.9), (b, 0.6), (t, 0.35)]  # 추정 제목과 맞는 후보는 관련도 기준만 넘으면 맨 앞
    from config import settings as _cfg
    _only = _cfg.auto_fetch_title_only
    _cfg.auto_fetch_title_only = False
    assert [x[0] for x in af.select_papers(ranked2)] == [t, b, a]
    _cfg.auto_fetch_title_only = True
    assert [x[0] for x in af.select_papers(ranked2)] == [t], "원 논문을 찾으면 그 1편만"
    assert [x[0] for x in af.select_papers(ranked)] == [b, a, c], "제목이 맞는 후보가 없으면 기존대로"
    low = NS(cited_by=1, title_match=1.0)  # 제목은 맞지만 관련도 미달(추정이 틀렸을 수 있음) → 다른 후보도 함께
    assert [x[0] for x in af.select_papers([(a, 0.99), (low, 0.0001)])] == [low, a]
    _cfg.auto_fetch_title_only = _only
    assert af.select_papers([(NS(cited_by=1, title_match=0.7), 0.1)]) == [], "제목이 비슷한 정도면 관련도 기준 적용"
    exact_low = NS(cited_by=1, title_match=1.0)
    assert af.select_papers([(exact_low, 0.01)])[0][0] is exact_low, "제목이 거의 정확히 맞으면 관련도 기준 면제"
    assert af.title_similar("Generative Adversarial Nets", "Generative Adversarial Networks"), "Nets/Networks 차이 허용"
    near, exact = NS(cited_by=1, title_match=0.8), NS(cited_by=1, title_match=1.0)
    assert af.select_papers([(near, 0.9), (exact, 0.5)])[0][0] is exact, "제목이 더 정확히 맞는 후보가 먼저"
    assert af.title_similar("Proximal Policy Optimization Algorithms", "Proximal policy optimization algorithms.")
    assert af.title_similar("Deep Residual Learning for Image Recognition", "Deep residual learning for image recognition")
    assert not af.title_similar("Proximal Policy Optimization Algorithms", "Policy Optimization for Robots with PPO")
    assert not af.title_similar("Vision Transformer", "Vision Transformer with Deformable Attention"), "짧은 추정은 파생 논문과 안 맞음"
    from config import settings as _s
    from rag.ingest.arxiv_source import openalex_params
    _k, _s.openalex_api_key = _s.openalex_api_key, ""
    assert openalex_params({"a": 1}) == {"a": 1}
    _s.openalex_api_key = "k"
    assert openalex_params({"a": 1}) == {"a": 1, "api_key": "k"}
    _s.openalex_api_key = _k
    print("  test_auto_fetch_decisions ✅")


def test_evict_over_budget() -> None:
    """예산 초과 시 오래 안 쓰인 자동 수집 논문부터 지우고, 수동·방금 수집한 논문은 남겨야 한다."""
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_pinecone import FakePineconeIndex

    import rag.ingest.auto_fetch as af
    from config import settings
    from rag.store.docstore import DocStore
    from rag.core.schemas import RetrievedChunk
    from rag.store.vectorstore import PineconeStore

    tmp = Path(tempfile.mkdtemp())
    st = PineconeStore(FakePineconeIndex(), "t", _OneHotEmbeddings(), DocStore(tmp / "d.jsonl"))
    rows = [("m", None, "2020-01-01"), ("vis", "auto", "2025-01-01"), ("old", "auto", "2026-01-01"),
            ("used", "auto", "2026-01-01"), ("new", "auto", "2026-11-11")]
    for src, origin, date in rows:
        meta = {"source": src, "fetched_at": date, **({"origin": origin} if origin else {})}
        st.upsert_vectors([f"{src}{i}" for i in range(2)], [[1.0, 0.0, 0.0]] * 2, [f"{src} 본문"] * 2, [meta] * 2)
    # 가장 오래된 자동 수집 논문이지만 표 청크가 있어 지우면 안 됨
    st.upsert_vectors(["vis_t"], [[1.0, 0.0, 0.0]], ["Table 1: x"], [{"source": "vis", "fetched_at": "2025-01-01",
                                                                      "origin": "auto", "kind": "table"}])
    orig = (settings.auto_usage_path, settings.auto_fetch_max_chunks)
    settings.auto_usage_path, settings.auto_fetch_max_chunks = str(tmp / "u.json"), 7
    try:
        af.record_usage([RetrievedChunk(content="x", source="used", origin="auto"),
                         RetrievedChunk(content="y", source="m")])
        assert list(af.load_usage()) == ["used"], "자동 수집 논문만 기록"
        evicted = af.evict_over_budget(st, protect={"new"})
        assert [e["source"] for e in evicted] == ["old"], evicted
        left = {m["source"] for m in st.get()["metadatas"]}
        assert left == {"m", "vis", "used", "new"}, f"표·그림·수식이 있는 논문은 남아야 함: {left}"
        assert af.evict_over_budget(st, protect={"new"}) == [], "예산 안이면 지우지 않음"
        # 오늘 넣은·오늘 쓰인 논문은 다음 수집 때 지우지 않음 — 지울 게 없으면 예산을 잠시 넘김(§5.51)
        today = af.dt.date.today().isoformat()
        st.upsert_vectors(["today0", "today1"], [[1.0, 0.0, 0.0]] * 2, ["t 본문"] * 2,
                          [{"source": "today", "fetched_at": today, "origin": "auto"}] * 2)
        settings.auto_fetch_max_chunks = 6
        evicted = af.evict_over_budget(st, protect={"newest"})
        assert [e["source"] for e in evicted] == ["new"], evicted
        left = {m["source"] for m in st.get()["metadatas"]}
        assert {"vis", "used", "today"} <= left, f"표·그림·오늘 논문은 남아야 함: {left}"
    finally:
        settings.auto_usage_path, settings.auto_fetch_max_chunks = orig
    print("  test_evict_over_budget ✅")


def test_paper_search_parsers() -> None:
    """arXiv Atom 응답 해석(직접 호출 경로)과 OpenAlex 초록 역색인 복원."""
    import rag.ingest.arxiv_source as A

    atom = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
      <id>http://arxiv.org/abs/2303.04137v5</id><published>2023-03-07T17:00:00Z</published>
      <title>Diffusion Policy:
        Visuomotor Policy Learning</title><summary>  We introduce   Diffusion Policy. </summary>
      <author><name>Cheng Chi</name></author><author><name>Shuran Song</name></author></entry></feed>"""

    class _Resp:
        status_code = 200
        content = atom

        def raise_for_status(self):
            pass

    class _Sess:
        def get(self, *a, **k):
            return _Resp()

    orig = A.http_session
    A.http_session = lambda: _Sess()
    try:
        papers = A.search_arxiv("diffusion policy", 3)
    finally:
        A.http_session = orig
    p = papers[0]
    assert (p.pid, p.title, p.summary) == ("2303.04137", "Diffusion Policy: Visuomotor Policy Learning",
                                           "We introduce Diffusion Policy."), p
    assert p.author_names == ["Cheng Chi", "Shuran Song"] and str(p.published) == "2023-03-07"
    assert A.paper_meta(p)["url"] == "https://arxiv.org/abs/2303.04137" and p.pdf_url.endswith("/pdf/2303.04137")
    assert A._abstract_from_inverted({"networks": [1], "KAN": [0], "are": [2]}) == "KAN networks are"
    print("  test_paper_search_parsers ✅")


def test_citation_graph() -> None:
    """인용 그래프: 코퍼스 안 선행·후속, 공통 참고문헌, arXiv ID 판별(네트워크 없이 임시 파일로)."""
    import tempfile

    from config import settings
    from rag.features import citations as C

    orig = settings.citation_graph_path
    with tempfile.TemporaryDirectory() as d:
        settings.citation_graph_path = str(Path(d) / "g.json")
        try:
            g = C._empty()
            g["papers"] = {
                "1706.03762": {"oa": "W1", "title": "Attention", "cited_by": 900, "year": 2017, "arxiv": "1706.03762", "refs": ["W9"]},
                "1810.04805": {"oa": "W2", "title": "BERT", "cited_by": 500, "year": 2018, "arxiv": "1810.04805", "refs": ["W1", "W9"]},
                "1907.11692": {"oa": "W3", "title": "RoBERTa", "cited_by": 100, "year": 2019, "arxiv": "1907.11692", "refs": ["W1", "W2", "W9"]},
            }
            C.save(g)
            rel = C.related_in_corpus("1810.04805")
            assert rel == {"prior": ["1706.03762"], "later": ["1907.11692"]}, rel
            assert C.related_in_corpus("9999.99999") == {"prior": [], "later": []}
            assert sorted(C.common_references(["1810.04805", "1907.11692"])) == [("W1", 2), ("W9", 2)]
        finally:
            settings.citation_graph_path = orig
    assert C.is_arxiv_id("2103.00020") and C.is_arxiv_id("math/0211159") and C.is_arxiv_id("quant-ph/9508027")
    assert not C.is_arxiv_id("local_ko_001")
    assert C._arxiv_of({"doi": "https://doi.org/10.48550/arXiv.2103.00020"}) == "2103.00020"
    assert C._arxiv_of({"doi": "x", "locations": [{"landing_page_url": "https://arxiv.org/abs/1512.03385v1"}]}) == "1512.03385"
    assert C._same_title("Attention Is All You Need", "attention is all you need.")
    print("  test_citation_graph ✅")


def test_compare_table() -> None:
    """비교표: 칸별 인용 점검, 표 형식, 논문별 전역 번호(LLM 없이 요약 함수를 바꿔 끼움)."""
    from rag.features import compare_table as T
    from rag.core.schemas import RetrievedChunk

    assert T.check_citations({"방법": "A를 제안 [2]", "데이터": "근거 없음", "결과": "좋음", "한계": "범위 밖 [9]"},
                             range(1, 4)) == ["결과", "한계"]
    chunks = {"a": [RetrievedChunk(content="m", source="a", title="Paper A", section="Method")],
              "b": [RetrievedChunk(content="r", source="b", title="Paper B | x")]}
    orig_pc, orig_sum = T._paper_chunks, T._summarize
    seen = []
    T._paper_chunks = lambda vs, srcs: chunks
    T._summarize = lambda title, ctx: (seen.append(ctx), {"방법": f"{title} 방법 [{len(seen)}]", "데이터": "근거 없음",
                                                          "결과": "수치 없음", "한계": "근거 없음"})[1]
    try:
        t = T.build_table(["a", "b", "a"], vectorstore=None, verify=False)
    finally:
        T._paper_chunks, T._summarize = orig_pc, orig_sum
    assert [r.source for r in t.rows] == ["a", "b"] and len(t.chunks) == 2
    assert seen[0].startswith("[1]") and seen[1].startswith("[2]"), "논문별 근거 번호가 이어져야 함"
    assert t.rows[0].uncited == ["결과"] and t.cells_uncited == 2 and t.cells_filled == 4
    md = t.to_markdown()
    assert md.splitlines()[0] == "| 논문 | 방법 | 데이터 | 결과 | 한계 |" and "Paper B \| x" in md
    T.flag_cells(t.rows, ["Paper A 방법 [1] 은 틀림", "전혀 다른 주장"])
    assert t.rows[0].flagged == ["방법"] and t.rows[1].flagged == []
    assert "⚠ Paper A 방법" in t.to_markdown(marks=True) and "⚠" not in t.to_markdown()
    print("  test_compare_table ✅")


def test_survey() -> None:
    """문헌 조사 초안: 근거 머리의 연도·선행 관계, 참고문헌은 코드가 생성, 검색 폭 설정 복원(LLM 없이)."""
    from config import settings
    from rag.features import survey as S
    from rag.core.schemas import RetrievedChunk

    g = {"papers": {"A": {"oa": "W1", "title": "A", "cited_by": 9, "year": 2017, "refs": []},
                    "B": {"oa": "W2", "title": "B", "cited_by": 1, "year": 2019, "refs": ["W1"]}},
         "works": {}, "later": {}, "missing": []}
    chunks = [RetrievedChunk(content="a", source="A", title="Paper A"), RetrievedChunk(content="b", source="B", title="Paper B")]
    ctx = S.format_context(chunks, g)
    assert "[1] Paper A, 2017\n" in ctx and "[2] Paper B, 2019 · 인용: [1]" in ctx, ctx
    assert S.year_of("2103.00020", g) == 2021 and S.year_of("math/0211159", g) == 2002
    assert S.year_of("quant-ph/9508027", g) == 1995 and S.year_of("A", g) == 2017 and S.year_of("x", g) is None
    keep = (settings.top_k_rerank, settings.rerank_max_per_source, settings.top_k_retrieve)
    orig_gather, orig_write = S.gather, S._write
    S.gather = lambda topic, vs: chunks
    S._write = lambda topic, ctx: "## 개요\nA가 먼저 나왔다 [1]."
    try:
        sv = S.build_survey("주제", None, verify=False)
    finally:
        S.gather, S._write = orig_gather, orig_write
    assert sv.cited == {1} and "(본문 미인용)" in sv.references_md().splitlines()[1]
    assert sv.to_markdown().startswith("# 문헌 조사 초안: 주제") and "## 참고문헌" in sv.to_markdown()
    with S._wide_retrieval():
        assert settings.top_k_rerank == S.SURVEY_CHUNKS and settings.rerank_max_per_source == 1
    assert (settings.top_k_rerank, settings.rerank_max_per_source, settings.top_k_retrieve) == keep
    print("  test_survey ✅")


def test_bm25_tokenize() -> None:
    """BM25 단어 나누기: 조사·문장부호가 붙은 영어 용어를 원래 용어로."""
    from config import settings
    from rag.stages.stage2_rerank import bm25_tokenize

    keep = settings.bm25_tokenizer
    try:
        settings.bm25_tokenizer = "regex"
        toks = bm25_tokenize("HyDE의 가설 문서는 RETRO, GPT-3와 다른가요?")
        assert {"hyde", "retro", "gpt", "3", "가설", "문서"} <= set(toks) and "의" not in toks and "와" not in toks, toks
        settings.bm25_tokenizer = "whitespace"
        assert bm25_tokenize("HyDE의 RETRO,") == ["HyDE의", "RETRO,"]
    finally:
        settings.bm25_tokenizer = keep
    print("  test_bm25_tokenize ✅")


def test_watch() -> None:
    """관심 분야 알림: 기간·이미 본 논문·DB 논문 제외, 관련도 기준, 두 번째 실행은 새 논문만(네트워크·LLM 없이)."""
    import datetime as dt
    import tempfile

    import rag.ingest.arxiv_source as A
    from config import settings
    from rag.features import watch as W

    today = dt.date(2026, 11, 12)
    papers = [A.Paper(pid="n1", title="RAG new", summary="rag", published=dt.date(2026, 11, 11)),
              A.Paper(pid="old", title="RAG old", summary="rag", published=dt.date(2026, 10, 1)),
              A.Paper(pid="indb", title="RAG db", summary="rag", published=dt.date(2026, 11, 11)),
              A.Paper(pid="off", title="Astro", summary="stars", published=dt.date(2026, 11, 11))]
    keep = (settings.watch_path, settings.watch_digest_dir)
    orig = (A.search_arxiv, W._summarize)
    calls = []
    with tempfile.TemporaryDirectory() as d:
        settings.watch_path, settings.watch_digest_dir = f"{d}/w.json", f"{d}/dg"
        A.search_arxiv = lambda q, n, **kw: (calls.append((q, kw.get("sort"))), papers)[1]
        W._summarize = lambda topic, title, abstract: f"요약:{title}"
        import rag.core.llm as L
        orig_rr = L.get_reranker
        L.get_reranker = lambda: type("R", (), {"predict": staticmethod(lambda pairs: [0.9 if "rag" in t.lower() else 0.0 for _, t in pairs])})()
        try:
            W.save({"topics": [{"topic": "RAG", "query": "retrieval augmented", "last_checked": None, "seen": []}]})
            digest = W.check_topic(W.load()["topics"][0], {"indb"}, 7, today)
            assert [i.pid for i in digest.items] == ["n1"] and digest.checked == 2, digest
            assert calls[0] == ("retrieval AND all:augmented", "submittedDate")
            md1, _ = W.run(None, today=today)
            assert "요약:RAG new" in md1 and "RAG old" not in md1 and "Astro" not in md1
            md, path = W.run(None, today=today)
            assert path.exists() and "새로 실을 만한 논문이 없습니다" in md, "두 번째 실행: 이미 본 논문은 다시 싣지 않음"
            assert W.latest_digest()[0] == "2026-11-12"
        finally:
            A.search_arxiv, W._summarize = orig
            L.get_reranker = orig_rr
            settings.watch_path, settings.watch_digest_dir = keep
    print("  test_watch ✅")


def test_visual_parse() -> None:
    """표·그림: 모델 응답 해석, 숫자 대조, 페이지 캡션에 없는 항목 버리기(모델 호출 없이)."""
    from rag.ingest import visual as V

    raw = chr(10).join(["### Table 2: Top-20 accuracy", "```markdown", "| Model | NQ |", "|---|---|", "| DPR | 78.4 |",
                   "| BM25 | 59.1 |", "```", "### Figure 1: Overview", "그림은 두 인코더를 보여 준다.", "### Table 9: 지어낸 표",
                   "| a | 12.5 |"])
    items = V.parse(raw, page=5)
    assert [(i.kind, i.label) for i in items] == [("table", "Table 2"), ("figure", "Figure 1"), ("table", "Table 9")]
    assert "```" not in items[0].body and items[0].text.startswith("Table 2: Top-20 accuracy")
    page = "Table 2: Top-20 accuracy ... DPR 78.4 79.4 BM25 59.1"
    assert V.number_match(items[0].body, page) == 1.0 and V.number_match("| x | 99.9 | 78.4 |", page) == 0.5
    assert [m.group(2) for m in V.CAPTION.finditer("Table 2: a" + chr(10) + "see Table 3. here" + chr(10) + "Figure1: b")] == ["2", "1"]

    from config import settings

    orig, orig_provider = V._ask, settings.llm_provider
    V._ask = lambda png, caps: raw
    settings.llm_provider = "local"   # 실제 모델은 부르지 않음(_ask 를 가짜로) — .env 와 무관하게
    try:
        import io

        import pdfplumber
        from pypdf import PdfWriter

        buf = io.BytesIO()
        w = PdfWriter()
        w.add_blank_page(200, 200)
        w.write(buf)
        orig_cp = V.page_labels
        V.page_labels = lambda pdf, kinds: [(1, ["Table 2", "Figure 1"], page)]
        got = V.extract(buf.getvalue(), kinds=("table", "figure"))
        V.page_labels = orig_cp
    finally:
        V._ask, settings.llm_provider = orig, orig_provider
    assert [i.label for i in got] == ["Table 2", "Figure 1"], "페이지 캡션에 없는 Table 9 는 버림"

    # 수식: 번호 붙은 수식 줄 찾기(본문 문장 끝 "(3)" 은 제외), LaTeX 검사
    eq_page = "C(r)= T(t)σ(r(t))c(r(t),d)dt, where T(t)=exp − σ(r(s))ds . (1)" + chr(10) + "as described in Section 2 (3)"
    assert V.equation_labels(eq_page) == ["Equation 1"]
    eq = V.parse("### Equation 1: expected color" + chr(10) + r"$$C(r)=\int T(t)\sigma(r(t))\,dt$$" + chr(10) + "C is the color.", 5)[0]
    assert (eq.kind, eq.label) == ("equation", "Equation 1")
    assert V.check_equation(eq.body, eq_page) == 1.0
    assert V.check_equation(r"$$\alpha + \beta$$", eq_page) is None, "관계 기호 없음"
    assert V.check_equation(r"$$y = \alpha \beta$$", eq_page) is None, "그리스 문자가 페이지에 없음"
    assert V.greek_match(r"\sigma", "𝜎(x)") == 1.0, "수학 이탤릭은 NFKC 로 맞춘다"
    print("  test_visual_parse ✅")


def test_visual_retrieval() -> None:
    """표·그림 청크: 본문 후보와 따로 나누기, 최소 점수 미만은 근거에서 빼기, 논문당 상한은 본문과 따로 세기."""
    import rag.core.llm as llm
    from langchain_core.documents import Document

    from config import settings
    from rag.core.schemas import RetrievedChunk
    from rag.stages.stage2_rerank import _mmr_select, _split_visual

    docs = [Document(page_content=f"d{i}", metadata={"kind": "table"} if i in (1, 3) else {}) for i in range(6)]
    body, visual = _split_visual(docs, 3)
    assert [d.page_content for d in body] == ["d0", "d2", "d4"] and [d.page_content for d in visual] == ["d1", "d3"]

    orig = llm.get_embeddings
    llm.get_embeddings = lambda: _FakeEmbeddings()
    try:
        cands = [RetrievedChunk(content="b1", source="P1", rerank_score=0.99),
                 RetrievedChunk(content="b2", source="P1", rerank_score=0.98),
                 RetrievedChunk(content="t1", source="P1", kind="table", rerank_score=0.97),
                 RetrievedChunk(content="o1", source="P2", rerank_score=0.5)]
        picked = _mmr_select(cands, k=4, lambda_=0.6, same_source_penalty=0.9, max_per_source=2)
        assert "t1" in [c.content for c in picked], "본문 2개가 뽑힌 논문의 표도 들어와야 함"
    finally:
        llm.get_embeddings = orig

    import rag.stages.stage2_rerank as s2

    class _R:
        def predict(self, pairs):
            return [0.9 if "body" in c else 0.05 for _, c in pairs]

    orig_r = s2.get_reranker
    s2.get_reranker = lambda: _R()
    saved = settings.rerank_diversity
    settings.rerank_diversity = False
    try:
        out = s2.rerank("q", [RetrievedChunk(content="body", source="A"),
                              RetrievedChunk(content="tab", source="B", kind="table")])
        assert [c.content for c in out] == ["body"], f"점수 {settings.visual_min_score} 미만 표는 빠져야 함"
    finally:
        s2.get_reranker = orig_r
        settings.rerank_diversity = saved
    from scripts.enrich_visuals import titled

    once = titled("Mamba", "Equation 4: discretization")
    assert once == "[Mamba] Equation 4: discretization" and titled("Mamba", once) == once, "제목은 한 번만 붙는다"
    assert titled("", "Table 1: x") == "Table 1: x"
    print("  test_visual_retrieval ✅")



def test_local_reasoning_stream() -> None:
    """로컬 서버의 delta.reasoning_content(thinking 추론)를 메시지 조각에 보존하고, 생성 단계가 본문과 따로 넘기는가."""
    from langchain_core.messages import AIMessageChunk
    from langchain_core.prompts import ChatPromptTemplate

    from rag.core.llm import _local_chat_class
    from rag.stages import stage3_generate as s3

    llm = _local_chat_class()(model="m", base_url="http://localhost:1/v1", api_key="x")
    gen = llm._convert_chunk_to_generation_chunk(
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": None, "reasoning_content": "생각"}}]},
        AIMessageChunk, None)
    assert gen.message.additional_kwargs.get("reasoning_content") == "생각" and gen.message.content == ""
    plain = llm._convert_chunk_to_generation_chunk({"choices": [{"index": 0, "delta": {"content": "답"}}]}, AIMessageChunk, None)
    assert "reasoning_content" not in plain.message.additional_kwargs and plain.message.content == "답"

    from langchain_core.runnables import RunnableGenerator

    def fake_stream(inputs):  # get_llm(raw=True) 대역: 추론 조각 2개 → 본문 조각 2개
        for _ in inputs:
            pass
        yield AIMessageChunk(content="", additional_kwargs={"reasoning_content": "음"})
        yield AIMessageChunk(content="", additional_kwargs={"reasoning_content": "…"})
        yield AIMessageChunk(content="답은 ")
        yield AIMessageChunk(content="4 [1].")

    orig = s3.get_llm
    s3.get_llm = lambda light=False, raw=False: RunnableGenerator(fake_stream)
    try:
        tokens, thoughts = [], []
        out = s3._run(ChatPromptTemplate.from_messages([("user", "{q}")]), {"q": "2+2"}, tokens.append, thoughts.append)
    finally:
        s3.get_llm = orig
    assert out == "답은 4 [1]." and thoughts == ["음", "…"], (out, thoughts)
    assert tokens[-1] == "답은 4 [1].", "on_token 은 누적 본문(추론 제외)"
    print("  test_local_reasoning_stream ✅")


def test_rerank_title_prefix() -> None:
    """리랭커 입력: 본문 청크 앞에 논문 제목, 이미 제목으로 시작하는 청크(표·그림)와 제목 없는 청크는 그대로."""
    from config import settings
    from rag.core.schemas import RetrievedChunk
    from rag.stages.stage2_rerank import rerank_input

    body = RetrievedChunk(content="We propose a dual encoder.", title="Dense Passage Retrieval")
    table = RetrievedChunk(content="Dense Passage Retrieval\nTable 2: ...", title="Dense Passage Retrieval", kind="table")
    assert rerank_input(body) == "Dense Passage Retrieval\nWe propose a dual encoder."
    assert rerank_input(table) == table.content
    assert rerank_input(RetrievedChunk(content="x")) == "x"
    keep = settings.rerank_title_prefix
    settings.rerank_title_prefix = False
    try:
        assert rerank_input(body) == body.content, "끄면 그대로"
    finally:
        settings.rerank_title_prefix = keep
    print("  test_rerank_title_prefix ✅")


def test_followup_helpers() -> None:
    """보충 검색: 여러 대상 중 근거에 없는 이름 찾기, 그 이름에 맞춘 검색어(다른 이름은 지움)."""
    from rag.core.schemas import RetrievedChunk
    from rag.stages.stage2_rerank import followup_query, missing_names

    q = "ColBERT와 DPR은 정확도-지연 트레이드오프가 어떻게 다른가요?"
    ev = [RetrievedChunk(content="late interaction ...", title="ColBERT: Efficient and Effective Passage Search")]
    assert missing_names(q, ev) == ["DPR"]
    assert missing_names(q, ev + [RetrievedChunk(content="DPR uses dual encoders. DPR is trained ...", title="x")]) == []
    assert missing_names("ColBERT는 어떻게 동작하나요?", []) == [], "대상이 하나면 보충하지 않음"
    fq = followup_query(q, "DPR")
    assert "DPR" in fq and "ColBERT" not in fq
    print("  test_followup_helpers ✅")


def test_citation_check() -> None:
    """4b 문장 단위 인용 확인: 문장 분리·인용 번호 해석·문장 상태(가짜 점수기)."""
    from rag.core.schemas import RetrievedChunk
    from rag.stages import citation_check as cc

    assert cc.parse_cites("A [1]. B [2, 3] C [4-6] D [근거 7]") == [1, 2, 3, 4, 5, 6, 7]
    assert not cc._is_claim("입력 데이터와 특징 표현 방식의 차이"), "소제목 줄은 주장 아님"
    assert not cc._is_claim("제공된 근거만을 바탕으로 답변드립니다")
    assert cc._is_claim("두 접근 모두 CNN의 효율성을 개선한다는 공통점이 있지만 목표가 다릅니다")
    answer = "\n".join([
        "## 요약",
        "DPR 은 질문과 문단을 따로 인코딩합니다 [1]. ColBERT 는 토큰 단위로 늦게 상호작용합니다 [1].",
        "- BM25 는 정확한 단어 일치에 강한 희소 검색입니다 [3].",
        "두 방식은 입력을 다루는 방식이 서로 다릅니다.",
        "| 표 | 행 |",
        "```", "code [1]", "```",
        "제공된 논문에서는 지연 시간 수치를 찾을 수 없습니다.",
        "검색 결과는 다음과 같습니다:",
        "없는 번호를 인용한 문장도 있습니다 [9].",
    ])
    sents = cc.split_sentences(answer)
    assert not any(s.startswith("|") or "code" in s for s in sents), "표·코드는 문장 아님"
    chunks = [RetrievedChunk(content="DPR dual encoder"), RetrievedChunk(content="ColBERT late interaction"),
              RetrievedChunk(content="BM25 lexical")]

    def scorer(pairs):   # 문장 핵심어가 근거에 있으면 높은 점수
        keys = {"DPR": "DPR", "ColBERT": "ColBERT", "BM25": "BM25"}
        return [0.9 if any(k in q and v in d for k, v in keys.items()) else 0.01 for q, d in pairs]

    rep = cc.check(answer, chunks, scorer=scorer, threshold=0.05, nli=False)
    status = {s.text.split()[0]: s.status for s in rep.sentences}
    assert status == {"DPR": "supported", "ColBERT": "miscited", "BM25": "supported",
                      "두": "summary", "없는": "bad_ref"}, status   # 인용 없는 짧은 종합 문장은 경보 아님(§5.47)
    colbert = next(s for s in rep.sentences if s.text.startswith("ColBERT"))
    assert colbert.best == 2, "더 맞는 근거 번호를 제안"
    summ = rep.summary()
    assert summ["claims"] == 5 and summ["cited"] == 4 and summ["supported"] == 2 and summ["precision"] == 0.5, summ
    j = rep.to_json()
    assert len(j["flagged"]) == 2 and j["threshold"] == 0.05
    assert cc.check("", chunks, scorer=scorer, nli=False).summary()["claims"] == 0

    # 인용 이어받기: 인용 없는 주제문이라도 바로 뒤 문장이 인용한 근거가 뒷받침하면 inherited(§5.47)
    irep = cc.check("이 논문은 DPR 이라는 밀집 검색기를 새로 제안합니다. DPR 은 질문과 문단을 따로 인코딩합니다 [1]. "
                    "ColBERT 라는 다른 모델도 따로 존재한다고 알려져 있습니다.", chunks, scorer=scorer, threshold=0.05, nli=False)
    assert [x.status for x in irep.sentences] == ["inherited", "supported", "uncited"], irep.sentences
    assert irep.summary()["recall"] == round(2 / 3, 3)
    # '근거가 없다'를 다르게 말한 문장·수식 줄은 주장이 아님
    assert not cc._is_claim("REALM의 비용 정보는 제공된 근거에 없습니다 [근거 없음]")
    assert not cc._is_claim("두 접근의 결합 효과를 입증할 근거가 부족합니다")
    assert not cc._is_claim("$$ \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \Delta B $$ 로 계산됩니다")

    # 숫자 대조: 근거에 없는 숫자, 다른 근거에 있는 숫자, 표기 차이(%·쉼표·million/만), 무시할 숫자
    assert cc.answer_numbers("Top-20 정확도 78.40%, GPT-4, 2020년, 3개, 1,024 토큰, 100만 시간, $x_{10}$") == \
        ["78.4", "1024", "1000000"]
    nums = cc.chunk_numbers("accuracy 0.784 with 1,024 tokens and 1 million hours")
    assert {"78.4", "1024", "1000000"} <= nums
    assert "10" in cc.chunk_numbers("impose an order-of-magnitude larger space footprint"), "영어 낱말 숫자"
    assert "10" not in cc.chunk_numbers("orders are placed"), "낱말 일부와 헷갈리지 않게"
    tchunks = [RetrievedChunk(content="Table 2: DPR top-20 accuracy 78.4, BM25 59.1"),
               RetrievedChunk(content="FlashAttention trains GPT-2 in 2.7 days, 3.5x faster")]
    ans = ("DPR 의 상위 20개 정확도는 78.4%입니다 [1]. BM25 는 68.1%입니다 [1]. "
           "학습은 2.7일 걸리고 3.5배 빠릅니다 [1]. 질문에 나온 상위 20개는 문제 삼지 않습니다 [2].")
    rep = cc.check(ans, tchunks, scorer=lambda pairs: [0.9] * len(pairs), threshold=0.05, question="상위 20개 정확도는?", nli=False)
    miss = {s.text.split()[0]: (s.numbers_missing, s.numbers_found_in) for s in rep.sentences}
    assert miss["DPR"] == ([], []) and miss["질문에"] == ([], []), miss
    assert miss["BM25"] == (["68.1"], []), "근거에 없는 숫자"
    assert miss["학습은"] == (["2.7", "3.5"], [2]), "다른 근거에 있는 숫자 → 그 번호 제안"
    s = rep.summary()
    assert s["numeric"] == 3 and s["number_mismatch"] == 2 and s["supported"] == 4, s
    assert len(rep.to_json()["flagged"]) == 2, "점수는 맞아도 숫자가 틀리면 확인 목록에"

    # 표 행: 머리글·구분선 빼고 데이터 행의 숫자만 대조(주장 문장 수에는 안 셈), 인용이 없으면 모든 근거와
    table_ans = "\n".join(["| 모델 | Top-20 |", "|---|---:|", "| DPR | 78.4 [1] |", "| BM25 | 61.0 [1] |",
                           "| GPT-2 학습 | 2.7일 |", "", "DPR 이 더 높습니다 [1]."])
    assert cc.table_rows(table_ans) == ["DPR | 78.4 [1]", "BM25 | 61.0 [1]", "GPT-2 학습 | 2.7일"]
    trep = cc.check(table_ans, tchunks, scorer=lambda pairs: [0.9] * len(pairs), threshold=0.05, nli=False)
    rows = {r.text.split()[0]: r.numbers_missing for r in trep.table_rows}
    assert rows == {"DPR": [], "BM25": ["61"], "GPT-2": []}, rows
    ts = trep.summary()
    assert ts["claims"] == 1 and ts["table_rows"] == 3 and ts["number_mismatch"] == 1, ts
    assert [f["status"] for f in trep.to_json()["flagged"]] == ["table_row"]

    # 함의(NLI): 리랭커가 뒷받침으로 본 문장 중 인용한 근거 모두와 모순인 것만 contradicted
    def fake_nli(pairs):   # 문장에 '낮' 이 있으면 모순
        return [0.1 if "낮" in h else 0.9 for _, h in pairs], [0.95 if "낮" in h else 0.01 for _, h in pairs]

    nrep = cc.check("DPR 정확도는 BM25 보다 높습니다 [1]. DPR 정확도는 BM25 보다 낮습니다 [1].", tchunks,
                    scorer=lambda pairs: [0.9] * len(pairs), threshold=0.05, nli=fake_nli)
    assert [s.status for s in nrep.sentences] == ["supported", "contradicted"]
    assert nrep.sentences[1].contradiction == 0.95 and nrep.summary()["contradicted"] == 1
    print("  test_citation_check ✅")


def test_sub_coverage() -> None:
    """다중 홉 몫 보장: 하위 질문의 최고 논문이 선택에 없으면 가장 많이 뽑힌 논문의 끝 청크와 바꾼다(§5.48)."""
    from rag.core.schemas import RetrievedChunk
    from rag.stages.stage2_rerank import ensure_sub_coverage

    a1, a2, a3 = (RetrievedChunk(content=f"RAG part {i}", source="A") for i in range(3))
    b1 = RetrievedChunk(content="REALM pretraining", source="B")
    c1 = RetrievedChunk(content="unrelated", source="C")
    cands = [a1, a2, a3, b1, c1]
    scorer = lambda pairs: [0.9 if q.split()[0] in d else 0.01 for q, d in pairs]  # noqa: E731
    out = ensure_sub_coverage(["RAG 는 무엇", "REALM 은 무엇"], cands, [a1, a2, a3], scorer=scorer)
    assert [c.source for c in out] == ["A", "A", "B"], out
    assert ensure_sub_coverage(["REALM 하나"], cands, [a1, a2], scorer=scorer) == [a1, a2], "하위 질문 1개면 그대로"
    low = ensure_sub_coverage(["RAG x", "없음 y"], cands, [a1, a2], scorer=scorer)
    assert low == [a1, a2], "하위 질문 최고 점수가 낮으면 보장하지 않음"
    print("  test_sub_coverage ✅")


def test_paper_store() -> None:
    """원문 저장소: 꺼져 있으면 조용히 건너뛰고, 수집은 PDF·전문을 넘기며, DB 가 있으면 저장·조회가 왕복한다."""
    from types import SimpleNamespace

    from config import settings
    from rag.ingest import arxiv_source as A
    from rag.store import paperstore

    url0 = settings.database_url
    try:
        settings.database_url = ""
        assert paperstore.save({"source": "x"}, "t", True) is False and paperstore.get_pdf("x") is None

        # 수집 경로: PDF 를 받으면 바이트와 전문을, 실패하면 초록(is_fulltext=False)을 저장소에 넘긴다
        calls = []
        save0, dl0 = paperstore.save, A.download_pdf_text
        paperstore.save = lambda meta, text, full, pdf=None, **k: calls.append((meta["source"], text, full, pdf))
        r = SimpleNamespace(title="T", summary="S", pdf_url="u", entry_id="e", authors=[],
                            published=__import__("datetime").date(2024, 1, 1), get_short_id=lambda: "2401.00001v1")

        def fake_dl(result, session, timeout=60, out=None):
            out["pdf"] = b"%PDF"
            return "1 Introduction\nbody text"
        try:
            A.download_pdf_text = fake_dl
            A.result_to_chunks(r, None, True)
            A.download_pdf_text = lambda *a, **k: ""
            A.result_to_chunks(r, None, True)
        finally:
            paperstore.save, A.download_pdf_text = save0, dl0
        assert calls[0][2] is True and calls[0][3] == b"%PDF" and "body text" in calls[0][1], calls[0]
        assert calls[1][2] is False and calls[1][3] is None and calls[1][1] == "T\n\nS", calls[1]
    finally:
        settings.database_url = url0

    # 자동 수집 후보: 원문 DB 에서 기준을 넘는 논문을 찾으면 사이트 검색을 하지 않는다
    from rag.core import llm as L
    from rag.ingest import auto_fetch as F
    from rag.ingest.arxiv_source import Paper

    saved = F.make_search_queries, F.paper_db_candidates, L.get_reranker, A.search_papers
    site_calls = []
    try:
        F.make_search_queries = lambda q: ["contrastive image text pretraining"]
        L.get_reranker = lambda: SimpleNamespace(predict=lambda pairs: [5.0 if "CLIP" in p[1] else -5.0 for p in pairs])
        A.search_papers = lambda *a, **k: site_calls.append(a) or [Paper(pid="9999.00001", title="site paper", found_by="arXiv")]
        F.paper_db_candidates = lambda qs, ex: [Paper(pid="2103.00020", title="CLIP: Learning Transferable", found_by="원문 DB")]
        fc = F.find_candidates("CLIP 은?", set(), title_guess=False, common_refs=False)
        assert fc["from_db"] and [p.pid for p, _ in fc["chosen"]] == ["2103.00020"] and not site_calls, fc
        F.paper_db_candidates = lambda qs, ex: [Paper(pid="1111.00001", title="unrelated", found_by="원문 DB")]
        fc = F.find_candidates("CLIP 은?", set(), title_guess=False, common_refs=False)
        assert not fc["from_db"] and site_calls, "원문 DB 에서 기준 미달이면 사이트 검색"
    finally:
        F.make_search_queries, F.paper_db_candidates, L.get_reranker, A.search_papers = saved

    why = _db_login_error()
    if why:
        print(f"  test_paper_store ✅ ({why} — DB 왕복은 건너뜀)")
        return
    src = "test-paperstore-0000"
    try:
        assert paperstore.save({"source": src, "title": "t"}, "1 Method\nzyxwquux transformer full", True, pdf=b"%PDF-1")
        assert paperstore.save({"source": src, "title": "t"}, "abstract only", False), "초록만이면 전문을 덮지 않음"
        assert paperstore.get_text(src, full_only=True) == "1 Method\nzyxwquux transformer full"
        assert paperstore.get_pdf(src) == b"%PDF-1" and src in paperstore.stored_sources(with_pdf=True)
        assert [r["source"] for r in paperstore.search(["zyxwquux"])] == [src], "원문 DB 검색"
        assert paperstore.search(["zyxwquux"], exclude={src}) == [], "벡터 DB 에 있는 논문은 제외"
        # 수집: 원문 DB 에 전문이 있으면 PDF 를 받지 않는다
        r2 = SimpleNamespace(**{**r.__dict__, "get_short_id": lambda: src})
        A.download_pdf_text = lambda *a, **k: (_ for _ in ()).throw(AssertionError("받으면 안 됨"))
        try:
            chunks, full = A.result_to_chunks(r2, None, True)
        finally:
            A.download_pdf_text = dl0
        assert full and "zyxwquux" in chunks[0].page_content
    finally:
        with paperstore.connect() as conn:
            conn.execute("DELETE FROM papers WHERE source = %s", (src,))
    print("  test_paper_store ✅ (DB 왕복 포함)")


def main() -> None:
    print("단위 테스트 실행:")
    test_rrf()
    test_split_lines()
    test_parse_json()
    test_contextualize_no_history()
    test_mmr_per_source_cap()
    test_pinecone_store_adapter()
    test_postgres_url()
    test_pgvector_store()
    test_paper_search_parsers()
    test_exclude_auto_view()
    test_auto_rerank_weight()
    test_auto_fetch_decisions()
    test_evict_over_budget()
    test_citation_graph()
    test_compare_table()
    test_survey()
    test_bm25_tokenize()
    test_watch()
    test_visual_parse()
    test_visual_retrieval()
    test_local_reasoning_stream()
    test_rerank_title_prefix()
    test_followup_helpers()
    test_citation_check()
    test_sub_coverage()
    test_original_signal()
    test_relation_claim()
    test_paper_store()
    print("모든 단위 테스트 통과 ✅")


if __name__ == "__main__":
    main()
