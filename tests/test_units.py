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
    from rag.stage2_rerank import _reciprocal_rank_fusion

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
    from rag.stage1_query_transform import _split_lines

    out = _split_lines("1. first\n- second\n\n  3) third  ")
    assert out == ["first", "second", "third"], out
    assert _split_lines("3D Gaussian Splatting\n4-bit quantization\n2. 8-bit") == [  # 숫자로 시작하는 검색어 보존
        "3D Gaussian Splatting", "4-bit quantization", "8-bit"]
    print("  test_split_lines ✅")


def test_parse_json() -> None:
    """_parse_json: 코드펜스/주변 텍스트가 있어도 JSON 추출."""
    from rag.stage4_verify import _parse_json

    assert _parse_json('{"grounded": true}')["grounded"] is True
    fenced = '```json\n{"faithfulness": 0.8}\n```'
    assert _parse_json(fenced)["faithfulness"] == 0.8
    noisy = '설명 텍스트 {"a": 1} 뒤에도 말이 있음'
    assert _parse_json(noisy)["a"] == 1
    print("  test_parse_json ✅")


def test_contextualize_no_history() -> None:
    """히스토리가 없으면 LLM 호출 없이 원문 그대로 반환해야 한다."""
    from rag.contextualize import contextualize

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
    import rag.llm as llm
    from rag.schemas import RetrievedChunk
    from rag.stage2_rerank import _mmr_select

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

    from rag.docstore import DocStore
    from rag.vectorstore import PineconeStore, get_all_documents

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

    store.delete(ids=["id-a"])
    assert store.count() == 1 and store.get()["ids"] == ["id-b"], "삭제가 인덱스·docstore 양쪽에 반영"
    ids = store.add_documents([Document(page_content="c", metadata={"source": "P3"})])
    assert store.similarity_search("c", k=1)[0].id == ids[0]
    print("  test_pinecone_store_adapter ✅")


def test_exclude_auto_view() -> None:
    """기준 코퍼스 보기: 자동 수집 청크를 숨긴 결과가 '처음부터 그 청크가 없던 스토어'와 같아야 한다."""
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fake_pinecone import FakePineconeIndex

    from rag.docstore import DocStore
    from rag.vectorstore import ExcludeOriginStore, PineconeStore, get_all_documents, get_vectors

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
    import rag.stage2_rerank as s2
    from config import settings
    from rag.schemas import RetrievedChunk

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
    from rag.auto_fetch import is_back_matter

    assert is_back_matter("References") and is_back_matter("Appendix A") and is_back_matter("BIBLIOGRAPHY")
    assert not is_back_matter("3.2 Method") and not is_back_matter(None)
    print("  test_auto_rerank_weight ✅")


def test_auto_fetch_decisions() -> None:
    """답변 신호는 대상이 근거에 없을 때만, 선정은 인용이 가장 많은 관련 논문을 먼저."""
    from types import SimpleNamespace as NS

    import rag.auto_fetch as af
    from rag.schemas import RetrievedChunk

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
    from rag.arxiv_source import openalex_params
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

    import rag.auto_fetch as af
    from config import settings
    from rag.docstore import DocStore
    from rag.schemas import RetrievedChunk
    from rag.vectorstore import PineconeStore

    tmp = Path(tempfile.mkdtemp())
    st = PineconeStore(FakePineconeIndex(), "t", _OneHotEmbeddings(), DocStore(tmp / "d.jsonl"))
    rows = [("m", None, "2020-01-01"), ("old", "auto", "2026-01-01"), ("used", "auto", "2026-01-01"), ("new", "auto", "2026-11-11")]
    for src, origin, date in rows:
        meta = {"source": src, "fetched_at": date, **({"origin": origin} if origin else {})}
        st.upsert_vectors([f"{src}{i}" for i in range(2)], [[1.0, 0.0, 0.0]] * 2, [f"{src} 본문"] * 2, [meta] * 2)
    orig = (settings.auto_usage_path, settings.auto_fetch_max_chunks)
    settings.auto_usage_path, settings.auto_fetch_max_chunks = str(tmp / "u.json"), 4
    try:
        af.record_usage([RetrievedChunk(content="x", source="used", origin="auto"),
                         RetrievedChunk(content="y", source="m")])
        assert list(af.load_usage()) == ["used"], "자동 수집 논문만 기록"
        evicted = af.evict_over_budget(st, protect={"new"})
        assert [e["source"] for e in evicted] == ["old"], evicted
        left = {m["source"] for m in st.get()["metadatas"]}
        assert left == {"m", "used", "new"}, left
        assert af.evict_over_budget(st, protect={"new"}) == [], "예산 안이면 지우지 않음"
    finally:
        settings.auto_usage_path, settings.auto_fetch_max_chunks = orig
    print("  test_evict_over_budget ✅")


def test_paper_search_parsers() -> None:
    """arXiv Atom 응답 해석(직접 호출 경로)과 OpenAlex 초록 역색인 복원."""
    import rag.arxiv_source as A

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
    from rag import citations as C

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
    from rag import compare_table as T
    from rag.schemas import RetrievedChunk

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
    from rag import survey as S
    from rag.schemas import RetrievedChunk

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
    from rag.stage2_rerank import bm25_tokenize

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

    import rag.arxiv_source as A
    from config import settings
    from rag import watch as W

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
        import rag.llm as L
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


def main() -> None:
    print("단위 테스트 실행:")
    test_rrf()
    test_split_lines()
    test_parse_json()
    test_contextualize_no_history()
    test_mmr_per_source_cap()
    test_pinecone_store_adapter()
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
    print("모든 단위 테스트 통과 ✅")


if __name__ == "__main__":
    main()
