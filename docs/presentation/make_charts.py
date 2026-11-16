"""
발표 자료용 그래프 생성 — docs/presentation/charts/*.png

수치는 docs/PROGRESS_REPORT.md·docs/ACHIEVEMENT.md 의 측정값을 옮긴 것이고, 07·08 은 eval/results 의 측정 결과 JSON 을
직접 읽는다(측정을 다시 하면 그림도 바뀜). LLM·네트워크 사용 없음.
실행: python docs/presentation/make_charts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "charts"
RESULTS = Path(__file__).resolve().parents[2] / "eval" / "results"

# 색: 검증된 기본 팔레트(dataviz) 고정 순서. 텍스트는 잉크 색만 쓴다.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 12,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlesize": 16, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("saved", OUT / name)


def achievement() -> None:
    """구분별 달성률(ACHIEVEMENT.md 요약표)."""
    rows = [("해결하려는 문제", 100, "현재"), ("접근 방식", 90, "현재"), ("현재 기능", 100, "현재"),
            ("확장 원칙 준수", 100, "현재"), ("단기 과제", 50, "로드맵"), ("중기 과제", 100, "로드맵"),
            ("다른 도메인 이전", 50, "로드맵"), ("장기 과제", 38, "로드맵")]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = range(len(rows))[::-1]
    for yi, (name, v, kind) in zip(y, rows):
        ax.barh(yi, v, height=0.6, color=BLUE if kind == "현재" else ORANGE,
                label=kind, edgecolor=SURFACE, linewidth=2)
        ax.text(v + 1.5, yi, f"{v}%", va="center", color=INK, fontsize=12)
    ax.set_yticks(list(y), [r[0] for r in rows], color=INK2)
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.grid(axis="y", visible=False)
    h, lab = ax.get_legend_handles_labels()
    uniq = dict(zip(lab, h))
    ax.legend([uniq["현재"], uniq["로드맵"]], ["현재 목표 (합계 97.5%)", "로드맵 (합계 63%)"],
              loc="lower right", frameon=False)
    ax.set_title("PROJECT_VISION 달성률 (2026-11-15)")
    save(fig, "01_achievement.png")


def query_transform() -> None:
    """ADR-001 번복: 질의 변환 설정별 recall@5 (§5.7, 490편 코퍼스)."""
    names = ["질의변환 없음\n벡터만", "multi_query\n벡터만", "multi_query\n하이브리드", "HyDE\n하이브리드"]
    full = [0.950, 0.968, 0.968, 0.923]
    hard = [0.871, 0.935, 0.935, 0.839]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = range(len(names))
    w = 0.36
    for off, vals, color, lab in [(-w / 2, full, BLUE, "전체 111문항"), (w / 2, hard, ORANGE, "어려운 31문항")]:
        bars = ax.bar([i + off for i in x], vals, width=w, color=color, label=lab, edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=10, color=INK2)
    ax.set_xticks(list(x), names, color=INK2)
    ax.set_ylim(0.80, 1.0)
    ax.set_ylabel("recall@5 (세로축 0.80부터)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("작은 표본으로 고른 HyDE가 확장 평가에서 가장 낮았다")
    save(fig, "02_query_transform.png")


def corpus_growth() -> None:
    """자동 수집량에 따른 검색 지표 하락과 예산(§5.18)."""
    chunks = [0, 1036, 2021, 3001, 4184]
    recall = [0, -0.0034, -0.0051, -0.0051, -0.0068]
    mrr = [0, 0.0017, -0.0048, -0.0088, -0.0088]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(chunks, recall, color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label="recall@5 변화")
    ax.plot(chunks, mrr, color=ORANGE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label="MRR 변화")
    ax.axhline(-0.01, color=CRITICAL, lw=1.2, ls="--")
    ax.text(4184, -0.0097, "허용 한도 -0.01", ha="right", va="bottom", color=INK2, fontsize=11)
    ax.axvline(2000, color=MUTED, lw=1.2, ls=":")
    ax.text(2050, 0.0015, "수집 예산 2,000청크", color=INK2, fontsize=11)
    ax.text(chunks[-1] + 60, recall[-1], f"{recall[-1]:+.4f}", va="center", color=INK2, fontsize=10)
    ax.text(chunks[-1] + 60, mrr[-1], f"{mrr[-1]:+.4f}", va="center", color=INK2, fontsize=10)
    ax.set_xlim(-150, 4700)
    ax.set_ylim(-0.012, 0.003)
    ax.set_xlabel("자동 수집 청크 수 (수동 2,436청크 기준)")
    ax.set_ylabel("자동 수집 제외 대비 변화")
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("코퍼스가 커질수록 검색 품질이 조금씩 내려간다 (11-11 측정)")
    save(fig, "03_corpus_growth.png")


def auto_fetch_progress() -> None:
    """자동 수집 개선 과정과 목표.
    원 논문 수집: 같은 20문항(기대 논문이 있는 문항) — §5.17 45% → §5.29 v4 50% → §5.51 v7 55% → §5.53 v9 75%.
    (§5.16 의 2/12=17% 는 12문항 기준이라 넣지 않는다.)
    코퍼스 안 수집 신호: §5.16·§5.17 은 실전 시험(hard 30), §5.53 은 B1 기록 재판정 — 측정 방식이 다르다.
    같은 재판정 기준으로는 3차 신호 전 10%(3/30, §5.31) → 후 13%(4/30)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    panels = [
        (axes[0], "원 논문 수집률 — 같은 20문항 (높을수록 좋음)", ["§5.17", "v4", "v7", "최종 v9"], [45, 50, 55, 75], 60, "목표 60%"),
        (axes[1], "코퍼스 안 질문의 수집 신호 (낮을수록 좋음)", ["실전 §5.16", "실전 §5.17", "재판정 §5.53"], [47, 23, 13], 5, "목표 5%"),
    ]
    for ax, title, labels, vals, target, tlab in panels:
        bars = ax.bar(labels, vals, width=0.5, color=[AXIS] * (len(vals) - 2) + [BLUE, BLUE], edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}%", ha="center", color=INK, fontsize=12)
        ax.axhline(target, color=CRITICAL, lw=1.2, ls="--")
        ax.text(-0.3, target + 1.5, tlab, ha="left", color=INK2, fontsize=11)
        ax.set_ylim(0, 85)
        ax.set_yticks([0, 20, 40, 60, 80], ["0", "20", "40", "60", "80%"])
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=13)
    axes[1].text(1.0, -16, "§5.53 은 측정 방식이 다름 — 같은 재판정 기준 3차 신호 전 10%", ha="center", color=INK2, fontsize=9)
    fig.suptitle("자동 수집: 원 논문 수집은 목표 달성, 코퍼스 안 신호는 아직 미달", x=0.02, ha="left", fontsize=16, fontweight="bold")
    save(fig, "04_auto_fetch_progress.png")


def latency() -> None:
    """단계별 평균 소요 시간(§5.9, 로컬 Qwen thinking off, hard 69문항)."""
    stages = [("질의 변환", 6.9, BLUE), ("검색·리랭크", 8.1, ORANGE), ("생성", 25.9, AQUA), ("검증", 21.1, YELLOW)]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    left = 0.0
    for name, sec, color in stages:
        ax.barh(0, sec, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=2)
        ax.text(left + sec / 2, 0.42, f"{name}\n{sec:.1f}초", ha="center", va="bottom", color=INK, fontsize=11)
        left += sec
    ax.text(left + 1, 0, f"합계 {left:.0f}초", va="center", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 72)
    ax.set_ylim(-0.4, 1.1)
    ax.set_yticks([])
    ax.set_xlabel("초")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_title("질문당 약 1분, 대부분은 생성과 검증")
    save(fig, "05_latency.png")


def kpi_table() -> None:
    """자동 수집 목표 수치표(ACHIEVEMENT.md §7)를 이미지로."""
    rows = [
        ("recall@5 하락", "0.01 이내", "-0.0034", "통과"),
        ("MRR 하락", "0.01 이내", "-0.0052", "통과"),
        ("faithfulness 하락", "0.05 이내", "0.985 → 0.980", "통과"),
        ("answer_relevance 하락", "0.05 이내", "0.640 → 0.642", "통과"),
        ("무관한 질문으로 추가된 논문", "0편", "0편", "통과"),
        ("같은 질문 재수집", "0회", "0회", "통과"),
        ("원 논문 수집", "60% 이상", "45% → 75%", "통과"),
        ("코퍼스 안 질문의 수집 신호", "5% 이하", "23% → 13%", "미달"),
    ]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.axis("off")
    tbl = ax.table(cellText=[list(r) for r in rows], colLabels=["항목", "기준", "결과", "판정"],
                   loc="center", cellLoc="left", colWidths=[0.42, 0.18, 0.24, 0.12])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)
        if r == 0:
            cell.set_text_props(weight="bold", color=INK2)
        elif c == 3:
            ok = cell.get_text().get_text() == "통과"
            cell.set_text_props(color="#006300" if ok else CRITICAL, weight="bold")
    ax.set_title("자동 수집 실전 시험 기준 (8개 중 7개 통과) — recall·MRR·faithfulness 는 11-11 측정", loc="left")
    save(fig, "06_kpi_table.png")


def citation_check() -> None:
    """문장 단위 인용 확인 기준값 보정(§5.40): 기준값별 통과율 — 맞는 인용은 통과, 다른·무관한 근거는 걸러야 한다."""
    d = json.loads((RESULTS / "citation_check_eval.json").read_text(encoding="utf-8"))
    grid = d["grid"]
    t = [g["threshold"] for g in grid]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for key, color, lab in [("pos_pass", BLUE, "맞는 인용 (통과해야 함)"), ("hard_neg_pass", ORANGE, "같은 질문의 다른 논문 근거"),
                            ("easy_neg_pass", AQUA, "무관한 근거 (걸러야 함)")]:
        v = [g[key] * 100 for g in grid]
        ax.plot(t, v, color=color, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, label=lab)
    chosen = d["threshold_used"]
    ax.axvline(chosen, color=MUTED, lw=1.2, ls=":")
    sel = next(g for g in grid if g["threshold"] == chosen)
    note = (f"기준 {chosen}: 맞는 인용 {sel['pos_pass']:.0%} 통과",
            f"다른 논문 {sel['hard_neg_pass']:.0%} · 무관 {sel['easy_neg_pass']:.1%}")
    ax.text(chosen + 0.01, 8, "\n".join(note), color=INK2, fontsize=11)
    ax.set_xscale("log")
    ax.set_xticks(t, [str(x) for x in t])
    ax.minorticks_off()
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.set_xlabel("리랭커 점수 기준값 (로그 눈금)")
    ax.set_ylabel("통과율")
    ax.legend(frameon=False, loc="center right")
    ax.set_title(f"문장 단위 인용 확인 — 실제 답변 {d['answers']}개, 인용 문장 {d['cited_sentences']}개로 보정")
    save(fig, "07_citation_check.png")


def reranker_title() -> None:
    """본문 청크 맥락 실험(§5.41·§5.42): 기준 대비 recall@5·MRR 변화. 1차(임베딩까지 바꿈) vs 2차(리랭커만, 적용)."""
    r1 = json.loads((RESULTS / "contextual_header_eval.json").read_text(encoding="utf-8"))["results"]
    r2 = json.loads((RESULTS / "contextual_header_eval_r2.json").read_text(encoding="utf-8"))["results"]
    base = r2["A_baseline"]["eval294"]
    nl = "\n"
    rows = [("임베딩", r1["B_embed"]), ("임베딩+BM25", r1["C_embed_bm25"]), (f"임베딩+BM25{nl}+리랭커", r1["D_embed_bm25_rerank"]),
            (f"리랭커만{nl}(적용)", r2["E2_rerank_title"]), (f"리랭커 점수{nl}평균", r2["G_rerank_blend"])]
    names = [n for n, _ in rows]
    d_rec = [r["eval294"]["recall"] - base["recall"] for _, r in rows]
    d_mrr = [r["eval294"]["mrr"] - base["mrr"] for _, r in rows]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = range(len(rows))
    w = 0.36
    for off, vals, color, lab in [(-w / 2, d_rec, BLUE, "recall@5 변화"), (w / 2, d_mrr, ORANGE, "MRR 변화")]:
        bars = ax.bar([i + off for i in x], vals, width=w, color=color, label=lab, edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.0006 if v >= 0 else -0.0006), f"{v:+.4f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.axvline(2.5, color=MUTED, lw=1, ls=":")
    ax.text(1.0, 0.027, "1차: 시험용 DB 사본", ha="center", color=INK2, fontsize=11)
    ax.text(3.5, 0.027, "2차: DB 그대로", ha="center", color=INK2, fontsize=11)
    ax.set_xticks(list(x), names, color=INK2)
    ax.set_ylim(-0.012, 0.031)
    ax.set_ylabel(f"기준(recall {base['recall']}, MRR {base['mrr']}) 대비")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("논문 제목은 리랭커에만 — MRR 은 오르고 recall 은 지킨다")
    save(fig, "08_reranker_title.png")


def title_candidates() -> None:
    """원 논문 제목 후보(§5.43): 원 논문이 제목 검색 후보에 드는 질문 비율 — 조정용·확인용(개선에 안 씀)."""
    d = json.loads((RESULTS / "title_candidates_eval.json").read_text(encoding="utf-8"))
    sets = list(d.items())   # (조정용, 확인용)
    names = ["제목 1개(기존)", "후보 3개", "둘 다(적용)"]
    keys = ["single", "multi", "union"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    w = 0.36
    for off, (label, v), color in zip((-w / 2, w / 2), sets, (BLUE, ORANGE)):
        vals = [v[k] / v["n"] * 100 for k in keys]
        bars = ax.bar([i + off for i in range(len(keys))], vals, width=w, color=color, edgecolor=SURFACE, linewidth=2,
                      label=f"{label} {v['n']}문항")
        for b, k, val in zip(bars, keys, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 1.5, f"{v[k]}/{v['n']}", ha="center", fontsize=10, color=INK2)
    ax.set_xticks(range(len(keys)), names, color=INK2)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.set_ylabel("원 논문이 후보에 든 질문")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("원 논문 찾기 — 제목 추정이 틀릴 때 후보를 더 받는다")
    save(fig, "09_title_candidates.png")


def answer_quality() -> None:
    """답변 품질 개선 단계 1·3(§5.48·§5.49): 기준 세트 판정 분포와 다중 홉 맞음. 판정은 Claude 초안(사람 검토 전)."""
    seq = ["#1c5aa3", "#6fa3e3", "#c7dbf4"]   # 맞음·부분·틀림 — 순서가 있는 값이라 한 색의 진하기로
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    rows = [("프롬프트 v1", (36, 12, 2)), ("프롬프트 v2", (41, 8, 1))]
    for yi, (name, vals) in zip((1, 0), rows):
        left = 0
        for v, c, lab in zip(vals, seq, ("맞음", "부분", "틀림")):
            a1.barh(yi, v, left=left, height=0.55, color=c, edgecolor=SURFACE, linewidth=2, label=lab if yi == 1 else None)
            if v >= 4:
                a1.text(left + v / 2, yi, str(v), ha="center", va="center", color=SURFACE if c == seq[0] else INK, fontsize=12)
            left += v
    a1.set_yticks([1, 0], [r[0] for r in rows], color=INK2)
    a1.set_xlim(0, 50)
    a1.set_xlabel("문항 수 (50)")
    a1.grid(axis="y", visible=False)
    a1.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    a1.set_title("기준 세트 50문항", pad=34)

    names, vals = ["v1", "v2", "v2 + 다중 홉 분해"], [5, 9, 11]
    bars = a2.bar(range(3), vals, width=0.55, color=seq[0], edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v}/18", ha="center", color=INK2, fontsize=11)
    a2.set_xticks(range(3), names, color=INK2, fontsize=10)
    a2.set_ylim(0, 18)
    a2.set_ylabel("맞음")
    a2.grid(axis="x", visible=False)
    a2.set_title("그중 다중 홉 18문항", pad=34)
    fig.suptitle("답변 품질 개선 — 판정은 Claude (사람 판정 10문항과는 5/10 일치, κ 0.00 — §5.52)", x=0.02, ha="left", fontsize=13, color=INK2)
    save(fig, "10_answer_quality.png")


if __name__ == "__main__":
    achievement()
    query_transform()
    corpus_growth()
    auto_fetch_progress()
    latency()
    kpi_table()
    citation_check()
    reranker_title()
    title_candidates()
    answer_quality()
