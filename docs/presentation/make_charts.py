"""
발표 자료용 그래프 생성 — docs/presentation/charts/*.png

수치는 docs/PROGRESS_REPORT.md·docs/ACHIEVEMENT.md 의 측정값을 그대로 옮긴 것이다(LLM·네트워크 사용 없음).
실행: python docs/presentation/make_charts.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "charts"

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
    rows = [("해결하려는 문제", 90, "현재"), ("접근 방식", 90, "현재"), ("현재 기능", 100, "현재"),
            ("확장 원칙 준수", 100, "현재"), ("단기 과제", 20, "로드맵"), ("중기 과제", 0, "로드맵"),
            ("다른 도메인 이전", 50, "로드맵"), ("장기 과제", 25, "로드맵")]
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
    ax.legend([uniq["현재"], uniq["로드맵"]], ["현재 목표 (합계 95%)", "로드맵 (합계 17%)"],
              loc="lower right", frameon=False)
    ax.set_title("PROJECT_VISION 달성률")
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
    ax.set_title("코퍼스가 커질수록 검색 품질이 조금씩 내려간다")
    save(fig, "03_corpus_growth.png")


def auto_fetch_progress() -> None:
    """자동 수집 개선 전후(§5.16 → §5.17)와 목표."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    panels = [
        (axes[0], "원 논문 수집률, 같은 12문항 (높을수록 좋음)", [17, 50], 60, "목표 60%"),
        (axes[1], "코퍼스 안 질문의 수집 신호 (낮을수록 좋음)", [47, 23], 5, "목표 5%"),
    ]
    for ax, title, vals, target, tlab in panels:
        bars = ax.bar(["개선 전", "개선 후"], vals, width=0.5, color=[AXIS, BLUE], edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}%", ha="center", color=INK, fontsize=12)
        ax.axhline(target, color=CRITICAL, lw=1.2, ls="--")
        ax.text(-0.3, target + 1.5, tlab, ha="left", color=INK2, fontsize=11)
        ax.set_ylim(0, 70)
        ax.set_yticks([0, 20, 40, 60], ["0", "20", "40", "60%"])
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=13)
    fig.suptitle("자동 수집: 좋아졌지만 두 목표는 아직 미달", x=0.02, ha="left", fontsize=16, fontweight="bold")
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
        ("원 논문 수집", "60% 이상", "45%", "미달"),
        ("코퍼스 안 질문의 수집 신호", "5% 이하", "23%", "미달"),
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
    ax.set_title("자동 수집 실전 시험 기준 (8개 중 6개 통과)", loc="left")
    save(fig, "06_kpi_table.png")


if __name__ == "__main__":
    achievement()
    query_transform()
    corpus_growth()
    auto_fetch_progress()
    latency()
    kpi_table()
