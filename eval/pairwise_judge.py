"""
eval/pairwise_judge.py — 두 실행(answer_quality --rerun 결과)의 답을 블라인드로 짝 비교 (정확도 실험 §5.56)

자동 지표(검증 v2·거절 문구·인용 경보)는 같은 설정 두 번 실행에서도 흔들리고(거절 8 vs 6) 검증 v2 는 변형을 잘 가르지 못했다
(완전 0.86 = 0.86). 그래서 판정자(사람 또는 다른 모델)가 어느 실행인지 모르게 X/Y 로 섞어 비교한다.

  --prep A B --out DIR      문항마다 질문·두 실행의 근거(합집합)·답 X/Y 를 묶은 판정 파일(묶음 단위)과 정답표(key.json)
  --score DIR               판정 결과(DIR/verdicts_*.json: {id: {"winner": "X"|"Y"|"tie", ...}})를 정답표로 풀어 집계

판정 파일·정답표에는 근거 원문이 들어가므로 storage/ 아래에 둔다(저장소에 올리지 않음).
판정 주체가 모델이면 결과는 '모델 판정 기준'으로만 쓴다(사람 판정으로 기록하지 않음).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.answer_quality import load_items, load_run  # noqa: E402

BATCH = 10
SNIP = 1800


def prep(a: str, b: str, out: str, seed: int = 7) -> None:
    items, ra, rb = load_items(), load_run(a), load_run(b)
    ids = [k for k in items if k in ra and k in rb]
    rnd = random.Random(seed)
    key, blocks = {}, []
    for k in ids:
        flip = rnd.random() < 0.5
        x, y = (rb[k], ra[k]) if flip else (ra[k], rb[k])
        key[k] = {"X": b if flip else a, "Y": a if flip else b}
        seen, ev = set(), []
        for c in (ra[k].get("sources") or []) + (rb[k].get("sources") or []):
            sig = (c["source"], c["content"][:120])
            if sig in seen:
                continue
            seen.add(sig)
            ev.append(f"- [{c['source']} / {c.get('section') or ''}] {c['content'][:SNIP].replace(chr(10), ' ')}")
        blocks.append((k, "\n".join([
            f"### {k}", f"질문: {items[k]['question']}", "근거(두 답이 받은 근거의 합집합, 순서 무관):", *ev,
            "", "[답 X]", x["answer"], "", "[답 Y]", y["answer"], ""])))
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(blocks), BATCH):
        (d / f"batch_{i // BATCH:02d}.md").write_text("\n".join(t for _, t in blocks[i:i + BATCH]), encoding="utf-8")
    (d / "key.json").write_text(json.dumps({"a": a, "b": b, "key": key}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{d}: {len(ids)}문항, 묶음 {(len(blocks) + BATCH - 1) // BATCH}개")


def sign_test(wins_a: int, wins_b: int) -> float:
    """무승부를 뺀 승패의 양측 부호 검정 p 값. 같은 설정 두 번(기준 1·2회차)도 10:15 로 갈렸다 — p<0.05 만 차이로 본다."""
    from math import comb

    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = max(wins_a, wins_b)
    return round(min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n), 4)


def score(out: str) -> dict:
    d = Path(out)
    meta = json.loads((d / "key.json").read_text(encoding="utf-8"))
    a, b, key = meta["a"], meta["b"], meta["key"]
    verdicts = {}
    for f in sorted(d.glob("verdicts_*.json")):
        verdicts.update(json.loads(f.read_text(encoding="utf-8")))
    tally = {a: 0, b: 0, "tie": 0}
    per = {}
    for k, v in verdicts.items():
        w = v.get("winner")
        name = "tie" if w == "tie" else key[k][w]
        tally[name] += 1
        per[k] = name
    n = sum(tally.values())
    res = {"a": a, "b": b, "judged": n, "wins": tally,
           "b_minus_a": tally[b] - tally[a], "sign_test_p": sign_test(tally[a], tally[b]), "per_item": per}
    print(json.dumps({k: v for k, v in res.items() if k != "per_item"}, ensure_ascii=False))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", nargs=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.prep:
        prep(*a.prep, a.out)
    elif a.score:
        score(a.out)


if __name__ == "__main__":
    main()
