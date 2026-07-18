# V17.5 — 공유 부분공간 top-k 제거 (탐색적 강건성 분석)

> **EN** — Exploratory robustness analysis: generalizes V17's rank-1 shared-axis removal to rank-k with leave-one-seed-out cross-validation and random-subspace controls, testing whether the factor-informative residual (claim C3) is a rank-1 artifact. Result: C3 holds; the shared subspace is low-rank (~1-2).

V17 동결 밖의 sensitivity 분석이다. 상태기계(cli loop)를 태우지 않고 단독 실행한다.
외부검토 약한고리 C("projection_gap이 rank-1 가정의 아티팩트일 수 있다")에 답한다.

## 무엇을 하나
- V17과 동일한 방식으로 요인 방향을 뽑되, 공유축을 rank-1이 아니라 rank-k로 제거.
- 교차검증(LOO): test seed를 뺀 나머지 seed 방향으로 공유 부분공간 U_k를 추정하고,
  held-out test seed에서 제거 후 gap 측정 → 같은 데이터로 추정·평가하는 낙관편향 제거.
- 동일 차원 random 정규직교 부분공간 제거를 대조로 병행.

## 실행 (edgexpert, 실모델)
```bash
cd ~/ouroboros/experiments/v17_5
# config.yaml은 v17 최종값 복사본 (mock:false / layer_frac:0.43 / revision 고정)
python topk_removal.py --kmax 6
```
mock 파이프라인 점검만 하려면 `python topk_removal.py --mock --kmax 6`.

## 결과 읽는 법 (판정은 스크립트가 사전약속 라벨로 출력)
- **k=1에서 gap 소멸** → MC1: C3는 평균축 아티팩트. V17의 "잔여구조" 주장 하향.
- **k=2~3에서 소멸** → MC2: 공유공간이 저차원. C3 약화.
- **top-k 제거 후에도 plateau + random 대비 우위(>0.05)** → MC3: C3 강건(rank-1 아님).
- **plateau는 있으나 random 대비 우위 미달** → 보류: 저차원 제거 일반효과 가능성.

## 이 분석이 말하지 않는 것 (spec.yaml claim_ceiling)
MC3가 나와도 "C3가 rank-1 아티팩트는 아니다"까지다. self-특이성(→ self/other 상호작용
실험 필요)도, nuisance 독립성(→ 약한고리 A 미해결)도 주장하지 않는다.
