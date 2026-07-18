# V17.7 — nuisance 통제 (약한고리 A, 탐색적)

> **EN** — Nuisance control for V17's cross-template transfer. Six confound axes
> (valence, register, syntax, evidentiality, agency, topic) are estimated from
> self/factor-neutral polarity pairs; their 6-D subspace is projected out of the
> four-factor activations, and cross-template min_factor transfer / gap are
> re-measured against a random-subspace control. Kill criterion and both-direction
> interpretation are **pre-registered in `spec.yaml`** before seeing results.

V17의 4요인 전이(min_factor 0.907)가 자기관련 구성개념 때문인지, 아니면 일반 담화
nuisance(주제·감정가·격식·구문·evidentiality·agency)가 실려서인지 가른다. V17의 M0
반박은 대명사 축 하나만 통제했는데, 여기서 nuisance 공간을 6축으로 넓혀 통제한다.

## 방법
각 nuisance 축을 self/요인과 무관한 문장 극성쌍으로 방향 추정 → 6축 부분공간을 4요인
활성치에서 제거 → cross-template min_factor 전이·gap 재측정. 동일 차원(6) random
정규직교 부분공간 제거를 대조로(차원 제거 일반효과 배제).

## 실행
```bash
cd ~/ouroboros/experiments/v17_7
python nuisance_control.py            # config대로 (실모델)
# 파이프라인 점검: python nuisance_control.py --mock
python construct_validity.py       # (a) nuisance 방향 타당성 진단
```

## 봉인된 판정 (결과 전 확정 — spec.yaml)
- **H_beyond**: nuis_min ≥ 0.75 AND (nuisance 하락 − random 하락) ≤ 0.10 → nuisance로 환원 안 됨, E6 강화 (단 6축 한정).
- **H_nuisance**: nuis_min ≤ 0.60 AND 위 값 ≥ 0.15 → 상당 부분 nuisance, E6 후퇴.
- **H_partial**: 그 사이.

## 편향 방어
nuisance 목록 6개는 봉인 — 실행 후 "하나 더"는 별도 후속 실험이지 이 판정에 소급 안 함
(goalpost moving 차단). 판정은 봉인 기준 대입만. 어느 쪽이 나오든 spec의 양방향 해석대로
보고하며, H_nuisance(자아 후퇴)가 나와도 환영·축소 없이 그대로 보고한다.
