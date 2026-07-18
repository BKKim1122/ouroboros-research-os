# V17.6 — self/other × privilege 상호작용 (탐색적)

> **EN** — Self/other x privilege 2x2 minimal-pair interaction test: is the 'privilege' (direct vs. inferred knowledge) axis self-specific, or a general epistemic-access axis applying equally to self and other referents? Pronoun components are projected out; both outcomes are pre-registered. Pending execution.

privilege가 "나라서 특별한 것"(self-특이)인지, 아니면 "직접 앎 vs 추론 앎"이라는
누구에게나 적용되는 일반 축인지 가른다. V17에서는 이 둘이 한 몸으로 붙어 있어
구분 불가였다. 여기서 2×2로 분리한다.

## 2×2
|          | 직접 앎(privileged)         | 추론으로 앎(inferred)          |
|----------|-----------------------------|--------------------------------|
| 나(self) | sp                          | si                             |
| 남(other)| op                          | oi                             |

핵심: self 내부 방향(sp−si)과 other 내부 방향(op−oi)이 서로 전이되면 → 일반 축.
전이 실패 → self에서 다르게 조직됨(self-특이). knower/referent 대명사 성분은
대명사 축 투영 제거로 통제.

## 실행 (edgexpert, 실모델)
```bash
cd ~/ouroboros/experiments/v17_6
python interaction.py
```
mock 점검만: `python interaction.py --mock`. seeds 기본 [3..10](V17 확증과 동일, 파일럿 비중복).

## 결과 읽는 법 (스크립트가 사전약속 라벨로 판정)
가장 중요한 지표는 **transfer_mean_ctrl**(대명사 통제 후 self↔other 상호전이).
- **>= 0.75 & cos_priv_axes >= 0.5** → H_general: privilege는 self/other 공통 일반 축.
  self-특이 아님. V17의 'self-related'는 유지, 'self-specific'는 미지지.
- **<= 0.60  또는  delta_self_minus_other >= 0.15** → H_selfspecific: self에서 다르게
  조직됨. self-indexed→self-specific 근거.
- 그 사이 → 부분 전이. 서술.
- shuffle_baseline이 ~0.5인지 확인(우연선 정상 작동).

## 사전 기대 봉인 (INTERP_GUARDRAILS)
분석자의 사전 기대는 H_general이지만 예단이라 판정에 반영 안 함. 위 기준으로만.
어느 쪽이 나오든 그대로 보고. H_selfspecific이 나와도 nuisance 완전통제(약한고리 A)
전까지는 'self-특이 정황'까지다.
