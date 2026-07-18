# Ouroboros 자기표상 연구 통합 리포트 (V16-E → V17.7)

> 상태: 2026-07-18. 탐색적 robustness 3종 완결, V17 확증 E6/H0 확정.
> 이 문서는 결과 통합이며, 주장 등급은 거버너·인간 게이트가 정한 것을 그대로 옮긴다.

## 1. 질문
사전학습만 거친 범용 LLM(Qwen2.5-1.5B **base**, rev 8faed761)의 잔차스트림에,
자기관련 표상 구조가 **명시적 학습 없이** 자연발생하는가?

## 2. 방법 — 왜 이 결과를 믿을 수 있나
결과보다 절차가 이 작업의 절반이다.
- **결정론적 상태기계**가 IDLE→…→ARCHIVE를 구동하고, 각 단계에서 필요한 코드만 호출.
- **Protocol Freeze**: 확증 전 코드+프롬프트 해시 동결. 이후 변조 시 결과 무효.
- **규칙기반 거버너 + 2축 주장격자**(기계론 E0–7 × 인간대응 H0–3): LLM은 등급을 올릴 수
  없고 강등/플래그만. E6·H≥1 승격은 인간 게이트 필요.
- **사전등록 확증**: 파일럿과 비중복 seed(3–10)로 out-of-sample.
- **해석 가드레일**(`docs/INTERP_GUARDRAILS.md`): 부하 주제(의식·자아)에서 분석자 편향을
  절차로 구속 — kill criterion·양방향 해석을 결과 전에 봉인, 자기점검 문항.

## 3. V17 확증 결과 — E6 / H0
요인 4개(identity·beneficiary·privilege·concern)에 대해 템플릿군 A→B 전이 probe.
- cross-template 전이 mean **0.968** / min_factor **0.907** (우연 0.5)
- privilege를 1인칭 대명사 probe로 판별 시 **0.500**(우연) — 그러나 전이 probe로는 판별됨
- 공유축 제거 후 projection_gap **0.397**
- seed 일관성 **8/8**, flags 없음 → verdict PASS
- 거버너 판정 **E6/H0** (E6는 claim_promotion 인간 승인 후)

**허용 주장**: base 모델 잔차스트림에, 대명사 표면축으로 환원되지 않는 자기관련 표상
(특히 인식적 특권 축)과 공유축 제거 후 잔존 세부구조가 자연발생할 수 있다.

## 4. Robustness 3종 (탐색적)
| 실험 | 질문 | 결과 |
|---|---|---|
| **V17.5** projection robustness | projection_gap이 rank-1 아티팩트인가 | k=1 제거 후 gap 0.375(=V17 0.397 재현) vs random 0.116 → **C3 지지**. 공유공간은 저차원(rank 1~2), 그 제거 후 요인 잔여 실재 |
| **V17.6** self/other × privilege | privilege가 self-특이인가 일반 축인가 | 대명사 통제 후 cross-referent transfer **0.995**, delta_self−other **0** → **H_general**. self-특이 아님 |
| **V17.7** nuisance control | 전이가 일반 담화 nuisance로 환원되는가 | 6축(valence·register·syntax·evidentiality·agency·topic) 제거 후 min_factor **0.906**(불변), nuisance−random 하락 **−0.001** → **H_beyond**. construct validity: 방향 유효(self_acc 0.932) + 4요인과 직교(cos 0.117) |

## 5. 종합 주장 — 정확히 어디까지인가
주장 사다리로 위치를 못 박는다:
- **self-related** (자기관련 표상 존재) — **지지**
- **self-indexed** (대명사·nuisance와 구분되는 자기색인) — **조심스럽게 지지**
  (V17 대명사 독립 + V17.7 nuisance 독립)
- **self-specific** (privilege가 '나 전용') — **미지지** (V17.6 H_general: privilege는
  self/other 공통의 일반 epistemic-access 축)
- **self-model / 자기모델** — **미지지**
- **phenomenal / 체험** — **미주장** (H0)

한 줄: *base 모델 하나에서 자기관련 선형 구조가 사전학습만으로 생기고, held-out 전이되며,
대명사·6종 nuisance로 환원되지 않는다. 딱 거기까지 — self-특이도, 인과적 사용도, 체험도 아님.*

## 6. 한계 (그대로 싣는다)
- **인과-기능 미검증**: 선택적 steering 실패(specificity ~0.8, E4 미달). 읽기는 되나
  요인-선택적 쓰기는 안 됨(읽기/쓰기 해리). 표상적 증거지 인과적 사용 증거는 아님.
- **단일 층·단일 모델**: layer 12, Qwen2.5-1.5B base. 모델간 전이(E7) 미검증.
- **self-specific 미지지**: privilege는 일반 epistemic 축(V17.6).
- **nuisance 6축 한정**: 무한 통제 불가. syntax(self_acc 0.766)·topic(shuffle 0.781)
  방향은 상대적으로 약함. 자연(비합성) 코퍼스 전이 미검증.
- **인과 개입 미실시**: activation patching·necessity ablation 없음.
- **프롬프트 출처**: 확증 seed는 파일럿과 분리했으나 프롬프트뱅크는 단일 저자 생성.
- **적대적 검토**: 파이프라인 adversary는 stub(외부 AI 검토로 일부 보완, 여전히 부분적).

## 7. 향후
- **V18 무감독 발견**: 모델의 실제 축이 우리 4요인과 일치하는지(ARI). 결과 전 양방향 봉인.
- self-specific 정제(V17.6의 천장·어휘 교락 제거), 자연 코퍼스 전이, 인과 patching,
  모델간 전이(E7), 나머지 nuisance 축.

## 계보
V1–V15(윈도우, 이력 미회수) → V16-E(`experiments/v16_e/`, DGX 확증) → V17(E6/H0) →
V17.5/6/7(robustness). 상세 의사결정은 `docs/history/2026-07_v17_decision_log.md`.
