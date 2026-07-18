# V16 방향 전환 및 착수 보고서 v0.1

## 1. 기준자료 검증

- 기존 최종 연구 기준: 논문 v4.1 / Evidence Package v1.4
- Evidence Package v1.4 SHA-256: 738/738 통과
- V16 Handoff Package v1.1 SHA-256: 5/5 통과
- 기존 v4.1 및 Evidence Package는 수정하지 않고 동결 상태로 유지

## 2. 이전 인수인계서에서 유지한 내용

1. V1–V15의 직접 결론과 주장 경계
2. task 성능, 상류 정보, 중간 표현, 최종 정책을 분리하는 측정 원칙
3. pilot과 confirmatory의 엄격한 분리
4. 확증 전 exact source snapshot과 protocol 동결
5. gain intervention, activation swap, blind/random control
6. 실패 seed와 교차효과를 숨기지 않는 운영
7. 새 출력 디렉터리를 실제로 재생성·감사하는 재현 규칙
8. 기존 논문과 후속 연구 트랙의 분리

## 3. 변경한 내용

기존 V16의 embodied identity/ownership/agency 분해는 폐기하지 않고 후속 V19 후보로 이관했다. V16의 1차 질문은 인간의 ‘봄’과 더 직접적으로 연결되는 다음 구조로 변경했다.

- 1차 주의 A: 모델이 실제로 어느 정보 흐름을 선택하는가
- 메타주의 M: 모델이 자신의 현재 주의분포를 내부적으로 추정하는가
- 보고 접근 R: 내부 추정을 외부 보고로 읽을 수 있는가
- 피드백 F: 내부 추정이 다음 주의배치를 실제로 수정하는가
- 복귀: 이탈한 주의가 피드백 뒤 목표로 돌아오는가

따라서 V16은 identity 연구의 단순 연장이 아니라 `attention -> attention-state model -> closed-loop correction`의 기능적 구성과 이중 해리를 시험한다.

## 4. Pilot v0.1 설계

두 개의 대칭 정보 흐름 A/B가 각각 6-bit 현재상태와 전이를 갖는다. 목표 cue가 한 흐름을 지정하며, 모델의 1차 attention score 뒤에 독립적인 drift perturbation을 주입한다. 모델은 다음을 동시에 수행한다.

1. corrected attention으로 목표 흐름의 다음 상태 예측
2. 상류 token 표현에서 두 흐름의 세계상태 예측
3. 목표와 무관한 neutral 관계 과제
4. 자신의 pre-correction attention distribution 추정
5. 추정된 attention state를 사용한 2차 attention 복구

Full monitor는 attention mechanism의 semantic trace를 bottleneck projection으로 받아 현재 attention distribution을 재구성한다. Blind cue control은 같은 출력용량을 갖지만 attention trace와 attended summary를 받지 않는다.

Pilot seed는 16100–16102이며, full monitor와 blind control 각각 700 step을 학습했다. Pilot seed는 향후 확증실험에 사용하지 않는다.

## 5. Pilot v0.1 핵심 결과

강한 drift(4.5)에서 3-seed 평균:

| 지표 | Full monitor | Blind cue control |
|---|---:|---:|
| pre-focus accuracy | 0.353 | 0.353 |
| meta attended-stream accuracy | 1.000 | 0.647 |
| attention-distribution MAE | 0.0058 | 0.4765 |
| feedback-off recovery | 0.000 | 0.000 |
| feedback-on recovery | 0.742 | 0.000 |
| feedback-on policy accuracy | 0.908 | 0.688 |
| world accuracy | 1.000 | 1.000 |
| neutral accuracy | 0.9997 | 0.9923 |

Full monitor의 seed별 강한-drift recovery는 0.998, 0.822, 0.405였다. 방향은 일관되었지만 강도에는 seed 변이가 남았다.

### 보고 접근과 피드백 사용의 해리

- report gain=0: meta-report는 약 chance(0.499)로 붕괴
- report gain=1: meta-report=1.000, attention MAE=0.0058
- 같은 동안 recovery와 policy는 report gain에 거의 변하지 않음

이는 현재 구조에서 ‘내부 meta-state의 causal use’와 ‘그 상태의 외부 report access’를 분리할 수 있음을 보여준다. 단, report gain=0은 meta-state 자체의 부재가 아니라 외부 readout 차단이다.

### 피드백의 선택적 효과

- feedback gain은 pre-focus와 world prediction을 바꾸지 않음
- 강한 drift에서 feedback-on은 recovery를 평균 0.742 높이고 policy accuracy를 0.679에서 0.908로 높임
- blind cue controller는 큰 correction activation을 출력해도 recovery가 0이었음

즉 단순한 목표 cue만으로는 변동하는 attention drift를 복구하지 못했고, 실제 attention-state trace를 가진 monitor가 필요했다.

## 6. 판정

Pilot v0.1은 V16의 핵심 구성 가능성을 지지한다.

- 주의상태의 내부 추정: 성공
- report access와 causal feedback use의 분리: 성공
- drift detection 후 closed-loop recovery: 성공
- 상류 world/neutral 기능 보존: 성공
- cue-only matched control 대비 우위: 성공

그러나 아직 확증실험으로 동결할 단계는 아니다.

## 7. 동결 전 보완할 사항

1. feedback gain 0.25–0.75에서 회복이 거의 나타나지 않고 1.0에서 급격히 나타난다. score/calibration을 조정해 연속 dose-response 또는 명시적 threshold dynamics 중 어느 가설을 검증할지 결정해야 한다.
2. strong-drift recovery가 seed 16102에서 0.405로 낮다. 학습 안정성 또는 controller capacity를 파일럿에서 추가 점검해야 한다.
3. 현재 opposite-target monitor swap은 donor와 drift 구성이 혼재되어 인과해석이 불명확하다. 동일 world·동일 target에서 focused/off-target attention state만 다른 matched donor swap으로 교체해야 한다.
4. monitor는 명시적 attention trace(efference-copy 성격)를 입력받는다. 따라서 현재 결과는 자연발생 awareness가 아니라 attention-state modeling이 가능한 construction proof다.
5. 시간축의 `이탈 -> 감지 -> 복귀`는 아직 1회성 2-pass 구조다. V17에서 recurrent/long-horizon dynamics로 확장한다.

## 8. 다음 즉시 작업

Pilot v0.2에서 다음만 조정한다.

- matched focus/off-target monitor swap
- feedback calibration 및 seed 안정화
- correction efficiency와 불필요한 correction 비용 측정
- full monitor / no-trace / cue-only / random-trace 대조 구분

v0.2 통과 후 confirmatory protocol, 성공기준, 신규 seed, exact frozen source를 동결한다.
