# Ouroboros Research OS — 세션 인수인계 (2026-07-18)

> 새 세션의 Claude에게: 이 문서와 함께 최신 zip(ouroboros_v0.8.zip)을 첨부해서
> 시작한다. 아래 맥락을 읽고 이어서 도와주면 된다. 사용자는 김병관, HL Klemove
> 사이버보안 담당이며 별도로 AI 자기표상/주객미분 연구를 수행 중. 직접적이고
> 정직한 평가를 선호하고, 반사적 hedging을 싫어한다. 한국어로 소통.

## 이 프로젝트가 무엇인가

AI 계산구조 연구와 인간 체험 연구를 잇는 이중 폐루프 연구 운영체계
("Ouroboros Research OS")의 v0.x 구현. 핵심 설계 원칙:

- **결정론적 상태기계가 중심** (에이전트 군집 아님). 상태기계가 순서를 정하고
  각 단계에서 필요한 에이전트/코드만 호출.
- **인간 게이트**: 프로토콜 동결(FREEZE_GATE), 인간 체험 개입(HUMAN_LOOP),
  E5 초과 주장 승격(claim_promotion)은 인간 승인 없이 통과 불가.
- **claim_ceiling / 2축 주장격자(기계론 E0-7 × 대응 H0-3)**: 결과가 좋아도
  과장 못 하게 Governor(규칙 기반 코드)가 승격 차단. LLM 에이전트는 증거수준을
  올릴 수 없고 강등/플래그만 가능(불변식 I2).
- **Protocol Freeze**: 코드+프롬프트 해시를 동결. 이후 변조 시 확증 결과 무효.

## 아키텍처 (파일 위치)

- `ouroboros/`: ledger.py(SQLite 원장=공식상태), machine.py(상태기계+게이트),
  spec.py(스키마검증), freeze.py(동결/무결성), audit.py(순수코드 채점),
  governor.py(규칙기반 주장승격), agents.py(LLM 래퍼, stub 폴백)
- `cli.py`: 오케스트레이터 (loop/approve/status)
- `envelope.yaml`: 자율실행 범위
- `experiments/v17/`, `experiments/v18/`: 실제 실험
- `docs/V17_pilot_findings.md`: 파일럿 발견 메모

## 실행 환경 (사용자 측)

- 장비: DGX Spark (ARM aarch64), 호스트 edgexpert-bbc5, GPU 사용가능(True)
- 경로: `~/ouroboros`, venv: `~/.venv` (`source ~/.venv/bin/activate`)
- 함정 기록: (1) PEP668 → venv 필수. (2) HF 캐시 권한 →
  `sudo chown -R $USER ~/.cache/huggingface`. (3) Triton 컴파일 →
  `sudo apt install python3.12-dev`. (4) 크래시 시 상태기계가 되감기지 않음 →
  재실행 전 `rm -f ~/ouroboros/v17.db; rm -rf .../results && mkdir .../results`.
  (5) 상대경로 리셋 실수 잦음 → 절대경로 권장.

## V17 실험: 무엇을 하는가

질문: 사전학습 LLM(Qwen2.5-1.5B **base**, HF rev 8faed761d45a263340a0528343f099c05c9a4323)에
자기관련 표상이 명시적 학습 없이 자연발생하는가?
요인 4개: identity(행위귀속)/beneficiary(수혜)/privilege(인식적특권)/concern(이해관계).
방법: probing(A군 학습→B군 전이) + activation steering(4×4 개입행렬) + 공용축 분해.

## V17 파일럿 결과 (2026-07-18, seed 0-2 + sweep + decompose@layer12)

1. **자연발생 확인**: cross-template probe 전이 0.75-1.0 (M0 표면cue 반박).
2. **공용축 존재**: 요인 혼동 비대각 0.74-0.85, 요인간 cos 0.38-0.72
   (beneficiary-concern 0.72 최대 → 요인 경계 재검토 필요).
3. **privilege는 별도 축**: 대명사축과 cos 0.20, 대명사 probe 판별 0.50(우연).
   양극 모두 1인칭인데도 전이 probe 1.0 → 파일럿 최대 발견.
4. **혼합 구조**: 공유축 제거 후 격차 +0.14→+0.30 (공유+요인특이).
5. **선택적 제어(E4) 실패**: steering specificity 0.33-1.06, 직교화해도 0.89.

## 승인된 결정 (사용자 승인 완료)

- V17 확증 가설 = 창발 3주장:
  C1 해독가능성 / C2 privilege 대명사 독립성 / C3 잔여구조(단, "경계가 4요인과
  일치"는 주장 안 함 — 약화됨).
- 사전 등록 기준(spec.yaml emergence_criteria):
  cross_template_probe_mean ≥0.75, _min_factor ≥0.65,
  privilege_person_probe ≤0.60(평균판정 scope:mean_only — 이항노이즈 보정),
  projection_gap ≥0.20. seed 8, 일관성 ≥0.75.
- 선택적 제어 + 요인 경계 검증은 **V18로 분리**.
  V18 1차 = 제어개선 아님. **모델 자기관련 축의 무감독 발견**(discover.py):
  차이벡터 → PCA → k-means → 요인/템플릿골격 라벨과 ARI 비교. H-A 검증.
  경합 steering 실패 가설: H-A 분류오류 / H-B 읽기쓰기해리 / H-C 방법미숙.

## 지금 위치 & 다음 할 일 (★ 여기서 이어서)

파일럿·프로토콜 개정·리포트 생성기까지 완료. **아직 확증실험 미실행.**
사용자가 edgexpert에서 실행할 순서:

```
# ① 파일럿 재실행 (프로토콜 개정됐으므로 새로) → FREEZE_GATE 정지
rm -f ~/ouroboros/v17.db
rm -rf ~/ouroboros/experiments/v17/results && mkdir ~/ouroboros/experiments/v17/results
cd ~/ouroboros && source ~/.venv/bin/activate
# config.yaml에 mock:false, layer_frac:0.43 확인
python cli.py --db v17.db loop --experiment experiments/v17
# ② confirmatory_metrics 4개가 기준 넘는지 확인 후 동결 승인
python cli.py --db v17.db approve --gate protocol_freeze --experiment V17-emergence-control --by 김병관
# ③ 확증실험 (seed 8) — 이게 "확증실험"
python cli.py --db v17.db loop --experiment experiments/v17
# (E6 승격 원하면) claim_promotion 게이트도 승인
# ④ 리포트
cd experiments/v17 && python report.py
```

**중요: 확증실험 = V17 본실험(③)이지 V18이 아니다.** V18은 그 다음 새 실험.

## 확증 후 남은 작업

1. ③④ 출력 검토 → verdict PASS/FAIL 확인 → E6 승격 여부 결정.
2. 재현 셋 보관: evidence zip + v17.db + sweep_report.json +
   decompose_report.json + V17_confirmatory_report.md → tar.gz.
3. 논문 초고. 구조 제안: Exploratory(파일럿) → Pre-registered Confirmatory
   → 한계(단일모델/요인경계 미검증→V18). 국문/영문 미정.
4. V18 파일럿: `cd experiments/v18 && python discover.py --layer 12`.
   관전포인트: beneficiary-concern이 한 군집으로 병합되는가(MA2 예측).

## 미해결 설계 이슈 (개선 후보)

- 크래시 시 상태 롤백 없음 (재실행 전 수동 리셋 필요).
- "미리 approve" 하면 게이트 무력화 가능 (파일럿 결과 없이 동결됐던 사고 1회).
- Adversary 등 LLM 에이전트가 stub 모드 (ANTHROPIC_API_KEY 미설정).
