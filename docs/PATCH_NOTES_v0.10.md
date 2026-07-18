# v0.10 패치 노트 (2026-07-18) — V17 확증 실행 전 수정

이전 세션의 handoff 대비, V17 확증실험을 실제로 유효하게 돌리기 위한 4개 수정.
mock 파이프라인으로 end-to-end 검증 완료(가드 정지 / seed 분리 / 감사 / adjudicate / 리포트).

## 1. 확증 단계 mock 금지 가드  (cli.py `_confirmatory`)
mock 백엔드는 사전등록 기준 4개를 무의미하게 통과해 verdict=PASS·E5까지 판정된다.
확증 진입 시 config.yaml의 `mock`을 검사해 true면 seed 실행 전에 중단한다.
- 검증: mock:true로 확증 진입 시 seed 생성 없이 즉시 정지 확인.

## 2. 파일럿/확증 seed 분리  (spec.yaml, cli.py `run_seeds`, freeze.py, report.py)
기존엔 확증이 `seeds` 전체(파일럿 [0,1,2] 포함)를 돌려, 결정론적 seed 특성상
확증 8개 중 3개가 기준을 튜닝한 탐색 데이터의 재현이었다(freeze엔 pilot_excluded:true라
적혀 있었으나 코드는 미구현).
- spec.yaml: `pilot_seeds:[0,1,2]`, `confirmatory_seeds:[3..10]`(신규 8개), `seeds`는 pool.
- cli.run_seeds: phase별로 위 집합 사용.
- freeze.py: 동결 기록에 두 집합 명시.
- report.py: 파일럿 seed 파일이 results/에 함께 있어도 확증 집합만 집계.
- 검증: 감사·리포트가 seed [3..10] 8개만 집계, 파일럿 0,1,2 미포함 확인.

## 3. E6 재판정용 `adjudicate` 서브커맨드  (cli.py)
loop는 confirmatory→adjudication을 한 번에 지나가 확증 수치를 본 뒤 E6를 결정할
틈이 없었다(E6를 받으려면 확증 전에 미리 승인해야 하는 하자). 판정만 독립 실행하는
`python cli.py adjudicate --experiment ...` 추가. audit_summary.json과 게이트 상태만
읽고 FSM은 건드리지 않는다.
- 검증: 확증 후 claim_promotion 승인 → adjudicate → E5에서 E6로 승격 확인.

## 4. 모델 revision 고정  (config.yaml, run_seed.py `HFBackend`, report.py)
E6는 특정 모델에 매인 주장인데 `from_pretrained(model_name)`이 캐시/최신을 로드했다.
config.yaml에 `revision` 추가(handoff의 8faed761…), HFBackend가 tokenizer·model 모두에
`revision=` 전달, 결과 JSON에 revision 기록, report는 config 고정값을 진실로 삼고
캐시 추정치와 불일치 시 경고 병기.

## config.yaml 최종값 (동결 전 확정 완료)
mock:false / model:Qwen2.5-1.5B / revision:8faed761… / layer_frac:0.43 / alpha:8.0
- layer_frac 0.43 = int(28*0.43)=layer 12. projection_gap 기준을 잰 층이라 변경 금지.
- alpha는 emergence 판정과 무관(steering 기술통계). 보고값으로만 의미.

## 남은(미패치) 설계 이슈 — 알고 넘어갈 것
- freeze는 experiment_dir + prompts만 해시한다. ouroboros/ 패키지(audit.py,
  governor.py 등 '분석 코드')는 동결 대상에 없다. 이번 확증엔 코드 수정을 먼저
  끝냈으니 실무상 문제 없으나, 엄밀히는 분석 코드도 동결 범위에 넣는 게 맞다.
- 크래시 시 FSM 롤백 없음 → 재실행 전 v17.db 삭제 + results/ 비우기(수동).
- 파일럿 실행 전에 approve하면 게이트가 무력화된다(순서만 지키면 무해).
- adversary는 stub 모드(ANTHROPIC_API_KEY 미설정) → 적대적 검토 미수행, 불변식 I3 no-op.
- envelope.allowed_code_paths는 v16으로 낡았고 코드가 참조하지 않음(문서 수준).
- CLI가 찍는 specificity 값은 emergence 판정과 무관(기술통계). verdict만 보면 됨.
