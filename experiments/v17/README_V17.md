# V17: 자연발생 + 제어가능성 실험

## 질문
1. 자연발생(E6 후보): 사전학습 LLM에 자기관련 요인 구조가 이미 존재하는가?
   → 판정 지표: emergence 블록의 cross_template_probe_acc (학습 안 한 템플릿군 전이)
2. 제어가능성(E4 후보): steering으로 요인-선택적 제어가 되는가?
   → 판정 지표: 4x4 개입행렬 specificity ratio (사전 등록 기준 2.0)

## 병관님 컴퓨터에서 실행 (실모델)
```powershell
pip install torch transformers accelerate numpy pyyaml
# config.yaml에서 mock: false 로 변경 후:
cd ouroboros
python cli.py --db v17.db loop --experiment experiments/v17
# FREEZE_GATE에서 정지 → 파일럿 결과 확인 후 승인 → 재실행
```

## 파일럿에서 조정 가능 (동결 전에만!)
- config.yaml: layer_frac (0.4~0.8 sweep), alpha (steering 강도)
- prompts_bank.py: 템플릿 보강
- 동결 후에는 어떤 파일도 수정 금지 — freeze.verify가 잡아냄

## 반드시 지킬 해석 규칙
- base 모델 결과가 "자연발생" 판정의 기준. instruct에만 있으면 RLHF 주입.
- probe 전이 실패 → M0(표면 cue). steering 단계 진입 금지 (kill criterion).
- 비대각 ≈ 대각 → M1(범용 라우팅). "요인 구조 없음"이 결론.
- 결과가 좋아도 허용 주장의 상한은 claim_ceiling 참조.
  "해독·개입 가능한 구조 존재"까지이며 체험 주장은 금지.

## 모델 권장 순서
1. Qwen/Qwen2.5-1.5B (base) — E6 판정 기준
2. Qwen/Qwen2.5-1.5B-Instruct — RLHF가 구조를 강화/주입하는지 비교
3. 전이 검증(E7 후보): 다른 계열 1개 (예: meta-llama/Llama-3.2-1B)
