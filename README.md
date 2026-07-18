# Ouroboros Research OS

> A deterministic, governor-gated research OS for computational self-representation
> experiments — protocol freeze, pre-registered confirmatory, two-axis claim ladder.

**[한국어](#한국어) · [English](#english)**

---

## 한국어

사전학습만 거친 LLM(Qwen2.5-1.5B **base**)에 자기관련 표상이 명시적 학습 없이
자연발생하는가를 검증하는 연구 운영체계. 결정론적 상태기계가 순서를 정하고 각
단계에서 필요한 코드/에이전트만 호출하며, 규칙기반 거버너가 2축 주장격자(기계론
E0–7 × 인간대응 H0–3)를 강제해 과대주장을 구조적으로 차단한다.

### 현재 상태
- **V17 (확증 완료)** — E6/H0. base 모델의 잔차스트림에, 대명사 표면축으로
  환원되지 않는 자기관련 표상(특히 인식적 특권 축)과 공유축 제거 후 잔존하는
  세부구조가 자연발생. 인간 체험 대응은 주장하지 않음(H0).
- **V17.5 (top-k 강건성)** — C3(공유축 제거 후 요인 잔여) 지지. 공유공간은 저차원.
- **V17.6 (self/other × privilege)** — privilege가 self-특이인지 일반 축인지 판별. 실행 대기.

### 구조
```
.
├── cli.py                 오케스트레이터 (loop / approve / adjudicate / status)
├── envelope.yaml          자율실행 범위
├── ouroboros/             ledger·machine·spec·freeze·audit·governor·agents
├── prompts/               에이전트 프롬프트
├── experiments/
│   ├── v17/               동결된 확증 실험 (E6/H0) — 수정 금지
│   ├── v17_5/             공유 부분공간 top-k 제거 (탐색적)
│   └── v17_6/             self/other × privilege 상호작용 (탐색적)
└── docs/                  HANDOFF · INTERP_GUARDRAILS · PATCH_NOTES · 파일럿 메모
```

### 실행 (요약)
```bash
source ~/.venv/bin/activate           # torch + transformers + numpy + pyyaml
python cli.py --db v17.db loop --experiment experiments/v17    # 확증 파이프라인
python experiments/v17_5/topk_removal.py --kmax 6              # 강건성
python experiments/v17_6/interaction.py                        # 상호작용
```
자세한 절차는 `docs/HANDOFF.md`, 실험별 `README.md` 참조.

### GitHub 업로드 (원클릭)
`gh auth login`을 한 번만 해둔 뒤: `./deploy_to_github.sh`
최초 실행은 public repo를 만들고 올리며, 이후 실행은 변경분을 커밋·푸시한다.

---

## English

A research operating system testing whether **self-related representations emerge
in a pretrained-only LLM** (Qwen2.5-1.5B base) without any explicit training.
A deterministic state machine drives the workflow, invoking only the code/agents
each stage needs, while a rule-based Governor enforces a two-axis claim ladder
(mechanistic E0–7 × human-correspondence H0–3) that structurally blocks
overclaiming: LLM agents can demote or flag claims but never promote them.

### Status
- **V17 (confirmatory, done)** — verdict **E6/H0**. In the residual stream of a
  base model, self-related representations that do not reduce to a surface
  pronoun axis (notably an *epistemic privilege* axis), plus factor-informative
  structure remaining after shared-axis removal, emerge from pretraining alone.
  Pre-registered criteria, 8 held-out seeds, 8/8 consistency. No claim of
  correspondence to human experience (H0); no claim of causal use or
  cross-model generality.
- **V17.5 (top-k robustness)** — the residual structure (C3) survives
  cross-validated top-k shared-subspace removal vs. random-subspace controls;
  the shared subspace itself is low-rank (~1–2).
- **V17.6 (self/other × privilege)** — a 2×2 minimal-pair interaction test asking
  whether the privilege axis is self-specific or a general epistemic-access
  axis. Pending execution; both interpretations pre-registered.

### Key design features
- **Human gates**: protocol freeze, human-experience loop, and claim promotion
  beyond E5 cannot pass without explicit human approval.
- **Protocol freeze**: code+prompt hashes are frozen before confirmatory runs;
  any post-freeze modification invalidates the result.
- **Interpretation guardrails** (`docs/INTERP_GUARDRAILS.md`): procedural rules
  against deflationary/inflationary bias on loaded topics (consciousness,
  emergence), incl. blinded external review and role-split adversarial review.

### Quick start
```bash
source ~/.venv/bin/activate           # torch + transformers + numpy + pyyaml
python cli.py --db v17.db loop --experiment experiments/v17    # confirmatory
python experiments/v17_5/topk_removal.py --kmax 6              # robustness
python experiments/v17_6/interaction.py                        # interaction
```
Most in-repo documentation is in Korean; experiment READMEs carry short English
abstracts, and this README is the bilingual entry point.

## License
MIT — see `LICENSE`.
