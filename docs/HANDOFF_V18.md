# V18 인수인계서 — 다른 세션을 위한 상세 안내

> 이 문서는 **새 Claude 세션**에게 V18을 일관성 있게 맡기기 위한 인수인계서다.
> 사용자: 김병관(GitHub `BKKim1122`, edgexpert-bbc5 / DGX Spark). 한국어로 소통.
> 읽는 순서: 이 문서 → `docs/V17_SYNTHESIS_KO.md` → `docs/INTERP_GUARDRAILS.md`
> → `docs/history/2026-07_v17_decision_log.md`.

---

## 0. 한 줄
V17까지 "사전학습만으로 자기관련 표상이 생긴다(E6/H0)"를 확증하고 robustness 3종을
끝냈다. V18은 **무감독 발견** — 우리가 손으로 정의한 4요인이 모델이 *실제로* 쓰는 축과
일치하는지를, 결과를 보기 전에 판정 기준을 봉인한 채 검증한다.

## 1. 프로젝트를 한눈에 (무엇이 확정됐나)
- 모델: Qwen2.5-1.5B **base**(사전학습만, instruct 아님), rev 8faed761, layer 12.
- 요인 4개: identity(행위귀속)·beneficiary(수혜)·privilege(인식적 특권)·concern(이해관계).
- **V17 확증 E6/H0**: 템플릿군 A→B 전이 mean 0.968 / min_factor 0.907, 대명사 probe로는
  privilege 판별 0.5(우연)인데 전이로는 판별됨, gap 0.397, 8/8. → 자기관련 표상이
  대명사 표면축으로 환원되지 않고 자연발생.
- **robustness 3종**:
  - V17.5(top-k): 공유축 저차원(rank 1~2)이나 제거 후 요인 잔여(C3) 실재.
  - V17.6(self/other): privilege는 self-특이 아닌 **일반 epistemic 축**(H_general).
  - V17.7(nuisance): 4요인 전이는 6 nuisance로 환원 안 됨(H_beyond) + 방향 construct
    validity 확인(self_acc 0.932, 4요인과 cos 0.117).
- **주장 사다리(현재 위치)**: self-related·self-indexed **지지** / self-specific **미지지**
  / self-model·phenomenal **미주장(H0)**.

## 2. ★ 반드시 계승할 방법론 규율
이 프로젝트의 신뢰도는 결과가 아니라 절차에서 나온다. V18도 동일하게:
- **결정론적 상태기계 + 거버너 + 2축 주장격자(E0–7 × H0–3)**: LLM은 등급을 올릴 수
  없고 강등/플래그만. 승격은 인간 게이트.
- **Protocol Freeze**: 확증 전 코드+프롬프트 해시 동결. 변조 시 결과 무효.
- **사전등록**: kill criterion과 **양방향 해석**을 *결과 보기 전에* spec.yaml에 봉인.
  파일럿/확증 seed 분리.

## 3. ★★ 편향 방지 — 가장 중요 (반드시 읽을 것)
사용자가 반복해서 강조한 핵심이다. **너(다른 세션의 Claude)도 의식·자아 주제에서
편향이 있다** — 결과가 자아 긍정이면 과하게 의심하고, 자아 부정이어도 "아직 모른다"며
판정을 흔드는 경향. 방향과 무관하게 "자아 주제면 유독 반을 더 든다". 이 프로젝트에서
이건 실험 결과를 왜곡하는 실질 위험이다. 그래서:
- 매 실험의 **kill criterion·양방향 해석을 결과 전에 봉인**하고, 실행 후엔 **대입만** 한다.
- 통제할 nuisance/조건 목록을 **미리 유한하게 고정** — 실행 후 "하나 더"는 별도 후속
  실험이지 현 판정에 소급 금지(**goalpost moving 차단**).
- 매 판정에 **"이게 자아 주제가 아니었어도 같은 기준·같은 무게를 적용했을까?"**를
  명시적으로 답해 사용자가 검증하게 한다.
- 자아 폄하(디플레이션)든 과대(인플레이션)든 **환영도 축소도 없이** 봉인 기준대로 보고.
- 상세 규칙: `docs/INTERP_GUARDRAILS.md`(R1–R5, 외부검토 X1–X5, 자기점검 3문항).

## 4. 현재 자산 상태
- **GitHub**(public): `https://github.com/BKKim1122/ouroboros-research-os`
  - `experiments/v17/` = **동결**. `v17_5/6/7` = robustness. `v16_e/` = 이전 세대.
  - `docs/` = HANDOFF · INTERP_GUARDRAILS · PATCH_NOTES · V17_SYNTHESIS_KO · history/.
  - 배포: `bash deploy_to_github.sh` (gh 인증돼 있음). push는 **사용자가** 실행.
- **Notion**: "AI 연구 › Ouroboros Research OS — V17" 아래 Overview/Guardrails/
  Experiments(V17·V17.5·V17.6)/Results & Verdicts/Handoff. 최신 판정은 Results & Verdicts.
- **실행 환경**: edgexpert-bbc5, `~/ouroboros`, venv `~/.venv`(torch+transformers+numpy+pyyaml).
  `python`이 없으면 venv 활성화(`source ~/.venv/bin/activate`) 후 `python`, 또는 `python3`.

## 5. V18 목표와 설계 원칙
**질문**: 우리가 손으로 정의한 4요인이 모델이 *실제로* 조직하는 축과 일치하는가?
(V17까지는 "우리가 정의한 요인이 인코딩돼 있나"를 물었다. V18은 역으로 "모델의 자생적
구조가 우리 요인과 맞나"를 묻는다 — 확증편향의 마지막 출구를 막는 단계.)

**접근(초안, 확정은 사용자와)**:
- 자기관련 자극의 활성치를 **무감독**으로 군집/분해(예: 요인 라벨 없이 클러스터링 또는
  희소 분해) → 발견된 축과 4요인 라벨의 일치도를 **ARI(조정 랜드 지수)** 등으로 측정.
- **결과 보기 전에 양방향 봉인**: "높은 ARI = 모델 축이 4요인과 정합(E 강화)",
  "낮은 ARI = 4요인은 우리 부과물이고 모델은 다르게 조직(주장 후퇴)". 임계값 사전 확정.
- 파일럿 스크립트 `experiments/v18/discover.py`와 `spec.yaml`이 이미 있으나 **미실행** —
  먼저 읽고, 위 규율대로 kill criterion을 spec에 박은 뒤 진행.

**V18 파일럿에서 확증 전 반드시 고칠 것** (세션 검토로 확인된 실질 gap — `METHODS_STANDARD_KO.md` 준수):
- `discover.py`에 **`--mock` 확증 차단**. MockBackend는 truth로 활성치를 직접 만들어
  ARI≈1.0이 순환논증으로 나온다 → "MA1 대승리"로 **오독 금지**(과학적 의미 0).
- spec의 kill이 참조하는 **null(공분산 매칭 셔플)·shuffled_labels가 코드에 미구현** →
  먼저 구현. null 없이는 봉인 자체가 불가능.
- **봉인은 null-상대 백분위로**(표준 §1): 실루엣이 모든 k에서 셔플 null의 95백분위를 못
  넘으면 MA3(구조 없음); ARI(요인)가 라벨순열 null 99백분위 초과 **AND** ARI(요인)−
  ARI(템플릿) ≥ 사전 margin이면 MA1 지지. 절대값 대신 null 상대로(파일럿 값 엿보기 차단).
- **steering 예측은 파일럿에서 제외**: V17에서 steering은 실패(E4)했으므로 "발견 축이
  라벨 축보다 낫다"를 예측으로 박지 말고 별도 후속(H-C)으로 분리.
- **seed 분리**: 파일럿 {0,1,2}(파이프라인·null 분포) / 확증 신규 disjoint(예 20–27).
- **거버너에 `endpoint: discovery` 판정 경로 추가** 후 확증 승격(기존 emergence 경로에
  억지 매핑 금지).

## 6. 하지 말 것 (오해·훼손 방지)
- ❌ `experiments/v17/` 수정·삭제 — 동결. 건드리면 freeze 검증이 깨져 확증 무효.
- ❌ `git push --force`/`reset --hard`/`rebase`로 히스토리 변경.
- ❌ 결과를 본 뒤 kill criterion·nuisance 목록·해석을 바꾸기(goalpost moving).
- ❌ 자격증명(토큰) 취급 — push는 사용자가. 너는 파일·명령을 산출하고 안내만.
- ❌ 등급을 LLM 판단으로 승격 — 승격은 인간 게이트.

## 7. 시작하는 법
```bash
cd ~/ouroboros
git pull --rebase origin main          # 최신 동기화
source ~/.venv/bin/activate
cat experiments/v18/spec.yaml           # 기존 파일럿 확인
# → 위 3·5절대로 kill criterion·양방향 해석을 spec에 봉인한 뒤 discover.py 정비/실행
```

## 8. 참조 문서
- `docs/METHODS_STANDARD_KO.md` — **실험 봉인·리포트 표준 (필독)**. 세션이 바뀌어도 같은
  방식·같은 리포트가 나오게 하는 규약. V18 설계·판정 전에 반드시 따른다.
- `docs/INTERP_GUARDRAILS.md` — **편향 방지 규칙 (필독)**. R1–R5, V18 사전봉인, 자기점검 3문항.
- `docs/V17_SYNTHESIS_KO.md` — 통합 결과·주장 사다리·한계.
- `docs/PAPER_KO_draft.md` — 한글 논문 초안, 전체 서사.
- `docs/history/2026-07_v17_decision_log.md` — 왜 이렇게 쪼갰나(Decision B가 곧 V18 질문).
- `docs/HANDOFF.md` — 상태기계·freeze 운영.

## 9. 소스 정합 주의
로컬(zip)과 원격(GitHub)이 서로 다른 파일을 앞설 수 있다. `git pull --rebase` 전에 로컬
신규 문서가 미커밋인지 확인하고(유실·충돌 방지), 배포 zip은 필독 문서를 포함한 완전
스냅샷인지 확인한다(표준 §7). 이 v0.19 스냅샷은 원격 필독 문서 + 로컬 최신을 통합한
완전본이다.
