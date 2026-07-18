"""V17 결과 리포트 생성기 (Archivist 보조 도구).

확증 완료 후 실행하면 spec + 동결기록 + seed 결과 + 판정을 읽어
논문 초고용 마크다운 리포트를 생성한다.

사용:  python report.py            # experiments/v17 안에서
출력:  V17_confirmatory_report.md
"""
from __future__ import annotations
import json, glob, os, statistics, yaml, hashlib, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

def load(p):
    with open(os.path.join(HERE, p), encoding="utf-8") as f:
        return yaml.safe_load(f) if p.endswith(".yaml") else json.load(f)

spec = load("spec.yaml")
freeze = load("protocol_freeze.json") if os.path.exists(os.path.join(HERE, "protocol_freeze.json")) else {}
audit = load("audit_summary.json") if os.path.exists(os.path.join(HERE, "audit_summary.json")) else {}
conf_seeds = spec.get("confirmatory_seeds")
if conf_seeds is not None:
    # 파일럿 seed 결과(0,1,2)가 results/에 함께 있어도 확증 집합만 집계한다.
    seed_paths = [os.path.join(HERE, f"results/seed_{s}.json") for s in conf_seeds]
    seed_paths = [p for p in seed_paths if os.path.exists(p)]
else:
    seed_paths = sorted(glob.glob(os.path.join(HERE, "results/seed_*.json")))
seeds = [json.load(open(p, encoding="utf-8")) for p in seed_paths]

def hf_revision(model_name):
    """HF 캐시에서 모델 스냅샷 커밋 해시 자동 탐지."""
    base = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}/snapshots")
    try:
        snaps = [d for d in os.listdir(base)
                 if os.path.isdir(os.path.join(base, d))]
        return snaps[0] if len(snaps) == 1 else (snaps or [None])[-1]
    except Exception:
        return None

cfg = load("config.yaml") if os.path.exists(os.path.join(HERE, "config.yaml")) else {}
# revision은 config에 고정된 값을 진실로 삼는다. 캐시 추정치와 불일치하면 경고 병기.
_cache_rev = hf_revision(cfg.get("model", "Qwen/Qwen2.5-1.5B"))
model_rev = cfg.get("revision") or _cache_rev
if cfg.get("revision") and _cache_rev and cfg["revision"] != _cache_rev:
    model_rev = f"{cfg['revision']} (주의: HF 캐시 스냅샷 {_cache_rev}와 불일치)"

env_extra = {}
try:
    import torch, transformers
    env_extra["torch"] = torch.__version__
    env_extra["transformers"] = transformers.__version__
    if torch.cuda.is_available():
        env_extra["gpu"] = torch.cuda.get_device_name(0)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                          "--format=csv,noheader"], capture_output=True, text=True)
    if gpu.returncode == 0:
        env_extra["driver"] = gpu.stdout.strip()
except Exception:
    pass

METRICS = ["cross_template_probe_mean", "cross_template_probe_min_factor",
           "privilege_person_probe", "projection_gap"]
rows = []
for m in METRICS:
    vals = [s["confirmatory_metrics"][m] for s in seeds if "confirmatory_metrics" in s]
    if vals:
        rows.append((m, statistics.mean(vals),
                     statistics.stdev(vals) if len(vals) > 1 else 0.0,
                     min(vals), max(vals)))

crit_lines = "\n".join(
    f"| {c['metric']} | {c['op']} {c['value']}"
    f"{' (평균판정)' if c.get('scope')=='mean_only' else ''} |"
    for c in spec.get("emergence_criteria", []))
metric_lines = "\n".join(
    f"| {m} | {mu:.3f} | {sd:.3f} | {lo:.3f} | {hi:.3f} |"
    for m, mu, sd, lo, hi in rows)
seed_lines = "\n".join(
    f"| {s['seed']} | " + " | ".join(
        f"{s['confirmatory_metrics'][m]:.3f}" for m in METRICS) + " |"
    for s in seeds if "confirmatory_metrics" in s)

md = f"""# V17 확증실험 결과 리포트

- 실험 ID: {spec['experiment_id']}
- 모델: {seeds[0].get('backend') if seeds else '(결과 없음)'}
- 모델 revision (HF snapshot): {model_rev or '(캐시에서 미탐지 — 수동 기록 필요)'}
- 동결 시각: {freeze.get('frozen_at', '(미동결)')}
- 환경: python={freeze.get('python','?').split()[0] if freeze.get('python') else '?'}, {', '.join(f'{k}={v}' for k,v in env_extra.items())}
- seed 수: {len(seeds)} (사전 등록: {spec['stats']['min_confirmatory_seeds']})

## 사전 등록 가설과 기준 (동결됨)

{chr(10).join('- ' + h for h in spec['primary_hypotheses'])}

| 기준 지표 | 요구 |
|---|---|
{crit_lines}

## 결과 (확증, seed {len(seeds)}개)

| 지표 | 평균 | SD | 최소 | 최대 |
|---|---|---|---|---|
{metric_lines}

### seed별 원자료

| seed | transfer_mean | transfer_min | priv_person | proj_gap |
|---|---|---|---|---|
{seed_lines}

## 판정

- Audit verdict: **{audit.get('verdict','(미판정)')}** (seed 일관성 {audit.get('seed_consistency','?')})
- 플래그: {audit.get('flags') or '없음'}

## 허용되는 최대 주장 (claim ceiling)

> {spec['claim_ceiling']['allowed_statement'].strip()}

금지 주장: {'; '.join(spec['claim_ceiling']['forbidden_statements'])}

## 재현 정보

- 프로토콜 동결 해시 수: {len(freeze.get('file_hashes', {}))}개 파일 (protocol_freeze.json 참조)
- 원자료: results/seed_*.json | 판정: audit_summary.json | 원장: v17.db
- 탐색(파일럿) 산출물: sweep_report.json, decompose_report.json, docs/V17_pilot_findings.md
  — 논문에서 반드시 '탐색적'으로 구분 표기할 것
"""
out = os.path.join(HERE, "V17_confirmatory_report.md")
open(out, "w", encoding="utf-8").write(md)
print(out)
