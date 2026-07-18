"""실험 사양(spec.yaml) 검증.

Experiment Architect(LLM 또는 인간)의 출력은 자유 서술이 아니라
이 스키마를 만족하는 YAML이어야 한다. claim_ceiling이 없으면 무효.
"""
from __future__ import annotations
import yaml

REQUIRED = [
    "experiment_id", "question", "competing_models",
    "discriminating_predictions", "kill_criteria", "controls",
    "seeds", "metrics", "analysis_plan", "claim_ceiling",
    "stats",  # 사전 등록 통계 사양
]

STATS_REQUIRED = ["min_confirmatory_seeds", "effect_size_min",
                  "multiple_comparison", "gpu_tolerance"]

CEILING_REQUIRED = ["max_e_level", "max_h_level",
                    "allowed_statement", "forbidden_statements"]


def load_spec(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    errors = [k for k in REQUIRED if k not in spec]
    if "claim_ceiling" in spec:
        errors += [f"claim_ceiling.{k}" for k in CEILING_REQUIRED
                   if k not in spec["claim_ceiling"]]
    if "stats" in spec:
        errors += [f"stats.{k}" for k in STATS_REQUIRED
                   if k not in spec["stats"]]
    if "competing_models" in spec and len(spec["competing_models"]) < 2:
        errors.append("competing_models: 최소 2개 (판별 대상이 없는 실험은 무효)")
    if errors:
        raise ValueError("spec 검증 실패: " + ", ".join(errors))
    return spec
