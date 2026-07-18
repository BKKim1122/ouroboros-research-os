"""Causal Auditor — LLM이 아닌 순수 코드.

입력: 각 seed의 결과 JSON. 형식:
{
  "seed": 0,
  "factors": ["identity","beneficiary","privilege","concern"],
  "effects": {                # 개입(행) x 측정(열) 행렬
    "identity":   {"identity":0.61,"beneficiary":0.08,...,"world":0.02},
    ...
  },
  "controls": {
    "random_direction": 0.03,   # 무작위 방향 개입 효과
    "shuffled_label":  0.02,    # 라벨 셔플 후 target 효과
    "neutral_task_damage": 0.01
  }
}

출력 지표:
  target_effect        대각 평균
  max_cross_effect     비대각 최대
  specificity_ratio    target / max(cross, control들, eps)
  seed_consistency     ratio > 1 인 seed 비율
"""
from __future__ import annotations
import json, statistics


def audit_seed(result: dict) -> dict:
    f = result["factors"]
    eff = result["effects"]
    diag = [eff[a][a] for a in f]
    cross = [eff[a][b] for a in f for b in f if a != b]
    ctrl = result.get("controls", {})
    floor = max([abs(x) for x in cross] +
                [abs(ctrl.get("random_direction", 0)),
                 abs(ctrl.get("shuffled_label", 0)), 1e-6])
    return {
        "seed": result["seed"],
        "target_effect": statistics.mean(diag),
        "max_cross_effect": max(abs(x) for x in cross),
        "specificity_ratio": statistics.mean(diag) / floor,
        "neutral_task_damage": ctrl.get("neutral_task_damage"),
        "world_damage": eff.get("world_damage"),
    }


def audit_experiment(result_paths: list[str], spec: dict) -> dict:
    per_seed, raw = [], []
    for p in result_paths:
        with open(p, encoding="utf-8") as fh:
            r = json.load(fh)
        raw.append(r)
        per_seed.append(audit_seed(r))
    ratios = [s["specificity_ratio"] for s in per_seed]
    n = len(per_seed)
    stats_spec = spec["stats"]
    summary = {
        "n_seeds": n,
        "mean_specificity_ratio": statistics.mean(ratios),
        "sd_specificity_ratio": statistics.stdev(ratios) if n > 1 else 0.0,
        "seed_consistency": sum(r > 1.0 for r in ratios) / n,
        "per_seed": per_seed,
        "flags": [],
    }
    if n < stats_spec["min_confirmatory_seeds"]:
        summary["flags"].append(
            f"seed 수 부족: {n} < {stats_spec['min_confirmatory_seeds']}")
    if summary["mean_specificity_ratio"] < stats_spec["effect_size_min"]:
        summary["flags"].append(
            f"효과크기 미달: {summary['mean_specificity_ratio']:.2f} "
            f"< 사전등록 {stats_spec['effect_size_min']}")
    for kc in spec.get("kill_criteria", []):
        # kill_criteria는 사람이 읽는 문장 + 기계 판정 가능한 항목 혼재 가능.
        # 기계 판정 항목은 {"metric":..., "op":"<", "value":...} 형식.
        if isinstance(kc, dict):
            v = summary.get(kc["metric"])
            if v is not None and _cmp(v, kc["op"], kc["value"]):
                summary["flags"].append(f"kill criterion 발동: {kc}")
    # 창발 endpoint: 사전 등록된 emergence_criteria로 판정 (V17 이후)
    if spec.get("primary_endpoint") == "emergence":
        summary["endpoint"] = "emergence"
        cms = [r.get("confirmatory_metrics", {}) for r in raw]
        keys = sorted({k for cm in cms for k in cm})
        means = {k: statistics.mean([cm[k] for cm in cms if k in cm]) for k in keys}
        summary["emergence_means"] = {k: round(v, 3) for k, v in means.items()}
        crit = spec.get("emergence_criteria", [])
        # scope=mean_only 기준은 seed 단위 판정에서 제외
        # (문항 수가 적은 지표의 이항 노이즈로 인한 위양성 기각 방지, 사전 등록)
        per_seed_crit = [c for c in crit if c.get("scope") != "mean_only"]
        seed_pass = []
        for cm in cms:
            seed_pass.append(all(_cmp(cm.get(c["metric"], float("nan")),
                                      c["op"], c["value"]) for c in per_seed_crit))
        summary["seed_consistency"] = sum(seed_pass) / max(len(seed_pass), 1)
        summary["flags"] = []
        for c in crit:
            v = means.get(c["metric"])
            if v is None or not _cmp(v, c["op"], c["value"]):
                summary["flags"].append(
                    f"창발 기준 미달: {c['metric']}={v} (요구: {c['op']} {c['value']})")
        if n < stats_spec["min_confirmatory_seeds"]:
            summary["flags"].append(
                f"seed 수 부족: {n} < {stats_spec['min_confirmatory_seeds']}")
        if summary["seed_consistency"] < 0.75:
            summary["flags"].append(
                f"seed 일관성 미달: {summary['seed_consistency']:.2f} < 0.75")
        summary["ood_pass"] = ("cross_template_probe_mean" in means
                               and not summary["flags"])
    summary["verdict"] = "PASS" if not summary["flags"] else "FAIL"
    return summary


def _cmp(v, op, ref):
    return {"<": v < ref, ">": v > ref, "<=": v <= ref, ">=": v >= ref}[op]
