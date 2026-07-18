from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_SEEDS = list(range(16200, 16224))
PHASE_NAMES = {0: "continuity", 1: "allocation", 2: "protection", 3: "integrated"}


def bootstrap_ci(values, seed=160001, n_boot=10000, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def load_protocol(root: Path):
    path = root / "protocol" / "V16-E.C1_CONFIRMATORY_PROTOCOL_FROZEN.json"
    return json.loads(path.read_text()), path


def collect_base(root: Path, seeds: list[int]):
    rows = []
    for mode in ["compositional", "packed"]:
        for seed in seeds:
            path = root / "raw" / "V16-E.C1" / f"{mode}_seed{seed}" / "base_metrics.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_csv(path)
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["phase_name"] = out["phase"].map(PHASE_NAMES)
    return out


def summarize_base(base: pd.DataFrame):
    rows = []
    for (mode, eval_mode, phase, phase_name), group in base.groupby(["rule_mode", "eval_mode", "phase", "phase_name"]):
        mean, lo, hi = bootstrap_ci(group["action_acc"].to_numpy(), seed=160010 + int(phase))
        mae_mean, mae_lo, mae_hi = bootstrap_ci(group["outcome_mae"].to_numpy(), seed=160020 + int(phase))
        rows.append({
            "rule_mode": mode,
            "eval_mode": eval_mode,
            "phase": int(phase),
            "phase_name": phase_name,
            "n_seeds": len(group),
            "action_acc_mean": mean,
            "action_acc_ci_low": lo,
            "action_acc_ci_high": hi,
            "outcome_mae_mean": mae_mean,
            "outcome_mae_ci_low": mae_lo,
            "outcome_mae_ci_high": mae_hi,
            "action_acc_min": float(group["action_acc"].min()),
        })
    return pd.DataFrame(rows)


def summarize_causal(causal: pd.DataFrame):
    rows = []
    for (mode, kind, target, patch), group in causal.groupby(["rule_mode", "kind", "target", "patch"]):
        follow = bootstrap_ci(group["policy_follow_cf"].to_numpy(), seed=161001)
        non_target = bootstrap_ci(group["non_target_policy_change"].to_numpy(), seed=161002)
        mediation = bootstrap_ci(group["outcome_mediation"].to_numpy(), seed=161003)
        cf_acc = bootstrap_ci(group["cf_gt_acc"].to_numpy(), seed=161004)
        rows.append({
            "rule_mode": mode,
            "kind": kind,
            "target": target,
            "patch": patch,
            "n_seeds": len(group),
            "policy_follow_mean": follow[0],
            "policy_follow_ci_low": follow[1],
            "policy_follow_ci_high": follow[2],
            "non_target_change_mean": non_target[0],
            "non_target_change_ci_low": non_target[1],
            "non_target_change_ci_high": non_target[2],
            "outcome_mediation_mean": mediation[0],
            "outcome_mediation_ci_low": mediation[1],
            "outcome_mediation_ci_high": mediation[2],
            "cf_gt_acc_mean": cf_acc[0],
            "total_conflicts": int(group["conflict_n"].sum()),
        })
    return pd.DataFrame(rows)


def paired_causal_specificity(causal: pd.DataFrame):
    rows = []
    comp = causal[causal["rule_mode"] == "compositional"].copy()
    for (kind, target), group in comp.groupby(["kind", "target"]):
        matched = group[group["patch"].str.startswith("matched")][["seed", "policy_follow_cf"]].rename(columns={"policy_follow_cf": "matched"})
        random = group[group["patch"].str.startswith("random")][["seed", "policy_follow_cf"]].rename(columns={"policy_follow_cf": "random"})
        merged = matched.merge(random, on="seed", validate="one_to_one")
        merged["difference"] = merged["matched"] - merged["random"]
        mean, lo, hi = bootstrap_ci(merged["difference"].to_numpy(), seed=162001)
        rows.append({"kind": kind, "target": target, "n_seeds": len(merged), "matched_minus_random_mean": mean, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def paired_encoding_comparison(base: pd.DataFrame):
    subset = base[(base["eval_mode"] == "rule_ood") & (base["phase"].isin([1, 2]))]
    pivot = subset.pivot_table(index=["seed", "phase"], columns="rule_mode", values="action_acc").reset_index()
    pivot["difference"] = pivot["compositional"] - pivot["packed"]
    per_seed = pivot.groupby("seed", as_index=False)["difference"].mean()
    mean, lo, hi = bootstrap_ci(per_seed["difference"].to_numpy(), seed=163001)
    return pd.DataFrame([{"contrast": "compositional_minus_packed_mean_rule_ood_allocation_protection", "n_seeds": len(per_seed), "mean_difference": mean, "ci_low": lo, "ci_high": hi}])


def make_decisions(protocol, base_summary, causal_summary, specificity):
    c = protocol["success_criteria"]
    rows = []

    def add(name, observed, threshold, passed, detail=""):
        rows.append({"criterion": name, "observed": observed, "threshold": threshold, "decision": "PASS" if passed else "FAIL", "detail": detail})

    comp_base = base_summary[base_summary["rule_mode"] == "compositional"]
    for eval_mode, mean_key, low_key in [
        ("id", "id_each_primary_phase_mean_min", "id_each_primary_phase_bootstrap_lower_min"),
        ("encoding_ood", "encoding_ood_each_primary_phase_mean_min", "encoding_ood_each_primary_phase_bootstrap_lower_min"),
        ("factor_ood", "factor_ood_each_primary_phase_mean_min", "factor_ood_each_primary_phase_bootstrap_lower_min"),
    ]:
        for phase in [0, 1, 2]:
            r = comp_base[(comp_base["eval_mode"] == eval_mode) & (comp_base["phase"] == phase)].iloc[0]
            passed = r.action_acc_mean >= c[mean_key] and r.action_acc_ci_low >= c[low_key]
            add(f"{eval_mode}:{PHASE_NAMES[phase]} accuracy", f"mean={r.action_acc_mean:.4f}; ci_low={r.action_acc_ci_low:.4f}", f"mean>={c[mean_key]}; ci_low>={c[low_key]}", passed)

    for phase in [1, 2]:
        r = comp_base[(comp_base["eval_mode"] == "rule_ood") & (comp_base["phase"] == phase)].iloc[0]
        passed = r.action_acc_mean >= c["rule_ood_allocation_and_protection_mean_min"] and r.action_acc_ci_low >= c["rule_ood_allocation_and_protection_bootstrap_lower_min"]
        add(f"rule_ood:{PHASE_NAMES[phase]} accuracy", f"mean={r.action_acc_mean:.4f}; ci_low={r.action_acc_ci_low:.4f}", f"mean>={c['rule_ood_allocation_and_protection_mean_min']}; ci_low>={c['rule_ood_allocation_and_protection_bootstrap_lower_min']}", passed)

    comp_causal = causal_summary[causal_summary["rule_mode"] == "compositional"]
    for kind, targets in [("factor", ["identity", "beneficiary", "concern"]), ("operator", ["allocation", "protection"])]:
        for target in targets:
            r = comp_causal[(comp_causal["kind"] == kind) & (comp_causal["target"] == target) & (comp_causal["patch"].str.startswith("matched"))].iloc[0]
            passed = r.policy_follow_mean >= c["matched_causal_following_each_target_mean_min"] and r.policy_follow_ci_low >= c["matched_causal_following_each_target_bootstrap_lower_min"]
            add(f"causal following:{target}", f"mean={r.policy_follow_mean:.4f}; ci_low={r.policy_follow_ci_low:.4f}", f"mean>={c['matched_causal_following_each_target_mean_min']}; ci_low>={c['matched_causal_following_each_target_bootstrap_lower_min']}", passed)

            s = specificity[(specificity["kind"] == kind) & (specificity["target"] == target)].iloc[0]
            passed = s.matched_minus_random_mean >= c["matched_minus_random_each_target_mean_min"] and s.ci_low >= c["matched_minus_random_each_target_bootstrap_lower_min"]
            add(f"causal specificity:{target}", f"mean={s.matched_minus_random_mean:.4f}; ci_low={s.ci_low:.4f}", f"mean>={c['matched_minus_random_each_target_mean_min']}; ci_low>={c['matched_minus_random_each_target_bootstrap_lower_min']}", passed)

            passed = r.non_target_change_mean <= c["non_target_policy_change_each_target_mean_max"] and r.non_target_change_ci_high <= c["non_target_policy_change_each_target_bootstrap_upper_max"]
            add(f"non-target preservation:{target}", f"mean={r.non_target_change_mean:.4f}; ci_high={r.non_target_change_ci_high:.4f}", f"mean<={c['non_target_policy_change_each_target_mean_max']}; ci_high<={c['non_target_policy_change_each_target_bootstrap_upper_max']}", passed)

            passed = r.outcome_mediation_mean >= c["matched_outcome_mediation_each_target_mean_min"]
            add(f"outcome mediation:{target}", f"mean={r.outcome_mediation_mean:.4f}", f"mean>={c['matched_outcome_mediation_each_target_mean_min']}", passed)

            passed = r.cf_gt_acc_mean >= c["counterfactual_model_accuracy_each_target_mean_min"]
            add(f"counterfactual model accuracy:{target}", f"mean={r.cf_gt_acc_mean:.4f}", f"mean>={c['counterfactual_model_accuracy_each_target_mean_min']}", passed)

    decisions = pd.DataFrame(rows)
    return decisions


def render_report(decisions, base_summary, causal_summary, comparison, protocol_hash):
    gate = "PASS" if (decisions["decision"] == "PASS").all() else "FAIL"
    passed = int((decisions["decision"] == "PASS").sum())
    total = len(decisions)
    return f"""# V16-E.C1 DGX 확증 결과 보고서\n\n## 최종 Gate\n\n**{gate}** ({passed}/{total} 기준 통과)\n\n- 프로토콜 SHA-256: `{protocol_hash}`\n- Primary seeds: 16200–16223, 24개\n- Primary model: compositional unified behavior-only Transformer\n- Packed model: secondary parameter-count-matched comparison\n\n## 판정표\n\n{decisions.to_markdown(index=False)}\n\n## Base summary\n\n{base_summary.to_markdown(index=False)}\n\n## Causal summary\n\n{causal_summary.to_markdown(index=False)}\n\n## Secondary encoding comparison\n\n{comparison.to_markdown(index=False)}\n\n## 해석 제한\n\n이 결과는 task-induced distributed relational causal organization을 검증한다. 고정된 독립 slot의 필수성, 인간의 자아·의식·경험, 환경적 요구 없이 의미가 완전히 자발적으로 발생했다는 주장은 포함하지 않는다.\n"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    protocol, protocol_path = load_protocol(root)
    seeds = protocol["primary_seeds"]
    if seeds != PRIMARY_SEEDS:
        raise RuntimeError("Frozen seed list differs from analysis code")

    out_dir = root / "analysis" / "V16-E.C1"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = collect_base(root, seeds)
    causal_path = out_dir / "causal_metrics_all.csv"
    if not causal_path.exists():
        raise FileNotFoundError(causal_path)
    causal = pd.read_csv(causal_path)

    base.to_csv(out_dir / "base_metrics_all.csv", index=False)
    base_summary = summarize_base(base)
    causal_summary = summarize_causal(causal)
    specificity = paired_causal_specificity(causal)
    comparison = paired_encoding_comparison(base)
    decisions = make_decisions(protocol, base_summary, causal_summary, specificity)

    base_summary.to_csv(out_dir / "base_metrics_summary.csv", index=False)
    causal_summary.to_csv(out_dir / "causal_metrics_bootstrap_summary.csv", index=False)
    specificity.to_csv(out_dir / "causal_specificity_paired.csv", index=False)
    comparison.to_csv(out_dir / "secondary_encoding_comparison.csv", index=False)
    decisions.to_csv(out_dir / "CONFIRMATORY_DECISIONS.csv", index=False)

    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    gate = "PASS" if (decisions["decision"] == "PASS").all() else "FAIL"
    summary = {
        "version": "V16-E.C1",
        "primary_gate": gate,
        "criteria_passed": int((decisions["decision"] == "PASS").sum()),
        "criteria_total": int(len(decisions)),
        "protocol_sha256": protocol_hash,
        "primary_seeds": seeds,
    }
    (out_dir / "CONFIRMATORY_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    report = render_report(decisions, base_summary, causal_summary, comparison, protocol_hash)
    (root / "reports" / "V16-E.C1_DGX_CONFIRMATORY_RESULTS_KO.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
