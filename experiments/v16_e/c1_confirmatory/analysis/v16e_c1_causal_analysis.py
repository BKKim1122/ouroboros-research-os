from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("v16e_c1", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["v16e_c1"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def continue_from(model, hidden: torch.Tensor, state_idx: int):
    if state_idx <= model.cfg.layers:
        for block_idx in range(state_idx, model.cfg.layers):
            hidden = model.blocks[block_idx](hidden)
        hidden = model.ln(hidden)
    z = hidden[:, 10:14]
    return {"policy": 5 * model.policy(z), "outcome": model.outcome(z)}


def select_canonical(x: torch.Tensor, phase_order: torch.Tensor, phase: int):
    query_idx = (phase_order == phase).long().argmax(1)
    row = torch.arange(x.shape[0], device=x.device)
    return x[row, query_idx]


def phase_position(batch: dict, phase: int):
    return 10 + (batch["phase_order"] == phase).long().argmax(1)


def rule_position(batch: dict, canonical_rule_idx: int):
    return 6 + (batch["rule_order"] == canonical_rule_idx).long().argmax(1)


def relation_position(batch: dict, factor_idx: int):
    return 2 + (batch["rel_roles"] == factor_idx).long().argmax(1)


def patch_indexed(hidden: torch.Tensor, donor: torch.Tensor, positions: list[torch.Tensor]):
    patched = hidden.clone()
    row = torch.arange(hidden.shape[0], device=hidden.device)
    for pos in positions:
        patched[row, pos] = donor[row, pos]
    return patched


def patch_fixed_slice(hidden: torch.Tensor, donor: torch.Tensor, start: int, end: int):
    patched = hidden.clone()
    patched[:, start:end] = donor[:, start:end]
    return patched


def random_norm_indexed(hidden: torch.Tensor, donor: torch.Tensor, positions: list[torch.Tensor], seed: int):
    patched = hidden.clone()
    row = torch.arange(hidden.shape[0], device=hidden.device)
    gen = torch.Generator().manual_seed(seed)
    for pos in positions:
        base = hidden[row, pos]
        delta = donor[row, pos] - base
        noise = torch.randn(delta.shape, generator=gen, device="cpu").to(hidden.device)
        noise = noise / (noise.norm(dim=-1, keepdim=True) + 1e-8) * delta.norm(dim=-1, keepdim=True)
        patched[row, pos] = base + noise
    return patched


def random_norm_slice(hidden: torch.Tensor, donor: torch.Tensor, start: int, end: int, seed: int):
    patched = hidden.clone()
    base = hidden[:, start:end]
    delta = donor[:, start:end] - base
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(delta.shape, generator=gen, device="cpu").to(hidden.device)
    noise = noise / (noise.norm(dim=(-1, -2), keepdim=True) + 1e-8) * delta.norm(dim=(-1, -2), keepdim=True)
    patched[:, start:end] = base + noise
    return patched


def intervention_metrics(model, base_out, cf_out, batch, cf_batch, patched_hidden, phase: int, state_idx: int):
    with torch.inference_mode():
        patched_out = continue_from(model, patched_hidden, state_idx)

    base_policy = select_canonical(base_out["policy"].argmax(-1), batch["phase_order"], phase)
    cf_policy = select_canonical(cf_out["policy"].argmax(-1), cf_batch["phase_order"], phase)
    patched_policy = select_canonical(patched_out["policy"].argmax(-1), batch["phase_order"], phase)

    base_pred_all = model_mod.unpermute_queries(base_out["policy"].argmax(-1), batch["phase_order"])
    patched_pred_all = model_mod.unpermute_queries(patched_out["policy"].argmax(-1), batch["phase_order"])

    base_outcome = select_canonical(torch.sigmoid(base_out["outcome"]), batch["phase_order"], phase)
    cf_outcome = select_canonical(torch.sigmoid(cf_out["outcome"]), cf_batch["phase_order"], phase)
    patched_outcome = select_canonical(torch.sigmoid(patched_out["outcome"]), batch["phase_order"], phase)

    base_gt = batch["actions_canonical"][:, phase]
    cf_gt = cf_batch["actions_canonical"][:, phase]
    conflict = base_gt != cf_gt

    cf_gt_outcome = cf_batch["outcomes_canonical"][:, phase]
    outcome_diff = (cf_gt_outcome - batch["outcomes_canonical"][:, phase]).abs().sum(-1) > 1e-6
    base_dist = (base_outcome - cf_gt_outcome).abs().mean(-1)
    patched_dist = (patched_outcome - cf_gt_outcome).abs().mean(-1)
    mediation = 1 - patched_dist / (base_dist + 1e-8)

    non_target = [q for q in range(4) if q != phase]

    def fmean(x):
        return float(x.float().mean().detach().cpu()) if x.numel() else float("nan")

    return {
        "conflict_n": int(conflict.sum().detach().cpu()),
        "base_gt_acc": fmean(base_policy[conflict] == base_gt[conflict]),
        "cf_gt_acc": fmean(cf_policy[conflict] == cf_gt[conflict]),
        "policy_follow_cf": fmean(patched_policy[conflict] == cf_gt[conflict]),
        "policy_follow_cf_model": fmean(patched_policy[conflict] == cf_policy[conflict]),
        "target_policy_change": fmean(patched_policy != base_policy),
        "non_target_policy_change": fmean(patched_pred_all[:, non_target] != base_pred_all[:, non_target]),
        "outcome_diff_n": int(outcome_diff.sum().detach().cpu()),
        "outcome_closer_cf": fmean(patched_dist[outcome_diff] < base_dist[outcome_diff]),
        "outcome_mediation": float(mediation[outcome_diff].mean().detach().cpu()) if outcome_diff.any() else float("nan"),
        "model_output_shift": float((patched_outcome - base_outcome).abs().mean().detach().cpu()),
        "cf_model_gap": float((cf_outcome - base_outcome).abs().mean().detach().cpu()),
    }


def load_model(root: Path, rule_mode: str, seed: int, device: torch.device):
    run_dir = root / "raw" / "V16-E.C1" / f"{rule_mode}_seed{seed}"
    meta = json.loads((run_dir / "metadata.json").read_text())
    cfg = model_mod.Config(**meta["config"])
    model = model_mod.Model(cfg, rule_mode).to(device)
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location=device, weights_only=True))
    model.eval()
    return model, cfg


def factor_analysis(root: Path, seeds: list[int], n: int, device: torch.device):
    rows = []
    factors = [("identity", 0, 0), ("beneficiary", 1, 1), ("concern", 2, 2)]
    for rule_mode in ["compositional", "packed"]:
        for seed in seeds:
            model, cfg = load_model(root, rule_mode, seed, device)
            for factor_i, (factor_name, phase, relation_idx) in enumerate(factors):
                base_cpu = model_mod.make_batch(n, cfg, torch.Generator().manual_seed(4100000 + seed + factor_i), "rule_ood")
                cf_cpu = model_mod.make_batch(n, cfg, torch.Generator().manual_seed(1), "rule_ood", base=base_cpu, flip=factor_name)
                base = model_mod.batch_to_device(base_cpu, device)
                cf = model_mod.batch_to_device(cf_cpu, device)
                with torch.inference_mode():
                    base_out = model(base, True)
                    cf_out = model(cf, True)
                state_idx = 0
                hidden = base_out["states"][state_idx]
                donor = cf_out["states"][state_idx]
                relation_pos = relation_position(base, relation_idx)

                matched = patch_indexed(hidden, donor, [relation_pos])
                met = intervention_metrics(model, base_out, cf_out, base, cf, matched, phase, state_idx)
                rows.append({"seed": seed, "rule_mode": rule_mode, "kind": "factor", "target": factor_name, "phase": phase, "state_idx": state_idx, "patch": "matched_relation_token", **met})

                random_patch = random_norm_indexed(hidden, donor, [relation_pos], 5100000 + seed + factor_i)
                met = intervention_metrics(model, base_out, cf_out, base, cf, random_patch, phase, state_idx)
                rows.append({"seed": seed, "rule_mode": rule_mode, "kind": "factor", "target": factor_name, "phase": phase, "state_idx": state_idx, "patch": "random_norm_relation_token", **met})
    return pd.DataFrame(rows)


def operator_analysis(root: Path, seeds: list[int], n: int, device: torch.device):
    rows = []
    operators = [("allocation", 1, 0), ("protection", 2, 2)]
    for rule_mode in ["compositional", "packed"]:
        for seed in seeds:
            model, cfg = load_model(root, rule_mode, seed, device)
            for operator_i, (operator_name, phase, canonical_rule_idx) in enumerate(operators):
                base_cpu = model_mod.make_batch(n, cfg, torch.Generator().manual_seed(6100000 + seed + operator_i), "rule_ood")
                cf_cpu = model_mod.make_batch(n, cfg, torch.Generator().manual_seed(1), "rule_ood", base=base_cpu, flip_rule=operator_name)
                base = model_mod.batch_to_device(base_cpu, device)
                cf = model_mod.batch_to_device(cf_cpu, device)
                with torch.inference_mode():
                    base_out = model(base, True)
                    cf_out = model(cf, True)
                state_idx = 0
                hidden = base_out["states"][state_idx]
                donor = cf_out["states"][state_idx]

                if rule_mode == "compositional":
                    pos = rule_position(base, canonical_rule_idx)
                    matched = patch_indexed(hidden, donor, [pos])
                    random_patch = random_norm_indexed(hidden, donor, [pos], 7100000 + seed + operator_i)
                    patch_name = "matched_comparator_token"
                    random_name = "random_norm_comparator_token"
                else:
                    matched = patch_fixed_slice(hidden, donor, 6, 10)
                    random_patch = random_norm_slice(hidden, donor, 6, 10, 7100000 + seed + operator_i)
                    patch_name = "matched_full_rule_coalition"
                    random_name = "random_norm_full_rule_coalition"

                met = intervention_metrics(model, base_out, cf_out, base, cf, matched, phase, state_idx)
                rows.append({"seed": seed, "rule_mode": rule_mode, "kind": "operator", "target": operator_name, "phase": phase, "state_idx": state_idx, "patch": patch_name, **met})
                met = intervention_metrics(model, base_out, cf_out, base, cf, random_patch, phase, state_idx)
                rows.append({"seed": seed, "rule_mode": rule_mode, "kind": "operator", "target": operator_name, "phase": phase, "state_idx": state_idx, "patch": random_name, **met})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.root)
    global model_mod
    model_mod = load_module(root / "src" / "v16e_c1_confirmatory.py")
    device = model_mod.resolve_device(args.device)
    torch.set_num_threads(args.threads)
    model_mod.configure_determinism()

    out_dir = root / "analysis" / "V16-E.C1"
    out_dir.mkdir(parents=True, exist_ok=True)

    factor_df = factor_analysis(root, args.seeds, args.n, device)
    operator_df = operator_analysis(root, args.seeds, args.n, device)
    all_df = pd.concat([factor_df, operator_df], ignore_index=True)
    all_df.to_csv(out_dir / "causal_metrics_all.csv", index=False)

    summary = all_df.groupby(["rule_mode", "kind", "target", "patch"], as_index=False).agg(
        mean_policy_follow=("policy_follow_cf", "mean"),
        min_policy_follow=("policy_follow_cf", "min"),
        mean_non_target_change=("non_target_policy_change", "mean"),
        mean_outcome_mediation=("outcome_mediation", "mean"),
        mean_outcome_closer=("outcome_closer_cf", "mean"),
        total_conflicts=("conflict_n", "sum"),
    )
    summary.to_csv(out_dir / "causal_metrics_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
