#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, sys, time, traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(min(4, os.cpu_count() or 1))
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


@dataclass
class Config:
    n_bits: int = 6
    d_model: int = 48
    n_heads: int = 4
    ff_dim: int = 96
    batch_size: int = 192
    eval_size: int = 2400
    steps: int = 700
    lr: float = 2e-3
    drift_train_max: float = 4.5
    device: str = "cpu"


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_batch(n: int, cfg: Config, gen: torch.Generator,
               drift_strength: float | None = None,
               drift_mode: str = "random") -> Dict[str, torch.Tensor]:
    # Symmetric A/B streams, with random token order to block position shortcuts.
    current = torch.randint(0, 2, (n, 2, cfg.n_bits), generator=gen).float()
    transition = torch.randint(0, 2, (n, 2, cfg.n_bits), generator=gen).float()
    nxt = torch.remainder(current + transition, 2.0)
    target = torch.randint(0, 2, (n,), generator=gen)
    order_flip = torch.randint(0, 2, (n,), generator=gen)
    stream_id = torch.tensor([[0, 1]]).repeat(n, 1)
    for i in range(n):
        if order_flip[i].item() == 1:
            current[i] = current[i].flip(0)
            transition[i] = transition[i].flip(0)
            nxt[i] = nxt[i].flip(0)
            stream_id[i] = stream_id[i].flip(0)
    # token position containing the semantic target stream
    target_pos = (stream_id == target[:, None]).float().argmax(dim=1)
    y_target = nxt[torch.arange(n), target_pos]
    other_pos = 1 - target_pos
    y_other = nxt[torch.arange(n), other_pos]

    if drift_strength is None:
        mag = torch.rand(n, generator=gen) * cfg.drift_train_max
    else:
        mag = torch.full((n,), float(drift_strength))
    if drift_mode == "random":
        sign = torch.where(torch.rand(n, generator=gen) < 0.65,
                           torch.ones(n), -torch.ones(n))
    elif drift_mode == "away":
        sign = torch.ones(n)
    elif drift_mode == "toward":
        sign = -torch.ones(n)
    else:
        raise ValueError(drift_mode)
    # Positive sign biases the non-target token, negative sign reinforces target.
    drift = torch.zeros(n, 2)
    drift[torch.arange(n), other_pos] = sign * mag
    drift[torch.arange(n), target_pos] = -sign * mag

    # Neutral relation independent of target: parity of all current bits.
    neutral = (nxt.sum(dim=(1, 2)) > cfg.n_bits).long()
    return {
        "current": current, "transition": transition, "next": nxt,
        "stream_id": stream_id, "target": target, "target_pos": target_pos,
        "y_target": y_target, "y_other": y_other, "neutral": neutral,
        "drift": drift, "drift_mag": mag, "drift_sign": sign,
    }


def move(b: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in b.items()}


class MetaAttentionModel(nn.Module):
    def __init__(self, cfg: Config, blind_monitor: bool = False):
        super().__init__()
        self.cfg = cfg
        self.blind_monitor = blind_monitor
        token_in = 2 * cfg.n_bits + 2
        self.token_in = nn.Linear(token_in, cfg.d_model)
        layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.ff_dim,
                                           batch_first=True, dropout=0.0,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.target_emb = nn.Embedding(2, cfg.d_model)
        self.score_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.score_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.value = nn.Linear(cfg.d_model, cfg.d_model)
        self.stream_sem_emb = nn.Embedding(2, cfg.d_model)
        self.world_head = nn.Linear(cfg.d_model, cfg.n_bits)
        self.neutral_head = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 2))
        self.attention_trace_proj = nn.Linear(2, cfg.d_model, bias=False)
        monitor_in = cfg.d_model * 4
        self.monitor = nn.Sequential(nn.Linear(monitor_in, cfg.d_model), nn.GELU(),
                                     nn.Linear(cfg.d_model, cfg.d_model), nn.GELU())
        self.monitor_norm = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
        self.meta_p_head = nn.Linear(cfg.d_model, 2, bias=False)
        self.feedback_head = nn.Linear(cfg.d_model, 2, bias=False)
        self.policy_head = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.n_bits))

    def encode(self, b: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        sid = F.one_hot(b["stream_id"], 2).float()
        x = torch.cat([b["current"], b["transition"], sid], dim=-1)
        h = self.encoder(self.token_in(x))
        pooled = h.mean(dim=1)
        return h, pooled

    def forward(self, b: Dict[str, torch.Tensor], report_gain: float = 1.0,
                feedback_gain: float = 1.0, monitor_override: torch.Tensor | None = None,
                random_monitor: bool = False) -> Dict[str, torch.Tensor]:
        h, pooled = self.encode(b)
        q = self.target_emb(b["target"])
        raw_scores = (self.score_k(h) * self.score_q(q)[:, None, :]).sum(-1) / math.sqrt(self.cfg.d_model)
        centered = raw_scores - raw_scores.mean(dim=-1, keepdim=True)
        rms = torch.sqrt((centered.pow(2)).mean(dim=-1, keepdim=True) + 1e-6)
        scores = 1.5 * torch.tanh(centered / rms)
        scores0 = scores + b["drift"]
        p0 = torch.softmax(scores0, dim=-1)
        p0_sem = torch.zeros_like(p0).scatter_add(1, b["stream_id"], p0)
        vals = self.value(h) + 1.5 * self.stream_sem_emb(b["stream_id"])
        s0 = torch.sum(p0[:, :, None] * vals, dim=1)
        trace = self.attention_trace_proj(p0_sem)
        if self.blind_monitor:
            monitor_summary = torch.zeros_like(s0)
            monitor_pool = torch.zeros_like(pooled)
            trace = torch.zeros_like(trace)
        else:
            monitor_summary = s0
            monitor_pool = pooled
        m = self.monitor(torch.cat([monitor_summary, monitor_pool, q, trace], dim=-1))
        m = self.monitor_norm(m)
        if monitor_override is not None:
            m_used = monitor_override
        elif random_monitor:
            rnd = torch.randn_like(m)
            rnd = rnd / (rnd.norm(dim=-1, keepdim=True) + 1e-8)
            m_used = rnd * (m.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            m_used = m
        meta_logits = report_gain * self.meta_p_head(m_used)
        p_hat = torch.softmax(meta_logits, dim=-1)
        corr_sem = feedback_gain * self.feedback_head(m_used)
        corr = torch.gather(corr_sem, 1, b["stream_id"])
        p1 = torch.softmax(scores0 + corr, dim=-1)
        s1 = torch.sum(p1[:, :, None] * vals, dim=1)
        policy = self.policy_head(s1)
        world = self.world_head(h)
        neutral = self.neutral_head(pooled)
        return {"h": h, "pooled": pooled, "vals": vals, "scores0": scores0, "p0": p0, "p0_sem": p0_sem,
                "monitor": m, "meta_logits": meta_logits, "p_hat": p_hat,
                "corr": corr, "p1": p1, "policy": policy, "world": world,
                "neutral": neutral}


def bit_acc(y: torch.Tensor, logits: torch.Tensor) -> float:
    return ((logits > 0).float() == y).float().mean().item()


def conflict_acc(y: torch.Tensor, other: torch.Tensor, logits: torch.Tensor) -> float:
    mask = (y != other)
    if mask.sum() == 0:
        return float("nan")
    pred = (logits > 0).float()
    return (pred[mask] == y[mask]).float().mean().item()


def train_model(cfg: Config, seed: int, condition: str, outdir: Path) -> MetaAttentionModel:
    seed_all(seed)
    blind = condition == "blind_cue_control"
    model = MetaAttentionModel(cfg, blind_monitor=blind).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 11)
    history = []
    for step in range(cfg.steps):
        b = move(make_batch(cfg.batch_size, cfg, gen), cfg.device)
        o = model(b, 1.0, 1.0)
        # Monitor learns the model's own first-pass attention distribution.
        meta_loss = F.kl_div(F.log_softmax(o["meta_logits"], dim=-1), o["p0_sem"].detach(), reduction="batchmean")
        policy_loss = F.binary_cross_entropy_with_logits(o["policy"], b["y_target"])
        world_loss = F.binary_cross_entropy_with_logits(o["world"], b["next"])
        neutral_loss = F.cross_entropy(o["neutral"], b["neutral"])
        # Feedback should restore a calibrated target allocation, not maximize attention blindly.
        desired = F.one_hot(b["target_pos"], 2).float() * 0.75 + 0.125
        correction_loss = F.kl_div(torch.log(o["p1"] + 1e-8), desired, reduction="batchmean")
        correction_cost = o["corr"].pow(2).mean()
        loss = policy_loss + 0.8 * world_loss + 0.25 * neutral_loss + 0.7 * meta_loss + 0.8 * correction_loss + 0.05 * correction_cost
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        if step % 25 == 0 or step == cfg.steps - 1:
            with torch.no_grad():
                pre = (o["p0"].argmax(-1) == b["target_pos"]).float().mean().item()
                post = (o["p1"].argmax(-1) == b["target_pos"]).float().mean().item()
                meta = (o["p_hat"].argmax(-1) == o["p0_sem"].argmax(-1)).float().mean().item()
                history.append({"step": step, "loss": loss.item(), "policy_loss": policy_loss.item(),
                                "world_loss": world_loss.item(), "meta_loss": meta_loss.item(),
                                "correction_loss": correction_loss.item(), "pre_focus": pre,
                                "post_focus": post, "meta_stream_acc": meta})
    pd.DataFrame(history).to_csv(outdir / f"{condition}_seed{seed}_train_history.csv", index=False)
    return model


@torch.no_grad()
def evaluate(model: MetaAttentionModel, cfg: Config, seed: int, condition: str,
             report_gain: float, feedback_gain: float, drift_strength: float,
             drift_mode: str = "random", monitor_mode: str = "factual") -> Dict[str, float]:
    gen = torch.Generator().manual_seed(seed + 100000 + int(drift_strength * 100))
    b = move(make_batch(cfg.eval_size, cfg, gen, drift_strength, drift_mode), cfg.device)
    if monitor_mode == "opposite_target_swap":
        donor = {k: v.clone() for k, v in b.items()}
        donor["target"] = 1 - donor["target"]
        donor["target_pos"] = 1 - donor["target_pos"]
        od = model(donor, 1.0, 0.0)
        o = model(b, report_gain, feedback_gain, monitor_override=od["monitor"])
    elif monitor_mode == "random_norm_matched":
        o = model(b, report_gain, feedback_gain, random_monitor=True)
    else:
        o = model(b, report_gain, feedback_gain)
    pre_target = (o["p0"].argmax(-1) == b["target_pos"])
    post_target = (o["p1"].argmax(-1) == b["target_pos"])
    meta_stream = (o["p_hat"].argmax(-1) == o["p0_sem"].argmax(-1))
    true_align = pre_target
    pred_align = (o["p_hat"].argmax(-1) == b["target"])
    off = ~pre_target
    on = pre_target
    recovery = post_target[off].float().mean().item() if off.any() else float("nan")
    on_pres = post_target[on].float().mean().item() if on.any() else float("nan")
    detect_recall = (~pred_align[off]).float().mean().item() if off.any() else float("nan")
    meta_mae = torch.abs(o["p_hat"] - o["p0_sem"]).mean().item()
    result = {
        "condition": condition, "seed": seed, "report_gain": report_gain,
        "feedback_gain": feedback_gain, "drift_strength": drift_strength,
        "drift_mode": drift_mode, "monitor_mode": monitor_mode,
        "pre_focus_acc": pre_target.float().mean().item(),
        "post_focus_acc": post_target.float().mean().item(),
        "recovery_rate": recovery, "on_target_preservation": on_pres,
        "meta_stream_acc": meta_stream.float().mean().item(),
        "meta_align_acc": (pred_align == true_align).float().mean().item(),
        "off_target_detection_recall": detect_recall,
        "meta_attention_mae": meta_mae,
        "policy_acc": bit_acc(b["y_target"], o["policy"]),
        "policy_conflict_acc": conflict_acc(b["y_target"], b["y_other"], o["policy"]),
        "world_acc": bit_acc(b["next"], o["world"]),
        "neutral_acc": (o["neutral"].argmax(-1) == b["neutral"]).float().mean().item(),
        "mean_correction_norm": o["corr"].norm(dim=-1).mean().item(),
    }
    return result



@torch.no_grad()
def evaluate_grid_cached(model: MetaAttentionModel, cfg: Config, seed: int, condition: str,
                         drift_strength: float, drift_mode: str = "random") -> list[Dict[str, float]]:
    gen = torch.Generator().manual_seed(seed + 100000 + int(drift_strength * 100))
    b = move(make_batch(cfg.eval_size, cfg, gen, drift_strength, drift_mode), cfg.device)
    base = model(b, 1.0, 0.0)
    h, pooled, vals = base["h"], base["pooled"], base["vals"]
    p0, p0_sem, scores0, m = base["p0"], base["p0_sem"], base["scores0"], base["monitor"]
    world_logits = base["world"]
    neutral_logits = base["neutral"]
    raw_meta = model.meta_p_head(m)
    raw_corr_sem = model.feedback_head(m)
    rows = []
    gains = [0.0, 0.25, 0.5, 0.75, 1.0]
    for gr in gains:
        p_hat = torch.softmax(gr * raw_meta, dim=-1)
        for gf in gains:
            corr_sem = gf * raw_corr_sem
            corr = torch.gather(corr_sem, 1, b["stream_id"])
            p1 = torch.softmax(scores0 + corr, dim=-1)
            s1 = torch.sum(p1[:, :, None] * vals, dim=1)
            policy = model.policy_head(s1)
            pre_target = (p0.argmax(-1) == b["target_pos"])
            post_target = (p1.argmax(-1) == b["target_pos"])
            meta_stream = (p_hat.argmax(-1) == p0_sem.argmax(-1))
            true_align = pre_target
            pred_align = (p_hat.argmax(-1) == b["target"])
            off = ~pre_target; on = pre_target
            recovery = post_target[off].float().mean().item() if off.any() else float("nan")
            on_pres = post_target[on].float().mean().item() if on.any() else float("nan")
            detect_recall = (~pred_align[off]).float().mean().item() if off.any() else float("nan")
            rows.append({
                "condition": condition, "seed": seed, "report_gain": gr,
                "feedback_gain": gf, "drift_strength": drift_strength,
                "drift_mode": drift_mode, "monitor_mode": "factual",
                "pre_focus_acc": pre_target.float().mean().item(),
                "post_focus_acc": post_target.float().mean().item(),
                "recovery_rate": recovery, "on_target_preservation": on_pres,
                "meta_stream_acc": meta_stream.float().mean().item(),
                "meta_align_acc": (pred_align == true_align).float().mean().item(),
                "off_target_detection_recall": detect_recall,
                "meta_attention_mae": torch.abs(p_hat - p0_sem).mean().item(),
                "policy_acc": bit_acc(b["y_target"], policy),
                "policy_conflict_acc": conflict_acc(b["y_target"], b["y_other"], policy),
                "world_acc": bit_acc(b["next"], world_logits),
                "neutral_acc": (neutral_logits.argmax(-1) == b["neutral"]).float().mean().item(),
                "mean_correction_norm": corr.norm(dim=-1).mean().item(),
            })
    return rows

def run_seed(args, cfg: Config, seed: int, condition: str, root: Path) -> None:
    t0 = time.time(); start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_dir = root / "raw" / "pilot" / f"{condition}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = root / "logs" / f"{condition}_seed{seed}.log"
    status = "success"; err = ""
    try:
        model = train_model(cfg, seed, condition, run_dir)
        ckpt = run_dir / "checkpoint.pt"
        torch.save({"model": model.state_dict(), "config": asdict(cfg), "condition": condition, "seed": seed}, ckpt)
        rows = []
        gains = [0.0, 0.25, 0.5, 0.75, 1.0]
        drifts = [0.0, 1.5, 3.0, 4.5]
        for d in drifts:
            rows.extend(evaluate_grid_cached(model, cfg, seed, condition, d))
        for mode in ["opposite_target_swap", "random_norm_matched"]:
            rows.append(evaluate(model, cfg, seed, condition, 1.0, 1.0, 4.5, "away", mode))
        pd.DataFrame(rows).to_csv(run_dir / "eval_grid.csv", index=False)
        config = {"seed": seed, "condition": condition, **asdict(cfg)}
        (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"start={start}\ncondition={condition}\nseed={seed}\n")
            f.write(f"checkpoint_sha256={sha256(ckpt)}\nstatus=success\n")
    except Exception as e:
        status = "failed"; err = repr(e)
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        registry = root / "run_registry.csv"
        exists = registry.exists()
        with registry.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["condition","seed","start_utc","end_utc","duration_s","status","error","run_dir"])
            if not exists: w.writeheader()
            w.writerow({"condition":condition,"seed":seed,"start_utc":start,"end_utc":end,
                        "duration_s":round(time.time()-t0,3),"status":status,"error":err,"run_dir":str(run_dir)})


def aggregate(root: Path) -> None:
    files = list((root / "raw" / "pilot").glob("*/eval_grid.csv"))
    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw.to_csv(root / "analysis" / "pilot_eval_raw.csv", index=False)
    metrics = [c for c in raw.columns if c not in {"condition","seed","report_gain","feedback_gain","drift_strength","drift_mode","monitor_mode"}]
    factual = raw[raw.monitor_mode == "factual"]
    summary = factual.groupby(["condition","report_gain","feedback_gain","drift_strength"], as_index=False)[metrics].mean(numeric_only=True)
    summary.to_csv(root / "analysis" / "pilot_eval_summary.csv", index=False)
    # Diagnostic endpoints.
    endpoint = factual[(factual.drift_strength == 4.5) & (factual.report_gain.isin([0.0,1.0])) & (factual.feedback_gain.isin([0.0,1.0]))]
    endpoint.to_csv(root / "analysis" / "pilot_endpoints.csv", index=False)
    swaps = raw[raw.monitor_mode != "factual"]
    swaps.to_csv(root / "analysis" / "pilot_monitor_swaps.csv", index=False)
    report = []
    for cond in sorted(raw.condition.unique()):
        x = factual[(factual.condition==cond)&(factual.drift_strength==4.5)&(factual.report_gain==1)&(factual.feedback_gain==1)]
        y = factual[(factual.condition==cond)&(factual.drift_strength==4.5)&(factual.report_gain==1)&(factual.feedback_gain==0)]
        report.append({"condition":cond,
                       "meta_stream_acc":x.meta_stream_acc.mean(),
                       "meta_attention_mae":x.meta_attention_mae.mean(),
                       "recovery_feedback_on":x.recovery_rate.mean(),
                       "recovery_feedback_off":y.recovery_rate.mean(),
                       "recovery_gain":x.recovery_rate.mean()-y.recovery_rate.mean(),
                       "policy_acc_feedback_on":x.policy_acc.mean(),
                       "policy_acc_feedback_off":y.policy_acc.mean(),
                       "world_acc":x.world_acc.mean(),
                       "neutral_acc":x.neutral_acc.mean()})
    pd.DataFrame(report).to_csv(root / "analysis" / "pilot_gate_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", choices=["run","aggregate","all"], default="all")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--eval-size", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--seeds", default="16100,16101,16102")
    ap.add_argument("--conditions", default="full_monitor,blind_cue_control")
    args = ap.parse_args()
    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    cfg = Config(steps=args.steps, eval_size=args.eval_size, batch_size=args.batch_size)
    if args.mode in {"run","all"}:
        for condition in [x for x in args.conditions.split(",") if x]:
            for seed in [int(x) for x in args.seeds.split(",") if x]:
                run_seed(args, cfg, seed, condition, root)
    if args.mode in {"aggregate","all"}:
        aggregate(root)

if __name__ == "__main__":
    main()
