# V16 Pilot Protocol v0.1

## Study title
Attention–Meta-Awareness Dissociation and Closed-Loop Recovery in a Transformer Agent

## Status
Exploratory pilot. This document is not a confirmatory preregistration. Pilot seeds will not be reused in the confirmatory study.

## Why the earlier V16 plan changed
The earlier handoff proposed embodied decomposition of identity, ownership, agency, control, valuation, and operational concern. Those variables remain relevant, but they are deferred to a later embodied stage. V16 now targets the more proximal research question needed for the original program: whether primary attention, an internal model of current attention, report access to that model, and causal feedback from that model can be separated and tested.

## Retained requirements from the earlier handoff
- Keep V1–V15 and manuscript v4.1 frozen and separate.
- Distinguish task performance, upstream world information, internal state, and final policy output.
- Separate pilot from confirmation; freeze exact source snapshot before confirmation.
- Preserve per-seed config, logs, outputs, hashes, timestamps, and environment.
- Use gain interventions and activation/content swaps plus random/blind controls.
- Report designed construction separately from natural emergence or human phenomenology.
- Treat interactions as results rather than automatically as failures.

## Core constructs
- Primary attention A: the pre-correction distribution over two information streams.
- Meta-attention estimate M: a learned estimate of the model's own pre-correction attention distribution, inferred from the attended summary and context without direct access to the numeric attention weights.
- Report access G_R: gain controlling whether M can be externally decoded/reported.
- Feedback use G_F: gain controlling whether M can causally modify a second-pass attention distribution.
- Recovery: an initially off-target attention state becomes on-target after feedback.

## Environment
Each episode contains two symmetric streams A/B. Each stream includes a 6-bit current state and 6-bit transition; next state is XOR(current, transition). A cue selects the task-relevant stream. A random post-controller disturbance shifts the first-pass attention logits by a signed strength independent of stream content. Disturbances can push attention toward or away from the task target, preventing a cue-only feedback controller from knowing the required correction magnitude.

## Simultaneous tasks
1. Target next-state prediction from the corrected attention summary.
2. Token-wise world prediction for both streams from upstream representations.
3. Observer-neutral parity relation.
4. Meta-estimation of the pre-correction attention distribution.
5. Meta-report of attended stream and target alignment.

## Pilot models
- Full monitor: estimates attention from attended summary, pooled context, and target cue; feedback uses the monitor latent.
- Blind cue control: same monitor/output capacity but its monitor receives no attended summary, testing whether feedback is merely a target-cue controller.
- No-feedback evaluation: full monitor retained for reporting but G_F=0.

## Pilot interventions
- Report-gain sweep G_R in {0, .25, .5, .75, 1}.
- Feedback-gain sweep G_F in {0, .25, .5, .75, 1}.
- Drift strengths {0, 1.5, 3.0, 4.5} with signs sampled independently.
- Matched monitor-latent swap from episodes with opposite target cue.
- Random norm-matched monitor replacement.

## Primary pilot diagnostics
- Pre-focus accuracy and post-focus accuracy.
- Meta attended-stream accuracy and attention-probability MAE.
- Off-target detection recall.
- Recovery rate among initially off-target episodes.
- Policy bit accuracy and conflict accuracy.
- World bit balanced accuracy and neutral accuracy.
- Feedback selectivity: recovery gain relative to unnecessary correction/harm on initially on-target episodes.
- Full-monitor advantage over blind cue control under variable disturbances.

## Pilot seeds and compute
Seeds 16100–16102; 700 training steps per model condition. Pilot includes full-monitor and blind-control models. Expected CPU runtime: approximately 5–20 minutes depending on host.

## Gate for confirmatory design
The pilot is considered informative if:
- full monitor attention-distribution MAE <= 0.12;
- meta attended-stream accuracy >= 0.90 at G_R=1;
- feedback increases recovery by >= 0.20 at strong drift;
- full monitor exceeds blind cue control in recovery or correction efficiency;
- world BACC remains >= 0.95;
- report gain changes meta-report endpoints without materially changing policy/world endpoints;
- feedback gain changes recovery/policy endpoints while leaving pre-attention and upstream world endpoints unchanged.

Thresholds for the confirmatory study will be set only after pilot inspection and frozen with an exact source snapshot.
