# ABC arms — diagnose repetition + weak knowledge recall on the CPT 30M

**Goal:** reproduce the failing CPT config at 15M scale, then test whether (A) capping dt or (C) replacing shared_first with per_iteration improves validation CE. (B) is a plain depth-1 control.

**Shared base (mirrors cpt_30m_bilingual_1b_20261117 run_spec):**
- hidden 160, 24 experts W128, kv 96, top-2 + 1 shared  → ~15M total / ~7M active
- block order, sigmoid dt, 512 window, 30 min per arm, same seed and data order
- `adaptive_halting=False` (run_arm.py line 65)
- lambda_cluster_locality 0.3 (inherited from base recipe), lambda_balance 0.005

| Arm | Change vs failing CPT | Rationale |
|---|---|---|
| baseline (REPRO) | depth=4, dt_init=0.5, dt_min=0.05, dt_max=1.0, shared_first | Reproduce the failing config. If this arm itself produces a learned dt > 0.5 we have confirmed the amplification risk. |
| **A** (dt-cap) | depth=4, dt_init=0.25, dt_max=0.5, else same | Cap effective per-group step so 4-iter total ≤ 2.0. Tests dt-amplification as the complaint. |
| **B** (plain) | depth=1 (plain 6-layer) | Removes loop entirely. If B ≥ baseline, the loop is paying its own cost without return. |
| **C** (per-iteration) | depth=4, dt_init=0.5, dt_max=1.0, `recurrent_attention_mode=per_iteration` | Lets K/V update every iteration. Tests shared_first as the limiter. |

**Pass rule:** ΔCE vs baseline at matched wall time is larger than the d1 seed noise observed in results_a/results_b (≈0.03 nats at step 800+).

**Run order:** baseline → A → B → C, sequentially.
