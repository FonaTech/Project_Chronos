"""Planner configs for the doc-mix packages (E=18, top-2 + 1 shared, 50M).

Usage: python demos/translation/scripts/plan_docmix.py demos/translation/recipes/docmix_plan.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


def main():
    from chronos.model.planner import PlanRequest, DatasetProfile, plan_training
    from chronos.config_schema import model_config_kwargs
    out = {}
    # Measured on the built sets (pretrain tokens; SFT record-length p50/p95 from trial_lengths.py).
    for name, seq, tokens, med, p95 in (("A", 4096, 185_519_423, 139.0, 1826), ("B", 1024, 62_496_360, 138.0, 695),
                                         ("C", 1024, 12_703_644, 140.0, 696)):
        prof = DatasetProfile(pretrain_tokens=tokens, pretrain_tokens_source="measured", vocab_size=16384,
                              seq_median_tokens=med, seq_p95_tokens=p95)
        res = plan_training(PlanRequest(target_total=50e6, num_experts=18, active_experts=2, seq_len=seq,
                                        ram_gb=128, token_budget=tokens * 4), prof)
        cfg = model_config_kwargs(res["config"])
        out[name] = {"model": cfg, "phases": res["longrun_phases"], "arch": res["report"]["architecture"],
                     "scaling": {k: res["report"]["scaling"].get(k) for k in ("chosen_lr", "chosen_batch_tokens")},
                     "memory": res["report"]["memory"]}
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
    for k, v in out.items():
        print(k, json.dumps(v["arch"], default=str)[:400])
        print("  phases", json.dumps(v["phases"]))
        print("  mem", json.dumps(v["memory"], default=str)[:400])
    print(json.dumps(out["A"]["model"], default=str))


if __name__ == "__main__":
    main()
