"""Explicit, auditable Chronos checkpoint migrations."""
from __future__ import annotations

import argparse
import os

from chronos.model.checkpoint import (
    chronos_config_from_checkpoint,
    load_checkpoint_state_dict,
    save_state_dict_with_config,
)
from chronos.model.model_chronos import ChronosForCausalLM


def migrate_v1_to_v2(source: str, output: str) -> str:
    cfg, _ = chronos_config_from_checkpoint(source, require_unsniffable=True)
    cfg.architecture_version = 2
    cfg.recurrent_enabled = True
    cfg.recurrent_depth = 1
    cfg.min_recurrent_depth = 1
    cfg.max_inference_depth = 1
    cfg.layer_aware_cache = False
    cfg.layer_aware_lookahead = False
    cfg.extra_metadata = {
        **dict(getattr(cfg, "extra_metadata", {}) or {}),
        "migration": {"from": "chronos-v1", "to": "chronos-v2", "source": os.path.abspath(source)},
    }
    model = ChronosForCausalLM(cfg)
    state = load_checkpoint_state_dict(source, map_location="cpu")
    result = model.load_state_dict(state, strict=False)
    unexpected = [k for k in result.unexpected_keys if not k.startswith("model.freqs_")]
    allowed_missing = (
        "model.recurrent_controller.", "model.multimodal_fusion.",
    )
    bad_missing = [k for k in result.missing_keys if not k.startswith(allowed_missing)]
    if unexpected or bad_missing:
        raise ValueError(f"migration topology mismatch: missing={bad_missing}, unexpected={unexpected}")
    save_state_dict_with_config(model, output, cfg, stage="migrated-v2")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Migrate an explicit Chronos v1 checkpoint to v2")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(migrate_v1_to_v2(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
