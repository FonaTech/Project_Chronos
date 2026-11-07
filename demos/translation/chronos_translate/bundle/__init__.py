"""Out-of-the-box translation model bundles (``chronos-translate bundle``).

A bundle is one self-contained directory:

    <bundle>/
      bundle.json              identity, provenance, file hashes, measured quality
      README.md / README_zh.md quick start
      NOTICE.md                data and code licences
      model/<name>.pth         weights (+ .config.json sidecar)
      model/tokenizer/         the model's own tokenizer
      eval/                    FLORES+ dev/devtest (CC BY-SA 4.0) for smoke tests
      recipes/*.json           ``chronos-translate run`` specs for SFT / continued pretraining
      recommended.json         tiered next-stage parameters with their sources
      examples/                tiny en-zh pairs to prove the fine-tune path

``chronos-translate bundle build`` creates it from a training run, ``... check``
verifies hashes and runs a CPU forward smoke test, ``... translate``
and ``... finetune`` are the two entry points a recipient needs.
"""
from .builder import BUNDLE_FORMAT, build_bundle, check_bundle, load_bundle_manifest

__all__ = ["BUNDLE_FORMAT", "build_bundle", "check_bundle", "load_bundle_manifest"]
