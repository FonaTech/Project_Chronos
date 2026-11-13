# Project Chronos: On-device low-latency lookahead dual-layer MoE inference
__version__ = "0.3.3"

# Register with HuggingFace AutoXXX factories so
# AutoModelForCausalLM.from_pretrained("path/to/chronos") works out of the box.
try:
    from chronos.model import hf_io as _hf_io  # noqa: F401
except Exception as _e:
    # Non-fatal: HF registration is a convenience, not a hard requirement.
    import warnings
    warnings.warn(f"Chronos HF registration skipped: {_e}")

