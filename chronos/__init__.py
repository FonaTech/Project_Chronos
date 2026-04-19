# Project Chronos: On-device low-latency lookahead dual-layer MoE inference
__version__ = "0.1.0"

# Bootstrap minimind dependency before any submodule imports
from chronos.deps import ensure_minimind as _ensure_minimind
_ensure_minimind()
