"""The en→zh demo's model topology and tokenizer special tokens.

``ENZH_50M_BASE`` is the fixed part of the 50M benchmark topology (moved out
of ``chronos.model.factory`` in 0.3.1a2, where it was ``ENZH_DEMO_BASE``).  The
searched part (hidden size, expert count and width) comes from the kernel's
``fit_config``::

    from chronos.model.factory import fit_config
    fit = fit_config(50e6, 12e6, vocab_size=16384, base=ENZH_50M_BASE)

The 72h recipe (``recipes/enzh_50m_72h.json``) stores the full ``model_config``
it was trained with; that file, not this search, is the reference.

``SPECIAL_TOKENS`` is the order the 0.3.1a1 tokenizer used (language tags in
ids 3-5), so a tokenizer trained with ``chronos tokenizer <data> --out <dir>
--extra-special "<|en|>" "<|zh|>" "<|translate|>"`` matches the 50M run.
"""
from __future__ import annotations

from chronos.tokenizer.model import special_tokens_with

LANG_TOKENS = ("<|en|>", "<|zh|>", "<|translate|>")
SPECIAL_TOKENS = special_tokens_with(LANG_TOKENS)

ENZH_50M_BASE = dict(
    architecture_version=3, num_hidden_layers=6, prelude_layers=1, coda_layers=1,
    recurrent_depth=4, min_recurrent_depth=1, max_inference_depth=4,
    recurrent_attention_mode="shared_first", adaptive_halting=True,
    depth_schedule="fixed_then_adaptive", depth_schedule_warmup_fraction=0.3,
    halting_threshold=0.95, num_experts_per_tok=2, num_shared_experts=1,
    kv_latent_dim=128, rope_dim=32, rope_layout="per_head", sliding_window_size=128,
    max_position_embeddings=1024,
    layer_aware_lookahead=True, layer_aware_cache=True, lookahead_steps=2,
    dropout=0.0, fallback_mask_prob=0.08, rope_theta=1e6,
    bos_token_id=1, eos_token_id=2, tie_word_embeddings=True,
)
ENZH_DEMO_BASE = ENZH_50M_BASE   # pre-0.3.1a2 name (chronos.model.factory)
