"""Long-context training kernels (0.3.2a1): exactness against the reference paths.

Chunked cross-entropy must give the loss and gradients of the full-logit
FP32 path, both for the stage loss (causal shift) and for LongRun's
pre-aligned labels, with -100 ignored.
"""
from __future__ import annotations

import os
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mlx.utils import tree_flatten  # noqa: E402


def _config(**kw):
    from chronos.model.config import ChronosConfig
    base = dict(architecture_version=3, hidden_size=64, num_hidden_layers=4, prelude_layers=1, coda_layers=1,
                recurrent_depth=3, min_recurrent_depth=2, num_attention_heads=4, num_key_value_heads=2,
                kv_latent_dim=16, rope_dim=8, num_experts=6, num_experts_per_tok=2, num_shared_experts=1,
                moe_intermediate_size=32, vocab_size=97, max_position_embeddings=128, sliding_window_size=8,
                dropout=0.0, fallback_mask_prob=0.0, adaptive_halting=True, lookahead_steps=2,
                layer_aware_lookahead=True)
    base.update(kw)
    return ChronosConfig(**base)


def _model(cfg):
    from chronos.mlx.model import ChronosMLXModel
    mx.random.seed(0)
    model = ChronosMLXModel(cfg)
    mx.eval(model.parameters())
    model.train()
    return model


def _batch(cfg, b=2, s=40):
    mx.random.seed(1)
    ids = mx.random.randint(3, cfg.vocab_size, (b, s))
    labels = mx.random.randint(3, cfg.vocab_size, (b, s))
    labels = mx.where(mx.arange(s)[None, :] < 7, -100, labels)    # a masked prompt
    return ids, labels


def _grads(model, loss_fn, ids, labels):
    loss, grads = nn.value_and_grad(model, loss_fn)(model, ids, labels)
    mx.eval(loss, grads)
    return loss, dict(tree_flatten(grads))


def _rel(a, b):
    return float((mx.abs(a - b).max() / mx.maximum(mx.abs(a).max(), 1e-12)).item())


@pytest.mark.parametrize("chunk", [1, 16, 33])
def test_chunked_ce_equals_full_logits_stage_loss(chunk):
    from chronos.mlx.training.trainer import _ce_stage_loss
    full_cfg, chunk_cfg = _config(), _config(ce_chunk_size=chunk)
    ids, labels = _batch(full_cfg)
    a = _grads(_model(full_cfg), _ce_stage_loss, ids, labels)
    b = _grads(_model(chunk_cfg), _ce_stage_loss, ids, labels)
    assert abs(a[0].item() - b[0].item()) < 1e-5
    assert a[1].keys() == b[1].keys()
    for name, g in a[1].items():
        assert _rel(g, b[1][name]) < 1e-4, name


def test_chunked_ce_aligned_labels_match_reference():
    from chronos.mlx.chunked_ce import chunked_lm_loss
    mx.random.seed(2)
    h = mx.random.normal((2, 50, 32))
    w = mx.random.normal((97, 32)) * 0.1
    lab = mx.random.randint(0, 97, (2, 50))
    lab = mx.where(mx.random.uniform(shape=lab.shape) < 0.3, -100, lab)
    lg = (h @ w.T).reshape(-1, 97)
    m = lab.reshape(-1) != -100
    ref = (nn.losses.cross_entropy(lg, mx.where(m, lab.reshape(-1), 0), reduction="none") * m).sum() / m.sum()
    for chunk in (1, 7, 64, 1000):
        assert abs(chunked_lm_loss(h, w, lab, chunk=chunk).item() - ref.item()) < 1e-5


def test_all_ignored_labels_give_zero_loss():
    from chronos.mlx.chunked_ce import chunked_lm_loss
    h = mx.ones((1, 9, 8))
    w = mx.ones((5, 8))
    assert chunked_lm_loss(h, w, mx.full((1, 9), -100), chunk=4).item() == 0.0


def test_ce_chunk_field_round_trips_through_the_schema():
    from chronos.config_schema import MODEL_FIELDS
    assert "ce_chunk_size" in MODEL_FIELDS
    assert _config(ce_chunk_size=4096).ce_chunk_size == 4096
    assert _config().ce_chunk_size == 0


@pytest.mark.parametrize("case", [{}, dict(recurrent_attention_mode="per_iteration"), dict(use_hybrid_attention=False),
                         dict(sliding_window_size=64)])
def test_blockwise_attention_model_equals_single_kernel(case):
    from chronos.mlx.training.trainer import _ce_stage_loss
    ref_cfg, blk_cfg = _config(**case), _config(attention_block_size=8, **case)
    ids, labels = _batch(ref_cfg)
    a = _grads(_model(ref_cfg), _ce_stage_loss, ids, labels)
    b = _grads(_model(blk_cfg), _ce_stage_loss, ids, labels)
    assert abs(a[0].item() - b[0].item()) < 1e-5
    for name, g in a[1].items():
        assert _rel(g, b[1][name]) < 1e-4, name


def test_blockwise_attention_keeps_decode_identical():
    from chronos.mlx.model import ChronosMLXModel
    ref_cfg, blk_cfg = _config(), _config(attention_block_size=8)
    ids, _ = _batch(ref_cfg, b=1, s=30)
    outs = []
    for cfg in (ref_cfg, blk_cfg):
        mx.random.seed(0)
        m = ChronosMLXModel(cfg)
        m.eval()
        logits, _, cache = m(ids[:, :24])
        step, _, _ = m(ids[:, 24:25], cache=cache)
        mx.eval(logits, step)
        outs.append((logits, step))
    assert _rel(outs[0][0], outs[1][0]) < 1e-5
    assert _rel(outs[0][1], outs[1][1]) < 1e-5


@pytest.mark.parametrize("block", [0, 8])
def test_document_masking_isolates_packed_documents(block):
    """With document masking, a document's logits inside a packed row equal
    the logits of that document alone (attention never crosses the separator)."""
    from chronos.mlx.model import ChronosMLXModel
    cfg = _config(attention_block_size=block, sliding_window_size=64, recurrent_depth=1, min_recurrent_depth=1,
                  rope_layout="per_head",
                  adaptive_halting=False)
    mx.random.seed(0)
    model = ChronosMLXModel(cfg)
    model.eval()
    from chronos.expert_clusters import segment_ids
    import numpy as np
    first = list(range(10, 23)) + [0]          # document 1 ends with its separator
    second = list(range(30, 49))
    row = mx.array([first + second])
    seg = mx.array(segment_ids(np.array(row)))
    packed, _, _ = model(row, segments=seg)
    alone, _, _ = model(mx.array([second]))
    mx.eval(packed, alone)
    # Document masking removes every cross-document term.  What remains is
    # the absolute-position offset: RoPE is relative in exact arithmetic,
    # but FP32 cos/sin tables at shifted rows round differently (measured
    # 8.8e-4 for the same document shifted by 14 positions, dense model).
    assert _rel(alone[0], packed[0, len(first):]) < 2e-3
    leaky, _, _ = model(row)
    assert _rel(alone[0], leaky[0, len(first):]) > 1e-3     # without masking the context differs


def test_router_pairs_do_not_cross_documents():
    from chronos.mlx.training import trainer as T
    seg = mx.array([[0, 0, 0, 1, 1]])
    T._PAIR_SEGMENTS[0] = seg
    try:
        m = T._pair_mask(None, 1)
    finally:
        T._PAIR_SEGMENTS[0] = None
    assert m.tolist() == [[1.0, 1.0, 0.0, 1.0]]
    assert T._pair_mask(None, 1) is None



# ---- 0.3.2a1 part 2: MoE chunks, selective recompute, RoPE, lookahead, curriculum, packing ----

def _grad_equal(a, b, tol=1e-4):
    assert abs(a[0].item() - b[0].item()) < 1e-5
    assert a[1].keys() == b[1].keys()
    for name, g in a[1].items():
        assert _rel(g, b[1][name]) < tol, name


@pytest.mark.parametrize("chunk", [7, 16])
@pytest.mark.parametrize("miss", [0.0, 0.4])
def test_moe_chunked_dispatch_equals_single_dispatch(chunk, miss):
    from chronos.mlx.training.trainer import _ce_stage_loss
    ref_cfg = _config(fallback_mask_prob=miss)
    chk_cfg = _config(moe_chunk_tokens=chunk, fallback_mask_prob=miss)
    ids, labels = _batch(ref_cfg)
    mx.random.seed(5)
    a = _grads(_model(ref_cfg), _ce_stage_loss, ids, labels)
    mx.random.seed(5)
    b = _grads(_model(chk_cfg), _ce_stage_loss, ids, labels)
    _grad_equal(a, b)


@pytest.mark.parametrize("policy", ["all", "mla", "swa", "alternate"])
def test_selective_recompute_keeps_loss_and_gradients(policy):
    from chronos.mlx.training.trainer import _ce_stage_loss
    ref_cfg = _config()
    rc_cfg = _config(activation_checkpointing=True, recompute_policy=policy)
    ids, labels = _batch(ref_cfg)
    a = _grads(_model(ref_cfg), _ce_stage_loss, ids, labels)
    b = _grads(_model(rc_cfg), _ce_stage_loss, ids, labels)
    _grad_equal(a, b, tol=5e-4)


def test_selective_recompute_policy_picks_the_blocks():
    from chronos.mlx.model import _recompute_block
    model = _model(_config())
    picked = {}
    for policy in ("all", "mla", "swa", "alternate"):
        model.config.recompute_policy = policy
        picked[policy] = [_recompute_block(model.config, i, b) for i, b in enumerate(model.layers)]
    assert all(picked["all"])
    assert picked["mla"] == [not v for v in picked["swa"]]
    assert any(picked["mla"]) and not all(picked["mla"])
    assert picked["alternate"] == [i % 2 == 0 for i in range(len(model.layers))]
    with pytest.raises(ValueError):
        _config(recompute_policy="nope")


@pytest.mark.parametrize("kind", ["linear", "ntk", "yarn"])
def test_rope_scaling_tables_match_between_mlx_and_torch(kind):
    pytest.importorskip("torch")
    import numpy as np
    from chronos.mlx.attention import _rope_freqs
    from chronos.model.kernel import precompute_freqs_cis
    sc = {"type": kind, "factor": 4, "original_max_position_embeddings": 64, "beta_fast": 32, "beta_slow": 1}
    c_t, s_t = precompute_freqs_cis(32, 256, 1e6, sc)
    c_m, s_m = _rope_freqs(32, max_len=256, base=1e6, rope_scaling=sc)
    assert np.abs(np.array(c_m) - c_t.numpy()).max() < 2e-5
    assert np.abs(np.array(s_m) - s_t.numpy()).max() < 2e-5
    base_c, _ = precompute_freqs_cis(32, 256, 1e6, None)
    assert np.abs(base_c.numpy() - c_t.numpy()).max() > 1e-3   # scaling really changes the table


def test_linear_rope_scaling_compresses_positions():
    import numpy as np
    from chronos.mlx.attention import _rope_freqs
    plain, _ = _rope_freqs(16, max_len=64, base=1e4, rope_scaling=None)
    lin, _ = _rope_freqs(16, max_len=64, base=1e4, rope_scaling={"type": "linear", "factor": 2})
    assert np.abs(np.array(lin[8]) - np.array(plain[4])).max() < 1e-5


def test_rope_train_scaling_sets_rope_scaling_and_round_trips():
    from chronos.config_schema import MODEL_FIELDS
    cfg = _config(rope_train_scaling={"type": "ntk", "factor": 8})
    assert cfg.rope_scaling == {"type": "ntk", "factor": 8}
    assert _config().rope_scaling is None
    new = {"rope_train_scaling", "recompute_policy", "moe_chunk_tokens", "reset_lookahead_at_documents"}
    assert new <= set(MODEL_FIELDS)


def test_lookahead_document_rule_is_decoupled_from_router_rule():
    from chronos.mlx.training import trainer as T
    base = dict(reset_router_at_documents=True, lambda_lookahead=1.0, lambda_temporal=1.0)
    ids, labels = _batch(_config())
    ids = mx.where(mx.arange(ids.shape[1])[None, :] % 9 == 8, 0, ids)      # separators inside each row
    cases = {"follow": {}, "off": dict(reset_lookahead_at_documents=False),
      "on": dict(reset_lookahead_at_documents=True)}
    vals = {}
    for name, extra in cases.items():
        model = _model(_config(**base, **extra))
        T._ce_stage_loss(model, ids, labels)
        vals[name] = {k: float(v.item()) for k, v in model._last_loss_components.items()}
    assert vals["follow"]["lookahead_loss"] == pytest.approx(vals["on"]["lookahead_loss"], rel=1e-6)
    assert vals["off"]["lookahead_loss"] != pytest.approx(vals["on"]["lookahead_loss"], rel=1e-6)
    assert vals["off"]["temporal_loss"] == pytest.approx(vals["on"]["temporal_loss"], rel=1e-6)
    assert T._LOOKAHEAD_SEGMENTS[0] is None and T._PAIR_SEGMENTS[0] is None


def test_length_curriculum_steps_up_with_progress():
    from chronos.mlx.training.longrun import PhaseSpec, curriculum_seq_len
    kw = dict(name="pretrain", budget_fraction=1.0, peak_lr=1e-3, tokens_per_step=1024)
    ph = PhaseSpec(seq_len=512, seq_len_curriculum=[[0.0, 512], [0.3, 2048], [0.7, 8192]], **kw)
    got = [curriculum_seq_len(ph, p) for p in (0.0, 0.29, 0.3, 0.69, 0.7, 1.0)]
    assert got == [512, 512, 2048, 2048, 8192, 8192]
    assert curriculum_seq_len(PhaseSpec(seq_len=256, **kw), 0.9) == 256


def _write_sft(folder, records, with_lmask):
    import numpy as np
    os.makedirs(folder, exist_ok=True)
    flat = np.concatenate([np.asarray(r, np.uint16) for r, _ in records])
    off = np.concatenate([[0], np.cumsum([len(r) for r, _ in records])]).astype(np.int64)
    flat.tofile(os.path.join(folder, "a.bin"))
    off.tofile(os.path.join(folder, "a.idx"))
    np.asarray([p for _, p in records], np.int32).tofile(os.path.join(folder, "a.plen"))
    if with_lmask:
        mask = np.concatenate([np.arange(len(r)) >= p for r, p in records]).astype(np.uint8)
        mask.tofile(os.path.join(folder, "a.lmask"))


@pytest.mark.parametrize("with_lmask", [False, True])
def test_sft_packing_fills_rows_with_separated_records(tmp_path, with_lmask):
    import numpy as np
    from chronos.data.shards import SFTRecords
    from chronos.expert_clusters import segment_ids
    recs = [([5, 6, 7, 8], 2), ([9, 10, 11], 1), ([12, 13, 14, 15, 16], 3)]
    _write_sft(str(tmp_path), recs, with_lmask)
    rec = SFTRecords(str(tmp_path), seed=0)
    ids, labels = rec.packed_batch(token_budget=64, seq_len=15)
    assert ids.shape == labels.shape == (4, 16)
    assert not np.any(ids[:, 0] == 0)    # a row starts with a record
    assert set(ids[ids != 0].tolist()) <= {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
    assert (ids == 0).sum() <= ids.shape[0] * 5        # at most one separator per 3+ record tokens
    assert not np.any((labels != -100) & (ids == 0))  # separators carry no loss
    allowed = {7, 8, 10, 11, 14, 15, 16}       # reply tokens of the three records
    assert set(labels[labels != -100].tolist()) <= allowed
    assert segment_ids(ids).max() >= 1  # several documents per row


def test_longrun_pretrain_source_follows_the_curriculum(tmp_path):
    import numpy as np
    from chronos.mlx.training.longrun import PretrainSource
    for split in ("train", "valid"):
        d = tmp_path / "pretrain" / split
        d.mkdir(parents=True)
        (np.arange(5000, dtype=np.uint16) % 90).tofile(d / "a.bin")
    src = PretrainSource(str(tmp_path), 64, seed=0)
    assert src.batch(256)[0].shape == (4, 64)
    src.set_seq_len(128)
    assert src.batch(256)[0].shape == (2, 128)


# ---- Torch parity (0.3.2a1): selective recompute on the Torch side ----

@pytest.mark.parametrize("policy", ["all", "mla", "swa", "alternate"])
def test_torch_selective_recompute_policy_keeps_gradients(policy):
    torch = pytest.importorskip("torch")
    from chronos.model.model_chronos import ChronosForCausalLM

    def run(**kw):
        torch.manual_seed(0)
        cfg = _config(**kw)
        model = ChronosForCausalLM(cfg)
        model.train()
        ids = torch.randint(3, cfg.vocab_size, (2, 24))
        out = model(ids, labels=ids)
        out.loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        return out.loss.item(), grads

    base_loss, base = run()
    loss, grads = run(activation_checkpointing=True, recompute_policy=policy)
    assert abs(loss - base_loss) < 1e-5
    assert base.keys() == grads.keys()
    for name, g in base.items():
        denom = max(float(g.abs().max()), 1e-12)
        assert float((g - grads[name]).abs().max()) / denom < 1e-3, name


def test_torch_recompute_policy_selects_expected_layers():
    pytest.importorskip("torch")
    from chronos.model.hybrid_attention import MLAAttention
    from chronos.model.model_chronos import ChronosModel
    model = ChronosModel(_config())
    is_mla = [isinstance(layer.self_attn, MLAAttention) for layer in model.layers]
    alternate = [i % 2 == 0 for i in range(len(is_mla))]
    expect = {"all": [True] * len(is_mla), "mla": is_mla, "swa": [not v for v in is_mla], "alternate": alternate}
    for policy, want in expect.items():
        model.config.recompute_policy = policy
        assert [model._recompute_block(layer) for layer in model.layers] == want
