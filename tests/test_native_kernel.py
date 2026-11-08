"""Native kernel (0.3.2+): no minimind, same numerics, working dense mode."""
from __future__ import annotations

import math
import os
import subprocess
import sys

import pytest
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos.model import kernel  # noqa: E402
from chronos.model.config import ChronosConfig  # noqa: E402
from chronos.model.model_chronos import ChronosForCausalLM  # noqa: E402

MODES = ("shared_first", "per_iteration", "first_only")


def test_import_needs_no_external_kernel(tmp_path):
    """``import chronos`` must not clone, download or edit sys.path."""
    code = (
        "import sys; before = list(sys.path); import chronos, chronos.model.model_chronos, chronos.deps;"
        "import chronos.trainer.chronos_trainer, chronos.trainer.sft_trainer;"
        "assert sys.path == before, set(sys.path) ^ set(before);"
        "bad = [m for m in ('model', 'trainer', 'dataset', 'model.model_minimind') if m in sys.modules];"
        "assert not bad, bad;"
        "assert not any('minimind' in p.lower() for p in sys.path)"
    )
    env = {**os.environ, "CHRONOS_BACKEND": "cpu", "PYTHONPATH": _repo_root,
           "HOME": str(tmp_path)}                  # no ~/.cache/chronos checkout to find
    env.pop("CHRONOS_MINIMIND_DIR", None)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]


def test_base_config_defaults_match_previous_releases():
    c = kernel.ChronosBaseConfig()
    assert (c.hidden_size, c.num_hidden_layers, c.vocab_size, c.bos_token_id, c.eos_token_id) == (768, 8, 6400, 1, 2)
    assert (c.num_attention_heads, c.num_key_value_heads, c.head_dim) == (8, 4, 96)
    assert c.intermediate_size == math.ceil(768 * math.pi / 64) * 64 == 2432
    assert (c.max_position_embeddings, c.rms_norm_eps, c.rope_theta) == (32768, 1e-6, 1e6)
    assert c.rope_scaling is None and c.tie_word_embeddings is True and c.hidden_act == "silu"
    assert (c.num_experts, c.num_experts_per_tok, c.norm_topk_prob, c.router_aux_loss_coef) == (4, 1, True, 5e-4)
    y = kernel.ChronosBaseConfig(hidden_size=512, num_attention_heads=8, inference_rope_scaling=True)
    assert y.head_dim == 64 and y.intermediate_size == 1664 and y.moe_intermediate_size == 1664
    # transformers >= 5.2x adds its own ``rope_type``/``rope_theta`` keys; the YaRN
    # fields (all the RoPE tables read) are the defaults, and the shared dict is untouched.
    assert {k: y.rope_scaling[k] for k in kernel.YARN_DEFAULT} == kernel.YARN_DEFAULT
    assert y.rope_scaling is not kernel.YARN_DEFAULT and set(kernel.YARN_DEFAULT) == {
        "beta_fast", "beta_slow", "factor", "original_max_position_embeddings", "attention_factor", "type"}
    long = 3 * kernel.YARN_DEFAULT["original_max_position_embeddings"]
    assert torch.equal(kernel.precompute_freqs_cis(32, long, 1e6, y.rope_scaling)[0],
                       kernel.precompute_freqs_cis(32, long, 1e6, kernel.YARN_DEFAULT)[0])
    assert isinstance(ChronosConfig(), kernel.ChronosBaseConfig)


def test_rope_closed_form_and_layout():
    cos, sin = kernel.precompute_freqs_cis(8, 16, rope_base=10000.0)
    assert cos.dtype == torch.float32 and cos.shape == (16, 8)
    inv = 1.0 / (10000.0 ** (torch.arange(0, 8, 2).float() / 8))
    ang = torch.arange(16).float()[:, None] * inv[None]
    torch.testing.assert_close(cos[:, :4], torch.cos(ang), rtol=0, atol=1e-6)
    torch.testing.assert_close(sin[:, 4:], torch.sin(ang), rtol=0, atol=1e-6)    # duplicated halves
    assert torch.equal(cos[0], torch.ones(8)) and torch.equal(sin[0], torch.zeros(8))


@pytest.mark.parametrize("scaling", [None, kernel.YARN_DEFAULT])
def test_shared_rope_tables_are_bit_identical_and_shared(scaling):
    kernel.clear_rope_tables()
    for length in (5, 64, 2048, 2049, 4100, 9000, 100):
        cos, sin = kernel.rope_tables(32, length, 1e6, scaling, "cpu")
        ref_cos, ref_sin = kernel.precompute_freqs_cis(32, length, 1e6, scaling)
        assert cos.shape[0] >= length
        assert torch.equal(cos[:length], ref_cos) and torch.equal(sin[:length], ref_sin), length
    a = kernel.rope_tables(32, 10, 1e6, scaling, "cpu")[0]
    b = kernel.rope_tables(32, 20, 1e6, scaling, "cpu")[0]
    assert a.data_ptr() == b.data_ptr()                                          # one table, not a copy


def test_model_layers_share_one_rope_table():
    kernel.clear_rope_tables()
    cfg = ChronosConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        vocab_size=97, num_experts=4, num_experts_per_tok=2, num_shared_experts=1,
                        max_position_embeddings=128, recurrent_depth=2, prelude_layers=1, coda_layers=1,
                        dropout=0.0, use_hybrid_attention=False)
    model = ChronosForCausalLM(cfg).eval()
    with torch.no_grad():
        model(torch.randint(3, 97, (1, 12)))
    ptrs = {m._rope_cos.data_ptr() for m in model.modules() if getattr(m, "_rope_cos", None) is not None}
    assert len(ptrs) == 1
    assert model.model.freqs_cos.data_ptr() in ptrs
    model.to(torch.bfloat16)
    assert model.model.freqs_cos.dtype == torch.float32                          # never rounded


def dense_model(mode, **kw):
    torch.manual_seed(7)
    cfg = ChronosConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        vocab_size=96, num_experts=4, num_experts_per_tok=2, num_shared_experts=1,
                        moe_intermediate_size=64, max_position_embeddings=128, recurrent_depth=3,
                        prelude_layers=1, coda_layers=1, dropout=0.0, fallback_mask_prob=0.0,
                        use_hybrid_attention=False, recurrent_attention_mode=mode, **kw)
    return ChronosForCausalLM(cfg).eval()


@pytest.mark.parametrize("mode", MODES)
def test_dense_gqa_mode_prefill_decode_matches_full(mode):
    """``use_hybrid_attention=False`` crashed before 0.3.2 in every recurrent mode."""
    from chronos.model.hybrid_attention import GlobalAttention
    model = dense_model(mode)
    assert all(isinstance(layer.self_attn, GlobalAttention) for layer in model.model.layers)
    ids = torch.randint(3, 96, (2, 20))
    with torch.no_grad():
        full = model(ids).logits
        out = model(ids[:, :12], use_cache=True)
        cache, steps = out.past_key_values, []
        for t in range(12, 20):
            out = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            steps.append(out.logits[:, -1])
        mask = torch.ones_like(ids)
        mask[1, :5] = 0
        masked = model(ids, attention_mask=mask).logits
    torch.testing.assert_close(torch.stack(steps, 1), full[:, 12:20], rtol=0, atol=1e-5)
    assert torch.isfinite(masked).all()
    torch.testing.assert_close(masked[0], full[0], rtol=0, atol=1e-5)          # unpadded row unchanged


def test_dense_gqa_int8_kv_and_speculative():
    from chronos.runtime.speculative import speculative_generate
    ref_model, q_model = dense_model("shared_first"), dense_model("shared_first", kv_cache_dtype="int8")
    ids = torch.randint(3, 96, (1, 9))
    with torch.no_grad():
        full = ref_model(ids).logits[:, -1]
        pre = q_model(ids[:, :8], use_cache=True)
        step = q_model(ids[:, 8:], past_key_values=pre.past_key_values, use_cache=True).logits[:, -1]
        assert (step - full).abs().max() < 5e-2
        seq = ids[0].tolist()
        out = ref_model(ids, use_cache=True)
        cache = out.past_key_values
        for _ in range(12):
            seq.append(int(out.logits[0, -1].argmax()))
            out = ref_model(torch.tensor([[seq[-1]]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
        spec, _stats = speculative_generate(ref_model, ids, max_new_tokens=12, gamma=4, draft_depth=1, gamma_max=8)
    assert spec[0].tolist() == seq


def test_logger_sinks_receive_lines(capsys):
    lines = []
    with kernel.add_log_sink(lines.append):
        kernel.Logger("step 1")
        kernel.Logger({"loss": 2})
    kernel.Logger("after")
    assert lines == ["step 1", "{'loss': 2}"]
    assert "after" in capsys.readouterr().out


def test_trainer_modules_use_the_shared_logger():
    import chronos.trainer.chronos_trainer as ct
    import chronos.trainer.sft_trainer as st
    from chronos.workbench.training import _trainer_logger_sink
    assert ct.Logger is kernel.Logger and st.Logger is kernel.Logger
    seen = []
    with _trainer_logger_sink(seen.append):
        ct.Logger("from trainer")                # name bound at import time
    assert seen == ["from trainer"]


def test_resume_state_roundtrip_keeps_dtype(tmp_path):
    net = torch.nn.Linear(4, 3).to(torch.bfloat16)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    net(torch.randn(2, 4, dtype=torch.bfloat16)).float().sum().backward()
    opt.step()
    path = kernel.save_resume_state(str(tmp_path / "sub" / "r.pth"), model=net, optimizer=opt,
                                    epoch=2, step=17, wandb_id="abc", scaler=None)
    assert not os.path.exists(path + ".tmp")
    data = kernel.load_resume_state(path)
    assert (data["epoch"], data["step"], data["world_size"], data["wandb_id"]) == (2, 17, 1, "abc")
    assert "scaler" not in data
    assert data["model"]["weight"].dtype == torch.bfloat16
    assert torch.equal(data["model"]["weight"], net.weight.detach())
    opt.load_state_dict(data["optimizer"])
    assert kernel.load_resume_state(str(tmp_path / "missing.pth")) is None


def test_legacy_tokenizer_is_bundled(monkeypatch, tmp_path):
    import chronos.deps as deps
    monkeypatch.delenv("CHRONOS_DEFAULT_TOKENIZER", raising=False)
    path = deps.get_tokenizer_path()
    for name in ("tokenizer.json", "tokenizer_config.json", "LICENSE", "NOTICE"):
        assert os.path.isfile(os.path.join(path, name)), name
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(path)
    assert len(tok) == 6400
    assert tok.decode(tok("hello 世界")["input_ids"], skip_special_tokens=True).strip() == "hello 世界"
    monkeypatch.setenv("CHRONOS_DEFAULT_TOKENIZER", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        deps.get_tokenizer_path()
    with pytest.warns(DeprecationWarning):
        assert deps.ensure_minimind() is None


def test_pretrain_save_keeps_bf16_release_checkpoint(tmp_path):
    """Before 0.3.2 the resume helper re-wrote chronos_<H>_moe.pth as FP16."""
    import types
    from chronos.trainer.chronos_trainer import ChronosTrainer
    cfg = ChronosConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        vocab_size=97, num_experts=4, num_experts_per_tok=2, num_shared_experts=1,
                        max_position_embeddings=64, recurrent_depth=2, prelude_layers=1, coda_layers=1, dropout=0.0)
    args = types.SimpleNamespace(device="cpu", dtype="bfloat16", learning_rate=1e-4, save_dir=str(tmp_path),
                                 epochs=1, accumulation_steps=1, grad_clip=1.0, log_interval=1, save_interval=1)
    trainer = ChronosTrainer(cfg, args)
    trainer._save(0, 3)
    state = torch.load(tmp_path / "chronos_64_moe.pth", map_location="cpu", weights_only=False)
    state = state.get("model", state)
    assert {v.dtype for v in state.values() if torch.is_tensor(v) and v.is_floating_point()} == {torch.bfloat16}
    resume = kernel.load_resume_state(str(tmp_path / "chronos_64_moe_resume.pth"))
    assert resume["step"] == 3 and resume["optimizer"] is not None


def test_rope_tables_under_meta_device_context():
    """The planner counts parameters under ``torch.device('meta')``."""
    kernel.clear_rope_tables()
    with torch.device("meta"):
        cos, _ = kernel.rope_tables(16, 32, 1e6, None, "cpu")
        model = ChronosForCausalLM(ChronosConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                                                 num_key_value_heads=2, vocab_size=97, num_experts=4,
                                                 num_experts_per_tok=2, max_position_embeddings=64))
    assert cos.device.type == "cpu" and torch.isfinite(cos).all()
    assert model.model.freqs_cos.device.type == "cpu"
    assert next(model.parameters()).device.type == "meta"
