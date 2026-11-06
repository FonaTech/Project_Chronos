"""Numerical and real frontend contracts that short forward smoke tests miss."""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch

from chronos.model.config import ChronosConfig
from chronos.model.model_chronos import ChronosForCausalLM
from chronos.model.recurrent import recurrent_auxiliary_state


def tiny(**overrides):
    args = dict(hidden_size=16, num_hidden_layers=4, num_attention_heads=2,
                num_key_value_heads=2, head_dim=8, kv_latent_dim=8, rope_dim=4,
                intermediate_size=32, moe_intermediate_size=32, num_experts=2,
                num_experts_per_tok=2, num_shared_experts=1, vocab_size=32,
                max_seq_len=16, recurrent_depth=3, min_recurrent_depth=2,
                prelude_layers=1, recurrent_groups=2, coda_layers=1,
                recurrent_enabled=True, layer_aware_lookahead=True, dropout=0.)
    args.update(overrides)
    return ChronosConfig(**args)


def test_shared_transition_consistency_gradient_penalizes_expansion():
    def objective(w):
        states = [(i*w).reshape(1, 1, 1) for i in range(1, 5)]
        return recurrent_auxiliary_state(states, [torch.full((1, 1), .3)]*4,
                                         min_depth=2, threshold=.99)[1].distill_loss
    w = torch.tensor(1., requires_grad=True)
    value = objective(w)
    value.backward()
    # The old detached teacher gave -6.667, increasing the very loss it claimed
    # to minimize. Check against the actual objective, not another implementation.
    numeric = (objective(w.detach()+1e-3)-objective(w.detach()-1e-3))/(2e-3)
    torch.testing.assert_close(w.grad, numeric, rtol=1e-3, atol=1e-3)
    assert objective(w.detach()-.01*w.grad) < value


@pytest.mark.parametrize("overrides", [
    {}, {"recurrent_enabled": False}, {"recurrent_depth": 1},
    {"num_hidden_layers": 3}, {"num_hidden_layers": 1},
    {"use_hybrid_attention": False}, {"tie_word_embeddings": False},
    {"use_moe": False}, {"intermediate_size": 0, "moe_intermediate_size": 0},
    {"modalities": ["image"], "image_patch_size": 8},
    {"modalities": ["audio"]}, {"modalities": ["image", "audio"]},
    {"layer_aware_lookahead": False, "num_experts_per_tok": 1},
])
def test_estimator_matches_allocated_unique_parameters(overrides):
    from ui.estimator import ArchConfig, total_params, active_param_bounds, memory_footprint
    cfg = tiny(**overrides)
    with torch.device("meta"):
        model = ChronosForCausalLM(cfg)
    arch = ArchConfig.from_config(cfg)
    assert total_params(arch) == sum(p.numel() for p in model.parameters())
    low, high = active_param_bounds(arch)
    assert 0 < low <= high <= total_params(arch)
    mem = memory_footprint(arch)
    assert mem["resident_weight_bytes"] + mem["cold_bytes"] == mem["total_bytes"]
    assert mem["mlx_training_state_bytes"] == (mem["total_bytes"] + mem["optimizer_bytes"]
        + mem["mlx_master_weight_bytes"] + mem["mlx_accumulated_gradient_bytes"])


def test_repeated_depth_changes_work_without_duplicating_blocks():
    from ui.estimator import ArchConfig, total_params, compute_params
    a, b = ArchConfig.from_config(tiny(recurrent_depth=2)), ArchConfig.from_config(tiny(recurrent_depth=5))
    assert total_params(b)-total_params(a) == 3*a.hidden_size  # loop embeddings only
    assert compute_params(b) > compute_params(a)


@pytest.mark.parametrize("depth,adaptive", [(1, False), (3, False), (3, True)])
def test_mlx_recurrent_graph_and_checkpoint_match_torch(depth, adaptive):
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    from chronos.mlx.model import ChronosMLXModel
    from chronos.mlx.training.io import mlx_state_to_torch
    cfg = tiny(recurrent_depth=depth, adaptive_halting=adaptive)
    torch.manual_seed(17)
    pt = ChronosForCausalLM(cfg).eval()
    # CPU execution isolates graph parity from Metal matmul rounding and MoE
    # near-tie selection. Other tests exercise real Metal updates in BF16.
    with mx.stream(mx.cpu):
        model = ChronosMLXModel.from_chronos_pytorch(pt, cfg).eval()
        cache_pt, cache_mx = None, None
        for ids in (torch.tensor([[1, 4, 2, 5]]), torch.tensor([[6]])):
            with torch.no_grad():
                out = pt(ids, past_key_values=cache_pt, use_cache=True)
            logits, prediction, cache_mx = model(mx.array(ids.numpy()), cache=cache_mx)
            mx.eval(logits, prediction)
            np.testing.assert_allclose(np.array(logits), out.logits.numpy(), atol=2e-5, rtol=2e-4)
            assert prediction.shape == (*ids.shape, cfg.lookahead_steps+1, cfg.recurrent_groups, cfg.num_experts)
            cache_pt = out.past_key_values
        pt.load_state_dict(mlx_state_to_torch(model, cfg), strict=True)


def test_mlx_fp32_masters_preserve_small_bf16_updates_and_match_adamw():
    mx = pytest.importorskip("mlx.core")
    from chronos.mlx.training.optim import MixedPrecisionAdamW
    from chronos.mlx.training.trainer import _add_grads
    from mlx.utils import tree_flatten
    opt = MixedPrecisionAdamW(learning_rate=1e-4, weight_decay=0)
    parameters = {"weight": mx.ones((2,), dtype=mx.bfloat16)}
    p = torch.nn.Parameter(torch.ones(2))
    reference = torch.optim.AdamW([p], lr=1e-4, betas=(.9, .95), weight_decay=0, eps=1e-8)
    for _ in range(40):
        grads = None
        for _ in range(8):
            grads = _add_grads(grads, {"weight": mx.full((2,), 1/8, dtype=mx.bfloat16)})
        parameters = opt.apply_gradients(grads, parameters)
        mx.eval(parameters, opt.state)
        p.grad = torch.ones_like(p)
        reference.step()
    assert parameters["weight"].dtype == mx.bfloat16
    assert parameters["weight"][0].item() < 1.
    torch.testing.assert_close(torch.tensor(opt.state["weight"]["master"].tolist()), p, rtol=0, atol=3e-6)
    assert all(v.dtype == mx.float32 for k, v in tree_flatten(opt.state) if k.endswith((".m", ".v", ".master")))


def test_long_prefill_is_causal_and_matches_chunked_and_token_decode():
    mx = pytest.importorskip("mlx.core")
    import numpy as np
    from chronos.mlx.model import ChronosMLXModel
    from ui.estimator import ArchConfig, kv_cache_bytes
    cfg = tiny(sliding_window_size=3, adaptive_halting=False)
    torch.manual_seed(19)
    pt = ChronosForCausalLM(cfg).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.no_grad(), mx.stream(mx.cpu):
        model = ChronosMLXModel.from_chronos_pytorch(pt, cfg).eval()
        full = pt(ids, use_cache=True).logits
        changed = ids.clone(); changed[:, 4:] = 9
        torch.testing.assert_close(pt(changed).logits[:, :4], full[:, :4], atol=1e-6, rtol=1e-5)
        for widths in ([1]*8, [2, 3, 3]):
            a, b, offset = None, None, 0
            for width in widths:
                chunk = ids[:, offset:offset+width]
                out = pt(chunk, past_key_values=a, use_cache=True)
                logits, _, b = model(mx.array(chunk.numpy()), cache=b)
                mx.eval(logits)
                torch.testing.assert_close(out.logits, full[:, offset:offset+width], atol=2e-5, rtol=2e-4)
                np.testing.assert_allclose(np.array(logits), out.logits.numpy(), atol=2e-5, rtol=2e-4)
                a = out.past_key_values
                offset += width
            arch = ArchConfig.from_config(cfg); arch.dtype = "float32"
            assert sum(t.nbytes for entry in b for t in entry if isinstance(t, mx.array)) == kv_cache_bytes(arch, 8)


def test_mlx_reports_real_components_and_preserves_checkpoint_on_finite_explosion(tmp_path, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from torch.utils.data import DataLoader, TensorDataset
    import chronos.mlx.training.trainer as trainer
    cfg = tiny()
    calls = 0
    original = trainer._ce_stage_loss
    def inject(model, ids, labels):
        nonlocal calls
        calls += 1
        loss = original(model, ids, labels)
        return loss if calls == 1 else loss*0 + 10000
    monkeypatch.setattr(trainer, "_ce_stage_loss", inject)
    checkpoint = tmp_path / "chronos_16_moe.pth"
    checkpoint.write_bytes(b"existing model must survive")
    ids = torch.tensor([[1, 2, 3, 4]]*3)
    events = []
    result = trainer.run_mlx_stage(stage="pretrain", config=cfg, checkpoint_path=None,
        save_dir=str(tmp_path), loader=DataLoader(TensorDataset(ids, ids), batch_size=1),
        args=SimpleNamespace(steps=3, epochs=1, log_interval=1, dtype="bfloat16", grad_clip=1.),
        progress_callback=events.append)
    assert result.failure_reason == "finite_loss_divergence"
    assert result.stopped and result.checkpoint_path.endswith(".recovery.pth")
    assert checkpoint.read_bytes() == b"existing model must survive"
    assert not any(e["event"] == "complete" for e in events)
    step = next(e for e in events if e["event"] == "step")
    assert step["loss"] == pytest.approx(sum(step["loss_components"].values()), abs=1e-5)
    assert step["ce"] < step["loss"]
    assert step["tokens_per_s"] > 0


def test_real_gradio_preset_edit_save_load_and_autotune_contract(tmp_path):
    gr = pytest.importorskip("gradio")
    if not hasattr(gr, "Blocks"):
        pytest.skip("real Gradio Blocks required")
    from gradio.state_holder import SessionState
    from ui.tabs.config_tab import build_config_tab
    from ui.tabs.autotune_tab import build_autotune_tab
    from ui.presets import get_preset, preset_names, CONFIG_INPUT_ORDER, values_in_input_order
    with gr.Blocks() as app:
        state, widgets, _ = build_config_tab()
        build_autotune_tab(state, widgets)
    session = SessionState(app)
    def handler(name):
        return next(fn for fn in app.fns.values() if fn.fn.__name__ == name)
    async def call(name, inputs):
        return await app.process_api(handler(name), inputs, state=session)
    async def exercise():
        for name in preset_names():
            result = await call("apply_preset", [name])
            assert len(result["data"]) == len(handler("apply_preset").outputs)
            assert session[state._id]["hidden_size"] == get_preset(name)["hidden_size"]
        await call("reset_minimind", [])
        cfg = dict(session[state._id], image_patch_size=8, lambda_router_locality=.031,
                   modalities=["image"], recurrent_strategy="fixed", recurrent_depth_jitter=.2,
                   max_position_embeddings=512)
        source = tmp_path / "loaded.json"
        source.write_text(json.dumps(cfg))
        await call("do_load", [str(source)])
        values = values_in_input_order(cfg)
        values[CONFIG_INPUT_ORDER.index("hidden_size")] = 256
        values[CONFIG_INPUT_ORDER.index("max_seq_len")] = 64
        # Save must see widget edits even if their input event is still queued.
        target = tmp_path / "saved.json"
        await call("do_save", [None, str(target)] + values)
        saved = json.loads(target.read_text())
        assert saved["hidden_size"] == 256
        assert saved["max_position_embeddings"] == saved["max_seq_len"] == 64
        assert saved["image_patch_size"] == 8 and saved["lambda_router_locality"] == .031
        assert saved["modalities"] == ["image"]
        await call("update_config", [None] + values)
        assert session[state._id]["hidden_size"] == 256
        await call("apply_best", [{"learning_rate": .0002}, None])
        assert session[state._id]["learning_rate"] == .0002
        assert session[state._id]["image_patch_size"] == 8
        result = await call("do_load", [str(tmp_path / "missing.json")])
        assert len(result["data"]) == len(handler("do_load").outputs)
        assert session[state._id]["image_patch_size"] == 8
    asyncio.run(exercise())
