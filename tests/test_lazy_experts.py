"""Architecture-adaptive lazy experts (chronos.lazy_experts) and the cluster-locality defaults."""
from __future__ import annotations

import json
import os
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos.expert_clusters import (  # noqa: E402
    DEFAULT_LAMBDA, inference_clusters, paging_tag, recommended_window, trained_with_clusters,
)
from chronos.lazy_experts import LazyOptions, checkpoint_cache_dir, page_layout  # noqa: E402


class Cfg:
    num_experts, num_experts_per_tok, cluster_locality_clusters = 8, 2, 0
    lambda_cluster_locality = 0.0


def _cfg(lam):
    c = Cfg()
    c.lambda_cluster_locality = lam
    return c


# ── layout selection ──────────────────────────────────────────────

def test_auto_paging_follows_how_the_model_was_trained():
    trained, plain = _cfg(0.3), _cfg(0.0)
    assert trained_with_clusters(trained) and not trained_with_clusters(plain)
    assert inference_clusters(trained) == [[0, 1, 2, 3], [4, 5, 6, 7]]       # ⌈8 / (2·2)⌉ = 2 clusters
    assert inference_clusters(plain) == [[i] for i in range(8)]
    assert inference_clusters(plain, "cluster") == inference_clusters(trained)
    assert inference_clusters(trained, "expert") == [[i] for i in range(8)]
    with pytest.raises(ValueError):
        inference_clusters(trained, "louvain")
    assert page_layout(trained)["resolved"] == "cluster" and page_layout(plain)["resolved"] == "expert"
    assert page_layout(trained)["reason"] == "trained with cluster locality"


def test_paging_tag_and_cache_dirs_never_mix_layouts(tmp_path):
    assert paging_tag([[i] for i in range(8)]) == "e"
    assert paging_tag([[0, 1, 2, 3], [4, 5, 6, 7]]) == "c2"
    ckpt = tmp_path / "m.pth"
    ckpt.write_bytes(b"x")
    cfg = _cfg(0.3)
    root = str(tmp_path / "cache")
    torch_e = checkpoint_cache_dir(str(ckpt), cfg, root=root)
    mlx_e = checkpoint_cache_dir(str(ckpt), cfg, backend="mlx", root=root)
    mlx_c = checkpoint_cache_dir(str(ckpt), cfg, backend="mlx", tag="c2", root=root)
    assert mlx_e == torch_e + "_mlx" and mlx_c == mlx_e + "_c2"            # per-expert names unchanged
    assert len({torch_e, mlx_e, mlx_c, checkpoint_cache_dir(str(ckpt), cfg, tag="c2", root=root)}) == 4


def test_recommended_window_from_record_lengths():
    assert recommended_window(None, 512) == 32
    assert recommended_window(91, 512) == 128
    assert recommended_window(5, 512) == 16
    assert recommended_window(4000, 512) == 512
    assert recommended_window(None, 8) == 8


def test_lazy_options_validate():
    assert LazyOptions().validated().generate_kwargs()["residency"] == "planned"
    for bad in (dict(paging="x"), dict(residency="queue"), dict(residency_decay=1.0),
                dict(decode_wait_s=-1), dict(ram_ratio=0)):
        with pytest.raises(ValueError):
            LazyOptions(**bad).validated()


# ── checkpoints and training defaults ─────────────────────────────

torch = pytest.importorskip("torch")


def _tiny(lam, num_experts=8):
    from chronos.model.config import ChronosConfig
    return ChronosConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         vocab_size=96, num_experts=num_experts, num_experts_per_tok=2, num_shared_experts=1,
                         moe_intermediate_size=32, max_position_embeddings=128, recurrent_depth=2,
                         prelude_layers=1, coda_layers=1, dropout=0.0, fallback_mask_prob=0.0,
                         kv_latent_dim=16, rope_dim=8, sliding_window_size=64, use_moe=True,
                         lambda_cluster_locality=lam, cluster_locality_window=16)


def _save(tmp_path, lam, name):
    from chronos.model.checkpoint import save_state_dict_with_config
    from chronos.model.model_chronos import ChronosForCausalLM
    torch.manual_seed(5)
    cfg = _tiny(lam)
    path = str(tmp_path / f"{name}.pth")
    save_state_dict_with_config(ChronosForCausalLM(cfg).eval(), path, cfg, stage="pretrain")
    return path


def test_checkpoint_records_lambda_and_legacy_sidecars_load_as_untrained(tmp_path):
    from chronos.model.checkpoint import checkpoint_config_path, chronos_config_from_checkpoint
    path = _save(tmp_path, 0.3, "trained")
    cfg, _ = chronos_config_from_checkpoint(path)
    assert cfg.lambda_cluster_locality == pytest.approx(0.3) and page_layout(cfg)["resolved"] == "cluster"
    side = checkpoint_config_path(path)
    meta = json.load(open(side, encoding="utf-8"))
    meta.pop("lambda_cluster_locality", None)                 # a sidecar written before the field existed
    if isinstance(meta.get("config"), dict):
        meta["config"].pop("lambda_cluster_locality", None)
    json.dump(meta, open(side, "w", encoding="utf-8"))
    legacy, _ = chronos_config_from_checkpoint(path)
    assert legacy.lambda_cluster_locality == 0.0 and page_layout(legacy)["resolved"] == "expert"


def test_class_default_is_off_and_new_config_entry_points_turn_it_on(monkeypatch):
    """Stored recipes without the field keep their objective; new configs get the measured default."""
    from chronos.model.config import ChronosConfig
    assert ChronosConfig().lambda_cluster_locality == 0.0
    import train_chronos
    monkeypatch.setattr(sys, "argv", ["train_chronos.py", "--data_path", "unused.jsonl"])
    assert train_chronos.parse_args().lambda_cluster_locality == DEFAULT_LAMBDA
    from ui.presets import MINIMIND_MOE_DEFAULTS
    assert MINIMIND_MOE_DEFAULTS["lambda_cluster_locality"] == DEFAULT_LAMBDA


def test_finetune_spec_turns_cluster_locality_on_by_default(tmp_path):
    from chronos.bundle.recommend import cluster_locality_config
    import numpy as np
    moe = {"num_experts": 36, "num_experts_per_tok": 2, "hidden_size": 64, "num_hidden_layers": 4}
    sft = tmp_path / "shards" / "sft" / "train"
    sft.mkdir(parents=True)
    # four records of 100 tokens → window 128 (power of two nearest the median)
    np.asarray([0, 100, 200, 300, 400], np.int64).tofile(sft / "train.x.00000.idx")
    on = cluster_locality_config(moe, str(tmp_path / "shards"), seq_len=512)
    assert on["values"] == {"lambda_cluster_locality": DEFAULT_LAMBDA, "cluster_locality_window": 128,
                            "cluster_locality_clusters": 0}
    assert "median SFT record (100 tokens)" in on["source"]["cluster_locality_window"]
    assert cluster_locality_config(moe, None, enabled=False)["values"]["lambda_cluster_locality"] == 0.0
    assert cluster_locality_config(moe, None, lambda_cluster=0.1, window=0)["values"][
        "cluster_locality_window"] == 0
    dense = cluster_locality_config(dict(moe, num_experts=4), None)          # 4 experts, k=2: one cluster
    assert dense["values"]["lambda_cluster_locality"] == 0.0


# ── engines: cluster paging is exact ──────────────────────────────

PROMPT = [5, 17, 33, 2, 61, 9, 40]


def _mlx_cpu():
    mx = pytest.importorskip("mlx.core")
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    return mx, previous


def _mlx_reference(path, n, penalty=1.0):
    """Full-memory MLX greedy (CTRL repetition penalty when != 1)."""
    import mlx.core as mx
    import numpy as np
    from chronos.mlx.model import ChronosMLXModel
    from chronos.model.checkpoint import chronos_config_from_checkpoint, load_checkpoint_into_model
    from chronos.model.model_chronos import ChronosForCausalLM
    cfg, _ = chronos_config_from_checkpoint(path)
    pt = ChronosForCausalLM(cfg)
    load_checkpoint_into_model(pt, path, map_location="cpu")
    model = ChronosMLXModel.from_chronos_pytorch(pt.eval(), cfg, dtype="float32").eval()
    seen = np.zeros(int(cfg.vocab_size), dtype=bool)
    seen[PROMPT] = True
    logits, _, cache = model(mx.array([PROMPT], dtype=mx.int32))
    out = []
    for _ in range(n):
        row = logits[0, -1].astype(mx.float32)
        if penalty != 1.0:
            row = mx.where(mx.array(seen), mx.where(row < 0, row * penalty, row / penalty), row)
        nxt = int(mx.argmax(row).item())
        out.append(nxt)
        seen[nxt] = True
        logits, _, cache = model(mx.array([[nxt]], dtype=mx.int32), cache=cache)
    return out


def _lazy_generate(path, tmp_path, opts, n=16, **gen):
    import mlx.core as mx
    from chronos.lazy_experts import build_mlx_lazy_engine, warm_ids
    from chronos.model.checkpoint import chronos_config_from_checkpoint
    cfg, _ = chronos_config_from_checkpoint(path)
    engine, info = build_mlx_lazy_engine(path, cfg, opts, cache_root=str(tmp_path / "cache"))
    try:
        engine.store.warm_up(warm_ids(cfg, info))
        toks = list(engine.generate(mx.array([PROMPT], dtype=mx.int32), max_new_tokens=n, temperature=0.0,
                                    top_p=1.0, **opts.generate_kwargs(), **gen))
        return toks, info, engine.store.stats()
    finally:
        engine.stop()


@pytest.mark.parametrize("residency", ["planned", "window"])
def test_mlx_lazy_engine_adapts_paging_and_matches_full_memory_greedy(tmp_path, residency):
    pytest.importorskip("mlx.core")
    mx, previous = _mlx_cpu()
    try:
        trained, plain = _save(tmp_path, 0.3, "trained"), _save(tmp_path, 0.0, "plain")
        ref = _mlx_reference(trained, 16)
        opts = LazyOptions(residency=residency, ram_ratio=0.5).validated()
        toks, info, st = _lazy_generate(trained, tmp_path, opts)
        assert toks == ref
        assert info["paging_resolved"] == "cluster" and info["num_pages"] == 2
        assert info["cache_dir"].endswith("_mlx_c2") and st["num_clusters"] == 2 and st["cluster_loads"] > 0
        toks2, info2, _ = _lazy_generate(trained, tmp_path, opts)               # same layout: cache reused
        assert toks2 == ref and info2["cache_reused"]
        forced = LazyOptions(paging="expert", residency=residency, ram_ratio=0.5).validated()
        toks3, info3, st3 = _lazy_generate(trained, tmp_path, forced)
        assert toks3 == ref and info3["num_pages"] == 8 and st3["num_clusters"] == 8
        assert info3["cache_dir"] != info["cache_dir"]
        toks4, info4, _ = _lazy_generate(plain, tmp_path, opts)                # untrained → per expert
        assert info4["paging_resolved"] == "expert" and info4["cache_dir"].endswith("_mlx")
        assert toks4 == _mlx_reference(plain, 16)
    finally:
        mx.set_default_device(previous)


def test_mlx_engine_repetition_penalty_and_waits(tmp_path):
    pytest.importorskip("mlx.core")
    mx, previous = _mlx_cpu()
    try:
        path = _save(tmp_path, 0.3, "trained")
        opts = LazyOptions(decode_wait_s=0.0, verify_wait_s=0.0).validated()
        toks, _info, _st = _lazy_generate(path, tmp_path, opts, n=12, repetition_penalty=1.3)
        assert toks == _mlx_reference(path, 12, penalty=1.3)
        with pytest.raises(ValueError):
            _lazy_generate(path, tmp_path, opts, n=4, repetition_penalty=1.3, speculative=True)
        spec, _i, _s = _lazy_generate(path, tmp_path, opts, n=12, speculative=True, draft_depth=1, gamma=3)
        assert spec == _mlx_reference(path, 12)
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("paging", ["auto", "expert"])
def test_torch_offload_pages_by_cluster_when_trained_and_stays_exact(tmp_path, monkeypatch, paging):
    monkeypatch.chdir(tmp_path)
    from chronos.model.checkpoint import chronos_config_from_checkpoint
    from ui.tabs.inference_tab import _clone_model_cfg, _run_torch_full_dram, _run_torch_inference
    path = _save(tmp_path, 0.3, "trained")
    cfg, _ = chronos_config_from_checkpoint(path)
    mask = [1] * len(PROMPT)
    ref, _ = _run_torch_full_dram("cpu", _clone_model_cfg(cfg), path, PROMPT, mask, 12, 1.0, None, do_sample=False)
    out, stats = _run_torch_inference("cpu", _clone_model_cfg(cfg), path, PROMPT, mask, 12, 1.0, None,
                                      do_sample=False, ram_load_ratio=0.5, paging=paging,
                                      miss_policy="sync_on_demand")      # exact: a miss loads, never falls back
    assert out == ref
    cache_root = tmp_path / "expert_cache"
    dirs = [d for d in os.listdir(cache_root)]
    if paging == "auto":
        assert stats["paging_resolved"] == "cluster" and stats["num_pages"] == 2
        assert any(d.endswith("_c2") for d in dirs)
        cdir = cache_root / next(d for d in dirs if d.endswith("_c2"))
        assert sorted(f for f in os.listdir(cdir) if f.endswith(".ctsr")) == ["cluster_0.ctsr", "cluster_1.ctsr"]
    else:
        assert stats["paging_resolved"] == "expert" and not any(d.endswith("_c2") for d in dirs)


@pytest.mark.parametrize("lam,resolved,pages", [(0.3, "cluster", 2), (0.0, "expert", 8)])
def test_serve_load_model_adapts_paging_and_reports_it(tmp_path, monkeypatch, lam, resolved, pages):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import transformers
    import chronos.serving.runtime as srt
    from chronos.serving.ollama_server import create_app
    path = _save(tmp_path, lam, "served")
    monkeypatch.setattr(srt, "_find_tokenizer_dir", lambda *_a, **_k: str(tmp_path))   # tokenizer not under test
    monkeypatch.setattr(transformers.PreTrainedTokenizerFast, "from_pretrained", classmethod(lambda cls, *_a, **_k: None))
    lm = srt.load_model(path, name="served", device="cpu", dtype="float32",
                        expert_cache_root=str(tmp_path / "cache"))
    try:
        assert lm.lazy_experts and lm.meta["paging_resolved"] == resolved and lm.meta["num_pages"] == pages
        cache = os.listdir(tmp_path / "cache")
        assert len(cache) == 1 and cache[0].endswith("_c2") == (resolved == "cluster")
        runtime = srt.ChronosRuntime(None, device="cpu")
        runtime.loaded["served"] = lm
        ps = TestClient(create_app(runtime)).get("/api/ps").json()["models"][0]["chronos"]
        assert ps["paging_resolved"] == resolved and ps["num_pages"] == pages and ps["paging"] == "auto"
    finally:
        lm.engine.teardown()
