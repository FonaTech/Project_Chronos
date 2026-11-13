"""Multimodal DPO / ORPO / GRPO / distill on MLX: data path, media effect,
tiling parity, rollout encoder reuse, and end-to-end runs through
``run_mlx_stage``."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

EMB = 20      # pre-encoded feature width (fusion fits it to hidden_size)


def _wav(path, n=4000, rate=16000):
    import wave
    pcm = (np.sin(np.arange(n) * 0.05) * 12000).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(pcm.tobytes())
    return str(path)


def _records(wav):
    """Pairs/prompts: image features, text-only, a wav file."""
    conv = lambda a: [{"role": "user", "content": "描述 describe"}, {"role": "assistant", "content": a}]
    g = np.random.default_rng(0)
    return [
        {"chosen": conv("北京 weather"), "rejected": conv("上海 call"),
         "conversations": conv("x"), "image_embeds": g.normal(size=(5, EMB)).round(3).tolist()},
        {"chosen": conv("weather 北京"), "rejected": conv("call 上海"), "conversations": conv("y")},
        {"chosen": conv("city 北京"), "rejected": conv("tool 上海"),
         "conversations": conv("z"), "audio": wav},
    ]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    from chronos.tokenizer import build_tokenizer
    from transformers import PreTrainedTokenizerFast
    root = tmp_path_factory.mktemp("mm_stages")
    with open(root / "corpus.jsonl", "w", encoding="utf-8") as fh:
        for i in range(200):
            fh.write(json.dumps({"text": f"描述 describe 北京 上海 weather call city tool {i}"}, ensure_ascii=False) + "\n")
    build_tokenizer([str(root / "corpus.jsonl")], str(root / "tok"), vocab_size=600, log=lambda *_: None)
    tok = PreTrainedTokenizerFast.from_pretrained(str(root / "tok"))
    path = root / "pairs.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for rec in _records(_wav(root / "a.wav")):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return tok, str(path), _records(str(root / "a.wav"))


def test_record_media_reads_embeds_and_audio_files(data):
    from chronos.data.flexible_dataset import record_media
    image, text, audio = data[2]
    media, mask = record_media(image)
    assert set(media) == {"image_embeds"} and media["image_embeds"].shape == (5, EMB)
    assert mask.tolist() == [1.0, 0.0]
    assert record_media(text) is None
    media, mask = record_media(audio)
    assert set(media) == {"audio"} and media["audio"].shape == (4000,) and mask.tolist() == [0.0, 1.0]


def test_preference_dataset_and_collate_keep_pair_shape_and_masks(data):
    from chronos.data.flexible_dataset import (MultimodalStreamingDPODataset, StreamingDPODataset,
                                               PREFERENCE_KEYS, dataset_modality_summary, preference_collate)
    tok, path, _ = data
    assert set(dataset_modality_summary(path)["detected_modalities"]) == {"image", "audio"}
    ds = MultimodalStreamingDPODataset(path, tok, max_length=24)
    plain = StreamingDPODataset(path, tok, max_length=24)
    items = [ds[i] for i in range(3)]
    for i, item in enumerate(items):                 # text part is the text-only dataset's, unchanged
        for key in PREFERENCE_KEYS:
            assert torch.equal(item[key], plain[i][key])
        assert int(item["mask_chosen"].sum()) > 0 and int(item["mask_rejected"].sum()) > 0
    assert "modality_inputs" not in items[1]
    batch = preference_collate(items)
    assert batch["x_chosen"].shape == (3, 23)
    assert batch["modality_mask"].tolist() == [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
    media = batch["modality_inputs"]
    assert media["image_embeds"].shape == (3, 5, EMB)
    assert media["image_embeds_mask"].tolist()[1] == [False] * 5
    assert media["audio"].shape == (3, 4000) and media["audio_mask"].tolist() == [False, False, True]
    assert "modality_inputs" not in preference_collate([items[1], items[1]])


def test_grpo_prompt_records_keep_media_lazily(data):
    from chronos.trainer.grpo_trainer import load_grpo_prompt_records, load_grpo_prompts
    _tok, path, _ = data
    recs = load_grpo_prompt_records(path)
    assert [isinstance(r, dict) for r in recs] == [True, False, True]
    assert [r["prompt"] if isinstance(r, dict) else r for r in recs] == load_grpo_prompts(path)


# ---- MLX -------------------------------------------------------------------

def _cfg(**kw):
    from chronos.model.config import ChronosConfig
    base = dict(hidden_size=32, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                vocab_size=600, num_experts=8, num_experts_per_tok=2, num_shared_experts=1,
                moe_intermediate_size=32, max_position_embeddings=64, recurrent_depth=2,
                prelude_layers=1, coda_layers=1, dropout=0.0, fallback_mask_prob=0.0,
                modalities=["image", "audio"], image_patch_size=8, max_image_tokens=12,
                max_audio_tokens=10, modality_dropout=0.0)
    base.update(kw)
    return ChronosConfig(**base)


def _torch_model(c, seed=0):
    from chronos.model.model_chronos import ChronosForCausalLM
    torch.manual_seed(seed)
    pt = ChronosForCausalLM(c).eval()
    with torch.no_grad():                 # gate 0 → sigmoid 0.5 everywhere; randomize so media matter
        fusion = pt.model.multimodal_fusion
        fusion.gate.fill_(1.5)
        for p in fusion.modality_embeddings.values():
            p.normal_(0, 0.5)
    return pt


def _mlx_model(c, seed=0):
    from chronos.mlx.model import ChronosMLXModel
    m = ChronosMLXModel.from_chronos_pytorch(_torch_model(c, seed), c)
    m.eval()
    return m


def _pair_batch(data):
    from chronos.data.flexible_dataset import MultimodalStreamingDPODataset, preference_collate
    tok, path, _ = data
    ds = MultimodalStreamingDPODataset(path, tok, max_length=24)
    return preference_collate([ds[i] for i in range(3)])


def _strip_media(batch):
    return {k: v for k, v in batch.items() if k not in ("modality_inputs", "modality_mask")}


def test_dpo_orpo_tiling_matches_per_side_forward_and_media_change_the_loss(data):
    mx = pytest.importorskip("mlx.core")
    from chronos.mlx.training.trainer import _dpo_stage_loss, _orpo_stage_loss, _tile_media, _ids_to_mx
    batch = _pair_batch(data)
    with mx.stream(mx.cpu):
        m = _mlx_model(_cfg())
        tiled = _tile_media(batch, 2)
        x = mx.concatenate([_ids_to_mx(batch["x_chosen"]), _ids_to_mx(batch["x_rejected"])], axis=0)
        both, _, _ = m(x, logits_dtype=mx.float32, **tiled)
        once = _tile_media(batch, 1)
        alone_c, _, _ = m(_ids_to_mx(batch["x_chosen"]), logits_dtype=mx.float32, **once)
        alone_r, _, _ = m(_ids_to_mx(batch["x_rejected"]), logits_dtype=mx.float32, **once)
        np.testing.assert_allclose(np.array(both), np.concatenate([np.array(alone_c), np.array(alone_r)]), atol=1e-5)
        text, _, _ = m(x, logits_dtype=mx.float32)
        rows = np.abs(np.array(both) - np.array(text)).max(axis=(1, 2))
        assert rows[[0, 2, 3, 5]].min() > 1e-3          # media rows (pairs 0 and 2, both sides) differ
        np.testing.assert_allclose(rows[[1, 4]], 0, atol=1e-5)   # the text-only pair is untouched
        for fn in (lambda b: _dpo_stage_loss(m, m, b), lambda b: _orpo_stage_loss(m, b)):
            with_media, without = float(fn(batch)), float(fn(_strip_media(batch)))
            assert np.isfinite(with_media) and abs(with_media - without) > 1e-6


def test_rollout_encodes_media_once_with_identical_logits():
    mx = pytest.importorskip("mlx.core")
    from chronos.mlx.training.trainer import _encoded_media
    g = torch.Generator().manual_seed(3)
    media = {"modality_inputs": {"image": torch.randn(1, 3, 16, 24, generator=g),
                                 "audio": torch.randn(1, 4000, generator=g)},
             "modality_mask": torch.tensor([[1.0, 1.0]])}
    ids = mx.array(np.random.default_rng(1).integers(3, 600, (1, 7)))
    with mx.stream(mx.cpu):
        m = _mlx_model(_cfg(), seed=2)
        enc = _encoded_media(m, media)
        assert set(enc["modality_inputs"]) == {"image_embeds", "audio_embeds"}
        raw, _, cache = m(ids[:, :-1], logits_dtype=mx.float32, **media)
        pre, _, cache2 = m(ids[:, :-1], logits_dtype=mx.float32, **enc)
        np.testing.assert_allclose(np.array(pre), np.array(raw), atol=1e-5)
        step_raw, _, _ = m(ids[:, -1:], cache=cache, logits_dtype=mx.float32, **media)
        step_pre, _, _ = m(ids[:, -1:], cache=cache2, logits_dtype=mx.float32, **enc)
        np.testing.assert_allclose(np.array(step_pre), np.array(step_raw), atol=1e-5)
        assert _encoded_media(m, None) == {}


class _Args:
    learning_rate = 1e-3
    weight_decay = 0.0
    steps = 2
    epochs = 1
    max_seq_len = 24
    max_gen_len = 4
    num_generations = 2
    temperature = 1.0
    log_interval = 1
    save_interval = 100
    accumulation_steps = 1
    grad_clip = 1.0
    dtype = "float32"
    reward = "toy"


@pytest.mark.parametrize("stage", ["dpo", "orpo", "grpo", "distill"])
def test_run_mlx_stage_trains_fusion_with_media(stage, data, tmp_path, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from torch.utils.data import DataLoader
    from chronos.data.flexible_dataset import (MultimodalStreamingDPODataset, MultimodalStreamingSFTDataset,
                                               multimodal_collate, preference_collate)
    from chronos.model.checkpoint import load_checkpoint_state_dict, save_state_dict_with_config
    from chronos.trainer.grpo_trainer import load_grpo_prompt_records
    from chronos.mlx.training import run_mlx_stage
    import chronos.deps
    tok, path, _ = data
    c = _cfg()
    ckpt = str(tmp_path / "init.pth")
    pt = _torch_model(c, seed=7)
    save_state_dict_with_config(pt, ckpt, c, stage="sft")
    before = {k: v.clone() for k, v in pt.state_dict().items() if k.startswith("model.multimodal_fusion.")}
    loader = prompts = None
    if stage in {"dpo", "orpo"}:
        loader = DataLoader(MultimodalStreamingDPODataset(path, tok, max_length=24), batch_size=2,
                            collate_fn=preference_collate)
    elif stage == "distill":
        loader = DataLoader(MultimodalStreamingSFTDataset(path, tok, max_length=24), batch_size=2,
                            collate_fn=multimodal_collate)
    else:
        prompts = [r for r in load_grpo_prompt_records(path) if isinstance(r, dict)]
        monkeypatch.setattr(chronos.deps, "get_tokenizer_path", lambda: str(tmp_path.parent))
        import transformers
        monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *_a, **_k: tok))
    events = []
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        result = run_mlx_stage(stage=stage, config=c, checkpoint_path=ckpt, save_dir=str(tmp_path / "out"),
                               loader=loader, prompts=prompts, teacher_path=ckpt if stage == "distill" else None,
                               args=_Args(), progress_callback=events.append)
    finally:
        mx.set_default_device(previous)
    assert not result.failure_reason and result.steps == 2 and result.checkpoint_saved, result
    assert all(np.isfinite(e["loss"]) for e in events if "loss" in e)
    after = load_checkpoint_state_dict(result.checkpoint_path, map_location="cpu")
    moved = [k for k, v in before.items() if not torch.equal(after[k].float(), v.float())]
    assert any("q_proj" in k or "in_proj" in k for k in moved), moved      # fusion received gradients
    if stage == "grpo":   # raw audio prompt: the encoder trains too
        assert any("audio_encoder" in k for k in moved), moved


def test_workbench_stage_collate_routes_media_batches():
    from chronos.data.flexible_dataset import multimodal_collate, preference_collate
    from chronos.workbench.training import _stage_collate
    pick = lambda mode, mods: _stage_collate(mode, mods, multimodal_collate, preference_collate)
    assert pick("sft", []) is None and pick("dpo", []) is None
    assert pick("pretrain", ["image"]) is multimodal_collate and pick("distill", ["audio"]) is multimodal_collate
    assert pick("dpo", ["image"]) is preference_collate and pick("orpo", ["audio"]) is preference_collate
    assert pick("grpo", ["image"]) is None        # GRPO iterates prompt records, not a DataLoader
