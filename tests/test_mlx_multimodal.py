"""MLX multimodal fusion: Torch parity, checkpoint round-trip, training step."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

mx = pytest.importorskip("mlx.core")


def _cfg(modalities=("image", "audio"), **kw):
    from chronos.model.config import ChronosConfig
    base = dict(hidden_size=32, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                vocab_size=48, num_experts=8, num_experts_per_tok=2, num_shared_experts=1,
                moe_intermediate_size=32, max_position_embeddings=64, recurrent_depth=2,
                prelude_layers=1, coda_layers=1, dropout=0.0, fallback_mask_prob=0.0,
                modalities=list(modalities), image_patch_size=8, max_image_tokens=12,
                max_audio_tokens=10, modality_dropout=0.0)
    base.update(kw)
    return ChronosConfig(**base)


def _torch_model(c, seed=0):
    from chronos.model.model_chronos import ChronosForCausalLM
    torch.manual_seed(seed)
    pt = ChronosForCausalLM(c).eval()
    with torch.no_grad():                 # gate 0 → sigmoid 0.5; randomize so parity is tested
        fusion = pt.model.multimodal_fusion
        fusion.gate.fill_(0.7)
        for p in fusion.modality_embeddings.values():
            p.normal_(0, 0.1)
        fusion.attn.in_proj_bias.normal_(0, 0.1)
        fusion.attn.out_proj.bias.normal_(0, 0.1)
    return pt


def _media(b=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {"image": torch.randn(b, 3, 16, 24, generator=g), "audio": torch.randn(b, 4000, generator=g)}


def _mlx(pt, c):
    from chronos.mlx.model import ChronosMLXModel
    m = ChronosMLXModel.from_chronos_pytorch(pt, c)
    m.eval()
    return m


CASES = {
    "image_audio": lambda b: (_media(b), None),
    "image_only": lambda b: ({"image": _media(b)["image"]}, None),
    "mixed_rows": lambda b: (_media(b), torch.tensor([[1., 0.], [0., 1.], [0., 0.]])[:b]),
    "scalar_mask": lambda b: (_media(b), torch.tensor([1., 0., 1.])[:b]),
    "embeds_token_mask": lambda b: ({"image_embeds": torch.randn(b, 5, 20),
                                     "image_embeds_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [0] * 5]).bool()[:b],
                                     "audio_embeds": torch.randn(b, 4, 40)},
                                    torch.tensor([[1., 1.], [1., 0.], [0., 0.]])[:b]),
    "frame_mask": lambda b: ({**_media(b), "audio_mask": torch.tensor([True, False, True])[:b]}, None),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_fusion_and_logits_match_torch(case):
    c = _cfg()
    pt = _torch_model(c)
    ids = torch.randint(3, 48, (3, 9))
    media, mask = CASES[case](3)
    with mx.stream(mx.cpu):
        m = _mlx(pt, c)
        text = torch.randn(3, 9, 32)
        ref, ref_states = pt.model.multimodal_fusion(text, media, mask)
        got, got_states = m.multimodal_fusion(mx.array(text.numpy()), media, mask)
        np.testing.assert_allclose(np.array(got), ref.detach().numpy(), atol=2e-5)
        assert sorted(got_states) == sorted(ref_states)
        for k in ref_states:
            np.testing.assert_allclose(np.array(got_states[k]), ref_states[k].detach().numpy(), atol=2e-5)
        with torch.no_grad():
            ref_logits = pt(ids, modality_inputs=media, modality_mask=mask).logits
        logits, _, _ = m(mx.array(ids.numpy()), modality_inputs=media, modality_mask=mask, logits_dtype=mx.float32)
        np.testing.assert_allclose(np.array(logits), ref_logits.numpy(), atol=2e-4)
    if case == "mixed_rows":   # the row without media is text-only
        with torch.no_grad():
            alone = pt(ids[2:], modality_inputs=None).logits
        np.testing.assert_allclose(ref_logits[2:].numpy(), alone.numpy(), atol=1e-5)


def test_checkpoint_round_trip_and_text_only_unchanged():
    from chronos.mlx.training.io import mlx_state_to_torch
    from chronos.model.model_chronos import ChronosForCausalLM
    c = _cfg()
    pt = _torch_model(c, seed=4)
    with mx.stream(mx.cpu):
        m = _mlx(pt, c)
        sd = mlx_state_to_torch(m, c)
    ref = pt.state_dict()
    fusion_keys = [k for k in ref if k.startswith("model.multimodal_fusion.")]
    assert fusion_keys and all(k in sd for k in fusion_keys)
    back = ChronosForCausalLM(c)
    back.load_state_dict(sd, strict=True)
    for k in fusion_keys:
        torch.testing.assert_close(back.state_dict()[k], ref[k], rtol=0, atol=0)
    with mx.stream(mx.cpu):                       # text-only batches skip the fusion block
        ids = torch.randint(3, 48, (2, 7))
        logits, _, _ = m(mx.array(ids.numpy()), logits_dtype=mx.float32)
        with torch.no_grad():
            np.testing.assert_allclose(np.array(logits), pt(ids).logits.numpy(), atol=2e-4)
    with pytest.raises(ValueError, match="no enabled modalities"):
        from chronos.mlx.model import ChronosMLXModel
        ChronosMLXModel(_cfg(modalities=()))(mx.array(ids.numpy()), modality_inputs=_media(2))


def test_ce_stage_trains_encoders_and_fusion_from_collated_batches():
    from chronos.data.flexible_dataset import multimodal_collate
    from chronos.mlx.training.trainer import _ce_stage_loss
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    c = _cfg()
    pt = _torch_model(c, seed=5)
    ids = torch.randint(3, 48, (9,))
    items = [{"input_ids": ids, "labels": ids, "modality_inputs": {"image": torch.randn(3, 16, 24)},
              "modality_mask": torch.tensor([1., 0.])},
             (ids, ids),
             {"input_ids": ids, "labels": ids, "modality_inputs": {"audio": torch.randn(4000)},
              "modality_mask": torch.tensor([0., 1.])}]
    batch = multimodal_collate(items)
    assert set(batch["modality_inputs"]) >= {"image", "audio"}
    from chronos.mlx.multimodal import batch_modalities
    with mx.stream(mx.cpu):
        m = _mlx(pt, c)
        m.train()

        def loss_fn(model, b):
            i, lab, media, mask = batch_modalities(b)
            return _ce_stage_loss(model, mx.array(i.numpy()), mx.array(lab.numpy()), media, mask)

        loss, grads = nn.value_and_grad(m, loss_fn)(m, batch)
        flat = dict(tree_flatten(grads))
        assert np.isfinite(float(loss))
        for key in ("multimodal_fusion.gate", "multimodal_fusion.q_proj.weight",
                    "multimodal_fusion.image_encoder.patch.weight",
                    "multimodal_fusion.audio_encoder.net.layers.0.weight"):
            assert float(mx.abs(flat[key]).sum()) > 0, key
        frozen = _mlx(_torch_model(_cfg(freeze_modality_encoders=True), seed=5), _cfg(freeze_modality_encoders=True))
        names = {k for k, _ in tree_flatten(frozen.trainable_parameters())}
        assert not any(".image_encoder." in k or ".audio_encoder." in k for k in names)
        assert "multimodal_fusion.q_proj.weight" in names
