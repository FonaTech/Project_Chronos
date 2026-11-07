"""``chronos bundle``: build → check → translate → prepare-data → fine-tune spec, on a tiny model."""
from __future__ import annotations

import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from chronos.bundle import builder  # noqa: E402
from chronos.bundle.builder import FLORES_REL, build_bundle, check_bundle, load_bundle_manifest  # noqa: E402
from chronos.model.checkpoint import (  # noqa: E402
    checkpoint_config_path, resolve_checkpoint_tokenizer, save_state_dict_with_config, tokenizer_metadata,
)
from chronos.model.config import ChronosConfig  # noqa: E402
from chronos.model.model_chronos import ChronosForCausalLM  # noqa: E402
from chronos.tokenizer import build_tokenizer  # noqa: E402

ZH = ["今天天气很好，我们去公园散步吧。", "经济增长放缓，央行维持利率不变。", "他说：我们明天再讨论这个问题。",
      "人工智能正在改变世界。", "请把这份文件翻译成英文。"]
EN = ["The weather is nice today, let's take a walk in the park.",
      "Economic growth slowed and the central bank kept rates unchanged.",
      "He said: we will discuss this question again tomorrow.",
      "Artificial intelligence is changing the world.", "Please translate this document into English."]


def _config(vocab: int) -> ChronosConfig:
    return ChronosConfig(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         vocab_size=vocab, num_experts=4, num_experts_per_tok=2, num_shared_experts=1,
                         moe_intermediate_size=64, max_position_embeddings=256, recurrent_depth=4,
                         min_recurrent_depth=2, prelude_layers=1, coda_layers=1, dropout=0.0,
                         fallback_mask_prob=0.0, recurrent_attention_mode="shared_first", sliding_window_size=16)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    """A finished-looking ``chronos longrun`` run directory (never trained; random weights)."""
    root = tmp_path_factory.mktemp("bundle_src")
    corpus = root / "pairs.jsonl"
    with open(corpus, "w", encoding="utf-8") as fh:
        for zh, en in zip(ZH * 40, EN * 40):
            fh.write(json.dumps({"translation": {"zh": zh, "en": en}}, ensure_ascii=False) + "\n")
    tok_dir = root / "tokenizer"
    tok = build_tokenizer([str(corpus)], str(tok_dir), vocab_size=700, log=lambda *_: None)
    vocab = ((tok.vocab_size + 63) // 64) * 64
    torch.manual_seed(5)
    cfg = _config(vocab)
    run = root / "runs" / "tiny_run"
    run.mkdir(parents=True)
    ckpt = run / "chronos_enzh_pretrain.best.pth"
    save_state_dict_with_config(ChronosForCausalLM(cfg).eval(), str(ckpt), cfg, stage="pretrain")
    side = checkpoint_config_path(str(ckpt))
    meta = json.load(open(side, encoding="utf-8"))
    meta["tokenizer"] = tokenizer_metadata(str(tok_dir))            # what the longrun exporter records
    json.dump(meta, open(side, "w", encoding="utf-8"), indent=2)
    raw = root / "raw"
    flores = raw / FLORES_REL
    flores.mkdir(parents=True)
    for split in ("dev", "devtest"):
        table = pa.table({"sentence_eng_Latn": EN, "sentence_zho_Hans": ZH})
        pq.write_table(table, str(flores / f"{split}-00000-of-00001.parquet"))
    spec = {"run_dir": str(run), "shards_dir": str(root / "shards"), "tokenizer_dir": str(tok_dir),
            "model_config": {k: getattr(cfg, k) for k in ("hidden_size", "num_hidden_layers", "vocab_size")},
            "phases": [{"name": "pretrain", "budget_fraction": 0.7, "peak_lr": 1e-3, "tokens_per_step": 65536,
                        "micro_tokens": 8192, "seq_len": 512, "warmup_fraction": 0.02, "min_lr_ratio": 0.1,
                        "weight_decay": 0.1, "grad_clip": 1.0},
                       {"name": "sft", "budget_fraction": 0.3, "peak_lr": 5e-4, "tokens_per_step": 65536,
                        "micro_tokens": 8192, "seq_len": 512, "warmup_fraction": 0.02, "min_lr_ratio": 0.1,
                        "weight_decay": 0.1, "grad_clip": 1.0}],
            "budget_hours": 1.0, "seed": 1234, "dtype": "bfloat16", "export_prefix": "chronos_enzh",
            "extra": {"raw_root": str(raw), "params": {"total": 123456, "active": 65432}}}
    json.dump(spec, open(run / "run_spec.json", "w", encoding="utf-8"), indent=2)
    with open(run / "metrics.jsonl", "w", encoding="utf-8") as fh:
        for rec in ({"event": "step", "phase": "pretrain", "step": 10, "total": 40, "tokens_seen": 1000},
                    {"event": "validation", "phase": "pretrain", "step": 10, "valid_ce": 3.0, "valid_ppl": 20.1},
                    {"event": "step", "phase": "pretrain", "step": 20, "total": 40, "tokens_seen": 2000},
                    {"event": "validation", "phase": "pretrain", "step": 20, "valid_ce": 2.5, "valid_ppl": 12.2},
                    {"event": "validation", "phase": "pretrain", "step": 30, "valid_ce": 2.7, "valid_ppl": 14.9}):
            fh.write(json.dumps(rec) + "\n")
        fh.write('{"event": "step", "phase": "pretr')                  # a line being written right now
    return run


@pytest.fixture(scope="module")
def bundle(run_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("bundles") / "tiny-bundle"
    return build_bundle(str(run_dir), str(out), name="tiny-bundle", log=lambda *_: None)


def test_build_records_best_validation_and_leaks_no_paths(bundle, run_dir):
    m = load_bundle_manifest(bundle)
    assert m["training"]["step"] == 20 and m["training"]["valid_ce"] == 2.5     # min valid CE, not the last
    assert m["training"]["tokens_seen"] == 2000 and m["prompt_format"] == "pretrain_tags"
    assert m["eval"]["flores_plus"] == ["dev-00000-of-00001.parquet", "devtest-00000-of-00001.parquet"]
    assert "model" in os.listdir(os.path.join(bundle, "third_party", "minimind"))
    home = os.path.expanduser("~")
    for rel in ("recipes/source_run_spec.json", m["model"]["config"], "bundle.json", "recommended.json"):
        text = open(os.path.join(bundle, rel), encoding="utf-8").read()
        assert str(run_dir) not in text and home not in text, rel
    side = json.load(open(os.path.join(bundle, m["model"]["config"]), encoding="utf-8"))
    assert side["tokenizer"]["path"] == "tokenizer"
    assert resolve_checkpoint_tokenizer(os.path.join(bundle, m["model"]["checkpoint"])) == \
        os.path.join(bundle, "model", "tokenizer")
    with pytest.raises(FileExistsError):
        build_bundle(str(run_dir), bundle, name="tiny-bundle", log=lambda *_: None)


def test_check_detects_tampering_and_smoke_translates(bundle, tmp_path):
    import shutil
    report = check_bundle(bundle, log=lambda *_: None)
    assert report["ok"] and report["missing"] == [] and report["smoke"]["device"] == "cpu"
    copy = tmp_path / "copy"
    shutil.copytree(bundle, copy)
    with open(copy / "examples" / "translate_en.txt", "a", encoding="utf-8") as fh:
        fh.write("tampered\n")
    os.remove(copy / "recommended.json")
    bad = check_bundle(str(copy), smoke=False, log=lambda *_: None)
    assert not bad["ok"] and bad["hash_mismatch"] == ["examples/translate_en.txt"]
    assert bad["missing"] == ["recommended.json"]


def test_translate_backends_and_speculative_agree(bundle):
    from chronos.bundle.runtime import BundleModel
    bm = BundleModel(bundle, device="cpu")
    src = EN[:3]
    plain = bm.translate(src, "en_zh", max_new_tokens=20)
    spec = bm.translate(src, "en_zh", max_new_tokens=20, speculative=True, draft_depth=1, gamma=3)
    assert spec == plain and bm.last_stats["speculative"]["verify_calls"] > 0
    with pytest.raises(ValueError, match="repetition_penalty"):
        bm.translate(src, "en_zh", speculative=True, repetition_penalty=1.2)
    with pytest.raises(ValueError, match="supports"):
        bm.prompt_ids("x", "fr_en")
    mx = pytest.importorskip("mlx.core")
    prev = mx.default_device()
    try:                                       # device="cpu" sets the process-wide MLX default
        ml = BundleModel(bundle, device="cpu", backend="mlx")
        assert ml.translate(src, "en_zh", max_new_tokens=20) == plain
        assert ml.translate(src, "en_zh", max_new_tokens=20, speculative=True, draft_depth=1) == plain
        assert ml.translate(src, "zh_en", max_new_tokens=12, repetition_penalty=1.2) == \
            bm.translate(src, "zh_en", max_new_tokens=12, repetition_penalty=1.2)
    finally:
        mx.set_default_device(prev)


def test_lazy_experts_translate_like_the_full_model(bundle, tmp_path):
    """SSD-lazy experts (MLX) give the same greedy output; the cache stays outside the bundle."""
    mx = pytest.importorskip("mlx.core")
    from chronos.bundle.runtime import BundleModel
    from chronos.lazy_experts import LazyOptions
    prev = mx.default_device()
    try:
        full = BundleModel(bundle, device="cpu", backend="mlx")
        src = EN[:2]
        ref = full.translate(src, "en_zh", max_new_tokens=16)
        for residency in ("planned", "window"):
            lazy = BundleModel(bundle, device="cpu", backend="mlx", lazy_experts=True,
                               lazy_options=LazyOptions(residency=residency, ram_ratio=0.5),
                               cache_root=str(tmp_path / "cache"))
            try:
                assert lazy.translate(src, "en_zh", max_new_tokens=16) == ref
                assert lazy.translate(src, "en_zh", max_new_tokens=16, speculative=True) == ref
                stats = lazy.last_stats["lazy"]
                assert stats["paging_resolved"] == "expert" and stats["residency"] == residency
            finally:
                lazy.close()
        assert not os.path.exists(os.path.join(bundle, "expert_cache"))
        with pytest.raises(ValueError, match="mlx"):
            BundleModel(bundle, device="cpu", backend="torch", lazy_experts=True)
    finally:
        mx.set_default_device(prev)


def test_prepare_data_and_finetune_spec(bundle, tmp_path):
    from chronos.bundle.finetune import prepare_data, write_finetune_spec
    data = tmp_path / "mine.jsonl"
    rows = [{"zh": f"{zh}第{i}句。", "en": f"{en} Sentence {i}."} for i in range(30) for zh, en in zip(ZH, EN)]
    rows += [{"zh": ZH[0], "en": EN[0]}]                                    # FLORES+ pair: must be removed
    with open(data, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    out = tmp_path / "prepared"
    report = prepare_data(bundle, [str(data)], str(out), workers=1, log=lambda *_: None)
    assert report["corpus"]["eval_leak_removed"] >= 1
    assert report["split_rows"]["valid"] >= 4
    assert report["shards"]["sft_train_records"] > 0 and report["shards"]["sft_valid_records"] > 0
    with pytest.raises(FileExistsError):
        prepare_data(bundle, [str(data)], str(out), workers=1, log=lambda *_: None)
    run = tmp_path / "ft_run"
    spec_path = write_finetune_spec(bundle, str(out / "shards"), str(run), tier="smoke", mode="sft")
    spec = json.load(open(spec_path, encoding="utf-8"))
    m = load_bundle_manifest(bundle)
    assert spec["init_checkpoint"] == os.path.join(bundle, m["model"]["checkpoint"])
    assert spec["tokenizer_dir"] == os.path.join(bundle, "model", "tokenizer")
    assert [p["name"] for p in spec["phases"]] == ["sft"] and spec["export_prefix"] == "tiny_bundle_ft"
    # the tiny model has 4 experts with k=2 (one cluster): the constraint is off and says why
    assert spec["model_config"]["lambda_cluster_locality"] == 0.0
    assert "fewer than two clusters" in spec["extra"]["cluster_locality_source"]["lambda_cluster_locality"]
    with pytest.raises(ValueError, match="pretrain"):
        write_finetune_spec(bundle, str(out / "shards"), str(tmp_path / "ft2"), tier="smoke", mode="pretrain")
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "latest.state.pkl").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        write_finetune_spec(bundle, str(out / "shards"), str(run), tier="smoke", mode="sft")


def test_pack_writes_archive_and_digest(bundle, tmp_path):
    import tarfile
    archive = builder.pack_bundle(bundle, str(tmp_path / "b.tar.gz"), log=lambda *_: None)
    digest = open(archive + ".sha256", encoding="utf-8").read().split()[0]
    assert digest == builder.sha256_file(archive)
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "tiny-bundle/bundle.json" in names and not any("__pycache__" in n for n in names)
    with pytest.raises(FileExistsError):
        builder.pack_bundle(bundle, archive, log=lambda *_: None)


def test_stage_tokenizer_guards(bundle, tmp_path):
    from types import SimpleNamespace
    from chronos.model.checkpoint import chronos_config_from_checkpoint
    from chronos.trainer.stage_utils import load_stage_tokenizer
    m = load_bundle_manifest(bundle)
    ckpt = os.path.join(bundle, m["model"]["checkpoint"])
    cfg, _ = chronos_config_from_checkpoint(ckpt)
    tok = load_stage_tokenizer(SimpleNamespace(tokenizer_path=None), cfg, ckpt)       # the bundle's own
    assert len(tok) <= cfg.vocab_size and tok.convert_tokens_to_ids("<|zh|>") is not None
    small = SimpleNamespace(vocab_size=64)
    with pytest.raises(ValueError, match="vocab_size"):
        load_stage_tokenizer(SimpleNamespace(tokenizer_path=None), small, ckpt)
    big = SimpleNamespace(vocab_size=cfg.vocab_size * 4)
    with pytest.raises(ValueError, match="--tokenizer_path"):
        load_stage_tokenizer(SimpleNamespace(tokenizer_path=None), big, ckpt)
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_tokenizer(ckpt, str(tmp_path))                            # no tokenizer.json there


def test_init_checkpoint_guard(bundle, tmp_path):
    pytest.importorskip("mlx.core")          # chronos.mlx.training.longrun is the MLX trainer
    from chronos.mlx.training.longrun import check_init_checkpoint
    from chronos.model.checkpoint import chronos_config_from_checkpoint
    m = load_bundle_manifest(bundle)
    ckpt = os.path.join(bundle, m["model"]["checkpoint"])
    cfg, _ = chronos_config_from_checkpoint(ckpt)
    check_init_checkpoint(ckpt, cfg)
    with pytest.raises(FileNotFoundError):
        check_init_checkpoint(str(tmp_path / "missing.pth"), cfg)
    wrong = _config(cfg.vocab_size)
    wrong.hidden_size = 96
    with pytest.raises(ValueError, match="hidden_size"):
        check_init_checkpoint(ckpt, wrong)
