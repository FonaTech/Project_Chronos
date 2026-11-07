"""Doc-mix shards: science-weighted selection, n-in-1 merges, length-proportional
sampling, both directions, document eval sets and the MLX doc-run wrapper."""
from __future__ import annotations

import json
import os
import random

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("tokenizers")

from chronos_translate.corpus import docmix  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|en|>", "<|zh|>"]
SCI_EN = ["The protein binds the cell membrane in the experiment.", "Quantum sensors measure the laser spectrum.",
          "The vaccine trial enrolled patients with the disease."]
SCI_ZH = ["蛋白质在实验中结合细胞膜。", "量子传感器测量激光光谱。", "疫苗试验招募了患有该疾病的患者。"]
GEN_EN = ["The committee met on Monday to discuss the budget.", "The general assembly adopted the agenda.",
          "Members thanked the chair for the report."]
GEN_ZH = ["委员会星期一开会讨论预算。", "大会通过了议程。", "成员感谢主席的报告。"]


def _tokenizer(root):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIALS,
                                  initial_alphabet=[f"<0x{i:02X}>" for i in range(256)])
    tok.train_from_iterator(SCI_EN + SCI_ZH + GEN_EN + GEN_ZH + ["翻译成中文，只输出译文：", "翻译成英文，只输出译文："] * 4,
                            trainer)
    os.makedirs(root, exist_ok=True)
    tok.save(os.path.join(root, "tokenizer.json"))
    with open(os.path.join(root, "tokenizer_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"fingerprint": "test", "vocab_size": tok.get_vocab_size()}, fh)
    return root


def _corpus(root, rows=2048, parts=2):
    """Ordered parts: runs of 8 rows repeat one topic sentence (adjacent overlap ≫ random)."""
    os.makedirs(root, exist_ok=True)
    rng = random.Random(0)
    for p in range(parts):
        en, zh = [], []
        for i in range(rows):
            topic = (i // 8) % 6
            pool_en, pool_zh = (SCI_EN, SCI_ZH) if topic < 2 else (GEN_EN, GEN_ZH)
            j = topic % 3
            run = "qv" + "".join(chr(97 + (i // 8 // 26 ** d) % 26) for d in range(4))   # one word per run
            en.append(f"{pool_en[j]} {run} {rng.randrange(99)}."); zh.append(pool_zh[j])
        table = pa.table({"zh": zh, "en": en, "pair_key": [f"{p}-{i}" for i in range(rows)],
                          "source": ["test"] * rows})
        pq.write_table(table, os.path.join(root, f"train.part{p}.parquet"))
    pq.write_table(pa.table({"zh": GEN_ZH * 20, "en": GEN_EN * 20, "pair_key": [str(i) for i in range(60)],
                             "source": ["test"] * 60}), os.path.join(root, "valid.part0.parquet"))
    return root


def test_science_hits_ignore_generic_words():
    assert docmix.science_hits(SCI_EN[0], SCI_ZH[0]) >= 4
    assert docmix.science_hits("The general secretary gave a physical copy.", "秘书长递交了文本。") == 0


def test_merge_plan_shares_cap_and_tiling():
    rng = random.Random(1)
    spec = docmix.DocMixSpec(fraction=1.0, ctx_len=4096)
    cost = np.asarray([rng.randint(20, 90) for _ in range(200_000)], np.int64)
    plan = docmix.merge_plan(cost, 4000, spec, random.Random(2))
    assert plan[0][0] == 0 and all(a[1] == b[0] for a, b in zip(plan, plan[1:]))
    assert all(cost[a:b].sum() <= 4000 for a, b, _k in plan)
    share = {k: sum(1 for p in plan if p[2] == k) / len(plan) for k in docmix.SIZE_KEYS}
    for k, want in docmix.DEFAULT_SHARES.items():
        assert abs(share[k] - want) < 0.01, (k, share[k])
    longs = [b - a for a, b, k in plan if k == "long" and b < len(cost)]     # the run's tail may be shorter
    assert min(longs) >= spec.long_min and max(longs) > 40     # long reaches toward the 4k cap


def test_merge_plan_skips_rows_over_cap():
    plan = docmix.merge_plan(np.asarray([10, 5000, 10, 10]), 100, docmix.DocMixSpec(1.0, 128), random.Random(0))
    assert all(not (a <= 1 < b) for a, b, _k in plan)


def _scan(tmp_path, rows=4096):
    pairs = _corpus(str(tmp_path / "pairs"), rows=rows)
    return pairs, docmix.scan_corpus(pairs, str(tmp_path / "scan"), block_rows=1024, workers=1, log=lambda *_: None)


def test_selection_fraction_and_science_weighting(tmp_path):
    _pairs, scan = _scan(tmp_path)
    base = dict(fraction=0.25, ctx_len=256, chunk_rows=16, block_rows=1024)
    flat = docmix.select_chunks(scan, docmix.DocMixSpec(**base, science_weights=(1.0, 1.0, 1.0)))["stats"]
    sci = docmix.select_chunks(scan, docmix.DocMixSpec(**base))["stats"]
    for st in (flat, sci):
        assert abs(st["rows_selected"] / st["train_rows_total"] - 0.25) < 0.01
    assert sci["mean_hits_per_row_selected"] > flat["mean_hits_per_row_selected"]
    assert sci["mean_hits_per_row_selected"] > sci["mean_hits_per_row_all"]
    assert sci["ordered_row_share_selected"] == 1.0              # topic runs are detected as ordered


def test_build_records_directions_and_sampler(tmp_path):
    pairs, _scan_out = _scan(tmp_path)
    tok_dir = _tokenizer(str(tmp_path / "tok"))
    out = str(tmp_path / "out")
    spec = docmix.DocMixSpec(fraction=0.5, ctx_len=256, chunk_rows=16, block_rows=1024, seed=5)
    m = docmix.build_docmix(pairs, tok_dir, out, spec, scan_dir=str(tmp_path / "scan"), workers=1,
                            log=lambda *_: None)
    tr = m["splits"]["train"]
    assert tr["records"] > 300 and tr["max_len"] <= 256 and m["format"] == "chronos-docmix-v1"
    assert abs(tr["dir_share_records"]["zh_en"] - 1 / 3) < 0.06
    assert tr["size_share_records"]["1"] > 0.3 and tr["size_share_records"]["long"] > 0.03
    folder = os.path.join(out, "sft", "train")
    recs = docmix.DocSFTRecords(folder, seed=3)
    assert len(recs) == tr["records"] and recs.total_tokens() == tr["sft_tokens"]
    assert recs.supervised_tokens() == tr["sft_supervised_tokens"]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
    for _ in range(40):
        ids, labels = recs.batch(1024)
        assert ids.shape == labels.shape and (ids.shape[1] - 1) % 64 == 0
        meta = recs.last_meta
        for row, (n, d, _k) in zip(range(len(ids)), meta):
            text = tok.decode(ids[row, :n].tolist(), skip_special_tokens=False).replace(" ", "").replace("▁", "")
            want = "翻译成英文" if d == 1 else "翻译成中文"
            assert want in text                                       # .dir matches the prompt
            sup = labels[row] != -100
            assert sup.any() and not sup[:4].any()                    # prompt masked, reply supervised
    s = recs.seen
    assert abs(s["dir"][1] / s["records"] - 1 / 3) < 0.12


def test_sampler_gives_equal_passes_per_record_at_every_length(tmp_path):
    folder = tmp_path / "sft"
    folder.mkdir()
    lens = [16] * 3000 + [160] * 300 + [1000] * 40
    toks = np.ones(sum(lens), np.uint16)
    docmix._write_arrays(str(folder / "a"), {
        ".bin": toks, ".idx": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        ".n": np.ones(len(lens), np.uint8), ".dir": np.zeros(len(lens), np.uint8),
        ".plen": np.full(len(lens), 4, np.int32)})
    recs = docmix.DocSFTRecords(str(folder), seed=0)
    assert recs.bin_tokens == {"1-128": 48000, "129-256": 48000, "513-1024": 40000}
    for _ in range(6000):
        ids, labels = recs.batch(16384)
        assert ids.size <= 16384 + 64 * 256 and ids.shape[0] == len(recs.last_meta)
    b = recs.seen["len_bins"]
    passes = [b["1-128"] / recs.bin_tokens["1-128"], b["129-256"] / recs.bin_tokens["129-256"],
              b["513-1024"] / recs.bin_tokens["513-1024"]]
    assert max(passes) / min(passes) < 1.12, passes                   # same passes per record at every length


def _raw_eval(root):
    flores = os.path.join(root, "facebook__flores__eng-zho", "data", "pair", "eng_Latn-zho_Hans")
    os.makedirs(flores)
    rows = [(i, f"u{i // 3}", GEN_EN[i % 3], GEN_ZH[i % 3]) for i in range(9)]
    pq.write_table(pa.table({"id": [r[0] for r in rows], "URL": [r[1] for r in rows],
                             "sentence_eng_Latn": [r[2] for r in rows], "sentence_zho_Hans": [r[3] for r in rows]}),
                   os.path.join(flores, "devtest-00000-of-00001.parquet"))
    wmt = os.path.join(root, "google__wmt24pp__en-zh_CN")
    os.makedirs(wmt)
    with open(os.path.join(wmt, "en-zh_CN.jsonl"), "w", encoding="utf-8") as fh:
        for seg, (doc, bad) in enumerate([("d1", False), ("d1", False), ("d2", True), ("d1", False)]):
            fh.write(json.dumps({"document_id": doc, "segment_id": 10 - seg, "source": SCI_EN[seg % 3],
                                 "target": SCI_ZH[seg % 3], "is_bad_source": bad, "domain": "news"}) + "\n")
    return root


def test_load_eval_documents_groups_and_orders(tmp_path):
    from chronos_translate.corpus.evalsets import load_eval_documents
    docs = load_eval_documents(_raw_eval(str(tmp_path / "raw")))
    flores = docs["flores_devtest_docs"]
    assert [d["doc_id"] for d in flores] == ["u0", "u1", "u2"] and all(len(d["segments"]) == 3 for d in flores)
    wmt = docs["wmt24pp_docs"]
    assert [d["doc_id"] for d in wmt] == ["d1"]                       # bad-source document dropped
    assert [s[0] for s in wmt[0]["segments"]] == [SCI_EN[0], SCI_EN[1], SCI_EN[0]][::-1]  # by segment_id
    assert "flores_dev_docs" not in docs                              # missing split is skipped


def _doc_spec(tmp_path, micro_fit="off", fraction=0.7):
    pairs = _corpus(str(tmp_path / "pairs"), rows=2048)
    tok_dir = _tokenizer(str(tmp_path / "tok"))
    shards = str(tmp_path / "shards")
    spec = docmix.DocMixSpec(fraction=0.5, ctx_len=128, chunk_rows=16, block_rows=1024, seed=5)
    docmix.build_docmix(pairs, tok_dir, shards, spec, scan_dir=str(tmp_path / "scan"), workers=1, log=lambda *_: None)
    from chronos.mlx.training.longrun import PhaseSpec
    from chronos_translate.run import TranslationRunSpec
    model = dict(architecture_version=3, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                 num_key_value_heads=2, vocab_size=640, num_experts=4, num_experts_per_tok=2,
                 num_shared_experts=1, moe_intermediate_size=64, max_position_embeddings=128,
                 sliding_window_size=64, recurrent_depth=2, prelude_layers=1, coda_layers=1, dropout=0.0,
                 fallback_mask_prob=0.0)
    phases = [PhaseSpec(name="pretrain", budget_fraction=0.5, peak_lr=1e-3, tokens_per_step=512,
                        micro_tokens=256, seq_len=128, warmup_fraction=0.1),
              PhaseSpec(name="sft", budget_fraction=0.5, peak_lr=5e-4, tokens_per_step=512,
                        micro_tokens=256, seq_len=128, warmup_fraction=0.1)]
    return TranslationRunSpec(run_dir=str(tmp_path / "run"), shards_dir=shards, tokenizer_dir=tok_dir, model_config=model,
                              phases=phases, budget_hours=1.0, bleu_samples=2, validate_every_s=1e9, checkpoint_every_s=1e9,
                              dtype="float32", final_eval_samples=0, tensor_stats_every_s=0, probe_samples=0,
                              extra={"raw_root": _raw_eval(str(tmp_path / "raw")), "micro_fit": micro_fit,
                                     "micro_fit_fraction": fraction, "doc_eval_docs": 2})


def test_doc_run_trains_both_phases_and_evaluates_both_directions(tmp_path):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("sacrebleu")
    import time
    from chronos_translate.doc_run import DocLongRun, DocSFTSource
    spec = _doc_spec(tmp_path)
    with mx.stream(mx.cpu):
        run = DocLongRun(spec, log=lambda *_: None)
        for pi in range(2):
            run.state.update(phase_index=pi, phase_step=0, phase_total=None, phase_budget_s=None, phase_wall_s=0.0)
            run.run_phase(pi, time.monotonic() + 600, max_steps=2)
            assert run.state["phase_step"] == 2
        assert spec.phases[1].micro_tokens == 256                     # recipe object restored
        src = DocSFTSource(spec.shards_dir, 0)
        vb = src.valid_batches(2, 256)
        assert all(ids.shape == lb.shape for ids, lb in vb)
        assert all(np.array_equal(a[0], b[0]) for a, b in zip(vb, src.valid_batches(2, 256)))   # fixed windows
        res = run.translation_eval(4)
    for key in ("bleu_zh", "chrf", "sent_zh_en_bleu", "doc_en_zh_bleu", "doc_zh_en_chrf", "wmt24pp_doc_en_zh_n"):
        assert key in res, (key, res)
    rows = [json.loads(l) for l in open(os.path.join(spec.run_dir, "samples.jsonl"), encoding="utf-8")]
    assert {r["direction"] for r in rows} == {"en_zh", "zh_en"}


def test_micro_fit_shrinks_under_a_tight_target(tmp_path):
    mx = pytest.importorskip("mlx.core")
    if not hasattr(mx, "device_info"):
        pytest.skip("mlx without device_info")
    from chronos_translate.doc_run import DocLongRun
    with mx.stream(mx.cpu):
        tight = DocLongRun(_doc_spec(tmp_path, micro_fit="auto", fraction=1e-12), log=lambda *_: None)
        assert tight._fit_micro_tokens(tight.spec.phases[1]) == 128   # one ctx row
        assert tight.state["micro_fit"]["sft"]["rows"] == 1
        loose = DocLongRun(_doc_spec(tmp_path / "b", micro_fit="auto", fraction=0.7), log=lambda *_: None)
        assert loose._fit_micro_tokens(loose.spec.phases[1]) == 256   # recipe size fits


def test_load_spec_resolves_relative_paths(tmp_path):
    pytest.importorskip("mlx.core")
    from chronos_translate.doc_run import load_spec
    recipe = {"run_dir": "run", "shards_dir": "data/shards", "tokenizer_dir": "data/tok", "model_config": {},
              "phases": [{"name": "pretrain", "budget_fraction": 1.0, "peak_lr": 1e-3, "tokens_per_step": 8}],
              "extra": {"raw_root": "data/raw"}, "_comment": "ignored"}
    path = tmp_path / "pkg" / "recipe.json"
    path.parent.mkdir()
    path.write_text(json.dumps(recipe), encoding="utf-8")
    spec = load_spec(str(path), budget_hours=2)
    assert spec.shards_dir == str(tmp_path / "pkg" / "data" / "shards") and spec.budget_hours == 2.0
    assert spec.extra["raw_root"] == str(tmp_path / "pkg" / "data" / "raw")


def test_target_passes_end_the_phase_and_shrink_the_time_budget(tmp_path):
    mx = pytest.importorskip("mlx.core")
    import time
    from chronos_translate.doc_run import DocLongRun
    spec = _doc_spec(tmp_path)
    spec.log_every_steps = 1
    spec.extra["target_passes"] = {"pretrain": 1e-9}
    with mx.stream(mx.cpu):
        run = DocLongRun(spec, log=lambda *_: None)
        run.state.update(phase_index=0, phase_step=0, phase_total=None, phase_budget_s=None, phase_wall_s=0.0)
        run.run_phase(0, time.monotonic() + 600, max_steps=50)
        assert run.state["phase_step"] == 1                          # first logged step reaches the target
        rows = [json.loads(l) for l in open(run.metrics_path, encoding="utf-8")]
        assert any(r.get("event") == "phase_end" and r.get("reason") == "target_passes" for r in rows)
        assert os.path.exists(os.path.join(spec.run_dir, f"{spec.export_prefix}_pretrain.pth"))   # phase export
        # Budget projection: 10% of the data in 100 s → 2 passes need 2000 s, clamped to the phase budget.
        run._phase_name = "sft"
        run._data_tokens["sft"] = 1000
        spec.extra["target_passes"] = {"sft": 2}
        run.state.update(tokens_seen=100, phase_start_tokens={"sft": 0}, phase_wall_s=100.0, phase_budget_s=5000.0,
                         phase_step=10)
        run._emit({"phase": "sft", "step": 10})
        assert run.state["phase_budget_s"] == pytest.approx(2000.0) and run.state["phase_total"] == 200
        run.state.update(phase_budget_s=1000.0, phase_budget_orig_s={})
        run._emit({"phase": "sft", "step": 10})
        assert run.state["phase_budget_s"] == pytest.approx(1000.0)  # never beyond the recipe's budget
