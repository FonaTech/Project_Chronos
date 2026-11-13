"""Auto-tune probe: real tokens from the dataset, loop knobs in the search space, WebUI wiring."""
import json
import os

import numpy as np
import pytest

optuna = pytest.importorskip('optuna')

from chronos.tuning import probe_data
from chronos.tuning.chronos_auto_tuner import ChronosAutoTuner, ChronosSearchSpaceConfig


def _dataset(tmp_path, records=40):
    path = tmp_path / 'probe.jsonl'
    line = 'The quick brown fox jumps over the lazy dog while the river runs past the old stone bridge. '
    with open(path, 'w', encoding='utf-8') as handle:
        for i in range(records):
            handle.write(json.dumps({'text': line * 4 + str(i)}) + '\n')
    return str(path)


class _Trial:
    def __init__(self):
        self.reports = []

    def report(self, value, step):
        self.reports.append((step, value))

    def should_prune(self):
        return False


def test_probe_tokens_come_from_the_dataset_with_the_bundled_vocabulary(tmp_path):
    tokens, vocab = probe_data.load_probe_tokens(_dataset(tmp_path), None)
    assert vocab == 32768
    assert tokens.dtype == np.int64 and tokens.size >= 512
    assert int(tokens.max()) < vocab and int(tokens.min()) >= 0


def test_probe_tokens_are_none_when_the_data_cannot_be_read(tmp_path):
    assert probe_data.load_probe_tokens(str(tmp_path / 'missing.jsonl'), None) is None
    tiny = tmp_path / 'tiny.jsonl'
    tiny.write_text(json.dumps({'text': 'too short'}) + '\n', encoding='utf-8')
    assert probe_data.load_probe_tokens(str(tiny), None) is None


def test_probe_batch_is_deterministic_and_advances_with_the_step():
    tokens = np.arange(5000, dtype=np.int64)
    first = probe_data.probe_batch(tokens, 0, 2, 64)
    assert first.shape == (2, 64)
    assert np.array_equal(first, probe_data.probe_batch(tokens, 0, 2, 64))
    assert not np.array_equal(first, probe_data.probe_batch(tokens, 1, 2, 64))
    assert np.array_equal(first[0], tokens[first[0][0]:first[0][0] + 64])


def test_loop_knobs_are_sampled_only_when_searched():
    tuner = ChronosAutoTuner()
    off = tuner._sample_params(optuna.create_study().ask(), ChronosSearchSpaceConfig())
    assert 'recurrent_order' not in off and 'recurrent_dt_init' not in off and 'recurrent_depth' not in off
    space = ChronosSearchSpaceConfig(tune_recurrent_order=True, tune_recurrent_dt_init=True, probe_recurrent_depth=6)
    on = tuner._sample_params(optuna.create_study().ask(), space)
    assert on['recurrent_order'] in ('layer', 'block')
    assert space.recurrent_dt_init_min <= on['recurrent_dt_init'] <= space.recurrent_dt_init_max
    assert on['recurrent_depth'] == 6
    one = ChronosSearchSpaceConfig(tune_recurrent_dt_init=True, probe_recurrent_depth=1)
    assert tuner._sample_params(optuna.create_study().ask(), one)['recurrent_depth'] == 2   # depth 1 has no loop to tune


def test_best_patch_carries_the_loop_knobs():
    from types import SimpleNamespace
    tuner = ChronosAutoTuner()
    tuner.result = SimpleNamespace(best_params={'recurrent_order': 'block', 'recurrent_dt_init': 0.3, 'lr': 1e-3})
    patch = tuner.get_best_chronos_config_patch()
    assert patch['recurrent_order'] == 'block' and patch['recurrent_dt_init'] == 0.3


def test_probe_trial_trains_a_looped_model_on_real_tokens(tmp_path, monkeypatch):
    import chronos.model.model_chronos as model_chronos
    seen = {}
    real_model = model_chronos.ChronosForCausalLM

    class Spy(real_model):
        def __init__(self, cfg, *args, **kwargs):
            seen['cfg'] = cfg
            super().__init__(cfg, *args, **kwargs)

    monkeypatch.setattr(model_chronos, 'ChronosForCausalLM', Spy)
    monkeypatch.setattr('chronos.tuning.probe_data.probe_batch', lambda *args: seen.setdefault('real_batch', True) and np.zeros((2, 64), dtype=np.int64) + 5)
    params = {'learning_rate': 1e-3, 'lambda_balance': 5e-4, 'lambda_temporal': 1e-3, 'lookahead_steps': 2,
            'hidden_size': 128, 'num_experts': 2, 'num_shared_experts': 1, 'kv_latent_dim': 32,
            'recurrent_depth': 2, 'recurrent_order': 'block', 'recurrent_dt_init': 0.3}
    tuner = ChronosAutoTuner()
    trial = _Trial()
    loss = tuner._run_probe_trial(trial, str(tmp_path / 'no_checkpoint.pth'), _dataset(tmp_path), 0.95, 3,
            str(tmp_path), 42, params, tokenizer_dir=None)
    cfg = seen['cfg']
    assert np.isfinite(loss) and len(trial.reports) == 3
    assert cfg.vocab_size == 32768
    assert cfg.recurrent_enabled and cfg.recurrent_depth == 2 and cfg.recurrent_order == 'block'
    assert cfg.recurrent_dt_init == 0.3
    assert seen.get('real_batch'), 'the probe did not draw its batches from the real tokens'


def test_probe_trial_without_loop_knobs_stays_a_plain_model(tmp_path, monkeypatch):
    import chronos.model.model_chronos as model_chronos
    seen = {}
    real_model = model_chronos.ChronosForCausalLM

    class Spy(real_model):
        def __init__(self, cfg, *args, **kwargs):
            seen['cfg'] = cfg
            super().__init__(cfg, *args, **kwargs)

    monkeypatch.setattr(model_chronos, 'ChronosForCausalLM', Spy)
    params = {'learning_rate': 1e-3, 'lambda_balance': 5e-4, 'lambda_temporal': 1e-3, 'lookahead_steps': 2,
            'hidden_size': 128, 'num_experts': 2, 'num_shared_experts': 1, 'kv_latent_dim': 32}
    missing = str(tmp_path / 'missing.jsonl')
    loss = ChronosAutoTuner()._run_probe_trial(_Trial(), str(tmp_path / 'no_checkpoint.pth'), missing, 0.95, 2,
            str(tmp_path), 42, params, tokenizer_dir=None)
    cfg = seen['cfg']
    assert np.isfinite(loss)
    assert cfg.recurrent_depth == 1 and cfg.recurrent_order == 'layer'
    assert cfg.vocab_size != 32768, 'unreadable data must fall back to the legacy random-id probe'


def test_webui_exposes_the_loop_knobs_to_the_search_and_maps_them_onto_the_space():
    from chronos.webui import schema
    choices = schema.FIELDS['tune.search'].choices
    assert 'recurrent_order' in choices and 'recurrent_dt_init' in choices
    assert 'recurrent_later_groups' in choices   # 0.3.3a1: searched with choices 0 and 2; the trial log carries its step time
    space = ChronosSearchSpaceConfig(**{f'tune_{name}': name in ('recurrent_order',) for name in choices})
    assert space.tune_recurrent_order and not space.tune_recurrent_dt_init
    base = os.path.join(os.path.dirname(schema.__file__), 'i18n')
    for lang in ('en', 'zh-Hans', 'zh-Hant', 'ja'):
        catalog = json.load(open(os.path.join(base, lang + '.json'), encoding='utf-8'))
        for name in ('recurrent_order', 'recurrent_dt_init'):
            assert f'field.{name}.label' in catalog, (lang, name)
