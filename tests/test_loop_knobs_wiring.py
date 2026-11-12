"""0.3.3a1: every loop knob in the planner, the schema and the auto-tune probe; the probe scores cross-entropy."""
import json
import os

import numpy as np
import pytest

optuna = pytest.importorskip('optuna')

from chronos.model.planner import PlanRequest, plan_training  # noqa: E402
from chronos.tuning.chronos_auto_tuner import LOOP_KEYS, ChronosAutoTuner, ChronosSearchSpaceConfig  # noqa: E402
from chronos.webui import schema  # noqa: E402

NEW_KNOBS = ('recurrent_dt_param', 'recurrent_dt_min', 'recurrent_dt_max', 'recurrent_later_topk',
    'recurrent_later_groups', 'lambda_iteration_cluster', 'lookahead_teacher_scope')
PICKS = dict(recurrent_order='block', recurrent_dt_param='sigmoid', recurrent_dt_min=0.15, recurrent_dt_max=0.9,
    recurrent_dt_init=0.1, recurrent_later_topk=1, recurrent_later_groups=2, lambda_iteration_cluster=0.3,
    lookahead_teacher_scope='iterations')
TINY = dict(learning_rate=1e-3, lambda_balance=5e-4, lambda_temporal=1e-3, lookahead_steps=2, hidden_size=64,
    num_experts=4, num_shared_experts=1, kv_latent_dim=16)


class _Trial:
    """Answers suggest_* with fixed picks (else the last choice / the lower bound) and records reports."""

    def __init__(self, picks=None):
        self.picks, self.reports, self.attrs = dict(picks or ()), [], dict()

    def suggest_categorical(self, name, choices):
        return self.picks.get(name, choices[-1])

    def suggest_float(self, name, lo, hi, log=False):
        return self.picks.get(name, lo)

    def report(self, value, step):
        self.reports.append(value)

    def should_prune(self):
        return False

    def set_user_attr(self, key, value):
        self.attrs[key] = value


def _space_with_every_loop_knob():
    return ChronosSearchSpaceConfig(tune_lr=False, tune_lambda_balance=False, tune_lambda_temporal=False,
        **{'tune_' + key: True for key in LOOP_KEYS})


def _spy_config(monkeypatch):
    import chronos.model.model_chronos as model_chronos
    seen = dict()
    real_model = model_chronos.ChronosForCausalLM

    class Spy(real_model):
        def __init__(self, cfg, *args, **kwargs):
            seen['cfg'] = cfg
            super().__init__(cfg, *args, **kwargs)

    monkeypatch.setattr(model_chronos, 'ChronosForCausalLM', Spy)
    return seen


def _probe(tmp_path, params, steps, seed=0):
    import torch
    torch.manual_seed(seed)
    trial = _Trial()
    score = ChronosAutoTuner()._run_probe_trial(trial, str(tmp_path / 'no_checkpoint.pth'), str(tmp_path / 'none.jsonl'),
            0.95, steps, str(tmp_path), 0, params, tokenizer_dir=None)
    return score, trial


def test_every_loop_knob_is_searchable_and_reaches_the_probe_model(tmp_path, monkeypatch):
    seen = _spy_config(monkeypatch)
    params = ChronosAutoTuner()._sample_params(_Trial(PICKS), _space_with_every_loop_knob())
    assert all(params[key] == value for key, value in PICKS.items())
    assert params['recurrent_depth'] == 4 and params['num_hidden_layers'] == 6
    score, trial = _probe(tmp_path, dict(params, **TINY), steps=2)
    cfg = seen['cfg']
    for key in LOOP_KEYS:
        if key != 'recurrent_dt_init':
            assert getattr(cfg, key) == PICKS[key], key
    assert cfg.recurrent_dt_init == pytest.approx(0.15)   # a drawn init below the drawn dt_min starts at dt_min
    assert cfg.recurrent_groups == 4 and cfg.prelude_layers == 1 and cfg.coda_layers == 1
    assert np.isfinite(score) and trial.attrs['step_time_s'] > 0


def test_later_groups_never_offers_one_and_needs_block_order():
    space = _space_with_every_loop_knob()
    assert 1 not in space.recurrent_later_groups_choices and 0 in space.recurrent_later_groups_choices
    layer = ChronosAutoTuner()._sample_params(_Trial(dict(PICKS, recurrent_order='layer')), space)
    assert layer['recurrent_order'] == 'layer' and 'recurrent_later_groups' not in layer
    only = ChronosSearchSpaceConfig(tune_recurrent_later_groups=True)
    drawn = ChronosAutoTuner()._sample_params(_Trial(), only)
    assert drawn['recurrent_order'] == 'block' and drawn['recurrent_later_groups'] in (0, 2)


def test_probe_scores_cross_entropy_and_trains_on_the_weighted_objective(tmp_path):
    low, trial_low = _probe(tmp_path, dict(TINY, lambda_lookahead=1e-3), steps=4)
    high, trial_high = _probe(tmp_path, dict(TINY, lambda_lookahead=1.0), steps=4)
    # Step 0 is scored before any update: equal scores mean the weighted terms are not in the score.
    assert trial_low.reports[0] == pytest.approx(trial_high.reports[0], abs=1e-6)
    # Later steps differ: lambda_lookahead now reaches training (0.3.3a0 ignored it).
    assert any(abs(x - y) > 1e-6 for x, y in zip(trial_low.reports[1:], trial_high.reports[1:]))


def test_best_patch_carries_every_loop_knob():
    from types import SimpleNamespace
    tuner = ChronosAutoTuner()
    tuner.result = SimpleNamespace(best_params=dict(PICKS, lr=1e-3))
    patch = tuner.get_best_chronos_config_patch()
    assert all(patch[key] == value for key, value in PICKS.items())


def test_webui_search_choices_cover_every_loop_knob_with_help_in_four_languages():
    choices = set(schema.FIELDS['tune.search'].choices)
    assert set(LOOP_KEYS) <= choices
    base = os.path.join(os.path.dirname(schema.__file__), 'i18n')
    for lang in ('en', 'zh-Hans', 'zh-Hant', 'ja'):
        catalog = json.load(open(os.path.join(base, lang + '.json'), encoding='utf-8'))
        for key in NEW_KNOBS:
            assert catalog.get('field.' + key + '.label') and catalog.get('field.' + key + '.help'), (lang, key)
            assert any(ch.isdigit() for ch in catalog['field.' + key + '.help']), (lang, key)   # carries the measured reason


def _plan(depth):
    return plan_training(PlanRequest(target_total=5e6, num_experts=4, active_experts=2, vocab_size=6400, seq_len=128,
        recurrent_depth=depth, memory_saving='none', token_budget=2e6, ram_gb=16), verify=False)


@pytest.mark.parametrize('depth', [2, 4, 8])
def test_planner_recommends_every_loop_knob_with_a_source(depth):
    plan = _plan(depth)
    config, sources = plan['config'], plan['report']['sources']
    for key in LOOP_KEYS:
        assert key in config and sources.get(key), key
        field = schema.FIELDS[key]
        if key != 'recurrent_dt_init':   # dt_init depends on depth; test_planner_loop.py covers it
            assert field.source == 'planner' and field.recommended == config[key], key
    assert config['recurrent_later_groups'] != 1 and 'never 1' in sources['recurrent_later_groups']


def test_plan_without_a_loop_sets_no_loop_knob():
    config = _plan(1)['config']
    assert not any(key in config for key in LOOP_KEYS)


def test_schema_payload_keeps_zero_values_and_drops_unset_flags():
    topk = schema.FIELDS['recurrent_later_topk'].as_dict()
    assert topk['min'] == 0 and topk['recommended'] == 0 and topk['default'] == 0
    assert schema.FIELDS['max_steps'].as_dict()['recommended'] == 0
    plain = schema.FIELDS['hidden_size'].as_dict()
    assert 'advanced' not in plain and 'nullable' not in plain and 'recommended' not in plain
