"""Planner recommendations for the loop knobs: block order, sigmoid step size, horizon-based initial dt."""
import pytest

from chronos.model.planner import PlanRequest, plan_training
from chronos.model.recurrent import recommended_dt_init


def _plan(depth):
    return plan_training(PlanRequest(target_total=5e6, num_experts=4, active_experts=2, vocab_size=6400, seq_len=128,
            recurrent_depth=depth, memory_saving='none', token_budget=2e6, ram_gb=16), verify=False)


@pytest.mark.parametrize('depth', [2, 4, 6, 8])
def test_looped_plan_recommends_block_order_sigmoid_and_the_horizon_init(depth):
    plan = _plan(depth)
    config, sources = plan['config'], plan['report']['sources']
    assert config['recurrent_order'] == 'block' and config['recurrent_dt_param'] == 'sigmoid'
    assert config['recurrent_dt_init'] == pytest.approx(round(recommended_dt_init(depth), 4))
    for key in ('recurrent_order', 'recurrent_dt_param', 'recurrent_dt_init'):
        assert sources[key], key
    assert 'estimated' in sources['recurrent_dt_init']      # horizon 2 / depth is not a fitted optimum


def test_plan_without_a_loop_keeps_the_plain_defaults():
    plan = _plan(1)
    config, sources = plan['config'], plan['report']['sources']
    assert not config['recurrent_enabled']
    for key in ('recurrent_order', 'recurrent_dt_param', 'recurrent_dt_init'):
        assert key not in config and key not in sources


def test_planned_loop_config_builds_a_model_and_validates():
    from chronos.config_schema import model_config_kwargs
    from chronos.model.config import ChronosConfig
    cfg = ChronosConfig(**model_config_kwargs(_plan(4)['config']))
    assert cfg.recurrent_order == 'block' and cfg.recurrent_dt_param == 'sigmoid'
    assert cfg.recurrent_dt_init == pytest.approx(0.5)


def test_schema_recommendations_match_what_the_planner_emits():
    from chronos.webui import schema
    plan = _plan(4)['config']
    for key in ('recurrent_order', 'recurrent_dt_param', 'recurrent_dt_init'):
        field = schema.FIELDS[key]
        assert field.source == 'planner' and field.recommended == pytest.approx(plan[key]), key
