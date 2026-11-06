"""Exercise stale Gradio requests before queueing, not just Python callbacks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest

gr = pytest.importorskip("gradio")
from fastapi.testclient import TestClient
from gradio.routes import App
from ui.event_contract import UIEventContract, event_contract_app_kwargs


@pytest.fixture
def event_app():
    contract = UIEventContract()
    applied = []
    with gr.Blocks(head=contract.head()) as blocks:
        state = gr.State({"hidden_size": 256})
        best = gr.JSON({"learning_rate": .0002})
        output = gr.JSON()
        button = gr.Button("Apply Best")
        def apply_best(values, previous):
            applied.append(values)
            updated = dict(previous, **values)
            return updated, updated
        button.click(apply_best, [best, state], [state, output])
        timer = gr.Timer(2)
        timer.tick(lambda: {}, outputs=best, api_name="poll_tune")
    contract.blocks = blocks
    blocks.chronos_event_contract = contract
    app = App.create_app(blocks, app_kwargs=event_contract_app_kwargs(blocks))
    with TestClient(app) as client:
        yield client, contract, applied


def test_old_timer_payload_cannot_execute_apply_best(event_app):
    client, contract, applied = event_app
    apply = next(fn for fn in contract.blocks.fns.values() if fn.api_name == "apply_best")
    # This is the reported request: a timer from an old page now points at a
    # two-input handler. Reject before Gradio creates a session or queues work.
    response = client.post("/gradio_api/queue/join", json={
        "fn_index": apply._id, "data": [], "session_hash": "old-page",
    })
    assert response.status_code == 409
    assert response.json()["reason"] == "input_count_mismatch"
    assert response.json()["code"] == "chronos_ui_outdated"
    assert applied == []
    assert "old-page" not in contract.blocks.state_holder.session_data


@pytest.mark.parametrize("path", ["/gradio_api/queue/join", "/gradio_api/api/apply_best", "/gradio_api/call/apply_best"])
def test_old_instance_is_rejected_even_when_argument_count_matches(event_app, path):
    client, contract, applied = event_app
    response = client.post(path, headers={"X-Chronos-UI-Contract": "previous-server"},
        json={"fn_index": 0, "data": [{"hidden_size": 512}, None], "session_hash": "old-page"})
    assert response.status_code == 409
    assert response.json()["reason"] == "server_restarted"
    assert applied == []
    assert "old-page" not in contract.blocks.state_holder.session_data


@pytest.mark.parametrize("browser_header", [True, False])
def test_current_browser_and_headerless_api_preserve_real_inputs(event_app, browser_header):
    client, contract, applied = event_app
    headers = {"X-Chronos-UI-Contract": contract.token} if browser_header else {}
    response = client.post("/gradio_api/api/apply_best", headers=headers,
        json={"data": [{"learning_rate": .0002}, None], "session_hash": "current-page"})
    assert response.status_code == 200, response.text
    assert applied == [{"learning_rate": .0002}]
    assert response.json()["data"][1] == {"hidden_size": 256, "learning_rate": .0002}
    # Proper empty-input polls still enter Gradio's normal queue.
    poll = next(fn for fn in contract.blocks.fns.values() if fn.api_name == "poll_tune")
    response = client.post("/gradio_api/queue/join", headers=headers,
        json={"fn_index": poll._id, "data": [], "session_hash": "current-page"})
    assert response.status_code == 200, response.text
    assert "event_id" in response.json()


def test_malformed_current_event_is_not_silently_filled_or_reloaded(event_app):
    client, contract, applied = event_app
    response = client.post("/gradio_api/queue/join", headers={"X-Chronos-UI-Contract": contract.token},
        json={"fn_index": 0, "data": [], "session_hash": "current-page"})
    assert response.status_code == 422
    assert response.json()["code"] == "chronos_invalid_event"
    assert applied == []


def test_contract_and_page_cannot_be_cached_across_restart(event_app):
    client, contract, _ = event_app
    response = client.get("/chronos-ui/contract")
    assert response.json() == {"token": contract.token}
    for path in ("/", "/config", "/chronos-ui/contract"):
        assert client.get(path).headers["cache-control"] == "no-store"
    assert UIEventContract().token != contract.token
