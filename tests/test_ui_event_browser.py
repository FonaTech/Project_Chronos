"""Browser regressions for timers and tabs surviving a Gradio event-graph change.

Uses an isolated localhost server and Chromium profile. Skips when Playwright
or a Chromium executable is unavailable; CHRONOS_TEST_CHROMIUM overrides it.
"""
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def browser_server(tmp_path_factory):
    sync = pytest.importorskip("playwright.sync_api")
    with sync.sync_playwright() as playwright:
        cached_shells = sorted((Path.home()/"Library/Caches/ms-playwright").glob(
            "chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell"), reverse=True)
        candidates = [os.environ.get("CHRONOS_TEST_CHROMIUM", ""), playwright.chromium.executable_path,
                      *map(str, cached_shells),
                      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                      "/usr/bin/chromium", "/usr/bin/chromium-browser"]
        executable = next((p for p in candidates if p and Path(p).is_file()), None)
        if executable is None:
            pytest.skip("Chromium unavailable; install Playwright Chromium or set CHRONOS_TEST_CHROMIUM")
        browser = playwright.chromium.launch(executable_path=executable, headless=True)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        log = (tmp_path_factory.mktemp("browser-server") / "server.log").open("w+")
        process = subprocess.Popen([sys.executable, "-u", "chronos_app.py", "--legacy", "--host", "127.0.0.1", "--port", str(port)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 40
            while True:
                try:
                    with urllib.request.urlopen(url+"/config", timeout=.5) as response:
                        config = json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        pytest.fail("Isolated WebUI failed to start:\n" + log.read())
                    time.sleep(.1)
            with urllib.request.urlopen(url+"/chronos-ui/contract") as response:
                token = json.load(response)["token"]
            yield browser, url, config, token
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            log.close()
            browser.close()


def wait_until(page, predicate, timeout=12):
    deadline = time.monotonic()+timeout
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("Browser condition timed out")
        page.wait_for_timeout(100)


def open_tab(page, label: str):
    """Click a top-level tab, also when Gradio moved it into the overflow ("…") menu."""
    page.get_by_role("tab").first.wait_for(timeout=30000)
    tab = page.get_by_role("tab", name=re.compile(label))
    if tab.count():
        tab.click()
        return
    bar = page.locator(".tab-wrapper").first                   # top-level bar (nested tabs follow it)
    bar.locator(":scope > .overflow-menu > button").click()
    bar.locator(":scope > .overflow-menu .overflow-dropdown button", has_text=re.compile(label)).click()


def test_actual_browser_sends_two_apply_inputs_and_empty_poll_inputs(browser_server):
    browser, url, config, token = browser_server
    deps = {d["id"]: d for d in config["dependencies"]}
    names = {d["api_name"]: d["id"] for d in config["dependencies"]}
    page = browser.new_page()
    requests, errors = [], []
    page.on("request", lambda r: requests.append((r.post_data_json, r.headers)) if "/queue/join" in r.url else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    try:
        page.goto(url, wait_until="domcontentloaded")
        open_tab(page, "Auto Tune")
        page.get_by_role("button", name=re.compile("Apply Best")).click()
        wait_until(page, lambda: {names["apply_best"], names["poll_tune"]}.issubset({r[0]["fn_index"] for r in requests}))
        for body, headers in requests:
            assert len(body["data"]) == len(deps[body["fn_index"]]["inputs"])
            assert headers["x-chronos-ui-contract"] == token
        assert not errors
    finally:
        page.close()


def test_old_timer_event_is_rejected_then_browser_reloads_once(browser_server):
    browser, url, config, token = browser_server
    names = {d["api_name"]: d["id"] for d in config["dependencies"]}
    page = browser.new_page()
    navigations, responses = [], []
    injected = False
    rejected = {}
    def old_timer(route):
        nonlocal injected
        data = route.request.post_data_json
        if data.get("fn_index") == names["poll_tune"] and not injected:
            injected = True
            # Reproduce the user's old numeric mapping: empty poll -> apply_best.
            data["fn_index"] = names["apply_best"]
            headers = dict(route.request.headers)
            headers.pop("x-chronos-ui-contract", None)
            headers.pop("content-length", None)
            response = route.fetch(post_data=json.dumps(data), headers=headers)
            rejected.update(status=response.status, **response.json())
            route.fulfill(response=response)
        else:
            route.continue_()
    page.route("**/queue/join*", old_timer)
    page.on("framenavigated", lambda f: navigations.append(f.url) if f == page.main_frame else None)
    page.on("response", lambda r: responses.append(r) if "/queue/join" in r.url else None)
    try:
        page.goto(url, wait_until="domcontentloaded")
        page.get_by_role("tab").first.wait_for(timeout=20000)
        wait_until(page, lambda: len(navigations) >= 2)
        assert rejected["status"] == 409
        assert rejected["reason"] == "input_count_mismatch"
        page.get_by_role("tab").first.wait_for()
        page.wait_for_timeout(3500)
        assert len(navigations) == 2
        assert any(r.status == 200 for r in responses)
    finally:
        page.close()


def test_idle_page_from_previous_server_instance_refreshes(browser_server):
    browser, url, _config, token = browser_server
    page = browser.new_page()
    navigations = []
    replaced = False
    def old_document(route):
        nonlocal replaced
        if not replaced:
            replaced = True
            response = route.fetch()
            route.fulfill(response=response, body=response.text().replace(token, "0"*32))
        else:
            route.continue_()
    page.route(url+"/", old_document)
    page.on("framenavigated", lambda f: navigations.append(f.url) if f == page.main_frame else None)
    try:
        page.goto(url, wait_until="domcontentloaded")
        wait_until(page, lambda: len(navigations) >= 2)
        page.get_by_role("tab").first.wait_for()
        page.wait_for_timeout(3500)
        assert len(navigations) == 2
    finally:
        page.close()
