"""Runner HTTP client helpers."""

import os

from zust_runner.http_client import merge_runner_headers


def test_merge_runner_headers_adds_ngrok_skip(monkeypatch):
    monkeypatch.setenv("ZUST_API_BASE", "https://abc.ngrok-free.dev")
    headers = merge_runner_headers({"Authorization": "Bearer t"})
    assert headers["Authorization"] == "Bearer t"
    assert headers["ngrok-skip-browser-warning"] == "69420"


def test_merge_runner_headers_unchanged_for_non_ngrok(monkeypatch):
    monkeypatch.setenv("ZUST_API_BASE", "https://api.zust.ai")
    headers = merge_runner_headers({"Authorization": "Bearer t"})
    assert "ngrok-skip-browser-warning" not in headers
