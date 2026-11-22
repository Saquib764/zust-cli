"""Runner HTTP client helpers."""

import os

import httpx

from zust_runner.http_client import is_retriable_api_error, merge_runner_headers


def test_merge_runner_headers_adds_ngrok_skip(monkeypatch):
    monkeypatch.setenv("ZUST_API_BASE", "https://abc.ngrok-free.dev")
    headers = merge_runner_headers({"Authorization": "Bearer t"})
    assert headers["Authorization"] == "Bearer t"
    assert headers["ngrok-skip-browser-warning"] == "69420"


def test_dead_tunnel_404_is_retriable():
    request = httpx.Request("POST", "https://tunnel.example/sync")
    response = httpx.Response(404, request=request)
    error = httpx.HTTPStatusError("missing", request=request, response=response)
    assert is_retriable_api_error(error) is True


def test_auth_failure_is_not_retriable():
    request = httpx.Request("POST", "https://tunnel.example/sync")
    response = httpx.Response(401, request=request)
    error = httpx.HTTPStatusError("unauthorized", request=request, response=response)
    assert is_retriable_api_error(error) is False


def test_merge_runner_headers_unchanged_for_non_ngrok(monkeypatch):
    monkeypatch.setenv("ZUST_API_BASE", "https://api.zust.ai")
    headers = merge_runner_headers({"Authorization": "Bearer t"})
    assert "ngrok-skip-browser-warning" not in headers
