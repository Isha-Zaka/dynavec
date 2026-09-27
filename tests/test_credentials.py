"""Tests for AWS credential/session resolution."""

from types import SimpleNamespace

import pytest

import dynavec.credentials as credentials_module
from dynavec.credentials import AWSCredentials, resolve_async_session
from dynavec.exceptions import MissingDependencyError


class _FakeAioboto3:
    def __init__(self) -> None:
        self.calls = []
        self.session = object()

    def Session(self, **kwargs):
        self.calls.append(kwargs)
        return self.session


def test_resolve_async_session_prefers_explicit_session(monkeypatch):
    explicit_session = object()

    def fail_import(name):
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(credentials_module.importlib, "import_module", fail_import)

    result = resolve_async_session(None, explicit_session)

    assert result is explicit_session


def test_resolve_async_session_missing_dependency(monkeypatch):
    def missing_import(name):
        raise ImportError(name)

    monkeypatch.setattr(credentials_module.importlib, "import_module", missing_import)

    with pytest.raises(MissingDependencyError) as exc_info:
        resolve_async_session(None, None)

    assert exc_info.value.feature == "AsyncDynavec"
    assert exc_info.value.package == "aioboto3"
    assert exc_info.value.extra == "async"


def test_resolve_async_session_default_chain(monkeypatch):
    fake_aioboto3 = _FakeAioboto3()
    monkeypatch.setattr(
        credentials_module.importlib,
        "import_module",
        lambda name: fake_aioboto3,
    )

    result = resolve_async_session(None, None)

    assert result is fake_aioboto3.session
    assert fake_aioboto3.calls == [{}]


def test_resolve_async_session_static_credentials(monkeypatch):
    fake_aioboto3 = _FakeAioboto3()
    monkeypatch.setattr(
        credentials_module.importlib,
        "import_module",
        lambda name: fake_aioboto3,
    )

    credentials = AWSCredentials(
        access_key_id="access",
        secret_access_key="secret",
        session_token="token",
        region="us-east-1",
    )

    result = resolve_async_session(credentials, None)

    assert result is fake_aioboto3.session
    assert fake_aioboto3.calls == [
        {
            "aws_access_key_id": "access",
            "aws_secret_access_key": "secret",
            "aws_session_token": "token",
            "region_name": "us-east-1",
        }
    ]


def test_resolve_async_session_profile(monkeypatch):
    fake_aioboto3 = _FakeAioboto3()
    monkeypatch.setattr(
        credentials_module.importlib,
        "import_module",
        lambda name: fake_aioboto3,
    )

    credentials = AWSCredentials(
        profile_name="dev",
        region="us-west-2",
    )

    resolve_async_session(credentials, None)

    assert fake_aioboto3.calls == [
        {
            "profile_name": "dev",
            "region_name": "us-west-2",
        }
    ]


def test_resolve_async_session_assume_role_reuses_sync_resolution(monkeypatch):
    fake_aioboto3 = _FakeAioboto3()
    monkeypatch.setattr(
        credentials_module.importlib,
        "import_module",
        lambda name: fake_aioboto3,
    )

    frozen = SimpleNamespace(
        access_key="assumed-access",
        secret_key="assumed-secret",
        token="assumed-token",
    )
    resolved = SimpleNamespace(
        get_frozen_credentials=lambda: frozen,
    )
    sync_session = SimpleNamespace(
        get_credentials=lambda: resolved,
    )

    monkeypatch.setattr(
        AWSCredentials,
        "session",
        lambda self: sync_session,
    )

    credentials = AWSCredentials(
        assume_role_arn="arn:aws:iam::123456789012:role/test",
        region="us-east-1",
    )

    result = resolve_async_session(credentials, None)

    assert result is fake_aioboto3.session
    assert fake_aioboto3.calls == [
        {
            "aws_access_key_id": "assumed-access",
            "aws_secret_access_key": "assumed-secret",
            "aws_session_token": "assumed-token",
            "region_name": "us-east-1",
        }
    ]
