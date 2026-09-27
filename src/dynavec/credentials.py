"""AWS credential handling — connect to the user's account many ways.

Supports (in order of precedence when multiple are given):
  * explicit access key / secret / optional session token
  * a named profile from ~/.aws/credentials
  * assume-role via STS (cross-account, with optional external id)
  * the default boto3 chain (env vars, instance role, SSO) when nothing is set

The resulting object is a frozen, side-effect-free description; call
:meth:`session` to materialize a ``boto3.Session``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from .exceptions import MissingDependencyError


@dataclass(frozen=True)
class AWSCredentials:
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    profile_name: str | None = None
    region: str | None = None

    # cross-account assume-role
    assume_role_arn: str | None = None
    role_session_name: str = "dynavec"
    external_id: str | None = None

    def session(self) -> Any:
        """Build a ``boto3.Session`` from this credential description."""
        import boto3

        if self.assume_role_arn:
            return self._assume_role_session(boto3)

        kwargs: dict[str, str] = {}
        if self.access_key_id and self.secret_access_key:
            kwargs["aws_access_key_id"] = self.access_key_id
            kwargs["aws_secret_access_key"] = self.secret_access_key
            if self.session_token:
                kwargs["aws_session_token"] = self.session_token
        if self.profile_name:
            kwargs["profile_name"] = self.profile_name
        if self.region:
            kwargs["region_name"] = self.region
        return boto3.Session(**kwargs)

    def _assume_role_session(self, boto3: Any) -> Any:
        # Base session used only to call STS.
        base_kwargs: dict[str, str] = {}
        if self.access_key_id and self.secret_access_key:
            base_kwargs["aws_access_key_id"] = self.access_key_id
            base_kwargs["aws_secret_access_key"] = self.secret_access_key
            if self.session_token:
                base_kwargs["aws_session_token"] = self.session_token
        if self.profile_name:
            base_kwargs["profile_name"] = self.profile_name
        if self.region:
            base_kwargs["region_name"] = self.region

        base = boto3.Session(**base_kwargs)
        sts = base.client("sts")

        assume_kwargs = {
            "RoleArn": self.assume_role_arn,
            "RoleSessionName": self.role_session_name,
        }
        if self.external_id:
            assume_kwargs["ExternalId"] = self.external_id
        creds = sts.assume_role(**assume_kwargs)["Credentials"]

        return boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            region_name=self.region,
        )


def resolve_session(credentials: AWSCredentials | None, boto_session: Any | None) -> Any:
    """Pick a boto3 session: explicit session > credentials > default chain."""
    if boto_session is not None:
        return boto_session
    if credentials is not None:
        return credentials.session()
    import boto3

    return boto3.Session()


def resolve_async_session(
    credentials: AWSCredentials | None,
    boto_session: Any | None,
) -> Any:
    """Pick an aioboto3 session: explicit session > credentials > default chain."""
    if boto_session is not None:
        return boto_session

    try:
        aioboto3: Any = importlib.import_module("aioboto3")
    except ImportError as exc:
        raise MissingDependencyError("AsyncDynavec", "aioboto3", "async") from exc

    if credentials is None:
        return aioboto3.Session()

    if credentials.assume_role_arn:
        # Reuse the existing synchronous STS assume-role flow once during setup.
        sync_session = credentials.session()
        resolved = sync_session.get_credentials()
        if resolved is None:
            raise RuntimeError("Unable to resolve assumed-role AWS credentials.")

        frozen = resolved.get_frozen_credentials()
        return aioboto3.Session(
            aws_access_key_id=frozen.access_key,
            aws_secret_access_key=frozen.secret_key,
            aws_session_token=frozen.token,
            region_name=credentials.region,
        )

    kwargs: dict[str, str] = {}

    if credentials.access_key_id and credentials.secret_access_key:
        kwargs["aws_access_key_id"] = credentials.access_key_id
        kwargs["aws_secret_access_key"] = credentials.secret_access_key

        if credentials.session_token:
            kwargs["aws_session_token"] = credentials.session_token

    if credentials.profile_name:
        kwargs["profile_name"] = credentials.profile_name

    if credentials.region:
        kwargs["region_name"] = credentials.region

    return aioboto3.Session(**kwargs)
