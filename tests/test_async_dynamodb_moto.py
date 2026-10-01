"""Moto-backed integration tests for the async DynamoDB store."""

from __future__ import annotations

from typing import Any

import aioboto3
import boto3
import pytest
from moto.server import ThreadedMotoServer

from dynavec.config import DynavecConfig
from dynavec.stores.async_dynamodb import AsyncDynamoDBStore


class MotoAsyncSession:
    """aioboto3 session that routes DynamoDB calls to Moto server."""

    def __init__(self, endpoint_url: str) -> None:
        self._endpoint_url = endpoint_url
        self._session = aioboto3.Session(
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
            region_name="us-east-1",
        )

    def resource(
        self,
        service_name: str,
        **kwargs: Any,
    ) -> Any:
        return self._session.resource(
            service_name,
            endpoint_url=self._endpoint_url,
            **kwargs,
        )


@pytest.fixture
def moto_dynamodb_endpoint() -> str:
    server = ThreadedMotoServer(port=0)
    server.start()

    try:
        _, port = server.get_host_and_port()
        endpoint_url = f"http://127.0.0.1:{port}"

        client = boto3.client(
            "dynamodb",
            region_name="us-east-1",
            endpoint_url=endpoint_url,
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )

        client.create_table(
            TableName="docs",
            KeySchema=[
                {
                    "AttributeName": "pk",
                    "KeyType": "HASH",
                }
            ],
            AttributeDefinitions=[
                {
                    "AttributeName": "pk",
                    "AttributeType": "S",
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        yield endpoint_url
    finally:
        server.stop()


async def test_async_dynamodb_roundtrip_with_moto(
    moto_dynamodb_endpoint: str,
) -> None:
    config = DynavecConfig(
        vector_bucket="test-bucket",
        index="test-index",
        table="docs",
        dimension=4,
        region="us-east-1",
    )

    session = MotoAsyncSession(
        moto_dynamodb_endpoint,
    )

    async with AsyncDynamoDBStore(
        config,
        session,
    ) as store:
        await store.put_many(
            "tenant-a",
            [
                (
                    "doc-1",
                    "Hello async world",
                    {
                        "topic": "async",
                        "source": "moto",
                    },
                )
            ],
        )

        documents = await store.get_many(
            "tenant-a",
            ["doc-1"],
        )

    assert documents == {
        "doc-1": {
            "text": "Hello async world",
            "metadata": {
                "topic": "async",
                "source": "moto",
            },
        }
    }
