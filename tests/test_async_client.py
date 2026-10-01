"""Tests for the high-level AsyncDynavec client."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from dynavec.async_client import AsyncDynavec
from dynavec.config import DynavecConfig
from dynavec.embeddings.base import Embedder
from dynavec.exceptions import (
    ConfigurationError,
    DimensionMismatchError,
)


def _config() -> DynavecConfig:
    return DynavecConfig(
        vector_bucket="bucket",
        index="index",
        table="docs",
        dimension=4,
    )


class AsyncOnlyEmbedder(Embedder):
    dimension = 4

    def __init__(self) -> None:
        self.async_calls: list[list[str]] = []
        self.async_query_calls: list[str] = []

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        raise AssertionError(
            "AsyncDynavec must not call sync embed_query()."
        )

    async def aembed_query(
        self,
        text: str,
    ) -> list[float]:
        self.async_query_calls.append(text)
        return [1.0, 2.0, 3.0, 4.0]

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        raise AssertionError(
            "AsyncDynavec must not call sync embed_documents()."
        )

    async def aembed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        self.async_calls.append(texts)

        return [
            [1.0, 2.0, 3.0, 4.0]
            for _ in texts
        ]


class FakeVectorStore:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.put_calls: list[
            tuple[list[tuple[str, list[float], dict[str, Any]]], int]
        ] = []

    async def __aenter__(self) -> FakeVectorStore:
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.exited = True

    async def put_vectors(
        self,
        vectors: list[
            tuple[str, list[float], dict[str, Any]]
        ],
        max_workers: int = 8,
    ) -> None:
        self.put_calls.append(
            (vectors, max_workers)
        )


class FakeDocumentStore:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.put_calls: list[
            tuple[
                str,
                list[
                    tuple[
                        str,
                        str | None,
                        dict[str, Any],
                    ]
                ],
            ]
        ] = []

    async def __aenter__(
        self,
    ) -> FakeDocumentStore:
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.exited = True

    async def put_many(
        self,
        namespace: str,
        items: list[
            tuple[
                str,
                str | None,
                dict[str, Any],
            ]
        ],
    ) -> None:
        self.put_calls.append(
            (namespace, items)
        )


def _install_fake_stores(
    client: AsyncDynavec,
    vectors: FakeVectorStore,
    documents: FakeDocumentStore,
) -> None:
    client._vectors = vectors  # type: ignore[assignment]
    client._docs = documents  # type: ignore[assignment]


def _client(
    *,
    embedder: Embedder | None = None,
) -> tuple[
    AsyncDynavec,
    FakeVectorStore,
    FakeDocumentStore,
]:
    client = AsyncDynavec(
        _config(),
        embedder=embedder,
        boto_session=object(),
    )

    vectors = FakeVectorStore()
    documents = FakeDocumentStore()

    _install_fake_stores(
        client,
        vectors,
        documents,
    )

    return client, vectors, documents


async def test_async_context_opens_and_closes_stores() -> None:
    client, vectors, documents = _client()

    async with client:
        assert vectors.entered
        assert documents.entered
        assert not vectors.exited
        assert not documents.exited

    assert vectors.exited
    assert documents.exited


async def test_aclose_closes_open_stores() -> None:
    client, vectors, documents = _client()

    await client.__aenter__()

    assert vectors.entered
    assert documents.entered

    await client.aclose()

    assert vectors.exited
    assert documents.exited

    # Closing twice should be safe.
    await client.aclose()


async def test_aupsert_requires_open_client() -> None:
    client, _, _ = _client()

    with pytest.raises(
        RuntimeError,
        match="not open",
    ):
        await client.aupsert(
            [
                {
                    "id": "a",
                    "vector": [1.0, 2.0, 3.0, 4.0],
                }
            ]
        )


async def test_aupsert_uses_async_embedding_and_writes_both_stores() -> None:
    embedder = AsyncOnlyEmbedder()
    client, vectors, documents = _client(
        embedder=embedder
    )

    async with client:
        result = await client.aupsert(
            [
                {
                    "id": "a",
                    "text": "hello",
                    "metadata": {
                        "topic": "test",
                    },
                }
            ],
            namespace="tenant",
        )

    assert result.count == 1
    assert result.ids == ["a"]

    assert embedder.async_calls == [
        ["hello"]
    ]

    assert len(vectors.put_calls) == 1
    assert len(documents.put_calls) == 1

    vector_payload, _ = vectors.put_calls[0]

    assert len(vector_payload) == 1
    assert vector_payload[0][1] == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]

    namespace, document_payload = (
        documents.put_calls[0]
    )

    assert namespace == "tenant"
    assert document_payload[0][0] == "a"
    assert document_payload[0][1] == "hello"


async def test_aupsert_passes_default_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _, documents = _client()

    monkeypatch.setattr(
        "dynavec.client_common.time.time",
        lambda: 1_700_000_000,
    )

    async with client:
        await client.aupsert(
            [
                {
                    "id": "a",
                    "vector": [1.0, 2.0, 3.0, 4.0],
                }
            ],
            namespace="tenant",
            ttl_seconds=60,
        )

    namespace, document_payload = documents.put_calls[0]

    assert namespace == "tenant"
    assert document_payload[0][2]["_ttl"] == 1_700_000_060


async def test_aupsert_rejects_non_positive_ttl() -> None:
    client, _, _ = _client()

    with pytest.raises(
        ValueError,
        match="ttl_seconds must be positive",
    ):
        await client.aupsert(
            [
                {
                    "id": "a",
                    "vector": [1.0, 2.0, 3.0, 4.0],
                }
            ],
            ttl_seconds=0,
        )


async def test_aupsert_runs_s3_and_dynamodb_writes_concurrently() -> None:
    s3_started = asyncio.Event()
    ddb_started = asyncio.Event()

    class CoordinatedVectors(
        FakeVectorStore
    ):
        async def put_vectors(
            self,
            vectors: list[
                tuple[
                    str,
                    list[float],
                    dict[str, Any],
                ]
            ],
            max_workers: int = 8,
        ) -> None:
            s3_started.set()

            await asyncio.wait_for(
                ddb_started.wait(),
                timeout=1,
            )

    class CoordinatedDocuments(
        FakeDocumentStore
    ):
        async def put_many(
            self,
            namespace: str,
            items: list[
                tuple[
                    str,
                    str | None,
                    dict[str, Any],
                ]
            ],
        ) -> None:
            ddb_started.set()

            await asyncio.wait_for(
                s3_started.wait(),
                timeout=1,
            )

    client = AsyncDynavec(
        _config(),
        boto_session=object(),
    )

    vectors = CoordinatedVectors()
    documents = CoordinatedDocuments()

    _install_fake_stores(
        client,
        vectors,
        documents,
    )

    async with client:
        await asyncio.wait_for(
            client.aupsert(
                [
                    {
                        "id": "a",
                        "vector": [
                            1.0,
                            2.0,
                            3.0,
                            4.0,
                        ],
                    }
                ]
            ),
            timeout=1,
        )

    assert s3_started.is_set()
    assert ddb_started.is_set()


async def test_partial_enter_failure_closes_first_store() -> None:
    vectors = FakeVectorStore()

    class FailingDocumentStore(
        FakeDocumentStore
    ):
        async def __aenter__(
            self,
        ) -> FailingDocumentStore:
            raise RuntimeError(
                "DynamoDB open failed"
            )

    client = AsyncDynavec(
        _config(),
        boto_session=object(),
    )

    documents = FailingDocumentStore()

    _install_fake_stores(
        client,
        vectors,
        documents,
    )

    with pytest.raises(
        RuntimeError,
        match="DynamoDB open failed",
    ):
        async with client:
            pass

    assert vectors.entered
    assert vectors.exited


async def test_resolve_query_vector_uses_precomputed_vector():
    client, _, _ = _client()

    vector = [0.1, 0.2, 0.3, 0.4]

    result = await client._resolve_query_vector(
        None,
        vector,
    )

    assert result is vector


async def test_resolve_query_vector_rejects_wrong_dimension():
    client, _, _ = _client()

    with pytest.raises(
        DimensionMismatchError,
        match="Query vector dimension",
    ):
        await client._resolve_query_vector(
            None,
            [0.1, 0.2],
        )


async def test_resolve_query_vector_requires_query_or_vector():
    client, _, _ = _client()

    with pytest.raises(
        ValueError,
        match="Provide either 'query' text or a 'vector'",
    ):
        await client._resolve_query_vector(
            None,
            None,
        )


async def test_resolve_query_vector_requires_embedder():
    client, _, _ = _client()

    with pytest.raises(
        ConfigurationError,
        match="Text query requires an embedder",
    ):
        await client._resolve_query_vector(
            "hello",
            None,
        )


async def test_resolve_query_vector_uses_async_embedder():
    embedder = AsyncOnlyEmbedder()
    client, _, _ = _client(embedder=embedder)

    result = await client._resolve_query_vector(
        "hello",
        None,
    )

    assert result == [1.0, 2.0, 3.0, 4.0]
    assert embedder.async_query_calls == ["hello"]
