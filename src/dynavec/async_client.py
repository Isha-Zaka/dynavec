"""Async Dynavec client for high-concurrency workloads."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from contextlib import AsyncExitStack
from types import TracebackType
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from .cache import BaseCache

from .client_common import (
    DDBPayload,
    HotPayload,
    S3Payload,
    apply_transform_pipeline,
    assign_embeddings,
    build_write_payloads,
    documents_to_embed,
    embedding_texts,
)
from .config import DynavecConfig
from .credentials import AWSCredentials, resolve_async_session
from .embeddings.base import Embedder
from .exceptions import ConfigurationError
from .hot import HotTier
from .models import Document, UpsertResult
from .stores.async_dynamodb import AsyncDynamoDBStore
from .stores.async_s3vectors import AsyncS3VectorsStore
from .telemetry import TelemetryRecorder
from .transforms import Transform, TransformPipeline, as_pipeline

TransformSpec = Union[TransformPipeline, Transform, Iterable[Transform]]


class AsyncDynavec:
    """Async Dynavec client backed by aioboto3."""

    def __init__(
        self,
        config: DynavecConfig,
        embedder: Embedder | None = None,
        *,
        credentials: AWSCredentials | None = None,
        boto_session: Any | None = None,
        transform: TransformSpec | None = None,
        cache: BaseCache | None = None,
        telemetry: TelemetryRecorder | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self._session = resolve_async_session(
            credentials,
            boto_session,
        )

        self._vectors = AsyncS3VectorsStore(
            config,
            boto_session=self._session,
        )
        self._docs = AsyncDynamoDBStore(
            config,
            boto_session=self._session,
        )

        self._default_transform = as_pipeline(transform)
        self._cache = cache
        self._telemetry = telemetry
        self._hot = (
            HotTier(config)
            if config.hot_tier
            else None
        )

        self._exit_stack: AsyncExitStack | None = None

        if (
            embedder is not None
            and embedder.dimension != config.dimension
        ):
            raise ConfigurationError(
                f"Embedder dimension ({embedder.dimension}) "
                f"!= index dimension ({config.dimension}). "
                "Fix the embedder or DynavecConfig.dimension."
            )

    async def __aenter__(self) -> AsyncDynavec:
        if self._exit_stack is not None:
            return self

        stack = AsyncExitStack()
        await stack.__aenter__()

        try:
            await stack.enter_async_context(self._vectors)
            await stack.enter_async_context(self._docs)
        except Exception:
            await stack.aclose()
            raise

        self._exit_stack = stack
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        stack = self._exit_stack

        if stack is None:
            return

        try:
            await stack.__aexit__(
                exc_type,
                exc_value,
                traceback,
            )
        finally:
            self._exit_stack = None

    async def aclose(self) -> None:
        stack = self._exit_stack

        if stack is None:
            return

        try:
            await stack.aclose()
        finally:
            self._exit_stack = None

    def _require_open(self) -> None:
        if self._exit_stack is None:
            raise RuntimeError(
                "AsyncDynavec is not open. "
                "Use it inside 'async with' before performing AWS operations."
            )

    def _invalidate_cache(self, namespace: str) -> None:
        if (
            self._cache is not None
            and self.config.cache_invalidate_on_write
        ):
            self._cache.invalidate(namespace)

    async def _prepare(
        self,
        docs: list[Document],
        namespace: str,
        auto_metadata: bool,
        transform: TransformSpec | None,
        default_ttl_seconds: int | None = None,
    ) -> tuple[
        list[S3Payload],
        list[DDBPayload],
        list[str],
        list[HotPayload],
    ]:
        pipeline = (
            as_pipeline(transform)
            or self._default_transform
        )

        apply_transform_pipeline(
            docs,
            namespace,
            pipeline,
        )

        to_embed = documents_to_embed(docs)

        if to_embed:
            if self.embedder is None:
                raise ConfigurationError(
                    "Some documents have no vector and no embedder "
                    "is configured. Pass an embedder to "
                    "AsyncDynavec(...) or provide precomputed vectors."
                )

            texts = embedding_texts(to_embed)

            vectors = await self.embedder.aembed_documents(
                texts
            )

            assign_embeddings(
                docs,
                to_embed,
                vectors,
            )

        return build_write_payloads(
            docs,
            self.config,
            namespace,
            auto_metadata,
            default_ttl_seconds=default_ttl_seconds,
        )

    async def aupsert(
        self,
        documents: Sequence[
            Document | dict[str, Any]
        ]
        | None = None,
        *,
        namespace: str = "default",
        auto_metadata: bool = False,
        transform: TransformSpec | None = None,
        ttl_seconds: int | None = None,
    ) -> UpsertResult:
        """Insert or overwrite documents asynchronously."""
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError(
                f"ttl_seconds must be positive, got {ttl_seconds}."
            )

        if not documents:
            return UpsertResult(
                count=0,
                ids=[],
            )

        self._require_open()

        docs = [
            document
            if isinstance(document, Document)
            else Document(**document)
            for document in documents
        ]

        (
            s3_payload,
            ddb_payload,
            ids,
            hot_payload,
        ) = await self._prepare(
            docs,
            namespace,
            auto_metadata,
            transform,
            default_ttl_seconds=ttl_seconds,
        )

        await asyncio.gather(
            self._vectors.put_vectors(
                s3_payload,
                max_workers=self.config.max_workers,
            ),
            self._docs.put_many(
                namespace,
                ddb_payload,
            ),
        )

        if self._hot is not None:
            self._hot.insert_many(
                namespace,
                hot_payload,
            )

        self._invalidate_cache(namespace)

        return UpsertResult(
            count=len(ids),
            ids=ids,
        )
