"""MemGPT-inspired virtual context pager and tri-factor memory retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque
from math import exp
from typing import Any, Callable
import re


@dataclass
class MemoryNode:
    content: str
    importance: float = 5.0
    embedding: list[float] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


class MemGPTPager:
    def __init__(
        self,
        token_budget: int = 4000,
        tau_warn: float = 0.70,
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        if token_budget <= 0:
            raise ValueError("token_budget must be positive.")
        if not 0 < tau_warn <= 1:
            raise ValueError("tau_warn must be in (0, 1].")
        self.token_budget = token_budget
        self.tau_warn = tau_warn
        self.core_memory: dict[str, str] = {}
        self.fifo_queue: deque[MemoryNode] = deque()
        self.archival_store: list[MemoryNode] = []
        self.interrupts: list[str] = []
        self.embedding_fn = embedding_fn or self._simple_embedding

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Lightweight deterministic approximation; not a tokenizer.
        return max(1, len(re.findall(r"\w+|[^\w\s]", text)))

    @property
    def current_token_usage(self) -> int:
        core = sum(self._estimate_tokens(k + " " + v) for k, v in self.core_memory.items())
        queued = sum(self._estimate_tokens(node.content) for node in self.fifo_queue)
        return core + queued

    @staticmethod
    def _simple_embedding(text: str) -> list[float]:
        """Small hashed bag-of-words vector; replace with a real embedding if available."""
        vector = [0.0] * 64
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[hash(word) % len(vector)] += 1.0
        return vector

    def core_memory_append(self, key: str, value: str) -> None:
        if not key.strip():
            raise ValueError("Memory key cannot be empty.")
        self.core_memory[key] = self.core_memory.get(key, "") + value
        self.check_memory_pressure()

    def core_memory_replace(self, key: str, value: str) -> None:
        if not key.strip():
            raise ValueError("Memory key cannot be empty.")
        self.core_memory[key] = value
        self.check_memory_pressure()

    def archival_memory_insert(
        self,
        content: str,
        importance: float = 5.0,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        node = MemoryNode(
            content=content,
            importance=max(0.0, min(10.0, float(importance))),
            embedding=list(embedding) if embedding is not None else self.embedding_fn(content),
            metadata=dict(metadata or {}),
        )
        self.archival_store.append(node)
        return node

    def add_to_fifo(self, content: str, importance: float = 5.0) -> MemoryNode:
        node = MemoryNode(content, importance, self.embedding_fn(content))
        self.fifo_queue.append(node)
        self.check_memory_pressure()
        return node

    def check_memory_pressure(self) -> bool:
        """Warn at tau_warn and archive the oldest half of the FIFO queue."""
        usage = self.current_token_usage
        if usage / self.token_budget < self.tau_warn:
            return False
        warning = (
            f"memory_pressure_warning: usage={usage}/{self.token_budget} "
            f"({usage / self.token_budget:.1%})"
        )
        self.interrupts.append(warning)
        flush_count = (len(self.fifo_queue) + 1) // 2
        if flush_count:
            consolidated = list(self.fifo_queue)[:flush_count]
            for _ in range(flush_count):
                self.fifo_queue.popleft()
            summary = "Archived FIFO batch: " + " | ".join(n.content for n in consolidated)
            avg_importance = sum(n.importance for n in consolidated) / len(consolidated)
            self.archival_memory_insert(
                summary, avg_importance,
                metadata={"source": "fifo_flush", "items_archived": len(consolidated)},
            )
        return True

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def score_memory(
        self,
        memory: MemoryNode,
        query_embedding: list[float],
        *,
        alpha_rec: float = 1.0,
        alpha_imp: float = 1.0,
        alpha_rel: float = 1.0,
        now: datetime | None = None,
    ) -> float:
        """Score = weighted recency decay + normalized importance + cosine relevance.

        The assignment gives gamma=0.99 and divides importance by 10. Here recency
        is gamma raised to hours since last access, so older memories decay smoothly.
        """
        now = now or datetime.now(timezone.utc)
        last = memory.last_accessed
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        hours = max(0.0, (now - last).total_seconds() / 3600.0)
        recency = 0.99 ** hours
        importance = memory.importance / 10.0
        relevance = self._cosine(memory.embedding, query_embedding)
        return alpha_rec * recency + alpha_imp * importance + alpha_rel * relevance

    def archival_memory_search(
        self,
        query: str,
        limit: int = 5,
        *,
        alpha_rec: float = 1.0,
        alpha_imp: float = 1.0,
        alpha_rel: float = 1.0,
    ) -> list[tuple[MemoryNode, float]]:
        query_embedding = self.embedding_fn(query)
        ranked = [
            (node, self.score_memory(
                node, query_embedding, alpha_rec=alpha_rec,
                alpha_imp=alpha_imp, alpha_rel=alpha_rel
            ))
            for node in self.archival_store
        ]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        now = datetime.now(timezone.utc)
        for node, _ in ranked[:max(0, limit)]:
            node.last_accessed = now
        return ranked[:max(0, limit)]


# Convenience functions matching the assignment's named API.
_default_pager = MemGPTPager()


def core_memory_append(key: str, value: str) -> None:
    _default_pager.core_memory_append(key, value)


def core_memory_replace(key: str, value: str) -> None:
    _default_pager.core_memory_replace(key, value)


def archival_memory_insert(content: str, importance: float = 5.0, **kwargs: Any) -> MemoryNode:
    return _default_pager.archival_memory_insert(content, importance, **kwargs)


def archival_memory_search(query: str, limit: int = 5) -> list[tuple[MemoryNode, float]]:
    return _default_pager.archival_memory_search(query, limit)


def check_memory_pressure() -> bool:
    return _default_pager.check_memory_pressure()


def score_memory(memory: MemoryNode, query_embedding: list[float], **kwargs: Any) -> float:
    return _default_pager.score_memory(memory, query_embedding, **kwargs)


if __name__ == "__main__":
    pager = MemGPTPager(token_budget=50, tau_warn=0.70)
    pager.core_memory_append("mission", "Build four agent architecture modules.")
    pager.add_to_fifo("Older context item one.")
    pager.add_to_fifo("Older context item two.")
    pager.add_to_fifo("Older context item three.")
    print("Usage:", pager.current_token_usage)
    print("Interrupts:", pager.interrupts)
    print("Archive search:", [(m.content, round(s, 3)) for m, s in pager.archival_memory_search("older context")])
