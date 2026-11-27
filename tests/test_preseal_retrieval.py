"""UPG-011 (BUG-037): Pre-seal literature retrieval, digest ledger, and stage-1 reuse."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from invar_agent.research.search import (
    LITERATURE_DEADLINE_S,
    LiteratureResult,
    compute_literature_digest,
    retrieve_before_sealing,
)


def test_compute_literature_digest_is_deterministic_and_order_independent():
    a = {"title": "Alpha", "url": "https://example.org/a", "source": "arxiv"}
    b = {"title": "Beta", "url": "https://example.org/b", "source": "semanticscholar"}
    d1 = compute_literature_digest([a, b])
    d2 = compute_literature_digest([b, a])
    assert d1 == d2
    assert len(d1) == 64


@pytest.mark.asyncio
async def test_retrieve_before_sealing_respects_deadline_and_returns_digest():
    assert LITERATURE_DEADLINE_S > 0

    slow_client = MagicMock()
    slow_client.offline = False
    slow_client.timeout = 2.0
    slow_client.cache_dir = None
    slow_client.search.side_effect = lambda *_a, **_kw: (time.sleep(0.4), [])[1]
    results, digest, status = await retrieve_before_sealing("test query", slow_client, deadline_s=0.05)
    assert results == []
    assert status == "DEADLINE"
    assert digest == compute_literature_digest([])

    fast_client = MagicMock()
    fast_client.offline = False
    fast_client.timeout = 2.0
    fast_client.cache_dir = None
    fast_client.search.return_value = [
        LiteratureResult(
            source="arxiv",
            title="Michaelis-Menten kinetics",
            summary="Enzyme kinetics benchmark",
            url="https://example.org/mm",
        )
    ]
    results2, digest2, status2 = await retrieve_before_sealing("Michaelis-Menten", fast_client, deadline_s=2.0)
    assert status2 == "OK"
    assert len(results2) == 1
    assert digest2 == compute_literature_digest(results2)
