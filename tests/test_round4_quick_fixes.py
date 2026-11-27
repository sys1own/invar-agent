"""Sprint 1 quick-fix regressions: BUG-038 (OEIS), BUG-039 (total_fsm), BUG-024 (tri-state gates)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from invar_agent.ir.total_fsm import InconclusiveReasonCode, classify_exception_to_reason
from invar_agent.research.search import LiteratureSearchClient


def test_bug038_oeis_handles_list_and_dict_payloads():
    client = LiteratureSearchClient()

    def _fake_resp(payload):
        m = MagicMock()
        m.read.return_value = json.dumps(payload).encode("utf-8")
        m.__enter__.return_value = m
        m.__exit__.return_value = False
        return m

    list_payload = [{"number": 45, "name": "Fibonacci numbers", "data": "0,1,1,2,3,5,8"}]
    with patch("urllib.request.urlopen", return_value=_fake_resp(list_payload)):
        res_list = client._search_oeis("fibonacci", limit=3)
    assert len(res_list) == 1
    assert "45" in res_list[0].title and "Fibonacci" in res_list[0].title

    dict_payload = {"results": [{"number": 45, "name": "Fibonacci numbers", "data": "0,1,1,2,3,5,8"}]}
    with patch("urllib.request.urlopen", return_value=_fake_resp(dict_payload)):
        res_dict = client._search_oeis("fibonacci", limit=3)
    assert len(res_dict) == 1
    assert "45" in res_dict[0].title

    with patch("urllib.request.urlopen", return_value=_fake_resp({"results": None})):
        assert client._search_oeis("empty", limit=3) == []


def test_bug039_total_fsm_classifier_uses_exception_types_and_word_boundaries():
    _state, reason = classify_exception_to_reason(TimeoutError("step exceeded budget"))
    assert reason == InconclusiveReasonCode.PROVIDER_TRANSPORT_FAULT

    # "inf" inside "information" or "nan" inside "finance" must NOT trigger SOLVER_STIFFNESS_BLOWUP
    _state2, reason2 = classify_exception_to_reason(RuntimeError("information about finance failed"))
    assert reason2 != InconclusiveReasonCode.SOLVER_STIFFNESS_BLOWUP

    _state3, reason3 = classify_exception_to_reason(ValueError("DIMENSIONAL_INCONSISTENCY: L vs T"))
    assert reason3 == InconclusiveReasonCode.DIMENSIONAL_INCONSISTENCY
