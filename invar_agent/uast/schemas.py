"""Declarative Pydantic v2 schemas for the ResearchComputeUAST contract."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _coerce_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("inf", "+inf", "infinity", "+infinity"):
            return float("inf")
        if s in ("-inf", "-infinity"):
            return float("-inf")
        try:
            return float(s)
        except ValueError:
            return default
    return default


def _sanitize_symbol(sym: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(sym or "").strip()).strip("_")
    if not cleaned:
        return "var_0"
    if cleaned[0].isdigit():
        cleaned = "v_" + cleaned
    return cleaned


class PhysicalQuantity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(..., pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    si_dimension: List[int] = Field(
        default_factory=lambda: [0, 0, 0, 0, 0, 0, 0],
        min_length=7,
        max_length=7,
        description="[Mass, Length, Time, Current, Temperature, Amount, Luminosity]",
    )
    interval: Tuple[float, float] = Field(default=(-float("inf"), float("inf")))
    is_state_variable: bool = Field(default=False)
    initial_value: float = Field(default=0.0)
    default_value: float = Field(default=1.0)
    unit: str = Field(default="")
    description: str = Field(default="")
    dimension_unspecified: bool = Field(default=False)
    semantic_tag: int = Field(
        default=0,
        ge=0,
        le=5,
        description="0=Physical, 1=Probability/Fraction, 2=Information/Entropy, 3=Population/Count, 4=StatisticalMetric, 5=EconomicUtility",
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def _clean_symbol(cls, v: Any) -> str:
        return _sanitize_symbol(str(v))

    @model_validator(mode="before")
    @classmethod
    def _normalize_quantity(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        if "si_dimension" not in d or d["si_dimension"] is None:
            d["si_dimension"] = [0, 0, 0, 0, 0, 0, 0]
            d["dimension_unspecified"] = True
        else:
            raw_dim = list(d["si_dimension"])
            if len(raw_dim) < 7:
                raw_dim.extend([0] * (7 - len(raw_dim)))
            elif len(raw_dim) > 7:
                raw_dim = raw_dim[:7]
            d["si_dimension"] = [int(round(_coerce_float(x, 0.0))) for x in raw_dim]
        if "interval" in d and isinstance(d["interval"], (list, tuple)) and len(d["interval"]) == 2:
            lo = _coerce_float(d["interval"][0], -float("inf"))
            hi = _coerce_float(d["interval"][1], float("inf"))
            if lo > hi:
                lo, hi = hi, lo
            d["interval"] = (lo, hi)
        else:
            d["interval"] = (-float("inf"), float("inf"))
        if "initial_value" in d:
            d["initial_value"] = _coerce_float(d["initial_value"], 0.0)
        if "default_value" in d:
            d["default_value"] = _coerce_float(d["default_value"], d.get("initial_value", 1.0))
        elif "value" in d:
            d["default_value"] = _coerce_float(d["value"], 1.0)
        return d


class OntologyAndParameterSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    quantities: Dict[str, PhysicalQuantity] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_quantities(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return {"quantities": {}}
        raw_q = data.get("quantities", data)
        norm: Dict[str, Any] = {}
        if isinstance(raw_q, list):
            for item in raw_q:
                if isinstance(item, dict) and "symbol" in item:
                    sym = _sanitize_symbol(item["symbol"])
                    norm[sym] = {**item, "symbol": sym}
        elif isinstance(raw_q, dict):
            for k, v in raw_q.items():
                if k in ("quantities",):
                    continue
                sym = _sanitize_symbol(k)
                if isinstance(v, PhysicalQuantity):
                    norm[sym] = v
                elif isinstance(v, dict):
                    if sym == "var_0" and isinstance(v.get("symbol"), str) and v["symbol"].strip():
                        sym = _sanitize_symbol(v["symbol"])
                    norm[sym] = {**v, "symbol": sym}
                elif isinstance(v, (int, float)):
                    norm[sym] = {
                        "symbol": sym,
                        "default_value": float(v),
                        "initial_value": float(v),
                        "dimension_unspecified": True,
                    }
        return {"quantities": norm}

    def state_variables(self) -> List[PhysicalQuantity]:
        return [q for q in self.quantities.values() if q.is_state_variable]

    def parameters(self) -> List[PhysicalQuantity]:
        return [q for q in self.quantities.values() if not q.is_state_variable]


class SymbolicGovernanceMap(BaseModel):
    model_config = ConfigDict(extra="ignore")

    equations: Dict[str, str] = Field(default_factory=dict)
    observables: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_governance(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return {"equations": {}, "observables": {}}
        if "equations" not in data and "observables" not in data:
            # Flat dict of state_var -> expr
            return {"equations": {str(k): str(v) for k, v in data.items()}, "observables": {}}
        eqs = {str(k): str(v) for k, v in (data.get("equations") or {}).items()}
        obs = {str(k): str(v) for k, v in (data.get("observables") or {}).items()}
        return {"equations": eqs, "observables": obs}


class InvariantContractSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    conserved_quantities: List[str] = Field(default_factory=list)
    monotonic_increasing: List[str] = Field(default_factory=list)
    custom_smt_assertions: List[str] = Field(default_factory=list)


class OperatorMutation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target_node_id: str
    new_csel_expression: str


class UASTPerturbationDelta(BaseModel):
    model_config = ConfigDict(extra="ignore")

    add_terms: Dict[str, str] = Field(default_factory=dict)
    modify_terms: List[OperatorMutation] = Field(default_factory=list)
    add_quantities: Dict[str, PhysicalQuantity] = Field(default_factory=dict)
    add_observables: Dict[str, str] = Field(default_factory=dict)


class ResearchComputeUAST(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = Field(default="Verified UAST Research Model")
    hypothesis: str = Field(default="Falsifiable dynamical system model.")
    ontology: OntologyAndParameterSpec
    governance: SymbolicGovernanceMap
    invariants: InvariantContractSpec = Field(default_factory=InvariantContractSpec)
    dt: float = Field(default=0.01, gt=0.0, le=1e12)
    steps: int = Field(default=1000, ge=1, le=200000)
    target_backend: Literal["rust_hin", "cpp", "multi_variant"] = "multi_variant"
    solver_override: Optional[str] = None

    @model_validator(mode="after")
    def _ensure_state_variables_consistent(self) -> "ResearchComputeUAST":
        # Every LHS in governance.equations must be marked is_state_variable=True
        for var in self.governance.equations:
            if var in self.ontology.quantities:
                self.ontology.quantities[var].is_state_variable = True
            else:
                self.ontology.quantities[var] = PhysicalQuantity(
                    symbol=var,
                    is_state_variable=True,
                    dimension_unspecified=True,
                )
        return self


class CounterexampleDiagnosticCard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target_equation_name: str
    failed_expression: str
    counterexample_vector: Dict[str, float] = Field(default_factory=dict)
    sub_expression_trace: List[str] = Field(default_factory=list)
    dimension_residual: str = ""
    smt_assertion_failed: str = ""
    jacobian_spectrum: Dict[str, Any] = Field(default_factory=dict)
    available_ontology_keys: List[str] = Field(default_factory=list)
    target_si_vector: List[int] = Field(default_factory=lambda: [0, 0, 0, 0, 0, 0, 0])
