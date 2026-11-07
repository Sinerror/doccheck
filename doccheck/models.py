"""Domain schemas shared by roles, orchestrator and report.

`reasoning` is the FIRST field of every LLM output schema: the UI streams it
while the rest of the JSON is still being generated. Enum values stay in English (stable API);
the report and the UI show Russian labels (VERDICT_RU).
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    number = "number"          # price, limit, percentage, quantity
    deadline = "deadline"      # term, date, period, response time
    condition = "condition"    # when / for whom / under what condition something applies
    reference = "reference"    # "according to clause 4.2 of the offer ..."
    feature = "feature"        # what is included / available in a plan or region
    other = "other"


class Claim(BaseModel):
    id: str = Field(description="Short id like C1, C2, ...")
    text: str = Field(description="The claim, self-contained, as close to the document's wording as possible")
    type: ClaimType
    quote: str = Field(description="The exact sentence of the document the claim was taken from")
    section: str = Field(default="", description="Heading of the document part where the claim is made, if any")


class ScoutClaimsOutput(BaseModel):
    reasoning: str = Field(description="Brief notes: what the document is, which statements are checkable")
    claims: list[Claim]


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    OVERSTATED = "OVERSTATED"
    UNVERIFIABLE = "UNVERIFIABLE"


VERDICT_RU = {
    "SUPPORTED": "подтверждено",
    "PARTIALLY_SUPPORTED": "подтверждено частично",
    "CONTRADICTED": "противоречит источникам",
    "OVERSTATED": "преувеличено",
    "UNVERIFIABLE": "не найдено в источниках",
}


class Stance(str, Enum):
    supports = "supports"
    contradicts = "contradicts"
    context = "context"


class Evidence(BaseModel):
    chunk_id: str = Field(description="Exact chunk_id from the provided context")
    quote: str = Field(description="VERBATIM substring of that chunk, <= 300 characters")
    stance: Stance


class AnalystVerdict(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning about the claim against the sources")
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence]
    correction: str | None = Field(default=None, description="If the claim is wrong: the correct statement "
                                                             "according to the sources, one sentence; else null")
    concerns: list[str]


class AnalystRevision(AnalystVerdict):
    response_to_critic: str = Field(description="Point-by-point answer to the critic: what you accept, "
                                                "what you reject and why")


class CriticStance(str, Enum):
    AGREE = "AGREE"
    CHALLENGE = "CHALLENGE"


IssueType = Literal[
    "ungrounded_quote", "overclaim", "missed_contradiction", "wrong_scope", "weak_evidence", "wrong_verdict",
    "outdated_source",
]
Severity = Literal["low", "medium", "high"]


class CounterEvidence(BaseModel):
    chunk_id: str
    quote: str = Field(description="VERBATIM substring of that chunk, <= 300 characters")


class Issue(BaseModel):
    type: IssueType
    severity: Severity
    detail: str
    counter_evidence: list[CounterEvidence] = Field(default_factory=list)


class Critique(BaseModel):
    reasoning: str = Field(description="Your audit of the analyst's verdict")
    stance: CriticStance
    issues: list[Issue]
    suggested_verdict: Verdict | None = Field(default=None,
                                              description="Verdict you believe is correct, or null if you agree")


class Summary(BaseModel):
    reasoning: str = Field(description="Short notes on the overall picture")
    summary: str = Field(description="3-6 sentences in Russian, grounded ONLY in the verified verdicts provided")


ClaimStatus = Literal["accepted", "revised", "disputed"]

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}
