"""Domain schemas shared by roles, orchestrator and report.

`reasoning` is the FIRST field of every LLM output schema: the UI streams it
while the rest of the JSON is still being generated.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    empirical = "empirical"
    theoretical = "theoretical"
    novelty = "novelty"
    comparison = "comparison"


class Claim(BaseModel):
    id: str = Field(description="Short id like C1, C2, ...")
    text: str = Field(description="The claim, as close to the paper's wording as possible")
    type: ClaimType
    section: str = Field(description="Section of the paper where the claim is made")


class ClaimWithQueries(Claim):
    search_queries: list[str] = Field(description="2-3 literature search queries that would find evidence for or against this claim")


class ScoutClaimsOutput(BaseModel):
    reasoning: str = Field(description="Brief notes on which claims are central and why")
    claims: list[ClaimWithQueries]


class ScoutRankOutput(BaseModel):
    reasoning: str = Field(description="Brief notes on the selection")
    selected: list[str] = Field(description="Candidate keys (e.g. P3) ordered by relevance, most relevant first")


class Verdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    OVERSTATED = "OVERSTATED"
    UNVERIFIABLE = "UNVERIFIABLE"


class Stance(str, Enum):
    supports = "supports"
    contradicts = "contradicts"
    context = "context"


class Evidence(BaseModel):
    chunk_id: str = Field(description="Exact chunk_id from the provided context")
    quote: str = Field(description="VERBATIM substring of that chunk, <= 300 characters")
    stance: Stance


class AnalystVerdict(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning about the claim against the evidence")
    verdict: Verdict
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence]
    concerns: list[str]


class AnalystRevision(AnalystVerdict):
    response_to_critic: str = Field(description="Point-by-point answer to the critic: what you accept, what you reject and why")


class CriticStance(str, Enum):
    AGREE = "AGREE"
    CHALLENGE = "CHALLENGE"


IssueType = Literal[
    "ungrounded_quote", "overclaim", "missed_contradiction", "wrong_scope", "weak_evidence", "wrong_verdict"
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
    suggested_verdict: Verdict | None = Field(default=None, description="Verdict you believe is correct, or null if you agree")


class MetaReview(BaseModel):
    reasoning: str = Field(description="Short notes on the overall picture")
    meta_review: str = Field(description="5-8 sentence meta-review grounded ONLY in the verified verdicts provided")


ClaimStatus = Literal["accepted", "revised", "disputed"]

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}
