from typing import List, Literal
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )


# =========================================
# Enums
# =========================================

class ClaimType(str, Enum):
    FACT = "fact"
    COMPETITIVE = "competitive"
    INFERENCE = "inference"
    CAUSAL = "causal"
    INDUSTRY = "industry"


class RecommendationCategory(str, Enum):
    JOB_DIRECTION = "job_direction"
    TECHNICAL_SKILL = "technical_skill"
    IMPORTANT_AREA = "important_area"
    INTERVIEW_TOPIC = "interview_topic"
    PRACTICAL_TASK = "practical_task"


# =========================================
# Evidence
# =========================================

class Evidence(StrictBaseModel):
    source_id: str
    source_type: str
    title: str
    url: str
    description: str | None = None
    snippet: str | None = None
    content: str | None = None


# =========================================
# Claim
# =========================================

class Claim(StrictBaseModel):
    statement: str

    claim_type: ClaimType

    evidence_ids: List[str]

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


# =========================================
# Recommendation
# =========================================

class Recommendation(StrictBaseModel):
    name: str

    rationale: str

    claim_refs: List[str]

    evidence_ids: List[str]

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    priority_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )

    priority_level: Literal[
        "P1",
        "P2",
        "P3",
    ] = "P3"


# =========================================
# Top Recommendation
# =========================================

class TopRecommendation(StrictBaseModel):

    name: str

    rationale: str

    category: RecommendationCategory

    claim_refs: List[str]

    evidence_ids: List[str]

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    priority_score: float = Field(
        ge=0.0,
        le=1.0
    )

    priority_level: Literal[
        "P1",
        "P2",
        "P3",
    ]


# =========================================
# Company Overview
# =========================================

class CompanyOverview(StrictBaseModel):

    summary: str

    evidence_ids: List[str]


# =========================================
# Signals
# =========================================

class GithubSignal(StrictBaseModel):

    project: str
    language: str | None
    technical_direction: str
    activity: str
    evidence: str

    relevance: Literal[
        "direct",
        "ecosystem"
    ]

    source_id: str


class NewsSignal(StrictBaseModel):

    title: str
    source: str
    published_at: str
    summary: str
    implication: str

    evidence: str

    relevance: str

    source_id: str


# =========================================
# Cross Analysis
# =========================================

class CrossAnalysis(StrictBaseModel):

    relationship: str

    evidence: str

    evidence_ids: List[str]

    claims: List[Claim] = Field(
        default_factory=list
    )


# =========================================
# Job Value
# =========================================

class JobValue(StrictBaseModel):

    job_directions: List[Recommendation]

    technical_skills: List[Recommendation]

    important_areas: List[Recommendation]


# =========================================
# Interview Preparation
# =========================================

class InterviewPreparation(StrictBaseModel):

    topics: List[Recommendation]

    practical_tasks: List[Recommendation]


# =========================================
# Reliability
# =========================================

class Reliability(StrictBaseModel):

    supported_conclusions: List[str]

    uncertain_conclusions: List[str]

    limitations: List[str]


# =========================================
# Final Report
# =========================================

class JobAnalysisReport(StrictBaseModel):

    company_overview: CompanyOverview

    github_signals: List[GithubSignal]

    news_signals: List[NewsSignal]

    cross_analysis: List[CrossAnalysis]

    job_value: JobValue

    interview_preparation: InterviewPreparation

    top_recommendations: List[TopRecommendation] = Field(
        default_factory=list
    )

    reliability: Reliability

    evidence: List[Evidence]