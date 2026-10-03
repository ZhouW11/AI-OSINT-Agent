from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.company import CompanyIntelligence
from models.report import JobAnalysisReport

from services.github_service import search_github
from services.news_service import search_news_articles
from llm import analyze


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


app = FastAPI(
    title="AI-OSINT-Agent",
    version="2.9.0",
    description=(
        "AI-powered OSINT intelligence analysis system "
        "for company research and job-seeking insights."
    ),
    openapi_tags=[
        {
            "name": "System",
            "description": "Service health and runtime status.",
        },
        {
            "name": "Analysis",
            "description": "Company OSINT analysis endpoints.",
        },
    ],
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check",
    description="Returns the current service health and API version.",
    response_description="Current service health status.",
    operation_id="health_check",
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="AI-OSINT-Agent",
        version="2.9.0",
    )


@app.post(
    "/analyze",
    response_model=JobAnalysisReport,
    tags=["Analysis"],
    summary="Analyze a target company",
    description=(
        "Collects GitHub and news intelligence for the target company, "
        "runs the structured analysis pipeline, validates the result, "
        "and returns a structured job-seeking intelligence report."
    ),
    response_description="Structured company intelligence and job-seeking analysis report.",
    operation_id="analyze_company",
)
def analysis(
    keyword: str = Query(
        ...,
        min_length=1,
        description="Target company or organization keyword, for example: OpenAI.",
    )
) -> JobAnalysisReport:
    github_data = search_github(keyword)
    news_data = search_news_articles(keyword)

    data = CompanyIntelligence(
        github=github_data,
        news=news_data,
    )

    result = analyze(
        data,
        target_company=keyword,
    )

    return result
