from models.report import JobAnalysisReport


def validate_top_recommendation_result_constraints(
    report: JobAnalysisReport,
    top_k: int = 5,
) -> bool:
    """
    Validate final TopRecommendation result constraints.

    Rules:
    1. top_k must be non-negative.
    2. The number of TopRecommendations must not exceed top_k.
    3. An empty result is allowed when no recommendations are available.
    """

    if top_k < 0:
        raise ValueError(
            "top_k must be non-negative."
        )

    count = len(
        report.top_recommendations
    )

    if count > top_k:
        raise ValueError(
            "TopRecommendation count exceeds "
            f"top_k: count={count}, top_k={top_k}"
        )

    return True