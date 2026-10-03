from models.report import (
    JobAnalysisReport,
    RecommendationCategory,
)


def validate_top_recommendation_integrity(
    report: JobAnalysisReport,
) -> bool:
    """
    Validate that every TopRecommendation is faithfully
    derived from an existing Recommendation.

    TopRecommendation is a derived presentation layer.
    It must not introduce or modify:

    - name
    - rationale
    - category
    - claim_refs
    - evidence_ids
    - confidence
    - priority_score
    - priority_level
    """

    # =========================================
    # Build Recommendation map
    # =========================================

    recommendation_map = {}

    recommendation_groups = [
        (
            report.job_value.job_directions,
            RecommendationCategory.JOB_DIRECTION,
        ),
        (
            report.job_value.technical_skills,
            RecommendationCategory.TECHNICAL_SKILL,
        ),
        (
            report.job_value.important_areas,
            RecommendationCategory.IMPORTANT_AREA,
        ),
        (
            report.interview_preparation.topics,
            RecommendationCategory.INTERVIEW_TOPIC,
        ),
        (
            report.interview_preparation.practical_tasks,
            RecommendationCategory.PRACTICAL_TASK,
        ),
    ]

    for group, category in recommendation_groups:

        for recommendation in group:

            key = (
                recommendation.name.strip().lower()
            )

            if key in recommendation_map:

                raise ValueError(
                    "Duplicate Recommendation name "
                    "detected while validating "
                    "TopRecommendation: "
                    f"{recommendation.name}"
                )

            recommendation_map[key] = (
                recommendation,
                category,
            )

    # =========================================
    # Validate every TopRecommendation
    # =========================================

    for top in report.top_recommendations:

        key = top.name.strip().lower()

        # -----------------------------------------
        # Source Recommendation must exist
        # -----------------------------------------

        if key not in recommendation_map:
            raise ValueError(
                "TopRecommendation does not correspond "
                "to an existing Recommendation: "
                f"{top.name}"
            )

        source, source_category = recommendation_map[key]

        # -----------------------------------------
        # name
        # -----------------------------------------

        if top.name != source.name:
            raise ValueError(
                "TopRecommendation name does not match "
                "source Recommendation: "
                f"{top.name}"
            )

        # -----------------------------------------
        # rationale
        # -----------------------------------------

        if top.rationale != source.rationale:
            raise ValueError(
                "TopRecommendation rationale does not "
                "match source Recommendation: "
                f"{top.name}"
            )

        # -----------------------------------------
        # category
        # -----------------------------------------

        if top.category != source_category:
            raise ValueError(
                "TopRecommendation category does not "
                "match source Recommendation: "
                f"{top.name}"
            )

        # -----------------------------------------
        # claim_refs
        # -----------------------------------------

        if top.claim_refs != source.claim_refs:
            raise ValueError(
                "TopRecommendation claim_refs do not "
                "match source Recommendation: "
                f"{top.name}"
            )

        # -----------------------------------------
        # evidence_ids
        # -----------------------------------------

        if top.evidence_ids != source.evidence_ids:

            raise ValueError(
                "TopRecommendation evidence_ids do not "
                "match source Recommendation: "
                f"{top.name}"
            )

        # -----------------------------------------
        # confidence
        # -----------------------------------------

        if top.confidence != source.confidence:

            raise ValueError(
                "TopRecommendation confidence does not "
                "match source Recommendation: "
                f"{top.name}"
            )

        # -----------------------------------------
        # priority_score
        # -----------------------------------------

        if top.priority_score != source.priority_score:

            raise ValueError(
                "TopRecommendation priority_score does not "
                "match source Recommendation: "
                f"{top.name}"
            )

        # -----------------------------------------
        # priority_level
        # -----------------------------------------

        if top.priority_level != source.priority_level:

            raise ValueError(
                "TopRecommendation priority_level does not "
                "match source Recommendation: "
                f"{top.name}"
            )

    return True