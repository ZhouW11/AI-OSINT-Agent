import re

from models.report import JobAnalysisReport


def validate_recommendation_claim_support(
    report: JobAnalysisReport,
) -> bool:
    """
    Validate semantic support between Recommendation
    and its referenced Claims.

    Rules:
    1. Every Recommendation must have claim_refs.
    2. Every claim_ref must exist.
    3. Recommendation must have meaningful semantic
       overlap with at least one referenced Claim.
    4. Multi-Claim recommendations are allowed as long
       as at least one referenced Claim supports the
       Recommendation.
    """

    # =========================================
    # Build Claim map
    # =========================================

    claim_map = {}

    for analysis_index, analysis in enumerate(
        report.cross_analysis
    ):

        for claim_index, claim in enumerate(
            analysis.claims
        ):

            claim_ref = (
                f"cross_analysis:"
                f"{analysis_index}"
                f".claims:"
                f"{claim_index}"
            )

            claim_map[claim_ref] = claim

    # =========================================
    # Recommendation groups
    # =========================================

    recommendation_groups = [
        report.job_value.job_directions,
        report.job_value.technical_skills,
        report.job_value.important_areas,
        report.interview_preparation.topics,
        report.interview_preparation.practical_tasks,
    ]

    # =========================================
    # Stop words
    # =========================================

    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "are",
        "is",
        "to",
        "of",
        "in",
        "on",
        "a",
        "an",

        "based",
        "related",
        "relevant",
        "important",
        "development",
        "technical",
        "skill",
        "skills",
        "area",
        "areas",
        "topic",
        "topics",
        "task",
        "tasks",
        "recommendation",

        "寤鸿",
        "鐩稿叧",
        "鎶€鏈?",
        "鏂瑰悜",
        "鑳藉姏",
        "棰嗗煙",
        "閲嶇偣",
        "涓婚",
        "浠诲姟",
    }

    # =========================================
    # Tokenizer
    # =========================================

    def meaningful_tokens(text: str) -> set[str]:

        text = (text or "").lower()

        tokens = set(
            re.findall(
                r"[a-zA-Z][a-zA-Z0-9_-]{2,}|"
                r"[\u4e00-\u9fff]{2,}",
                text,
            )
        )

        return {
            token
            for token in tokens
            if token not in stop_words
        }

    # =========================================
    # Validate recommendations
    # =========================================

    for group in recommendation_groups:

        for recommendation in group:

            if not recommendation.claim_refs:

                raise ValueError(
                    "Recommendation has no claim_refs: "
                    f"{recommendation.name}"
                )

            recommendation_text = " ".join([
                recommendation.name,
                recommendation.rationale,
            ])

            recommendation_tokens = (
                meaningful_tokens(
                    recommendation_text
                )
            )

            if not recommendation_tokens:

                raise ValueError(
                    "Recommendation has no meaningful "
                    f"semantic tokens: {recommendation.name}"
                )

            supported = False

            for claim_ref in recommendation.claim_refs:

                # =====================================
                # Validate claim_ref
                # =====================================

                if claim_ref not in claim_map:

                    raise ValueError(
                        "Recommendation references "
                        f"invalid claim_ref: {claim_ref}"
                    )

                claim = claim_map[claim_ref]

                claim_tokens = meaningful_tokens(
                    claim.statement
                )

                # =====================================
                # Semantic overlap
                # =====================================

                overlap = (
                    recommendation_tokens
                    & claim_tokens
                )

                if overlap:

                    supported = True

                    break

            # =========================================
            # No supporting Claim
            # =========================================

            if not supported:

                raise ValueError(
                    "Recommendation is not semantically "
                    "supported by referenced Claims: "
                    f"{recommendation.name} "
                    f"claim_refs={recommendation.claim_refs}"
                )

    return True
