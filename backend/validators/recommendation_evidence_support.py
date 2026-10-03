from models.report import JobAnalysisReport, Evidence


def derive_recommendation_evidence_ids(
    recommendation,
    claim_map,
) -> list[str]:
    """
    Derive Recommendation.evidence_ids exclusively
    from the evidence_ids of referenced Claims.

    Recommendation.evidence_ids is therefore a derived field,
    not an independently trusted LLM field.
    """

    if not recommendation.claim_refs:
        raise ValueError(
            "Recommendation has no claim_refs: "
            f"{recommendation.name}"
        )

    derived_evidence_ids = []

    for claim_ref in recommendation.claim_refs:

        if claim_ref not in claim_map:
            raise ValueError(
                "Recommendation references "
                f"invalid claim_ref: {claim_ref}"
            )

        claim = claim_map[claim_ref]

        if not claim.evidence_ids:
            raise ValueError(
                "Referenced Claim has no evidence_ids: "
                f"{claim_ref}"
            )

        for source_id in claim.evidence_ids:

            if source_id not in derived_evidence_ids:
                derived_evidence_ids.append(
                    source_id
                )

    if not derived_evidence_ids:
        raise ValueError(
            "Recommendation has no evidence derived "
            "from claim_refs: "
            f"{recommendation.name}"
        )

    return derived_evidence_ids


def validate_recommendation_evidence_support(
    report: JobAnalysisReport,
    evidence: list[Evidence],
) -> bool:

    valid_evidence_ids = {
        item.source_id
        for item in evidence
    }

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

    recommendation_groups = [
        report.job_value.job_directions,
        report.job_value.technical_skills,
        report.job_value.important_areas,
        report.interview_preparation.topics,
        report.interview_preparation.practical_tasks,
    ]

    for group in recommendation_groups:

        for recommendation in group:

            if not recommendation.claim_refs:

                raise ValueError(
                    "Recommendation has no claim_refs: "
                    f"{recommendation.name}"
                )

            # =========================================
            # Recommendation evidence_ids is derived
            # exclusively from claim_refs.
            #
            # Never trust the LLM-provided
            # recommendation.evidence_ids.
            # =========================================

            derived_evidence_ids = (
                derive_recommendation_evidence_ids(
                    recommendation,
                    claim_map,
                )
            )

            # =========================================
            # Validate derived evidence exists
            # =========================================

            for source_id in derived_evidence_ids:

                if source_id not in valid_evidence_ids:

                    raise ValueError(
                        "Recommendation references invalid "
                        f"derived evidence_id: {source_id}"
                    )

            # =========================================
            # Overwrite LLM-provided evidence_ids
            # =========================================

            recommendation.evidence_ids = (
                derived_evidence_ids
            )

    return True