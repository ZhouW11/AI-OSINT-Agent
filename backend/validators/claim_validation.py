from models.report import JobAnalysisReport, Evidence


def validate_claims(
    report: JobAnalysisReport,
    evidence: list[Evidence],
):
    """
    Validate Claim objects.

    Responsibilities:
    - every Claim has evidence
    - evidence_ids exist
    - Claim evidence belongs to the enclosing analysis
    """

    valid_source_ids = {
        item.source_id
        for item in evidence
    }

    for analysis in report.cross_analysis:

        analysis_evidence = set(
            analysis.evidence_ids
        )

        for claim in analysis.claims:

            # =================================
            # Every claim must have evidence
            # =================================

            if not claim.evidence_ids:

                raise ValueError(
                    "Claim has no evidence_ids: "
                    f"{claim.statement}"
                )

            # =================================
            # Claim evidence must exist
            # =================================

            for source_id in claim.evidence_ids:

                if source_id not in valid_source_ids:

                    raise ValueError(
                        "Claim references invalid "
                        f"source_id: {source_id}"
                    )

            # =================================
            # Claim evidence must belong to
            # enclosing CrossAnalysis evidence
            # =================================

            if not set(
                claim.evidence_ids
            ).issubset(
                analysis_evidence
            ):

                raise ValueError(
                    "Claim references evidence "
                    "not present in enclosing "
                    f"cross_analysis: {claim.statement}"
                )

    return True