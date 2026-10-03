from models.report import JobAnalysisReport
from validators.target_consistency import get_evidence_relevance


def validate_claim_semantics(
    report: JobAnalysisReport,
):
    """
    Validate semantic compatibility between
    Claim type and Evidence relevance.
    """

    for analysis_index, analysis in enumerate(
        report.cross_analysis
    ):

        for claim_index, claim in enumerate(
            analysis.claims
        ):

            claim_type = claim.claim_type

            for source_id in claim.evidence_ids:

                relevance = get_evidence_relevance(
                    report,
                    source_id
                )

                # =================================
                # Direct evidence
                # =================================

                if relevance == "direct":
                    continue

                # =================================
                # Competitive evidence
                # =================================

                if relevance == "competitive":

                    if claim_type != "competitive":

                        raise ValueError(
                            "Competitive evidence cannot "
                            "support non-competitive Claim: "
                            f"cross_analysis:"
                            f"{analysis_index}.claims:"
                            f"{claim_index}"
                        )

                # =================================
                # Industry evidence
                # =================================

                elif relevance == "industry":

                    if claim_type != "industry":

                        raise ValueError(
                            "Industry evidence cannot "
                            "support non-industry Claim: "
                            f"cross_analysis:"
                            f"{analysis_index}.claims:"
                            f"{claim_index}"
                        )

                # =================================
                # Ecosystem evidence
                # =================================

                elif relevance == "ecosystem":

                    if claim_type not in {
                        "industry",
                        "inference",
                    }:

                        raise ValueError(
                            "Ecosystem evidence cannot "
                            "support direct company Claim: "
                            f"cross_analysis:"
                            f"{analysis_index}.claims:"
                            f"{claim_index}"
                        )