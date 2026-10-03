from models.report import JobAnalysisReport


def get_evidence_relevance(
    report: JobAnalysisReport,
) -> dict[str, str]:
    """
    Resolve authoritative relevance for every evidence item
    based on the LLM's validated signals.

    Returns:
        {
            source_id: "direct" | "competitive" | "industry" | "ecosystem"
        }
    """

    relevance_map = {}

    # -----------------------------------------
    # GitHub signals
    # -----------------------------------------

    for signal in report.github_signals:

        relevance_map[
            signal.source_id
        ] = signal.relevance

    # -----------------------------------------
    # News signals
    # -----------------------------------------

    for signal in report.news_signals:

        relevance_map[
            signal.source_id
        ] = signal.relevance

    return relevance_map


def validate_target_consistency(
    report: JobAnalysisReport,
    target_company: str,
):
    target = (
        target_company
        .strip()
        .lower()
    )

    relevance_map = get_evidence_relevance(
        report
    )

    # =========================================
    # Company Overview
    # =========================================

    for source_id in (
        report.company_overview.evidence_ids
    ):

        relevance = relevance_map.get(
            source_id
        )

        if relevance in (
            "competitive",
            "industry",
        ):

            raise ValueError(
                "Company overview cannot use "
                f"{relevance} evidence as direct "
                f"company fact: {source_id}"
            )

    # =========================================
    # GitHub signals
    # =========================================

    for signal in report.github_signals:

        if signal.relevance == "direct":

            source_id = (
                signal.source_id.lower()
            )

            if not source_id.startswith(
                f"github:{target}/"
            ):

                raise ValueError(
                    "Direct GitHub signal does not "
                    f"belong to target company "
                    f"{target_company}: "
                    f"{signal.source_id}"
                )

    # =========================================
    # News signals
    # =========================================

    for signal in report.news_signals:

        if signal.relevance == "direct":

            text = " ".join([
                signal.title,
                signal.summary,
                signal.implication,
            ]).lower()

            if target not in text:

                raise ValueError(
                    "Direct news signal does not "
                    f"explicitly mention target company "
                    f"{target_company}: "
                    f"{signal.title}"
                )

    return True