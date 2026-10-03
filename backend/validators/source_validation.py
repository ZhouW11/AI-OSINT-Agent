from models.report import Evidence


def validate_source_ids(
    report,
    evidence
):
    """
    Validate:
    1. source_id exists
    2. github_signals use github source_id
    3. news_signals use news source_id
    4. all evidence_ids exist
    """

    # -----------------------------------------
    # Normalize report
    # -----------------------------------------

    if isinstance(report, dict):
        parsed_data = report

    elif hasattr(report, "model_dump"):
        parsed_data = report.model_dump()

    else:
        raise TypeError(
            "validate_source_ids() expects a dict "
            "or a Pydantic model"
        )

    # -----------------------------------------
    # Build authoritative evidence map
    # -----------------------------------------

    evidence_map = {
        item.source_id: item
        for item in evidence
    }

    valid_source_ids = set(
        evidence_map.keys()
    )

    # -----------------------------------------
    # company_overview
    # -----------------------------------------

    company_overview = parsed_data.get(
        "company_overview",
        {}
    )

    overview_ids = company_overview.get(
        "evidence_ids",
        []
    )

    # -----------------------------------------
    # Validate company overview IDs
    # -----------------------------------------

    for source_id in overview_ids:

        if source_id not in valid_source_ids:

            raise ValueError(
                f"Invalid company_overview source_id: "
                f"{source_id}"
            )

    # -----------------------------------------
    # github_signals
    # -----------------------------------------

    for item in parsed_data.get(
        "github_signals",
        []
    ):

        source_id = item.get("source_id")

        if not source_id:
            continue

        # existence
        if source_id not in valid_source_ids:

            raise ValueError(
                f"Invalid GitHub source_id: "
                f"{source_id}"
            )

        # type
        evidence_item = evidence_map[source_id]

        if evidence_item.source_type != "github":

            raise ValueError(
                f"GitHub signal referenced "
                f"non-GitHub source_id: "
                f"{source_id}"
            )

    # -----------------------------------------
    # news_signals
    # -----------------------------------------

    for item in parsed_data.get(
        "news_signals",
        []
    ):

        source_id = item.get("source_id")

        if not source_id:
            continue

        # existence
        if source_id not in valid_source_ids:

            raise ValueError(
                f"Invalid News source_id: "
                f"{source_id}"
            )

        # type
        evidence_item = evidence_map[source_id]

        if evidence_item.source_type != "news":

            raise ValueError(
                f"News signal referenced "
                f"non-News source_id: "
                f"{source_id}"
            )

    # -----------------------------------------
    # cross_analysis
    # -----------------------------------------

    for item in parsed_data.get(
        "cross_analysis",
        []
    ):

        for source_id in item.get(
            "evidence_ids",
            []
        ):

            if source_id not in valid_source_ids:

                raise ValueError(
                    f"Invalid cross_analysis "
                    f"source_id: {source_id}"
                )

    return True