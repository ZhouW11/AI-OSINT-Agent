from models.report import JobAnalysisReport


def validate_evidence_content_consistency(
    report: JobAnalysisReport,
    evidence,
):
    """
    Validate that signals are consistent with the
    authoritative Evidence metadata.

    Responsibilities:
    - GitHub signal.project must match Evidence.title
    - News signal.title must match Evidence.title

    This validator does NOT evaluate semantic truth
    of summaries or implications. It only verifies
    source-to-signal metadata consistency.
    """

    # =========================================
    # Build authoritative evidence map
    # =========================================

    evidence_map = {
        item.source_id: item
        for item in evidence
    }

    # =========================================
    # Helper
    # =========================================

    def normalize(text: str) -> str:
        """
        Normalize text for metadata consistency comparison.

        This is NOT semantic normalization.
        It only removes harmless formatting differences such as:
        - Unicode normalization differences
        - Curly quotes vs straight quotes
        - Different dash characters
        - Non-breaking spaces
        - Repeated whitespace
        - Case differences
        """

        import re
        import unicodedata

        if not text:
            return ""

        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Normalize punctuation that commonly differs
        # between source data and LLM output.
        replacements = {
            "’": "'",
            "‘": "'",
            "“": '"',
            "”": '"',
            "–": "-",
            "—": "-",
            "−": "-",
            "\u00A0": " ",  # non-breaking space
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip().lower()

    # =========================================
    # GitHub signals
    # =========================================

    for signal in report.github_signals:

        source_id = signal.source_id

        if source_id not in evidence_map:

            raise ValueError(
                "GitHub signal references unknown "
                f"evidence: {source_id}"
            )

        authoritative = evidence_map[source_id]

        expected_project = normalize(
            authoritative.title
        )

        actual_project = normalize(
            signal.project
        )

        if actual_project != expected_project:

            raise ValueError(
                "GitHub signal project does not match "
                f"authoritative evidence title: "
                f"source_id={source_id}, "
                f"expected={authoritative.title}, "
                f"actual={signal.project}"
            )

    # =========================================
    # News signals
    # =========================================

    for signal in report.news_signals:

        source_id = signal.source_id

        if source_id not in evidence_map:

            raise ValueError(
                "News signal references unknown "
                f"evidence: {source_id}"
            )

        authoritative = evidence_map[source_id]

        expected_title = normalize(
            authoritative.title
        )

        actual_title = normalize(
            signal.title
        )

        if actual_title != expected_title:

            raise ValueError(
                "News signal title does not match "
                f"authoritative evidence title: "
                f"source_id={source_id}, "
                f"expected={authoritative.title}, "
                f"actual={signal.title}"
            )

    return True
