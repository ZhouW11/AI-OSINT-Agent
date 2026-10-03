from models.report import JobAnalysisReport, Evidence


def validate_evidence_content_support(
    report: JobAnalysisReport,
    evidence: list[Evidence],
):
    """
    Validate that signal.evidence is reasonably supported
    by authoritative Evidence content.

    This is a lightweight lexical consistency check.
    It is NOT full semantic entailment.
    """

    evidence_map = {
        item.source_id: item
        for item in evidence
    }

    def normalize(text: str) -> str:
        return " ".join(
            (text or "")
            .lower()
            .strip()
            .split()
        )

    def tokens(text: str) -> set[str]:

        text = normalize(text)

        # Remove basic punctuation
        for char in [
            ",", ".", ":", ";",
            "!", "?", "(", ")", "[", "]",
            "{", "}", "\"", "'", "-",
        ]:
            text = text.replace(char, " ")

        words = text.split()

        # Ignore very short/common words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "on",
            "for",
            "is",
            "are",
            "was",
            "were",
            "that",
            "this",
            "with",
            "from",
            "by",
            "as",
            "it",
            "new",
            "newly",
            "recent",
            "recently",
            "very",
            "completely",
            "major",
            "important",
            "different",
            "latest",
        }

        return {
            word
            for word in words
            if len(word) > 2
            and word not in stop_words
        }

    def validate_signal(
            source_id: str,
            signal_evidence: str,
            signal_type: str,
            relevance: str | None = None,
    ):

        if source_id not in evidence_map:
            raise ValueError(
                f"{signal_type} signal references "
                f"unknown evidence: {source_id}"
            )

        authoritative = evidence_map[source_id]

        source_text = " ".join([
            authoritative.title or "",
            authoritative.description or "",
            authoritative.snippet or "",
        ])

        source_tokens = tokens(source_text)
        evidence_tokens = tokens(
            signal_evidence
        )

        if not evidence_tokens:
            raise ValueError(
                f"{signal_type} evidence is empty: "
                f"{source_id}"
            )

        overlap = (
            source_tokens
            & evidence_tokens
        )

        # -------------------------------------
        # Evidence coverage
        # -------------------------------------

        coverage = (
            len(overlap)
            / len(evidence_tokens)
        )

        # -------------------------------------
        # Minimum lexical support
        # -------------------------------------

        if signal_type == "GitHub" and relevance == "ecosystem":

            minimum_supported = (
                    len(overlap) >= 1
            )

            print(
                "[DEBUG ecosystem]",
                "source_id=", source_id,
                "relevance=", repr(relevance),
                "signal_type=", repr(signal_type),
                "overlap=", sorted(overlap),
                "coverage=", coverage,
                "minimum_supported=", minimum_supported,
            )

        else:

            if len(evidence_tokens) <= 4:
                minimum_supported = (
                        len(overlap) >= 2
                )

            elif len(evidence_tokens) <= 10:
                minimum_supported = (
                        len(overlap) >= 2
                        and coverage >= 0.40
                )

            else:
                minimum_supported = (
                        len(overlap) >= 3
                        and coverage >= 0.40
                )

        if not minimum_supported:
            print("\n========== DEBUG EVIDENCE SUPPORT FAILURE ==========")
            print("source_id       =", source_id)
            print("signal_type     =", signal_type)
            print("relevance       =", relevance)
            print("source_text     =", source_text)
            print("source_tokens   =", sorted(source_tokens))
            print("signal_evidence =", signal_evidence)
            print("evidence_tokens =", sorted(evidence_tokens))
            print("overlap         =", sorted(overlap))
            print("coverage        =", coverage)
            print("minimum_supported =", minimum_supported)
            print("====================================================\n")

            raise ValueError(
                f"{signal_type} evidence is not sufficiently "
                f"supported by authoritative source: "
                f"{source_id}; "
                f"coverage={coverage:.2f}; "
                f"overlap={sorted(overlap)}"
            )

    # =========================================
    # GitHub
    # =========================================

    for signal in report.github_signals:
        print("\n========== DEBUG GITHUB SIGNAL ==========")
        print("project   =", signal.project)
        print("source_id =", signal.source_id)
        print("relevance =", signal.relevance)
        print("evidence  =", signal.evidence)

        validate_signal(
            signal.source_id,
            signal.evidence,
            "GitHub",
            signal.relevance,
        )

    # =========================================
    # News
    # =========================================

    for signal in report.news_signals:

        validate_signal(
            signal.source_id,
            signal.evidence,
            "News",
        )

    return True