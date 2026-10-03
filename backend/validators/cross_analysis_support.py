from models.report import JobAnalysisReport

def validate_cross_analysis_evidence_support(
    report: JobAnalysisReport,
    evidence,
    target_company: str,
):
    """
    Validate whether cross_analysis evidence actually supports
    the type of conclusion being made.

    This validation works together with:

    1. validate_source_ids()
       -> source_id must exist

    2. validate_cross_analysis_evidence()
       -> relevance rules

    3. validate_cross_analysis_causality()
       -> causal relationship rules

    4. validate_target_consistency()
       -> target company consistency

    This function focuses specifically on:
    whether the referenced evidence is capable of supporting
    the kind of claim made in cross_analysis.
    """

    # =========================================
    # Build evidence map
    # =========================================

    evidence_map = {
        item.source_id: item
        for item in evidence
    }

    target = target_company.strip().lower()

    # =========================================
    # Helper
    # =========================================

    def get_relevance(source_id):

        item = evidence_map.get(source_id)

        if item is None:
            raise ValueError(
                f"Cross analysis references unknown evidence: "
                f"{source_id}"
            )

        return getattr(
            item,
            "relevance",
            None
        )

    # =========================================
    # Cross analysis
    # =========================================

    for analysis in report.cross_analysis:

        relationship = (
            analysis.relationship or ""
        ).lower()

        evidence_text = (
            analysis.evidence or ""
        ).lower()

        source_ids = analysis.evidence_ids

        if not source_ids:
            continue

        # =====================================
        # Collect evidence types
        # =====================================

        relevances = []

        for source_id in source_ids:

            relevance = get_relevance(
                source_id
            )

            if relevance is not None:
                relevances.append(
                    relevance
                )

        # =====================================
        # Determine claim type
        # =====================================

        causal_keywords = [
            "because",
            "caused",
            "cause",
            "causing",
            "led to",
            "resulted in",
            "driven by",
            "due to",
            "response to",
            "in response",
            "由于",
            "因为",
            "导致",
            "造成",
            "推动",
            "驱动",
            "原因",
            "因此",
        ]

        temporal_keywords = [
            "then",
            "after",
            "before",
            "followed by",
            "subsequently",
            "later",
            "随后",
            "之后",
            "此前",
            "之前",
        ]

        competitive_keywords = [
            "competition",
            "competitive",
            "competitor",
            "competes",
            "rival",
            "rivals",
            "competitive pressure",
            "竞争",
            "竞争对手",
            "竞争压力",
            "竞品",
        ]

        industry_keywords = [
            "industry",
            "market",
            "sector",
            "ecosystem",
            "industry trend",
            "行业",
            "市场",
            "生态",
            "趋势",
        ]

        is_causal = any(
            keyword in relationship
            or keyword in evidence_text
            for keyword in causal_keywords
        )

        is_temporal = any(
            keyword in relationship
            or keyword in evidence_text
            for keyword in temporal_keywords
        )

        is_competitive = any(
            keyword in relationship
            or keyword in evidence_text
            for keyword in competitive_keywords
        )

        is_industry = any(
            keyword in relationship
            or keyword in evidence_text
            for keyword in industry_keywords
        )

        if is_temporal:
            is_causal = True

        # =====================================
        # Rule 1
        # Industry evidence cannot support
        # a direct company fact
        # =====================================

        if "industry" in relevances:

            # If the claim explicitly presents
            # the target company as the actor,
            # industry evidence alone is insufficient.

            if target in evidence_text:

                if (
                    is_causal
                    or not is_industry
                ):

                    raise ValueError(
                        "Cross analysis uses industry "
                        "evidence to support a direct "
                        f"company claim about {target_company}: "
                        f"{source_ids}"
                    )

        # =====================================
        # Rule 2
        # Competitive evidence must remain
        # competitive
        # =====================================

        if "competitive" in relevances:

            if not is_competitive:

                raise ValueError(
                    "Cross analysis uses competitive "
                    "evidence without explicitly "
                    "establishing a competitive relationship: "
                    f"{source_ids}"
                )

        # =====================================
        # Rule 3
        # Competitive evidence cannot alone
        # establish a target-company causal fact
        # =====================================

        if (
            "competitive" in relevances
            and all(
                relevance == "competitive"
                for relevance in relevances
            )
        ):

            if target in evidence_text and is_causal:

                raise ValueError(
                    "Cross analysis makes a causal "
                    f"claim about {target_company} "
                    "using only competitive evidence: "
                    f"{source_ids}"
                )

        # =====================================
        # Rule 4
        # Causal claims require direct evidence
        # involving the target company
        # =====================================

        if is_causal:

            has_direct = (
                "direct" in relevances
            )

            if not has_direct:

                raise ValueError(
                    "Cross analysis makes a causal "
                    "claim without direct evidence "
                    f"about {target_company}: "
                    f"{source_ids}"
                )




        # =====================================
        # Rule 5
        # Competitive + direct evidence
        # is allowed only when the relationship
        # explicitly describes competition.
        # =====================================

        if (
            "direct" in relevances
            and "competitive" in relevances
        ):

            if not is_competitive:

                raise ValueError(
                    "Cross analysis combines direct "
                    "and competitive evidence without "
                    "explicitly establishing competition: "
                    f"{source_ids}"
                )

    return True

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
