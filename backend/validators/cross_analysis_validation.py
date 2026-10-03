def validate_cross_analysis_evidence(
    report,
    evidence,
    target_company: str,
):
    """
    Validate cross_analysis evidence relevance.

    Rules:
    1. All CrossAnalysis evidence_ids must exist.
    2. All Claim evidence_ids must exist.
    3. Claim evidence_ids must belong to enclosing CrossAnalysis.
    4. Direct evidence can support direct company facts.
    5. Competitive evidence cannot create arbitrary company facts.
    6. Industry evidence cannot create arbitrary company facts.
    7. Ecosystem GitHub evidence cannot create direct company facts.
    8. Ecosystem evidence may only support ecosystem / industry conclusions.
    """

    target = target_company.strip().lower()

    evidence_map = {
        item.source_id: item
        for item in evidence
    }

    news_relevance_map = {
        signal.get("source_id"): signal.get("relevance")
        for signal in report.get("news_signals", [])
        if signal.get("source_id")
    }

    github_relevance_map = {
        signal.get("source_id"): signal.get("relevance")
        for signal in report.get("github_signals", [])
        if signal.get("source_id")
    }

    def get_relevance(source_id: str):
        if source_id.startswith("news:"):
            return news_relevance_map.get(source_id)

        if source_id.startswith("github:"):
            return github_relevance_map.get(source_id)

        return None

    def classify_sources(source_ids):
        has_direct = False
        has_competitive = False
        has_industry = False
        has_ecosystem = False

        for source_id in source_ids:

            if source_id not in evidence_map:
                raise ValueError(
                    "Unknown source_id: "
                    f"{source_id}"
                )

            relevance = get_relevance(source_id)

            if relevance == "direct":
                has_direct = True

            elif relevance == "competitive":
                has_competitive = True

            elif relevance == "industry":
                has_industry = True

            elif relevance == "ecosystem":
                has_ecosystem = True

        return (
            has_direct,
            has_competitive,
            has_industry,
            has_ecosystem,
        )

    def is_ecosystem_conclusion(text: str):

        ecosystem_keywords = [
            "ecosystem",
            "industry",
            "market",
            "sector",
            "trend",
            "open source",
            "open-source",
            "developer ecosystem",
            "ai ecosystem",
            "technology ecosystem",
            "broader ecosystem",
            "surrounding ecosystem",

            "鐢熸€佺郴缁?",
            "琛屼笟",
            "甯傚満",
            "瓒嬪娍",
            "寮€婧?",
            "寮€婧愮ぞ鍖?",
            "AI鐢熸€佺郴缁?",
        ]

        text = (text or "").lower()

        return any(
            keyword in text
            for keyword in ecosystem_keywords
        )

    def is_competitive_conclusion(text: str):

        competitive_keywords = [
            # English
            "competition",
            "competitive",
            "competitor",
            "competes",
            "competing",
            "competitive pressure",
            "rival",
            "rivals",
            "rivalry",
            "criticism",
            "criticize",
            "criticized",
            "criticizes",
            "critique",
            "public criticism",
            "external criticism",
            "competitive landscape",

            # Chinese
            "竞争",
            "竞争压力",
            "竞争对手",
            "竞品",
            "竞争格局",
            "竞争态势",
            "竞争关系",
            "公开批评",
            "外部批评",
            "行业批评",
            "批评",
            "质疑",
            "指责",
            "争议",
        ]

        text = (text or "").lower()

        return any(
            keyword.lower() in text
            for keyword in competitive_keywords
        )

    def is_industry_conclusion(text: str):

        industry_keywords = [
            # English
            "industry",
            "market",
            "sector",
            "trend",
            "ai industry",
            "technology industry",
            "technology market",
            "market trend",
            "industry trend",
            "industry landscape",
            "regulatory environment",
            "regulatory landscape",
            "policy environment",
            "policy landscape",
            "regulatory pressure",
            "industry environment",
            "broader market",
            "broader industry",
            "broader sector",
            "industry-wide",
            "sector-wide",
            "market-wide",

            # Chinese
            "行业",
            "行业环境",
            "行业趋势",
            "行业格局",
            "行业发展",
            "市场",
            "市场环境",
            "市场趋势",
            "市场格局",
            "市场发展",
            "产业",
            "产业环境",
            "产业趋势",
            "产业格局",
            "技术行业",
            "技术市场",
            "监管环境",
            "监管趋势",
            "监管政策",
            "政策环境",
            "政策趋势",
            "政策影响",
            "行业监管",
            "市场监管",
        ]

        text = (text or "").lower()

        return any(
            keyword.lower() in text
            for keyword in industry_keywords
        )

    # =========================================
    # Validate CrossAnalysis
    # =========================================

    for analysis_index, analysis in enumerate(
        report.get("cross_analysis", [])
    ):

        relationship = analysis.get(
            "relationship",
            ""
        )

        evidence_text = analysis.get(
            "evidence",
            ""
        )

        analysis_text = " ".join([
            relationship,
            evidence_text,
        ])

        analysis_evidence_ids = analysis.get(
            "evidence_ids",
            []
        )

        # =====================================
        # CrossAnalysis evidence must exist
        # =====================================

        for source_id in analysis_evidence_ids:

            if source_id not in evidence_map:

                raise ValueError(
                    "Cross analysis references "
                    "unknown source_id: "
                    f"{source_id}"
                )

        # =====================================
        # Classify CrossAnalysis evidence
        # =====================================

        (
            has_direct,
            has_competitive,
            has_industry,
            has_ecosystem,
        ) = classify_sources(
            analysis_evidence_ids
        )

        # =====================================
        # CrossAnalysis-level validation
        # =====================================

        if (
            has_ecosystem
            and not has_direct
            and not is_ecosystem_conclusion(
                analysis_text
            )
        ):

            raise ValueError(
                "Cross analysis cannot use "
                "ecosystem GitHub evidence as a direct "
                f"company fact for {target_company}: "
                f"{analysis_evidence_ids}"
            )

        if (
            has_competitive
            and not has_direct
            and not is_competitive_conclusion(
                analysis_text
            )
        ):

            raise ValueError(
                "Cross analysis cannot use "
                "competitive evidence as a direct "
                f"company fact for {target_company}: "
                f"{analysis_evidence_ids}"
            )

        if (
            has_industry
            and not has_direct
            and not is_industry_conclusion(
                analysis_text
            )
        ):

            raise ValueError(
                "Cross analysis cannot use "
                "industry evidence as a direct "
                f"company fact for {target_company}: "
                f"{analysis_evidence_ids}"
            )

        # =====================================
        # Validate individual Claims
        # =====================================

        for claim_index, claim in enumerate(
            analysis.get("claims", [])
        ):

            claim_statement = claim.get(
                "statement",
                ""
            )

            claim_type = claim.get(
                "claim_type",
                ""
            )

            claim_evidence_ids = claim.get(
                "evidence_ids",
                []
            )

            # =================================
            # Claim must have evidence
            # =================================

            if not claim_evidence_ids:

                raise ValueError(
                    "Claim has no evidence_ids: "
                    f"cross_analysis:"
                    f"{analysis_index}.claims:"
                    f"{claim_index} - "
                    f"{claim_statement}"
                )

            # =================================
            # Claim evidence must exist
            # =================================

            for source_id in claim_evidence_ids:

                if source_id not in evidence_map:

                    raise ValueError(
                        "Claim references unknown "
                        f"source_id: {source_id}"
                    )

            # =================================
            # Claim evidence must belong to
            # enclosing CrossAnalysis
            # =================================

            if not set(
                claim_evidence_ids
            ).issubset(
                set(analysis_evidence_ids)
            ):

                raise ValueError(
                    "Claim references evidence "
                    "not present in enclosing "
                    f"cross_analysis: "
                    f"{claim_statement}"
                )

            # =================================
            # Classify Claim evidence
            # =================================

            (
                claim_has_direct,
                claim_has_competitive,
                claim_has_industry,
                claim_has_ecosystem,
            ) = classify_sources(
                claim_evidence_ids
            )

            # =================================
            # Ecosystem GitHub restriction
            # =================================

            if (
                claim_has_ecosystem
                and not claim_has_direct
            ):

                if claim_type in {
                    "fact",
                    "inference",
                    "causal",
                }:

                    if not is_ecosystem_conclusion(
                        claim_statement
                    ):

                        raise ValueError(
                            "Claim cannot use "
                            "ecosystem GitHub evidence "
                            "as a direct company conclusion "
                            f"for {target_company}: "
                            f"{claim_statement}"
                        )

            # =================================
            # Competitive restriction
            # =================================

            if (
                claim_has_competitive
                and not claim_has_direct
            ):

                if claim_type != "competitive":

                    if not is_competitive_conclusion(
                        claim_statement
                    ):

                        raise ValueError(
                            "Claim cannot use "
                            "competitive evidence as a "
                            f"direct company conclusion "
                            f"for {target_company}: "
                            f"{claim_statement}"
                        )

            # =================================
            # Industry restriction
            # =================================

            if (
                claim_has_industry
                and not claim_has_direct
            ):

                if claim_type != "industry":

                    if not is_industry_conclusion(
                        claim_statement
                    ):

                        raise ValueError(
                            "Claim cannot use "
                            "industry evidence as a "
                            f"direct company conclusion "
                            f"for {target_company}: "
                            f"{claim_statement}"
                        )

    return True