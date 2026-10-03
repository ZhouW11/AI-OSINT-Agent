
def validate_cross_analysis_causality(
    report,
    evidence,
    target_company: str,
):
    """
    Validate causal claims in cross_analysis.

    This validator focuses ONLY on causality.

    It does NOT validate:
    - source_id existence
    - source type
    - relevance
    - target-company ownership

    Those responsibilities belong to:
    - validate_source_ids()
    - validate_cross_analysis_evidence()
    - validate_target_consistency()

    Rules:

    1. A simple relationship is allowed.

    2. Competitive evidence can support:
       - competition
       - competitive pressure
       - competitive relationship

    3. Competitive evidence alone cannot support:
       - "because competitor X, target company did Y"
       - "target company changed Y because of competitor X"

    4. If a causal relationship is claimed,
       the evidence must contain enough direct evidence
       about the target company's action and the stated cause.
    """

    target = target_company.strip().lower()

    # =========================================
    # Build evidence map
    # =========================================

    evidence_map = {
        item.source_id: item
        for item in evidence
    }

    # =========================================
    # Causal keywords
    # =========================================

    causal_keywords = [
        # English
        "because",
        "due to",
        "due to the",
        "as a result of",
        "caused by",
        "driven by",
        "in response to",
        "because of",
        "prompted by",
        "resulted from",
        "led to",
        "therefore",
        "thus",

        # Chinese
        "因为",
        "由于",
        "因此",
        "导致",
        "促使",
        "源于",
        "受到",
        "回应",
        "为了应对",
        "因竞争",
        "受竞争压力",
    ]

    competitive_keywords = [
        # English
        "competition",
        "competitive",
        "competitor",
        "competes",
        "competing",
        "competitive pressure",

        # Chinese
        "竞争",
        "竞争对手",
        "竞争压力",
        "竞争关系",
    ]

    # =========================================
    # Helper
    # =========================================

    def contains_any(
        text: str,
        keywords: list[str],
    ) -> bool:

        text = (text or "").lower()

        return any(
            keyword.lower() in text
            for keyword in keywords
        )

    # =========================================
    # Validate cross_analysis
    # =========================================

    for item in report.get(
        "cross_analysis",
        []
    ):

        relationship = item.get(
            "relationship",
            ""
        )

        evidence_text = item.get(
            "evidence",
            ""
        )

        combined_text = " ".join([
            relationship,
            evidence_text,
        ])

        evidence_ids = item.get(
            "evidence_ids",
            []
        )

        if not evidence_ids:
            continue

        # =====================================
        # Detect causal claim
        # =====================================

        is_causal = contains_any(
            combined_text,
            causal_keywords
        )

        if not is_causal:
            continue

        # =====================================
        # Classify referenced evidence
        # =====================================

        evidence_roles = set()

        for source_id in evidence_ids:

            if source_id not in evidence_map:
                continue

            for signal in report.get(
                    "news_signals",
                    []
            ):

                if signal.get("source_id") != source_id:
                    continue

                relevance = (
                        signal.get("relevance")
                        or ""
                ).strip().lower()

                if relevance:
                    evidence_roles.add(relevance)

            for signal in report.get(
                    "github_signals",
                    []
            ):

                if signal.get("source_id") != source_id:
                    continue

                relevance = (
                        signal.get("relevance")
                        or ""
                ).strip().lower()

                if relevance:
                    evidence_roles.add(relevance)

        has_direct = "direct" in evidence_roles
        has_competitive = "competitive" in evidence_roles
        has_industry = "industry" in evidence_roles
        has_ecosystem = "ecosystem" in evidence_roles

        # =====================================
        # Rule 1
        #
        # Competitive evidence alone
        # cannot establish causality
        # =====================================

        print("\n[DEBUG CAUSALITY]")
        print(f"relationship = {relationship}")
        print(f"evidence_ids = {evidence_ids}")
        print(f"evidence_roles = {sorted(evidence_roles)}")
        print(f"has_direct = {has_direct}")
        print(f"has_competitive = {has_competitive}")
        print(f"has_industry = {has_industry}")
        print(f"has_ecosystem = {has_ecosystem}")
        print(f"is_causal = {is_causal}")

        if has_competitive and not has_direct:
            raise ValueError(
                "Cross analysis makes a causal claim "
                "about the target company using only "
                "competitive evidence: "
                f"{evidence_ids}"
            )

        # =====================================
        # Rule 2
        #
        # Industry evidence alone
        # cannot establish causality
        # =====================================

        if has_industry and not has_direct:

            raise ValueError(
                "Cross analysis makes a causal claim "
                "about the target company using only "
                "industry evidence: "
                f"{evidence_ids}"
            )

        # =====================================
        # Rule 3
        #
        # Competitive + direct evidence:
        #
        # allow only if the relationship is
        # explicitly framed as competition.
        #
        # Example:
        #
        # "OpenAI reduced prices in response
        # to competitive pressure."
        #
        # This still requires direct evidence
        # for OpenAI's action.
        # =====================================

        # =====================================
        # Rule 3
        #
        # Direct + competitive evidence
        #
        # Mere coexistence of:
        #
        #   direct evidence
        #   +
        #   competitive evidence
        #
        # is NOT enough to establish causality.
        #
        # Example:
        #
        # "OpenAI lowered prices because
        # Anthropic launched a new model."
        #
        # We know:
        #
        #   OpenAI lowered prices.
        #   Anthropic launched a model.
        #
        # But we do NOT know:
        #
        #   Anthropic caused OpenAI to lower prices.
        #
        # Therefore reject unless the evidence itself
        # explicitly supports the causal relationship.
        # =====================================

        if has_direct and has_competitive:

            # ---------------------------------
            # Look for explicit causal language
            # in the evidence itself.
            #
            # IMPORTANT:
            #
            # Do NOT use relationship/evidence text
            # generated by the LLM here.
            #
            # Only the actual source-backed signal
            # fields can establish causality.
            # ---------------------------------

            explicit_causal_support = False

            for source_id in evidence_ids:

                print(
                    "[DEBUG SOURCE]",
                    source_id
                )

                if source_id not in evidence_map:
                    print(
                        "[DEBUG SOURCE] NOT FOUND IN EVIDENCE MAP"
                    )

                    continue

                # -----------------------------
                # Check news signal
                # -----------------------------

                for signal in report.get(
                        "news_signals",
                        []
                ):

                    if signal.get(
                            "source_id"
                    ) != source_id:
                        continue

                    source_text = " ".join([
                        signal.get("title", ""),
                        signal.get("summary", ""),
                        signal.get("implication", ""),
                    ])

                    # The source itself must explicitly
                    # express causality.
                    if contains_any(
                            source_text,
                            causal_keywords
                    ):
                        explicit_causal_support = True

                # -----------------------------
                # Check github signal
                # -----------------------------

                for signal in report.get(
                        "github_signals",
                        []
                ):

                    if signal.get(
                            "source_id"
                    ) != source_id:
                        continue

                    source_text = " ".join([
                        signal.get("project", ""),
                        signal.get("technical_direction", ""),
                        signal.get("activity", ""),
                        signal.get("evidence", ""),
                    ])

                    if contains_any(
                            source_text,
                            causal_keywords
                    ):
                        explicit_causal_support = True

            # ---------------------------------
            # Reject unsupported causality
            # ---------------------------------

            if not explicit_causal_support:
                raise ValueError(
                    "Cross analysis makes an unsupported "
                    "causal claim involving competitive "
                    "evidence. Direct evidence confirms "
                    "the target-company action, but the "
                    "evidence does not establish that "
                    "the competitor caused that action: "
                    f"{evidence_ids}"
                )

        # =========================================
        # Validate individual Claims
        # =========================================

        for claim in item.get("claims", []):

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

            if not claim_evidence_ids:
                continue

            claim_evidence_items = []

            for source_id in claim_evidence_ids:

                if source_id not in evidence_map:
                    raise ValueError(
                        "Claim references unknown "
                        f"source_id: {source_id}"
                    )

                claim_evidence_items.append(
                    evidence_map[source_id]
                )

            claim_has_direct = False
            claim_has_competitive = False
            claim_has_industry = False
            claim_has_ecosystem = False

            for evidence_item in claim_evidence_items:

                source_id = evidence_item.source_id

                if source_id.startswith("news:"):

                    for signal in report.get(
                            "news_signals",
                            []
                    ):

                        if signal.get(
                                "source_id"
                        ) == source_id:

                            relevance = signal.get(
                                "relevance"
                            )

                            if relevance == "direct":
                                claim_has_direct = True

                            elif relevance == "competitive":
                                claim_has_competitive = True

                            elif relevance == "industry":
                                claim_has_industry = True

                elif source_id.startswith("github:"):

                    for signal in report.get(
                            "github_signals",
                            []
                    ):

                        if signal.get(
                                "source_id"
                        ) == source_id:

                            relevance = signal.get(
                                "relevance"
                            )

                            if relevance == "direct":
                                claim_has_direct = True

                            elif relevance == "ecosystem":
                                claim_has_ecosystem = True

            # =====================================
            # Ecosystem GitHub cannot support
            # direct company fact
            # =====================================

            if (
                    claim_has_ecosystem
                    and not claim_has_direct
                    and claim_type
                    in {"fact", "inference", "causal"}
            ):
                raise ValueError(
                    "Claim cannot use ecosystem "
                    "GitHub evidence as a direct "
                    f"company conclusion for "
                    f"{target_company}: "
                    f"{claim_statement}"
                )


    return True
