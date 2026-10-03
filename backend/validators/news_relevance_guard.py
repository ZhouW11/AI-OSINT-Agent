from __future__ import annotations

import re


def _normalize(text: str) -> str:
    if not text:
        return ""

    text = str(text).lower()
    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in patterns
    )


def _detect_expected_role(
    title: str,
    description: str,
    snippet: str,
    target_company: str,
) -> str | None:

    target = _normalize(target_company)

    if not target:
        return None

    source_text = " ".join(
        [
            title or "",
            description or "",
            snippet or "",
        ]
    )

    text = _normalize(source_text)

    target_pattern = re.escape(target)

    # ========================================================
    # Competitive role
    #
    # Only classify as competitive when the source explicitly
    # establishes rivalry / competition involving the target.
    # ========================================================

    competitive_patterns = (
        rf"(rivals?|competitors?|competition|competitive).{{0,120}}{target_pattern}",
        rf"{target_pattern}.{{0,120}}(rivals?|competitors?|competition|competitive)",
        rf"(ahead of|beat|beats|outpace|outpaces).{{0,120}}(rivals?|competitors?|{target_pattern})",
    )

    if _matches_any(
        text,
        competitive_patterns,
    ):
        return "competitive"

    # ========================================================
    # Industry / external-actor role
    #
    # These patterns require an external actor performing an
    # action, rather than merely mentioning a government system,
    # researcher, or agency somewhere in the article.
    # ========================================================

    industry_patterns = (
        r"\bwhite house\b.{0,140}\b("
        r"asked|asks|wants|seeks|required|requires|requested|"
        r"told|ordered|review|reviewed|testing|tested|test"
        r")\b",

        r"\b(?:u\.s\.|us|american)\s+government\b.{0,140}\b("
        r"asked|asks|wants|seeks|required|requires|requested|"
        r"told|ordered|review|reviewed|testing|tested|test|"
        r"restricted|restricts|blocked|blocks|approved|approves"
        r")\b",

        r"\bgovernment\b.{0,140}\b("
        r"reviewed|reviews|testing|tested|test|regulated|regulates|"
        r"required|requires|requested|restricted|restricts|"
        r"blocked|blocks|approved|approves"
        r")\b",

        r"\b(?:regulator|regulators)\b.{0,120}\b("
        r"reviewed|reviews|tested|testing|test|required|requires|"
        r"ordered|orders|blocked|blocks|approved|approves|"
        r"investigated|investigates"
        r")\b",

        r"\b(?:security researcher|security researchers|researcher|researchers)\b"
        r".{0,120}\b("
        r"tested|test|evaluated|evaluate|benchmarked|benchmark|"
        r"discovered|discover|found|find|disclosed|disclose"
        r")\b",

        r"\b(?:third-party|third party)\b.{0,120}\b("
        r"tested|test|evaluated|evaluate|audited|audit|"
        r"benchmarked|benchmark"
        r")\b",

        r"\b(?:auditor|auditors|audit)\b.{0,120}\b("
        r"tested|test|evaluated|evaluate|reviewed|review|"
        r"assessed|assessment"
        r")\b",

        r"\bred team\b.{0,120}\b("
        r"tested|test|evaluated|evaluate|attacked|attack|"
        r"assessed|assessment"
        r")\b",
    )

    if _matches_any(
        text,
        industry_patterns,
    ):
        return "industry"

    return None


def normalize_cross_analysis_scope(
    parsed_data: dict,
) -> bool:
    """
    Propagate authoritative evidence roles into CrossAnalysis.

    Rules:
    - industry-only evidence -> claim_type = industry
    - competitive-only evidence -> claim_type = competitive
    - ecosystem-only evidence -> claim_type = industry
    - industry-only CrossAnalysis -> explicitly frame as industry context
    - competitive-only CrossAnalysis -> explicitly frame as competitive context
    - mixed direct/non-direct evidence is not rewritten here
    """

    role_by_source = {}

    for signal in parsed_data.get(
        "news_signals",
        [],
    ):
        source_id = signal.get("source_id")

        if source_id:
            role_by_source[source_id] = (
                signal.get("relevance")
                or ""
            ).strip().lower()

    for signal in parsed_data.get(
        "github_signals",
        [],
    ):
        source_id = signal.get("source_id")

        if source_id:
            role_by_source[source_id] = (
                signal.get("relevance")
                or ""
            ).strip().lower()

    for index, analysis in enumerate(
        parsed_data.get(
            "cross_analysis",
            [],
        )
    ):
        evidence_ids = analysis.get(
            "evidence_ids",
            [],
        )

        roles = {
            role_by_source.get(source_id)
            for source_id in evidence_ids
            if role_by_source.get(source_id)
        }

        if not roles:
            continue

        # ----------------------------------------------------
        # Propagate role to individual Claims.
        # ----------------------------------------------------

        for claim in analysis.get(
            "claims",
            [],
        ):
            claim_evidence_ids = claim.get(
                "evidence_ids",
                [],
            )

            claim_roles = {
                role_by_source.get(source_id)
                for source_id in claim_evidence_ids
                if role_by_source.get(source_id)
            }

            if claim_roles == {"industry"}:
                current = claim.get("claim_type")

                if current != "industry":
                    print(
                        "[CROSS-ANALYSIS ROLE GUARD] "
                        f"analysis={index} "
                        f"claim_type {current} -> industry"
                    )

                    claim["claim_type"] = "industry"

            elif claim_roles == {"competitive"}:
                current = claim.get("claim_type")

                if current != "competitive":
                    print(
                        "[CROSS-ANALYSIS ROLE GUARD] "
                        f"analysis={index} "
                        f"claim_type {current} -> competitive"
                    )

                    claim["claim_type"] = "competitive"

            elif claim_roles == {"ecosystem"}:
                current = claim.get("claim_type")

                if current not in {
                    "industry",
                    "inference",
                }:
                    print(
                        "[CROSS-ANALYSIS ROLE GUARD] "
                        f"analysis={index} "
                        f"claim_type {current} -> industry"
                    )

                    claim["claim_type"] = "industry"

        # ----------------------------------------------------
        # Propagate role to CrossAnalysis relationship.
        # This gives existing validators an explicit semantic
        # context without weakening their rules.
        # ----------------------------------------------------

        if roles == {"industry"}:
            relationship = (
                analysis.get(
                    "relationship",
                    ""
                )
                or ""
            )

            normalized = _normalize(
                relationship
            )

            if "industry" not in normalized:
                analysis["relationship"] = (
                    "Industry policy and regulatory context: "
                    + relationship
                )

                print(
                    "[CROSS-ANALYSIS ROLE GUARD] "
                    f"analysis={index} "
                    "framed as industry context"
                )

        elif roles == {"competitive"}:
            relationship = (
                analysis.get(
                    "relationship",
                    ""
                )
                or ""
            )

            normalized = _normalize(
                relationship
            )

            if "competitive" not in normalized:
                analysis["relationship"] = (
                    "Competitive context: "
                    + relationship
                )

                print(
                    "[CROSS-ANALYSIS ROLE GUARD] "
                    f"analysis={index} "
                    "framed as competitive context"
                )

        elif roles == {"ecosystem"}:
            relationship = (
                analysis.get(
                    "relationship",
                    ""
                )
                or ""
            )

            normalized = _normalize(
                relationship
            )

            if (
                "ecosystem" not in normalized
                and "industry" not in normalized
            ):
                analysis["relationship"] = (
                    "Ecosystem and industry context: "
                    + relationship
                )

                print(
                    "[CROSS-ANALYSIS ROLE GUARD] "
                    f"analysis={index} "
                    "framed as ecosystem context"
                )

    return True


def normalize_news_relevance_scope(
    parsed_data: dict,
    evidence,
    target_company: str,
) -> bool:
    """
    Establish the authoritative News relevance -> Evidence role
    boundary before Pydantic and downstream business validation.

    Source-backed evidence metadata has priority over an LLM
    relevance label when an explicit external actor / competitive
    relationship is detectable.

    This function intentionally does NOT weaken validators.
    It normalizes parsed LLM output so downstream validators
    operate on the corrected evidence role.
    """

    evidence_map = {
        item.source_id: item
        for item in evidence
    }

    news_relevance = {}

    # ========================================================
    # 1. Normalize NewsSignal relevance
    # ========================================================

    for signal in parsed_data.get(
        "news_signals",
        [],
    ):
        source_id = signal.get("source_id")

        if not source_id:
            continue

        source = evidence_map.get(source_id)

        if source is None:
            continue

        expected_role = _detect_expected_role(
            title=source.title,
            description=source.description or "",
            snippet=source.snippet or "",
            target_company=target_company,
        )

        current_role = (
            signal.get("relevance")
            or ""
        ).strip().lower()

        if expected_role is not None:
            if current_role != expected_role:
                print(
                    "[NEWS ROLE GUARD] "
                    f"{source_id}: "
                    f"{current_role or 'missing'} "
                    f"-> {expected_role}"
                )

                signal["relevance"] = expected_role

        news_relevance[source_id] = (
            signal.get("relevance")
            or ""
        ).strip().lower()

    # ========================================================
    # 2. Rebuild company_overview evidence scope
    #
    # Only direct News / direct GitHub evidence survives.
    # ========================================================

    overview = parsed_data.get(
        "company_overview"
    )

    if isinstance(overview, dict):

        original_ids = list(
            overview.get(
                "evidence_ids",
                [],
            )
        )

        github_relevance = {
            signal.get("source_id"): (
                signal.get("relevance")
                or ""
            ).strip().lower()
            for signal in parsed_data.get(
                "github_signals",
                []
            )
            if signal.get("source_id")
        }

        allowed_ids = []

        for source_id in original_ids:

            if source_id.startswith("news:"):
                role = news_relevance.get(
                    source_id
                )
            elif source_id.startswith("github:"):
                role = github_relevance.get(
                    source_id
                )
            else:
                role = None

            if role == "direct":
                allowed_ids.append(source_id)
            else:
                print(
                    "[COMPANY OVERVIEW SCOPE GUARD] "
                    f"dropped non-direct evidence: {source_id} "
                    f"role={role or 'unknown'}"
                )

        overview["evidence_ids"] = allowed_ids

    normalize_cross_analysis_scope(
        parsed_data
    )

    return True
