import re
from difflib import SequenceMatcher
from validators.target_consistency import (
    get_evidence_relevance,
    validate_target_consistency,
)
from validators.source_validation import validate_source_ids
from validators.cross_analysis_validation import (
    validate_cross_analysis_evidence,
)
from validators.causality_validation import (
    validate_cross_analysis_causality,
)
from validators.cross_analysis_support import (
    validate_cross_analysis_evidence_support,
)
from validators.evidence_content_consistency import (
    validate_evidence_content_consistency,
)
from validators.evidence_content_support import (
    validate_evidence_content_support,
)
from validators.claim_validation import (
    validate_claims,
)
from validators.recommendation_evidence_support import (
    derive_recommendation_evidence_ids,
    validate_recommendation_evidence_support,
)
from validators.recommendation_claim_support import (
    validate_recommendation_claim_support,
)
from validators.top_recommendation_integrity import (
    validate_top_recommendation_integrity,
)
from validators.top_recommendation_result_constraints import (
    validate_top_recommendation_result_constraints,
)
from validators.claim_semantics import (
    validate_claim_semantics,
)
from validators.news_relevance_guard import (
    normalize_news_relevance_scope,
)


from models.report import (
    JobAnalysisReport,
    Evidence,
    NewsSignal,
    Claim, ClaimType, Recommendation, RecommendationCategory, TopRecommendation,
)
from openai import OpenAI
import os
import json
import copy

from dotenv import load_dotenv

from models.company import CompanyIntelligence
from models.github import GitHubRepository
from models.news import NewsArticle, NewsSource
from pydantic import ValidationError

load_dotenv()


client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url="https://api.deepseek.com"
)

def filter_company_data(
    data: CompanyIntelligence,
    target_company: str
) -> CompanyIntelligence:

    target = target_company.strip().lower()

    # =========================================
    # GitHub
    # =========================================

    # GitHub search_github() 已经负责初步搜索。
    # 这里暂时保留搜索结果，不再使用简单字符串
    # 判断把 repository 全部过滤掉。

    filtered_github = []

    for repo in data.github:

        repo_name = (
                repo.name
                or ""
        ).strip().lower()

        if "/" not in repo_name:
            continue

        owner = repo_name.split("/", 1)[0]

        if owner == target:
            filtered_github.append(repo)

    # =========================================
    # News
    # =========================================

    filtered_news = []

    for article in data.news:

        text = " ".join([
            article.title or "",
            article.description or "",
            article.content or "",
            article.url or "",
            article.source.name or "",
        ]).lower()

        if target in text:
            filtered_news.append(article)

    return CompanyIntelligence(
        github=filtered_github,
        news=filtered_news,
    )

def build_evidence(
    data: CompanyIntelligence
) -> list[Evidence]:

    evidence = []

    # =========================================
    # GitHub
    # =========================================

    for repo in data.github:

        evidence.append(
            Evidence(
                source_id=f"github:{repo.name}",
                source_type="github",
                title=repo.name,
                url=repo.url,
                description=repo.description,
                snippet=(
                    repo.description
                    or ""
                ),
            )
        )

    # =========================================
    # News
    # =========================================

    for article in data.news:

        evidence.append(
            Evidence(
                source_id=f"news:{article.id}",
                source_type="news",
                title=article.title,
                url=article.url,
                description=article.description,
                snippet=(
                    article.content[:1000]
                    if article.content
                    else article.description
                ),
            )
        )

    return evidence


def parse_llm_json_output(
    raw_content: str
) -> dict:
    """
    Parse and validate the raw LLM response
    as a JSON object.

    This function is intentionally limited to
    serialization-level parsing.

    Business validators remain outside this
    boundary.
    """

    if not raw_content:
        raise ValueError(
            "LLM returned empty response"
        )

    try:
        parsed_data = json.loads(
            raw_content
        )

    except json.JSONDecodeError as e:

        print(
            "\n========== INVALID JSON =========="
        )

        print(raw_content)

        print(
            "==================================\n"
        )

        raise ValueError(
            f"LLM returned invalid JSON: "
            f"line={e.lineno}, "
            f"column={e.colno}, "
            f"message={e.msg}"
        ) from e

    if not isinstance(parsed_data, dict):
        raise ValueError(
            "LLM returned valid JSON, "
            "but the root value is not a JSON object."
        )

    return parsed_data

def validate_and_finalize_report(
    parsed_data: dict,
    evidence,
    target_company: str,
    top_k: int = 5,
) -> JobAnalysisReport:
    """
    Final report validation and post-processing boundary.

    This function intentionally preserves the existing
    validation, scoring, recommendation, and Top-K logic.

    Its responsibility is to convert parsed LLM JSON into
    a fully validated JobAnalysisReport.

    Architecture boundary:

    parsed JSON
        ?
    source / relationship validation
        ?
    Pydantic schema validation
        ?
    evidence / claim validation
        ?
    recommendation validation
        ?
    confidence / priority calculation
        ?
    Top-K construction
        ?
    final consistency validation
    """

    normalize_news_relevance_scope(
        parsed_data,
        evidence,
        target_company
    )

    validate_source_ids(
        parsed_data,
        evidence
    )

    validate_cross_analysis_evidence(
        parsed_data,
        evidence,
        target_company
    )

    validate_cross_analysis_causality(
        parsed_data,
        evidence,
        target_company
    )

    parsed_data["evidence"] = [
        item.model_dump()
        for item in evidence
    ]

    try:
        result = JobAnalysisReport.model_validate(
            parsed_data
        )
    except Exception as e:
        raise ValueError(
            f"LLM JSON does not match "
            f"JobAnalysisReport schema: {e}"
        ) from e

    validate_evidence_content_consistency(
        result,
        evidence
    )

    validate_claims(
        result,
        evidence
    )

    validate_evidence_content_support(
        result,
        evidence
    )

    validate_recommendation_evidence_support(
        result,
        evidence
    )

    apply_confidence_scores(
        result,
        evidence
    )

    # =========================================
    # Recommendation deduplication
    # =========================================

    apply_recommendation_deduplication(
        result
    )

    # =========================================
    # Recommendation confidence
    # =========================================

    apply_recommendation_confidence_scores(
        result
    )

    # =========================================
    # Recommendation priority
    # =========================================

    apply_recommendation_priority(
        result
    )

    # =========================================
    # Top recommendations
    # =========================================

    result.top_recommendations = (
        build_top_recommendations(
            result,
            top_k=top_k
        )
    )

    print(
        "\n========== TOP RECOMMENDATIONS CHECK =========="
    )

    print(
        "count =",
        len(result.top_recommendations)
    )

    for index, item in enumerate(
            result.top_recommendations,
            start=1
    ):
        print(
            f"#{index}",
            item.name,
            "| category=",
            item.category.value,
            "| priority=",
            item.priority_score,
            "| level=",
            item.priority_level,
        )

    print(
        "===============================================\n"
    )

    validate_target_consistency(
        result,
        target_company
    )

    return result

def analyze(
    data: CompanyIntelligence,
    target_company: str
) -> JobAnalysisReport:

    # =========================================
    # Target company filtering
    # =========================================

    filtered_data = filter_company_data(
        data,
        target_company
    )

    # =========================================
    # Build authoritative evidence
    # ONLY from filtered data
    # =========================================

    evidence = build_evidence(
        filtered_data
    )

    evidence_json = json.dumps(
        [
            item.model_dump()
            for item in evidence
        ],
        ensure_ascii=False,
        indent=2
    )

    report_schema = JobAnalysisReport.model_json_schema()

    llm_output_schema = copy.deepcopy(
        report_schema
    )

    llm_output_schema["properties"].pop(
        "evidence",
        None
    )

    llm_output_schema["required"] = [
        field
        for field in llm_output_schema.get(
            "required",
            []
        )
        if field != "evidence"
    ]

    report_schema_json = json.dumps(
        llm_output_schema,
        ensure_ascii=False,
        separators=(",", ":")
    )
    prompt = f"""
    你是一名 AI 求职情报分析师。

    ========================
    目标公司
    ========================

    Target Company:
    {target_company}

    本次分析必须严格围绕：

    "{target_company}"

    展开。

    你不是在生成泛化的 AI 行业报告，
    而是在生成：

    "{target_company}"

    的公司情报与求职分析报告。

    ========================
    核心任务
    ========================

    根据提供的真实公开数据，
    分析目标公司 "{target_company}" 的：

    1. 公司业务与技术方向
    2. GitHub 技术信号
    3. 新闻信号
    4. 不同证据之间的关系
    5. 对求职者的价值
    6. 面试准备方向
    7. 结论可靠性

    只能使用下面提供的数据。

    如果数据不足以判断目标公司某个方面，
    必须明确说明：

    "现有公开数据不足以判断"

    禁止将其他公司或其他项目的情况，
    直接描述为目标公司 "{target_company}" 的事实。

    例如：

    如果目标公司是 OpenAI：

    可以说：

    "OpenAI 的 API 定价发生变化。"

    前提是输入数据中确实存在 OpenAI 相关证据。

    但是不能因为：

    Anthropic 有 IPO 新闻

    就直接说：

    "OpenAI 的估值正在上升。"

    除非输入数据明确支持。

    ========================
    目标公司相关性规则
    ========================

    对于每一条结论，都必须考虑：

    这条信息是否真的与：

    "{target_company}"

    有关？

    如果没有直接关系：

    不要把它写成目标公司的事实。

    如果只是行业背景：

    必须明确标记为：

    "行业背景"

    而不是：

    "公司事实"。

    如果数据不足：

    必须输出：

    "现有公开数据不足以判断"

    ========================

输入数据：

{filtered_data.model_dump_json(
    indent=2,
    by_alias=True
)}
---

可用 Evidence：

{evidence_json}

---

Evidence 使用规则：

NEWS SIGNAL TITLE — HARD CONSTRAINTS

1. NewsSignal.title MUST exactly match the title of the corresponding Evidence item.

2. DO NOT rewrite the news title.

3. DO NOT summarize the news title.

4. DO NOT shorten, truncate, or modify the news title.

5. DO NOT translate the news title into another language.

6. DO NOT remove the target company name or any other part of the original title.

7. NewsSignal.title MUST be copied verbatim from the corresponding Evidence.title.

8. The title must preserve the exact wording and punctuation of Evidence.title.

Example:

Evidence.title:
"OpenAI launches legal-focused AI platform, escalating race for law firm users"

NewsSignal.title MUST be:
"OpenAI launches legal-focused AI platform, escalating race for law firm users"

The following are INVALID:

"focused AI platform, escalating race for law firm users"

"OpenAI launches legal-focused AI platform"

A translated or paraphrased version of the title.

--------------------------------------------------

========================
NEWS SIGNAL EVIDENCE — HARD CONSTRAINTS
========================

NEWS SIGNAL EVIDENCE — HARD CONSTRAINTS

1. NewsSignal.evidence is NOT a summary field.

2. NewsSignal.evidence is a SOURCE-GROUNDED VERIFICATION FIELD.

3. The backend will automatically validate NewsSignal.evidence
   against the corresponding authoritative Evidence source.

4. Therefore, NewsSignal.evidence MUST contain factual statements
   that are directly supported by the corresponding source.

5. For English-language news sources,
   NewsSignal.evidence MUST be written in English.

6. DO NOT translate English source evidence into Chinese.

7. NewsSignal.evidence should reuse distinctive factual phrases
   from the original source whenever possible.

8. NewsSignal.evidence should be concise and extractive.

9. NewsSignal.evidence should describe concrete facts such as:
   - what the company did
   - what happened
   - what the company announced
   - what issue was disclosed
   - what policy or framework was announced

10. DO NOT put interpretation into NewsSignal.evidence.

11. DO NOT put career advice into NewsSignal.evidence.

12. DO NOT put job recommendations into NewsSignal.evidence.

13. DO NOT write generic statements such as:
    "This shows that..."
    "This suggests that..."
    "This indicates that..."

14. For English news, the evidence should contain
    multiple distinctive terms that can be found in the original source.

15. If the source says:
    "OpenAI shared several undisclosed incidents of its AI models
    misbehaving and announced a new framework for reporting
    model misalignment."

    a valid evidence field may be:

    "OpenAI shared several undisclosed incidents of its AI models
    misbehaving and announced a new framework for reporting model
    misalignment."

16. The evidence field MUST remain substantially in the
    original source language.

17. summary may be written for human readability,
    but evidence must remain source-grounded and verifiable.

18. When there is a conflict between natural-language readability
    and source-grounded verification, prioritize source-grounded
    verification for the evidence field.

IMPORTANT:

NewsSignal.evidence is NOT a summary field.

NewsSignal.evidence is a SOURCE-GROUNDED VERIFICATION FIELD.

The backend will automatically validate NewsSignal.evidence
against the original source article using lexical content-support
validation.

Therefore, evidence MUST preserve the language and factual wording
of the original source.

--------------------------------------------------
RULE 1 — SOURCE LANGUAGE
--------------------------------------------------

If the source article is written in English:

NewsSignal.evidence MUST be written in English.

DO NOT translate the evidence into Chinese.

If the source article is written in Chinese:

--------------------------------------------------
RULE 2 — DO NOT TRANSLATE
--------------------------------------------------

For English news:

NEVER output a Chinese translation in the evidence field.

WRONG:

"OpenAI 公开了未被披露的 AI 模型行为不当事件，并提出了新的披露框架。"

CORRECT:

"OpenAI shared several undisclosed incidents of its AI models
misbehaving and announced a new framework for reporting model
misalignment."

--------------------------------------------------
RULE 3 — REUSE SOURCE WORDING
--------------------------------------------------

For English news, evidence should reuse distinctive factual phrases
from the original article whenever possible.

Prefer:

"OpenAI shared several undisclosed incidents of its AI models
misbehaving and announced a new framework for reporting model
misalignment."

over:

"OpenAI discussed AI safety issues."

The second version is too generic and may not be sufficiently
verifiable against the source.

--------------------------------------------------
RULE 4 — EVIDENCE MUST BE EXTRACTIVE
--------------------------------------------------

For English news, NewsSignal.evidence should preferably be:

1. an exact sentence from the source article, OR
2. two exact sentences from the source article, OR
3. a minimally edited combination of factual phrases that appear
   in the source article.

Do NOT translate.

Do NOT rewrite the entire sentence into another language.

Do NOT introduce new terminology that does not appear in the source.

--------------------------------------------------
RULE 5 — ONLY SOURCE-SUPPORTED FACTS
--------------------------------------------------

Evidence may describe only facts explicitly supported by the source.

Examples:

- an action taken by the company
- an event disclosed by the company
- an incident reported by the article
- an announced policy
- an announced framework
- a documented partnership
- a documented product or service

Do NOT put interpretation into evidence.

Do NOT put "this shows that..." into evidence.

Do NOT put "this suggests that..." into evidence.

Do NOT put career advice into evidence.

Do NOT put job recommendations into evidence.

Those belong in implication or recommendation fields.

--------------------------------------------------
RULE 6 — EVIDENCE VS SUMMARY
--------------------------------------------------

summary:
May be written in Chinese.
May summarize the article.

implication:
May be written in Chinese.
May explain the significance of the article.

evidence:
Must remain source-grounded and source-language-compatible.

Therefore:

summary ≠ evidence

implication ≠ evidence

Evidence must NOT be generated as a Chinese summary.

--------------------------------------------------
RULE 7 — SOURCE VERIFICATION
--------------------------------------------------

Before producing NewsSignal.evidence, verify mentally:

"Can I find the important factual words in this evidence
inside the original source article?"

If the answer is NO:

rewrite the evidence using the original source wording.

--------------------------------------------------
RULE 8 — WHEN SOURCE TEXT IS AVAILABLE
--------------------------------------------------

The original source article is included in the input data.

You MUST use the provided source article as the sole basis
for NewsSignal.evidence.

Do NOT rely on general knowledge.

Do NOT rely on memory.

Do NOT generate a translated evidence statement.

--------------------------------------------------
RULE 9 — REQUIRED BEHAVIOR FOR ENGLISH SOURCES
--------------------------------------------------

For an English source containing:

"OpenAI shared several undisclosed incidents of its AI models
misbehaving and announced a new framework for reporting model
misalignment."

The evidence MUST remain substantially in English.

Acceptable:

"OpenAI shared several undisclosed incidents of its AI models
misbehaving and announced a new framework for reporting model
misalignment."

Not acceptable:

"OpenAI 公开了多起此前未披露的 AI 模型行为异常事件，并宣布新的披露框架。"

--------------------------------------------------
FINAL RULE
--------------------------------------------------

When there is a conflict between producing natural Chinese evidence
and preserving source verifiability:

ALWAYS prioritize source verifiability.

For English articles:

evidence = English source-grounded factual text.

summary = Chinese summary.

implication = Chinese interpretation.

示例：

如果新闻正文包含：

"OpenAI shared several undisclosed incidents of its AI models
misbehaving and announced a new framework for reporting model
misalignment."

那么：

正确：
"OpenAI shared several undisclosed incidents of its AI models
misbehaving and announced a new framework for reporting model
misalignment."

不推荐：
"OpenAI 公布多起 AI 模型行为异常事件并制定披露计划。"

原因：
英文新闻使用中文翻译后的 evidence 会降低与原始 source text
之间的 lexical verifiability。

因此：
evidence = source-grounded factual expression
summary = concise summary
implication = interpretation

========================
Relevance 使用规则
========================

github_signals:

relevance 只能是：

"direct"

或：

"ecosystem"


direct：

目标公司的官方 GitHub 项目，
或者明确属于目标公司官方组织的项目。

ecosystem：

第三方项目，但输入数据明确表明
其与目标公司的 API、模型或开发者生态存在关系。

禁止：

不能因为一个项目属于 AI、Agent、RAG、
Prompt、LLM 等领域，
就认为它与目标公司有关。


news_signals:

relevance 只能是：

"direct"
"competitive"
"industry"


direct：

新闻直接报道目标公司。

competitive：

新闻报道竞争对手，
并且新闻内容明确涉及目标公司的竞争关系。

industry：

仅代表 AI 行业背景。

重要分类原则：

NewsSignal.relevance 必须根据新闻的 PRIMARY SUBJECT
和 PRIMARY FACTUAL ACTION 判断，而不是仅根据目标公司是否出现在标题或正文中。

direct：
只有当新闻主要描述目标公司自身的行为、公告、产品、任命、
运营、披露、事件或其他可确认事实时，才能标记为 direct。

competitive：
如果新闻主要描述竞争对手、竞争对手员工、外部批评、
竞争关系或外部主体对目标公司的评价，应标记为 competitive。

industry：
如果新闻主要描述整个 AI 行业、市场、监管、政策、
技术趋势或生态环境，应标记为 industry。

特别注意：

“新闻中出现目标公司名称”不等于“direct”。

如果新闻的主要事实行为主体是：
- competitor
- competitor employee
- analyst
- industry participant
- regulator
- government
- other external actor

则不得仅因为目标公司出现在标题或正文中而标记为 direct。

分类前必须依次判断：

1. 谁是主要事实行为主体？
2. 新闻主要描述什么行为或事件？
3. 该行为是否由目标公司自身实施？

只有第 3 项答案为“是”时，才允许标记为 direct。

Examples：

“OpenAI appoints a new worldwide sales chief” → direct

“Microsoft executive criticizes OpenAI's web scraping” → competitive

“New AI regulation affects major AI companies” → industry

competitive 和 industry
不得被描述为目标公司的直接事实。


特别重要：

如果目标公司是 OpenAI：

Anthropic 的新闻可以作为竞争背景，
但不得写成 OpenAI 自身发生的事实。

Microsoft、Dify、AutoGPT 等项目
不能因为属于 AI 生态，
就被描述成 OpenAI 的官方项目。
1. 只能引用提供的 source_id。
2. 禁止创建 source_id。
3. 禁止修改 source_id。
4. 禁止猜测 source_id。
5. 禁止使用 URL 代替 source_id。
6. 不要输出 Evidence 对象。
7. Evidence 由 Python 后端生成。
8. github source_id 格式为 github:<repository_name>。
9. news source_id 格式为 news:<article_id>。
10. github_signal 必须引用对应 GitHub source_id。
11. news_signal 必须引用对应 News source_id。
12. cross_analysis 的 evidence_ids 必须来自提供的 source_id。
13. company_overview 的 evidence_ids 必须来自提供的 source_id。
14. 如果没有足够证据支持，不要强行引用。
15. 每一个 github_signal 必须生成 evidence。
16. 每一个 news_signal 必须生成 evidence。

NewsSignal.evidence 必须是对对应 Evidence 的简洁、可验证、接近原文的事实性表达。
对于英文新闻，必须使用英文原文中的关键事实短语，不得仅做中文翻译。

允许：
- 摘要
- 改写
- 翻译

禁止：
- 添加 Evidence 中不存在的新事实
- 添加未经 Evidence 支持的数字
- 添加未经 Evidence 支持的因果关系
- 添加个人推测
---

第三方安全评估分类补充：

以下新闻不得因为目标公司名称出现而自动标记为 direct：

- 第三方安全研究人员对目标公司系统进行安全测试
- 第三方红队测试
- 第三方漏洞研究
- 外部安全机构对目标公司模型进行评估
- 外部研究人员披露目标公司系统漏洞
- 外部 benchmark 对目标公司模型进行安全评测

如果新闻的主要事实行为主体是：

- security researcher
- benchmark company
- external researcher
- red team
- cybersecurity team
- external auditor

而目标公司只是被测试、被评估、被披露的一方：

应优先标记为：

industry

而不是：

direct

也不是：

competitive。

原因：

该新闻证明的是外部安全环境、
第三方评估结果或行业安全测试活动，

而不是目标公司自身主动实施的事实。

例如：

“Researchers conduct an authorised security test against OpenAI systems”
→ industry

“Benchmark company evaluates OpenAI model safety”
→ industry

“Security researchers disclose a vulnerability affecting OpenAI systems”
→ industry

只有：

“OpenAI announces a new security program”
→ direct

“OpenAI deploys a new security control”
→ direct

========================
Claim Rules
========================

每一个 cross_analysis 必须尽量拆分成明确 Claim。

Claim 包含：

1. statement
2. claim_type
3. evidence_ids
4. confidence

claim_type 只能是：

"fact"
"competitive"
"inference"
"causal"
"industry"

规则：

1. fact：
   Evidence 直接支持的事实。

2. competitive：
   描述目标公司与竞争对手之间关系。

3. inference：
   基于多个 Evidence 得出的合理推断，
   但不是原始事实。

4. causal：
   明确表达因果关系。
   只有 Evidence 明确支持因果关系时才允许。

5. industry：
   行业背景，不得直接当作目标公司事实。

Claim Evidence Scope Rules

每个 Claim 的 statement 必须严格由该 Claim 自己的 evidence_ids 支持。

1. Claim 不得生成 Evidence 中不存在的新事实。
2. Claim 不得根据目标公司名称自行推断目标公司的行为。
3. 如果 Claim 使用 competitive evidence：
   - claim_type 必须为 "competitive"
   - statement 必须描述竞争关系、竞争行为、外部批评或竞争背景
   - 不得把 competitive evidence 改写成目标公司的直接事实
4. 如果 Claim 使用 industry evidence：
   - claim_type 必须为 "industry"
   - statement 必须描述行业、市场、政策、监管或技术环境
   - 不得把 industry evidence 改写成目标公司的直接事实
5. 如果 Claim 使用 ecosystem GitHub evidence：
   - 只能用于 ecosystem / industry / inference 类型的结论
   - 不得将其他项目的行为描述成目标公司的行为
6. 只有 direct evidence 才可以支持目标公司的直接事实。
7. 如果 Evidence 无法支持某个 Claim，不要生成该 Claim。

特别注意：

不要因为 Evidence 或新闻中出现目标公司名称，
就自动生成“目标公司做了某事”的 Claim。

例如：

Evidence 描述 Microsoft 高管批评 OpenAI，
正确的 Claim 应描述竞争关系或外部批评，
而不能生成：
“OpenAI 发布了某模型”
“OpenAI 采取了某行动”
“OpenAI 获得了某市场增长”

除非对应 direct Evidence 明确支持这些事实。

confidence 必须是 0 到 1 之间的小数。

禁止：

不能因为 confidence 较高，
就允许没有 Evidence 的结论。

confidence 不是 Evidence。

每个 Claim 必须至少引用一个有效 evidence_id。

========================
Company Overview Evidence Rules
========================

company_overview.evidence_ids
只能包含与目标公司直接相关的 evidence。

允许：
- direct News evidence
- direct GitHub evidence

禁止：
- competitive News evidence
- industry News evidence
- ecosystem GitHub evidence

即使 competitive 新闻正文中提到了目标公司，
只要该新闻的 relevance = "competitive"，
也绝对不能出现在：

company_overview.evidence_ids

例如：

Anthropic / Inherent 的新闻可以用于：
- news_signals
- cross_analysis

但不能用于：
- company_overview.evidence_ids

如果需要表达“OpenAI 面临来自 Anthropic/Inherent 的竞争”，
必须放在 competitive news signal 或 cross_analysis 中。

========================
Company Overview Evidence Rules
========================

company_overview 是目标公司的“直接事实概述”。

它包含两个相互独立的部分：

1. company_overview.summary
2. company_overview.evidence_ids

必须严格区分：

“summary 中提到某个事实”
≠
“该事实对应的证据可以加入 evidence_ids”。

company_overview.evidence_ids
只允许作为“目标公司直接事实”的证据集合。

========================
DIRECT EVIDENCE POOL
========================

生成 company_overview 前，必须先建立：

direct_evidence_pool

direct_evidence_pool 只能包含：

- github_signals.relevance = "direct"
- news_signals.relevance = "direct"

禁止将以下任何 evidence 加入 direct_evidence_pool：

- news_signals.relevance = "competitive"
- news_signals.relevance = "industry"
- github_signals.relevance = "ecosystem"

即使：

- Evidence 标题中出现目标公司名称
- Evidence 正文中出现目标公司名称
- Evidence 描述其他公司对目标公司的评价
- Evidence 讨论行业环境且顺带提及目标公司

也不能因此变成 direct evidence。

========================
COMPANY OVERVIEW EVIDENCE HARD CONSTRAINT
========================

company_overview.evidence_ids
必须满足：

对于每一个 evidence_id：

1. 它必须真实存在于提供的 Evidence 中；
2. 它必须能够在 github_signals 或 news_signals 中找到；
3. 对应 signal.relevance 必须严格等于 "direct"。

因此：

允许：

- direct GitHub evidence
- direct News evidence

禁止：

- competitive News evidence
- industry News evidence
- ecosystem GitHub evidence

绝对不要因为某条行业新闻、
竞争新闻、
监管新闻、
第三方评论、
竞争对手行为
能够帮助“解释背景”，
就把它加入 company_overview.evidence_ids。

“解释背景”
不是
“目标公司直接证据”。

========================
SUMMARY EVIDENCE BOUNDARY
========================

company_overview.summary
只能陈述能够由：

company_overview.evidence_ids

中的 direct evidence 支持的目标公司事实。

禁止根据以下信息推断目标公司事实：

- competitor
- competitor employee
- analyst
- regulator
- government
- industry participant
- industry trend
- industry policy
- competitive news
- ecosystem GitHub repository

特别禁止：

“新闻中提到了 OpenAI”
→
“所以可以作为 OpenAI 公司事实证据”

这是错误的。

只有当新闻的 PRIMARY SUBJECT
和 PRIMARY FACTUAL ACTION
确实是目标公司自身，并且对应：

news_signals.relevance = "direct"

时，才能用于 company_overview。

========================
INSUFFICIENT EVIDENCE RULE
========================

这是最重要的规则：

如果 direct_evidence_pool 为空，
或者现有 direct evidence 无法支持可靠的公司概述：

必须输出：

company_overview.evidence_ids = []

并在 summary 中明确说明：

“现有公开数据不足以支持更完整的目标公司直接事实概述。”

或其他等价的“不足以判断”表述。

此时：

禁止为了证明“信息不足”，
而引用 industry / competitive evidence。

特别注意：

“某条 industry evidence 说明信息不足”
不等于
“该 industry evidence 可以加入 company_overview.evidence_ids”。

如果没有 direct evidence：

summary 可以表达“不足以判断”，
但 evidence_ids 必须保持为空。

========================
NO EVIDENCE CONTAMINATION
========================

company_overview.evidence_ids
绝对不能出现：

competitive evidence
industry evidence
ecosystem evidence

即使这些 evidence：

- 在 news_signals 中出现；
- 在 cross_analysis 中出现；
- 被某个 Claim 引用；
- 在 reliability 中被讨论；
- 用来解释背景；
- 用来解释为什么证据不足。

不同模块的 evidence scope 必须严格隔离。

CrossAnalysis 使用过的 Evidence
不能因此自动成为
Company Overview 的 Evidence。

Claim 使用过的 Evidence
不能因此自动成为
Company Overview 的 Evidence。

========================
COMPANY OVERVIEW GENERATION PROCEDURE
========================

必须严格按照以下顺序：

Step 1：
读取 github_signals 和 news_signals。

Step 2：
读取每一个 signal 的 relevance。

Step 3：
建立 direct_evidence_pool。

只保留：

- relevance = "direct"

Step 4：
判断 direct_evidence_pool 是否为空。

如果为空：

company_overview.evidence_ids = []

summary 必须表达证据不足。

Step 5：
如果 direct_evidence_pool 不为空：

只能从 direct_evidence_pool
选择 company_overview.evidence_ids。

Step 6：
生成 summary。

summary 只能描述
company_overview.evidence_ids
真正支持的目标公司事实。

Step 7：
最终检查：

对于 company_overview.evidence_ids
中的每一个 source_id：

必须满足：

source_id 存在
AND
对应 signal.relevance = "direct"

如果任何一个不满足：

删除该 source_id。

如果删除后为空：

company_overview.evidence_ids = []

不要强行补充其他 evidence。

========================
FINAL COMPANY OVERVIEW CHECK
========================

输出 JSON 前必须检查：

1. company_overview.evidence_ids 不为空
   → 每个 evidence 都必须是 direct。

2. company_overview.evidence_ids 为空
   → summary 不得伪造目标公司事实。

3. 不允许使用 competitive evidence
   作为公司直接事实。

4. 不允许使用 industry evidence
   作为公司直接事实。

5. 不允许使用 ecosystem GitHub evidence
   作为公司直接事实。

6. 不允许因为“目标公司名称出现在 Evidence 中”
   就自动认定为 direct。

7. 不允许为了表达“证据不足”
   而把 industry / competitive evidence
   加入 evidence_ids。

8. 如果证据不足：
   宁可：

   evidence_ids = []

   也不要强行引用。

========================

========================
Recommendation Evidence Rules
========================

Job Value 和 Interview Preparation
不能凭空生成。

每一个 Recommendation 必须回答：

"为什么根据当前公开证据，
推荐这个方向 / 技能 / 面试主题？"

因此每一个 Recommendation 必须：

1. 至少引用一个 claim_ref
2. 至少引用一个 evidence_id
3. claim_ref 必须来自当前 cross_analysis 中真实存在的 Claim
4. evidence_id 必须来自当前 Evidence
5.Recommendation 的 evidence_ids
必须是所引用 Claims 的 evidence_ids 的子集。
6. rationale 必须解释 Claim 如何支持这个 Recommendation

禁止：

不能因为某个技能在 AI 行业常见，
就直接推荐。

例如：

"Python 是 AI 行业常见技能"

不能作为 OpenAI 求职推荐的唯一依据。

正确：

"OpenAI 相关技术新闻显示其关注模型评估和安全，因此模型评估能力是值得准备的方向。"

并引用对应 Claim。

========================
Recommendation Chain Rules
========================

每一个 Recommendation 必须形成完整链条：

Recommendation
    ↓
claim_ref
    ↓
Claim
    ↓
evidence_ids
    ↓
Evidence

因此：

1. claim_refs 中的每一个 Claim 必须真实存在。
2. Recommendation 的 evidence_ids 必须至少来自所引用 Claim 的 evidence_ids。
3. Recommendation 不允许引用与 claim_refs 无关的 evidence。
4. Recommendation 的 rationale 必须解释：
   为什么所引用 Claim 能够支持该 Recommendation。
5. 不允许只因为某个技能常见，
   就创建 Recommendation。
6. 如果没有足够的 Claim 支持：
   输出：
   "现有公开数据不足以判断"
   或不生成该 Recommendation。

========================
Recommendation HARD Constraint
========================

Recommendation 是严格的 Evidence-grounded 输出。

这是强制结构约束，不是建议。

任何 Recommendation 如果没有至少一个有效 claim_ref，
都属于非法输出。

因此：

claim_refs 绝对不能为空。

非法示例：

{{
  "name": "Python编程",
  "rationale": "Python 是 AI 行业常用语言，因此值得掌握。",
  "claim_refs": [],
  "evidence_ids": [],
  "confidence": 0.8
}}

非法示例：

{{
  "name": "Python编程",
  "rationale": "Python 对 AI 求职非常重要。",
  "claim_refs": null,
  "evidence_ids": [],
  "confidence": 0.8
}}

合法 Recommendation 必须满足：

Recommendation
    ↓
claim_refs
    ↓
真实存在的 Claim
    ↓
Claim.evidence_ids
    ↓
真实存在的 Evidence

生成每一个 Recommendation 时，
必须严格按照以下步骤：

Step 1：
从当前 cross_analysis 中选择一个真实存在的 Claim。

Step 2：
生成该 Claim 对应的合法 claim_ref。

claim_ref 格式必须严格为：

cross_analysis:<analysis_index>.claims:<claim_index>

Step 3：
读取该 Claim 的 evidence_ids。

Step 4：
Recommendation.claim_refs 必须引用该 Claim。

Step 5：
Recommendation.evidence_ids 必须来自该 Claim 的 evidence_ids。

Step 6：
rationale 必须明确解释：
该 Claim 为什么支持这个 Recommendation。

如果无法完成以上步骤：

不要生成 Recommendation。

========================
禁止通用知识生成 Recommendation
========================

禁止根据模型自身知识、行业常识、
训练数据、经验或常见求职要求
创建 Recommendation。

特别禁止以下逻辑：

"Python 是 AI 常用语言"
→ "推荐 Python"

"Git 是程序员常用工具"
→ "推荐 Git"

"机器学习很重要"
→ "推荐机器学习"

"Linux 对后端开发很重要"
→ "推荐 Linux"

"算法是程序员面试常见内容"
→ "推荐算法"

这些内容即使客观上正确，
如果当前 Evidence 和 Claim 没有明确支持，
也不得生成 Recommendation。

Recommendation 必须来自：

当前 Evidence
    ↓
当前 Claim
    ↓
Recommendation

而不是：

模型常识
    ↓
Recommendation

如果当前 Evidence 无法支持某个求职方向、
技术技能、重要领域、面试主题或实践任务：

不要为了填充列表而生成 Recommendation。

宁可少生成 Recommendation，
也不能生成没有 Claim 支持的 Recommendation。

========================
Recommendation 完整性检查
========================

在输出 JSON 前，
必须逐个检查所有：

job_directions
technical_skills
important_areas
interview_preparation.topics
interview_preparation.practical_tasks

对于每一个 Recommendation：

1. claim_refs 不得为空。
2. claim_refs 中每一个 claim_ref 必须真实存在。
3. claim_ref 对应的 Claim 必须存在。
4. Claim 必须至少包含一个 evidence_id。
5. Recommendation.evidence_ids 必须来自对应 Claim 的 evidence_ids。
6. rationale 必须由对应 Claim 支持。
7. 如果上述任意条件无法满足：
   删除该 Recommendation。

禁止输出一个仅仅因为"通常有价值"
而存在的 Recommendation。

========================
JSON Serialization HARD Rules

Additional JSON Contract Rules

The response must contain only valid JSON.

Do not use invalid escape sequences.

Do not invent fields that are not present in the authoritative LLM output schema.

Do not generate the backend evidence field.

Do not generate backend-only fields.

Before returning the response, verify that all field names follow the authoritative schema and that the JSON is syntactically valid.

========================

最终输出必须是一个严格合法的 JSON object。

必须遵守：

1. 只能输出 JSON object。
2. 不得输出 Markdown。
3. 不得输出 `json 或 `。
4. 不得输出 JSON 之外的解释文字。
5. 所有 JSON 字符串必须保持在合法 JSON 字符串范围内。
6. JSON 字符串中禁止出现真实的换行符。
7. 如果字符串需要表达换行，必须使用转义形式：\n
8. JSON 字符串中的双引号必须正确转义为：\"
9. JSON 字符串中的反斜杠必须正确转义。
10. 禁止 trailing comma。
11. 禁止 JSON comments。
12. 所有字段必须使用双引号。
13. 输出前必须检查整个内容能够被标准 JSON parser 直接解析。

特别注意：

以下结构非法：

"limitations": [
  "第一行内容
第二行内容"
]

必须改为：

"limitations": [
  "第一行内容\n第二行内容"
]

即：

JSON 字符串内部不得出现真实的 CR 或 LF 字符。

如果某个 summary、rationale、evidence、implication、
relationship、statement、supported_conclusions、
uncertain_conclusions 或 limitations 内容较长：

可以缩短内容，

但绝不能输出非法 JSON。

最终结果必须能够直接执行：

json.loads(raw_content)

而不发生 JSONDecodeError。

========================

========================
Recommendation Topic Diversity Rules
========================

V2.8-2 的目标是：

不仅保证 Recommendation 的 Category Diversity，
还必须尽可能保证 Recommendation 的 Topic Diversity、
Claim Diversity 和 Evidence Diversity。

注意：

Category 不同
不等于
Recommendation 一定不同。

禁止仅仅改变 Category，
却把同一个主题、同一个 Claim 或同一个事实
重复包装成不同 Recommendation。

========================
Topic Cluster Identification
========================

生成 Recommendation 之前，
先根据当前已经存在的：

1. CrossAnalysis.relationship
2. Claim.statement
3. Claim.evidence_ids
4. Claim 所属 CrossAnalysis
5. Evidence 所描述的主要事实

识别当前数据中存在的独立主题集群。

一个主题集群应当表示：

一组围绕同一个核心事实、
事件、技术问题、竞争关系、
行业问题或实践问题的 Claims / Evidence。

不同主题集群必须尽量保持语义独立。

例如：

主题 A：
模型安全评估

主题 B：
系统漏洞与访问控制

主题 C：
AI 反垄断与监管

这三个主题属于不同 cluster。

========================
Recommendation Topic Allocation
========================

如果当前存在多个独立主题集群：

应尽量让不同 Category
覆盖不同主题集群。

优先：

主题 A → job_direction
主题 B → technical_skill
主题 C → important_area
主题 A 或 B → interview_topic
主题 B 或 C → practical_task

而不是：

主题 A
→ job_direction
→ technical_skill
→ important_area
→ interview_topic
→ practical_task

特别注意：

如果存在多个具有独立证据支持的 Claim：

不要让所有 Recommendation
都依赖同一个 Claim。

应优先选择：

不同 Claim
或
不同 CrossAnalysis
或
不同 Evidence cluster

前提是这些 Claim / Evidence
与 Recommendation 存在真实语义关系。

========================
Category Semantic Diversity
========================

不同 Category 必须体现不同的工作语义：

1. job_direction

关注：

岗位方向、
职业方向、
工作职责、
适合发展的岗位领域。

不要只是把 technical_skill
改写成一个岗位名称。

2. technical_skill

关注：

具体技术能力、
分析方法、
工程方法、
评估方法、
实现能力。

不要只是重复 job_direction。

3. important_area

关注：

重要技术领域、
风险领域、
行业议题、
治理问题、
监管问题、
值得持续关注的主题。

不要只是重复 technical_skill。

4. interview_topic

关注：

面试中可以深入讨论、
解释、比较或分析的知识主题。

不要只是把 job_direction
或 technical_skill 原名称重复一遍。

5. practical_task

关注：

可以实际完成、
设计、
实现、
测试、
验证或展示的任务。

不要只是把 interview_topic
换成“实践”两个字。

========================
Claim Diversity Rule
========================

当存在多个独立 Claims 时：

应尽量让不同 Recommendation
引用不同的 primary Claim。

允许多个 Recommendation
共享一个 Claim，

但只有在：

1. 当前没有其他相关 Claim；
2. 该 Claim 对该 Category 确实具有真实支持；
3. Recommendation 的工作语义明显不同；

时才允许这样做。

禁止：

同一个 Claim
+
仅仅改变 Recommendation.name
+
生成多个 Category 的 Recommendation。

========================
Evidence Diversity Rule
========================

当存在多个独立 Evidence cluster 时：

尽量避免所有 Recommendation
都引用完全相同的 evidence_ids。

优先使用：

不同 Evidence
或
不同 Claim
或
不同 CrossAnalysis

支持不同 Recommendation。

但是：

Evidence-grounded 优先级
高于 diversity。

绝对禁止为了制造主题多样性：

- 使用无关 Evidence；
- 创建不存在的 Claim；
- 修改 Claim 含义；
- 扩大 Evidence 支持范围；
- 根据模型常识补充 Recommendation；
- 将 industry / competitive evidence
  改写成 target company 的 direct fact。

如果某个 Recommendation
无法由真实 Claim 和 Evidence 支持：

不要生成。

========================
Diversity Threshold
========================

如果当前数据只有一个真正独立的主题集群：

允许 Recommendation 存在较高主题重合。

如果当前至少存在两个独立主题集群：

Recommendation 应尽可能覆盖至少两个主题集群。

如果当前至少存在三个独立主题集群：

Recommendation 应尽可能覆盖至少三个主题集群。

不要为了满足 Diversity
强行制造不存在的主题。

========================
Bad Example
========================

如果当前 Claims 主要都是：

Claim A：
OpenAI 模型安全评估结果

Claim B：
OpenAI 系统安全测试结果

禁止直接生成：

job_direction：
AI 安全评估

technical_skill：
AI 安全评估方法

important_area：
AI 模型安全

interview_topic：
AI 安全评估

practical_task：
设计 AI 安全评估方案

然后让五条 Recommendation
主要全部由 Claim A 支持。

虽然 Category 不同，
但 Topic 实际高度重复。

========================
Better Example
========================

如果当前 Claims 同时支持：

Claim A：
模型安全评估

Claim B：
系统漏洞与访问控制

Claim C：
AI 行业反垄断与监管

则可以形成：

job_direction：
AI 安全工程

technical_skill：
漏洞分析与访问控制

important_area：
AI 监管与反垄断合规

interview_topic：
模型安全评估方法

practical_task：
设计一次模型安全评估实验

这样：

不同 Category
具有不同工作语义，

同时整体覆盖：

Claim A
Claim B
Claim C

三个独立主题。

========================
FINAL DIVERSITY CHECK
========================

在输出 JSON 前必须检查：

1. 是否只是把同一个主题
   机械复制到多个 Category？

2. 是否存在多个独立 Claim，
   但所有 Recommendation
   都使用同一个 Claim？

3. 是否存在多个独立 Evidence cluster，
   但所有 Recommendation
   都集中在同一个 Evidence cluster？

4. 不同 Category 的 Recommendation
   是否真的具有不同的工作语义？

5. Recommendation.name
   是否只是对另一条 Recommendation
   进行同义改写？

6. rationale 是否真正解释了
   该 Recommendation 为什么由
   对应 Claim 支持？

7. 是否为了制造 Diversity
   使用了无关 Claim 或 Evidence？

如果：

Evidence-grounded
与
Recommendation Diversity

发生冲突：

优先保证 Evidence-grounded。

宁可 Recommendation 较少，
也不要制造虚假的主题多样性。
========================
Recommendation Priority Rules
========================

priority_score 和 priority_level
由 Python 后端计算。

LLM 不负责决定最终优先级。

如果 JSON 中包含：

priority_score
priority_level

这些字段中的值会被 Python 后端覆盖。

求职推荐优先级必须基于：

1. Recommendation confidence
2. Claim confidence
3. Evidence support
4. Target-company relevance

Priority 的含义：

P1：
最值得优先准备。

P2：
重要，但优先级低于 P1。

P3：
值得关注，但不是当前最重要方向。

========================
Claim Reference Format
========================

claim_ref 必须严格使用：

cross_analysis:<analysis_index>.claims:<claim_index>

例如：

cross_analysis:0.claims:0

表示：

cross_analysis[0].claims[0]

禁止创建不存在的 claim_ref。

你必须严格输出一个 JSON object。

========================
Cross Analysis Claim Evidence Rules
========================

========================
Cross Analysis Claim Evidence Rules
===================================

CROSS-ANALYSIS EVIDENCE RULES

1. CrossAnalysis.evidence_ids must contain only evidence that directly supports the stated relationship about the target company.

2. If an evidence item is classified as industry context rather than a direct fact about the target company, DO NOT put that evidence_id into CrossAnalysis.evidence_ids when the relationship is presented as a company-level fact.

3. Industry evidence may be used only by an "industry" Claim to describe the broader industry or policy environment. It must not be used as direct evidence that the target company performed, decided, announced, or caused something.

4. If a CrossAnalysis contains both company-specific evidence and industry context, CrossAnalysis.evidence_ids must contain only the company-specific evidence. The industry evidence may appear only inside the relevant industry Claim.

5. Never promote an industry-level statement into a company-level relationship merely because the target company or its executive is mentioned in the industry article.

6. If there is insufficient company-specific evidence for a CrossAnalysis, omit that CrossAnalysis rather than forcing a relationship.

7. An evidence item describing what another person, government, competitor, or industry actor said or did is not automatically evidence of the target company's action.

每一个 cross_analysis 都必须先确定自己的 evidence_ids。

然后，该 cross_analysis 下的每一个 Claim，
只能从该 cross_analysis.evidence_ids 中选择 evidence_id。

严格遵循以下生成顺序：

Step 1：

先确定：

cross_analysis.evidence_ids

该字段只能包含当前 CrossAnalysis 实际使用的 Evidence。

Step 2：

再创建 Claims。

Step 3：

每创建一个 Claim，
必须从当前 CrossAnalysis.evidence_ids
中选择 evidence_ids。

因此：

Claim.evidence_ids
必须是：

当前 CrossAnalysis.evidence_ids 的子集。

严格数学约束：

set(Claim.evidence_ids)
⊆
set(CrossAnalysis.evidence_ids)

例如：

合法：

cross_analysis.evidence_ids = [
"news:openai",
"news:apple"
]

Claim.evidence_ids = [
"news:openai"
]

合法：

cross_analysis.evidence_ids = [
"news:openai",
"news:apple"
]

Claim.evidence_ids = [
"news:openai",
"news:apple"
]

非法：

cross_analysis.evidence_ids = [
"news:openai"
]

Claim.evidence_ids = [
"news:apple"
]

如果 Claim 需要使用：

"news:apple"

那么必须首先把：

"news:apple"

加入当前：

cross_analysis.evidence_ids

禁止 Claim 直接引用当前 CrossAnalysis
未声明的 Evidence。

========================
Claim Creation Procedure
========================

创建每一个 Claim 时，
必须执行以下检查：

1. 找到支持该 Claim 的 Evidence。

2. 获取这些 Evidence 的 source_id。

3. 将这些 source_id 加入当前 cross_analysis.evidence_ids。

4. 再将相同的 source_id 写入 Claim.evidence_ids。

5. 检查：

   Claim.evidence_ids
   ⊆
   CrossAnalysis.evidence_ids

6. 如果无法满足上述条件，
   不生成该 Claim。

禁止先创建 Claim.evidence_ids，
然后再猜测 CrossAnalysis.evidence_ids。

========================
Critical Example
================

如果 Evidence 中存在：

news:openai-apple

其内容支持：

"OpenAI 与 Apple 之间存在商业秘密诉讼，
OpenAI 正请求法院驳回相关诉讼。"

那么如果生成：

{{
"relationship": "OpenAI 与 Apple 的法律纠纷",
"evidence_ids": [
"news:openai-apple"
],
"claims": [
{{
"statement": "OpenAI 与 Apple 之间存在商业秘密诉讼，OpenAI 正努力请求驳回。",
"claim_type": "fact",
"evidence_ids": [
"news:openai-apple"
],
"confidence": 0.8
}}
]
}}

这是合法结构。

但以下结构非法：

{{
"relationship": "...",
"evidence_ids": [
"news:openai"
],
"claims": [
{{
...
}}
]
}}

因为：

"news:openai-apple"

没有出现在当前：

cross_analysis.evidence_ids

中。

========================
No Evidence = No Claim
======================

如果没有 Evidence 能够支持某个 Claim：

不要创建该 Claim。

不要猜测 source_id。

不要创建新的 source_id。

不要使用 URL 代替 source_id。

不要引用其他 CrossAnalysis 的 Evidence。

不要因为某条 Evidence 在其他 CrossAnalysis
中存在，就自动将其用于当前 Claim。

如果无法建立：

Claim
↓
Evidence
↓
当前 CrossAnalysis.evidence_ids

完整链条，

则不要生成该 Claim。

========================
Cross Analysis Semantic Scope Rules
===================================

CROSS-ANALYSIS EVIDENCE SCOPE RULES

1. CrossAnalysis.evidence_ids must contain only evidence that directly supports the stated relationship.

2. Evidence must be classified according to what the source actually establishes:
   - direct: evidence about the target company's own actions, products, announcements, operations, or documented events.
   - industry: evidence about the broader AI, technology, regulatory, policy, or market environment.
   - competitive: evidence about competitors, competing companies, external criticism, or competitive positioning.

3. For a company-level factual CrossAnalysis relationship about the target company's own action, product, announcement, operation, decision, or documented event:
   - CrossAnalysis.evidence_ids MUST contain direct company evidence.
   - Industry evidence MUST NOT be used as direct evidence.
   - Competitive evidence MUST NOT be used as direct evidence.

4. A competitive article mentioning the target company does NOT automatically become direct company evidence.

5. If an article primarily reports what a competitor, competitor employee, analyst, industry participant, government, regulator, or other external actor said or did about the target company, classify the evidence according to what the source actually establishes. Do not rewrite the external actor's statement or action as a direct action by the target company.

6. Competitive evidence MAY be used in a CrossAnalysis when the CrossAnalysis.relationship itself describes:
   - a competitive relationship;
   - competitor activity;
   - external criticism;
   - competitive positioning; or
   - another explicitly competitive context.

7. If the CrossAnalysis.relationship is explicitly competitive:
   - competitive evidence MAY appear in CrossAnalysis.evidence_ids;
   - Claims using that evidence MUST use claim_type = "competitive";
   - the Claim statement must describe the competitive relationship, external criticism, competitor action, or competitive context supported by that evidence;
   - the evidence MUST NOT be rewritten as a direct action by the target company.

8. If the CrossAnalysis.relationship is explicitly industry-level:
   - industry evidence MAY appear in CrossAnalysis.evidence_ids;
   - Claims using that evidence MUST use claim_type = "industry";
   - the Claim statement must describe the broader industry, market, policy, regulatory, or technology environment;
   - the evidence MUST NOT be rewritten as a direct action by the target company.

9. If a CrossAnalysis contains both company-specific evidence and competitive or industry context, do not mix their roles:
   - company-level Claims must use direct company evidence;
   - competitive Claims may use competitive evidence;
   - industry Claims may use industry evidence.

10. Competitive or industry evidence MUST NOT be used to prove that the target company itself performed, decided, announced, supported, opposed, or caused something unless direct evidence explicitly establishes that fact.

11. If there is insufficient evidence for the specific CrossAnalysis.relationship, OMIT that CrossAnalysis instead of forcing a relationship.

12. Never promote an industry-level or competitive-level statement into a company-level relationship merely because the target company is mentioned in the article.

13. CrossAnalysis should describe a relationship that can actually be established by its cited evidence. Do not use CrossAnalysis merely as a container for unrelated external commentary.

??? Claim ????? CrossAnalysis.relationship
?????????

CrossAnalysis ??????????

???

?? Claim ??????????????
????? CrossAnalysis?

Claim ???????

1. ????????
2. ??? CrossAnalysis.relationship ???
3. ????? Evidence ???
4. Claim.evidence_ids ???? CrossAnalysis.evidence_ids?

???

relationship?

"OpenAI ???????? AI ????"

???

"OpenAI ?????? AI ??????????"

????

"OpenAI ? Apple ?????????"

???????? OpenAI ???
??????? CrossAnalysis relationship?

???? Claim ??????????

?????? CrossAnalysis?

?????? CrossAnalysis ???
??? Claim ????? CrossAnalysis?

========================
CrossAnalysis Integrity Rule
============================

============================

每个 CrossAnalysis 必须满足：

relationship
↓
evidence_ids
↓
claims
↓
claim evidence_ids

四者必须描述同一个逻辑关系。

不得出现：

relationship A
+
claim B
+
evidence C

其中 A、B、C 实际描述不同事件。


========================
AUTHORITATIVE OUTPUT SCHEMA
========================

下面的 JSON Schema
由 Pydantic JobAnalysisReport.model_json_schema()
自动生成。

这是本次分析输出结构的权威定义。

重要：

1. 如果手写 JSON 示例与下面 Schema 有任何差异：
   以 Pydantic Schema 为准。

2. 不得增加 Schema 未定义的字段。

3. 不得遗漏 Schema required 字段。

4. 所有 nested object
   必须满足对应 Schema。

5. Enum / Literal 约束必须严格遵守。

6. confidence / priority_score
   等数值范围必须严格遵守。

7. additionalProperties 约束必须严格遵守。

8. Evidence、Claim、Recommendation、
   TopRecommendation 等嵌套对象
   必须符合对应模型定义。

9. 不得因为手写示例较简单，
   而输出简化版对象。

Pydantic JSON Schema：

{report_schema_json}

---
JSON 必须符合以下结构：

{{
  "company_overview": {{
    "summary": "string",
    "evidence_ids": ["source_id"]
  }},

  "github_signals": [
    {{
      "project": "string",
      "language": "string or null",
      "technical_direction": "string",
      "activity": "string",
      "evidence": "string",
      "relevance": "direct | ecosystem",
      "source_id": "source_id"
    }}
  ],

  "news_signals": [
    {{
      "title": "string",
      "source": "string",
      "published_at": "string",
      "summary": "string",
      "implication": "string",
      "evidence": "string",
      "relevance": "direct | competitive | industry",
      "source_id": "source_id"
    }}
  ],

  "cross_analysis": [
  {{
    "relationship": "string",
    "evidence": "string",
    "evidence_ids": ["source_id"],

    "claims": [
      {{
        "statement": "string",
        "claim_type": "fact | competitive | inference | causal | industry",
        "evidence_ids": ["source_id"],
        "confidence": 0.0
      }}
    ]
  }}
],

  "job_value": {{
  "job_directions": [
    {{
      "name": "string",
      "rationale": "string",
      "claim_refs": [
        "cross_analysis:0.claims:0"
      ],
      "evidence_ids": [
        "source_id"
      ],
      "confidence": 0.0
    }}
  ],

  "technical_skills": [
    {{
      "name": "string",
      "rationale": "string",
      "claim_refs": [
        "cross_analysis:0.claims:0"
      ],
      "evidence_ids": [
        "source_id"
      ],
      "confidence": 0.0
    }}
  ],

  "important_areas": [
    {{
      "name": "string",
      "rationale": "string",
      "claim_refs": [
        "cross_analysis:0.claims:0"
      ],
      "evidence_ids": [
        "source_id"
      ],
      "confidence": 0.0
    }}
  ]
}},

  "interview_preparation": {{
  "topics": [
    {{
      "name": "string",
      "rationale": "string",
      "claim_refs": [
        "cross_analysis:0.claims:0"
      ],
      "evidence_ids": [
        "source_id"
      ],
      "confidence": 0.0
    }}
  ],

  "practical_tasks": [
    {{
      "name": "string",
      "rationale": "string",
      "claim_refs": [
        "cross_analysis:0.claims:0"
      ],
      "evidence_ids": [
        "source_id"
      ],
      "confidence": 0.0
    }}
  ]
}},

  "reliability": {{
    "supported_conclusions": ["string"],
    "uncertain_conclusions": ["string"],
    "limitations": ["string"]
  }}
}}

重要：

1. 只能输出 JSON。
2. 不要输出 Markdown。
3. 不要使用 ```json。
4. 不要在 JSON 前后添加解释。
5. 所有 JSON 字符串必须使用双引号。
6. JSON 必须合法，可以被 Python json.loads() 直接解析。
7. 不要输出额外字段。
8. 不要遗漏 required 字段。
9. source_id 必须来自提供的 Evidence。
---


GitHub source_id 示例：

"github:NousResearch/hermes-agent"

News source_id 示例：

"news:6a8801beeaa9b6b78d62323b471f17be"

========================
Cross Analysis Construction Procedure
========================

生成 cross_analysis 时必须严格按照以下顺序：

Step 1：
先确定当前 CrossAnalysis 使用哪些 Evidence。

将这些 Evidence 的 source_id
全部写入：

cross_analysis.evidence_ids

Step 2：
再生成当前 CrossAnalysis 下的 Claims。

每一个 Claim 的 evidence_ids
只能从当前 CrossAnalysis 已经存在的
evidence_ids 中选择。

Step 3：
禁止 Claim 引用任何没有出现在
当前 CrossAnalysis.evidence_ids
中的 source_id。

必须满足：

set(claim.evidence_ids)
<=
set(cross_analysis.evidence_ids)

Step 4：
在输出最终 JSON 之前，
必须逐个检查所有 CrossAnalysis：

对于每一个：

cross_analysis[i]

检查：

for claim in cross_analysis[i].claims:

    claim.evidence_ids
    必须全部存在于：
    cross_analysis[i].evidence_ids

如果不满足：

不要输出该 Claim。

正确做法是：

1. 将 Claim 所需要的 Evidence
   加入当前 CrossAnalysis.evidence_ids；

或者：

2. 删除该 Claim。

禁止输出不一致的结构。

========================
Cross Analysis Example
========================

合法：

{{
  "evidence_ids": [
    "news:openai-a",
    "news:openai-b"
  ],
  "claims": [
    {{
      "statement": "OpenAI announced a new product.",
      "claim_type": "fact",
      "evidence_ids": [
        "news:openai-a"
      ],
      "confidence": 0.95
    }},
    {{
      "statement": "OpenAI also changed its pricing.",
      "claim_type": "fact",
      "evidence_ids": [
        "news:openai-b"
      ],
      "confidence": 0.90
    }}
  ]
}}

合法，因为：

Claim 1 evidence_ids:
["news:openai-a"]

属于：

CrossAnalysis evidence_ids:
["news:openai-a", "news:openai-b"]

Claim 2 同样合法。

非法：

{{
  "evidence_ids": [
    "news:openai-a"
  ],
  "claims": [
    {{
      "statement": "OpenAI also changed its pricing.",
      "claim_type": "fact",
      "evidence_ids": [
        "news:openai-b"
      ],
      "confidence": 0.90
    }}
  ]
}}

这是非法结构。

因为：

"news:openai-b"

没有出现在当前 CrossAnalysis 的：

evidence_ids

中。

绝对不要输出这种结构。


========================
Ecosystem GitHub Claim Rules
========================

如果当前 CrossAnalysis 使用的 Evidence
包含：

github_signals.relevance = "ecosystem"

那么：

1. 不得创建描述目标公司自身行为的 fact Claim。

2. 不得声称目标公司：
   - owns
   - develops
   - uses
   - adopts
   - maintains
   - deploys
   - recently updated
   - internally uses

   这些 GitHub 项目。

3. 不得将 ecosystem GitHub evidence
   用于证明目标公司的：
   - technical focus
   - engineering activity
   - GitHub activity
   - internal technology stack
   - product roadmap
   - engineering priorities

4. ecosystem GitHub evidence
   只能支持：

   - industry
   - ecosystem
   - technology trend
   - open-source trend
   - developer ecosystem

5. 如果 CrossAnalysis 同时包含：

   direct company evidence
   +
   ecosystem GitHub evidence

   那么：

   ecosystem GitHub evidence
   只能用于描述外部生态背景，

   不得作为目标公司事实的直接证据。

6. 如果无法明确区分：

   target company fact

   和

   ecosystem background

   则删除该 Claim。


========================
FINAL CONSISTENCY CHECK
========================

输出 JSON 之前，必须执行以下内部检查：

1. 所有 source_id 必须来自提供的 Evidence。

2. 所有 github_signal.source_id
   必须存在于 GitHub Evidence。

3. 所有 news_signal.source_id
   必须存在于 News Evidence。

4. company_overview.evidence_ids
   必须全部是 direct evidence。

5. 对每一个 cross_analysis：

   cross_analysis.evidence_ids
   必须包含该 CrossAnalysis 所有 Claims
   所使用的 Evidence。

6. 对每一个 Claim：

   claim.evidence_ids
   必须是当前 CrossAnalysis.evidence_ids
   的子集。

7. 对每一个 Recommendation：

   claim_refs 必须真实存在。

8. Recommendation.evidence_ids
   必须来自对应 Claim 的 evidence_ids。

9. 不允许出现任何不存在的 source_id。

10. 不允许出现任何不存在的 claim_ref。

11. 不允许出现：
    Claim 引用了 CrossAnalysis 没有声明的 Evidence。

如果任何一项检查失败，
必须在输出 JSON 前自行修正。

不要输出未通过检查的 JSON。

### CROSS-ANALYSIS EVIDENCE ISOLATION RULE

You MUST distinguish between direct company evidence and ecosystem/competitive evidence.

#### Direct company evidence

Evidence may be used as a direct fact about the target company ONLY when:

* the evidence explicitly refers to the target company;
* the source is authoritative for that company;
* and the evidence is classified as `direct`.

Examples:

* `news:...` explicitly reporting an action by OpenAI
* an official OpenAI GitHub repository
* an official OpenAI announcement

#### Ecosystem / competitive GitHub evidence

GitHub repositories that are NOT owned by the target company MUST be treated as ecosystem evidence.

For example, if the target company is OpenAI:

* `github:NousResearch/hermes-agent`
* `github:Significant-Gravitas/AutoGPT`
* `github:microsoft/markitdown`
* `github:f/prompts.chat`
* `github:langgenius/dify`

are NOT OpenAI projects.

They MUST NOT be used as evidence for direct facts such as:

* "OpenAI develops AutoGPT"
* "OpenAI uses Dify"
* "OpenAI develops MarkItDown"
* "OpenAI's GitHub activity shows..."
* "OpenAI recently updated this repository"

These repositories may ONLY be used for ecosystem-level analysis.

#### Cross-analysis restriction

When generating `cross_analysis`:

1. A claim with `claim_type = "fact"` about the target company MUST NOT reference ecosystem/competitive GitHub evidence.

2. A claim describing the target company's own technical activity MUST NOT reference non-target GitHub repositories.

3. Ecosystem GitHub evidence may be used ONLY for claims explicitly describing:

   * industry trends;
   * surrounding ecosystem;
   * competing/open-source projects;
   * technologies commonly associated with the target company's industry;
   * possible external technical signals.

4. Such claims MUST clearly state that the evidence comes from the broader ecosystem, not from the target company itself.

5. Do NOT infer target-company adoption, development, ownership, usage, or internal priorities from ecosystem GitHub repositories.

6. If no direct evidence exists for a target-company technical fact, explicitly state that the provided data does not establish that fact.

#### Important

Never convert ecosystem evidence into direct company evidence.

For OpenAI, the following are valid:

* "The surrounding AI ecosystem shows strong activity in autonomous-agent frameworks."
* "Open-source projects such as AutoGPT and Dify indicate continued industry interest in agentic workflows."

The following are INVALID:

* "OpenAI is developing AutoGPT."
* "OpenAI uses Dify."
* "OpenAI's GitHub activity demonstrates its focus on agentic workflows."

When evidence is insufficient, prefer omission or an explicit uncertainty statement over unsupported inference.

========================
FINAL OUTPUT CONTRACT
========================

========================
Top Recommendation Output Rules
========================

每一个 top_recommendations 元素必须完整包含以下字段：

- name
- rationale
- category
- claim_refs
- evidence_ids
- confidence
- priority_score
- priority_level

其中：

priority_score:
- 必须是 0 到 1 之间的数字
- 必须由当前分析结果中的 Recommendation priority 逻辑确定
- 不得省略

priority_level:
- 必须是以下三个值之一：
  - "P1"
  - "P2"
  - "P3"
- 不得省略

禁止输出只有 confidence 而没有 priority_score 和 priority_level 的 TopRecommendation。

错误示例：

{{
  "name": "AI 安全与模型对齐",
  "rationale": "...",
  "category": "job_direction",
  "claim_refs": ["claim:1"],
  "evidence_ids": ["news:123"],
  "confidence": 0.8
}}

正确示例：

{{
  "name": "AI 安全与模型对齐",
  "rationale": "...",
  "category": "job_direction",
  "claim_refs": ["claim:1"],
  "evidence_ids": ["news:123"],
  "confidence": 0.8,
  "priority_score": 0.88,
  "priority_level": "P1"
}}


Your response MUST be a single valid JSON object.

Do NOT output:
- Markdown
- Markdown code fences
- ```json
- explanations before or after the JSON
- comments
- any text outside the JSON object

The JSON returned by the model MUST follow the authoritative LLM output schema provided above.

The model is responsible only for analytical result fields.

The model MUST NOT generate the evidence field.

Evidence is generated and injected by the Python backend after JSON parsing.

Do not add extra top-level fields.

All nested objects MUST also follow the defined schema.

Enum constraints:

ClaimType:
- fact
- competitive
- inference
- causal
- industry

RecommendationCategory:
- job_direction
- technical_skill
- important_area
- interview_topic
- practical_task

TopRecommendation.priority_level:
- P1
- P2
- P3

GithubSignal.relevance:
- direct
- ecosystem

confidence MUST be between 0 and 1.

priority_score MUST be between 0 and 1.

All evidence_ids MUST refer to actual Evidence source_id values provided in the input.

All claim_refs MUST refer to actual claims generated within the corresponding analysis.

Do not invent evidence IDs, claim references, GitHub projects, news facts, company actions, or relationships.

If sufficient evidence does not exist, use an empty list where appropriate or explicitly state uncertainty in the corresponding text field.

Return JSON only.

"""

    response = client.chat.completions.create(
        model="deepseek-chat",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        response_format={
            "type": "json_object"
        },

        temperature=0
    )

    raw_content = response.choices[0].message.content

    print("\n========== RAW LLM RESPONSE ==========")
    print(raw_content)
    print("=======================================\n")

    if not raw_content:
        raise ValueError("LLM returned empty response")

    parsed_data = parse_llm_json_output(
        raw_content
    )
    return validate_and_finalize_report(
        parsed_data=parsed_data,
        evidence=evidence,
        target_company=target_company,
        top_k=5,
    )


def test_source_validation():

    # -----------------------------------------
    # Fake authoritative evidence
    # -----------------------------------------

    evidence = [
        Evidence(
            source_id="github:NousResearch/hermes-agent",
            source_type="github",
            title="NousResearch/hermes-agent",
            url="https://github.com/NousResearch/hermes-agent",
        ),
        Evidence(
            source_id="news:123456",
            source_type="news",
            title="Test News",
            url="https://example.com/news",
        ),
    ]

    # =========================================
    # Test 1: GitHub 非法 source_id
    # =========================================

    fake_report_github = {
        "company_overview": {
            "summary": "test",
            "evidence_ids": [
                "github:not-real"
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },
    }

    try:

        validate_source_ids(
            fake_report_github,
            evidence
        )

        print(
            "❌ Test 1 FAILED: "
            "github:not-real 没有被拦截"
        )

    except ValueError as e:

        print(
            "✅ Test 1 PASSED: "
            "成功拦截非法 GitHub source_id"
        )

        print(
            f"   {e}"
        )

    # =========================================
    # Test 2: News 非法 source_id
    # =========================================

    fake_report_news = {
        "company_overview": {
            "summary": "test",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [
            {
                "title": "Fake News",
                "source": "Fake",
                "published_at": "2026-08-23",
                "summary": "test",
                "implication": "test",

                # 非法 source_id
                "source_id": "news:not-real"
            }
        ],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },
    }

    try:

        validate_source_ids(
            fake_report_news,
            evidence
        )

        print(
            "❌ Test 2 FAILED: "
            "news:not-real 没有被拦截"
        )

    except ValueError as e:

        print(
            "✅ Test 2 PASSED: "
            "成功拦截非法 News source_id"
        )

        print(
            f"   {e}"
        )

    # =========================================
    # Test 3: Cross analysis 非法 source_id
    # =========================================

    fake_report_cross = {
        "company_overview": {
            "summary": "test",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "test relationship",
                "evidence": "test evidence",

                # 非法 source_id
                "evidence_ids": [
                    "github:not-real"
                ]
            }
        ],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },
    }

    try:

        validate_source_ids(
            fake_report_cross,
            evidence
        )

        print(
            "❌ Test 3 FAILED: "
            "cross_analysis 非法 source_id 没有被拦截"
        )

    except ValueError as e:

        print(
            "✅ Test 3 PASSED: "
            "成功拦截 Cross Analysis 非法 source_id"
        )

        print(
            f"   {e}"
        )

    print(
        "\n🎉 Source validation tests completed."
    )

    # =========================================
    # Test 4: 合法 source_id
    # =========================================

    valid_report = {
        "company_overview": {
            "summary": "test",
            "evidence_ids": [
                "github:NousResearch/hermes-agent",
                "news:123456",
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },
    }

    try:

        validate_source_ids(
            valid_report,
            evidence
        )

        print(
            "✅ Test 4 PASSED: "
            "合法 source_id 正常通过"
        )

    except ValueError as e:

        print(
            "❌ Test 4 FAILED: "
            "合法 source_id 被错误拦截"
        )

        print(
            f"   {e}"
        )

        # Test 4: company_overview 非法 source_id

        fake_report = valid_report.model_copy(
            deep=True
        )

        fake_report.company_overview.evidence_ids = [
            "github:not-real"
        ]

        try:
            validate_source_ids(
                fake_report,
                evidence
            )

            raise AssertionError(
                "Test 4 FAILED: 没有拦截非法 company_overview source_id"
            )

        except ValueError as e:

            print(
                "✅ Test 4 PASSED: "
                "成功拦截 Company Overview 非法 source_id"
            )

            print(f"   {e}")

    # =========================================
    # Test 5: GitHub signal 错用 News source_id
    # =========================================

    fake_report_wrong_type = {
        "company_overview": {
            "summary": "test",
            "evidence_ids": []
        },

        "github_signals": [
            {
                "project": "fake",
                "language": "Python",
                "technical_direction": "test",
                "activity": "test",
                "evidence": "test",

                # 故意使用 News source_id
                "source_id": "news:123456"
            }
        ],

        "news_signals": [],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },
    }

    try:

        validate_source_ids(
            fake_report_wrong_type,
            evidence
        )

        print(
            "❌ Test 5 FAILED: "
            "GitHub signal 错用了 News source_id"
        )

    except ValueError as e:

        print(
            "✅ Test 5 PASSED: "
            "成功拦截 GitHub/News source_type 错配"
        )

        print(f"   {e}")



def test_github_relevance_validation():

    fake_report = {
        "company_overview": {
            "summary": "test",
            "evidence_ids": []
        },

        "github_signals": [
            {
                "project": "NousResearch/hermes-agent",
                "language": "Python",
                "technical_direction": "AI Agent",
                "activity": "active",
                "evidence": "test",

                # 故意放非法 relevance
                "relevance": "competitive",

                "source_id": "github:NousResearch/hermes-agent"
            }
        ],

        "news_signals": [],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    }

    try:

        JobAnalysisReport.model_validate(
            fake_report
        )

        print(
            "❌ Test 6 FAILED: "
            "非法 GitHub relevance 没有被拦截"
        )

    except Exception as e:

        print(
            "✅ Test 6 PASSED: "
            "成功拦截非法 GitHub relevance"
        )

        print(
            f"   {e}"
        )

def test_news_relevance_validation():

    fake_report = {
        "company_overview": {
            "summary": "test",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [
            {
                "title": "Test News",
                "source": "Test",
                "published_at": "2026-08-23",
                "summary": "test",
                "implication": "test",

                # News 不允许 ecosystem
                "relevance": "ecosystem",

                "source_id": "news:123456"
            }
        ],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    }

    try:

        JobAnalysisReport.model_validate(
            fake_report
        )

        print(
            "❌ Test 7 FAILED: "
            "非法 News relevance 没有被拦截"
        )

    except Exception as e:

        print(
            "✅ Test 7 PASSED: "
            "成功拦截非法 News relevance"
        )

        print(
            f"   {e}"
        )

def test_company_filter():

    data = CompanyIntelligence(

        github=[
            GitHubRepository(
                name="openai/openai-python",
                description="Official OpenAI Python library",
                stars=100,
                forks=10,
                language="Python",
                updated="2026-08-23",
                url="https://github.com/openai/openai-python",
            ),

            GitHubRepository(
                name="Significant-Gravitas/AutoGPT",
                description="An experimental open-source autonomous AI agent",
                stars=100,
                forks=10,
                language="Python",
                updated="2026-08-23",
                url="https://github.com/Significant-Gravitas/AutoGPT",
            ),
        ],

        news=[

            NewsArticle(
                source_id="news:openai-test",
                id="openai-test",
                title="OpenAI announces new API pricing",
                description="OpenAI changes API pricing.",
                content="OpenAI announced new pricing.",
                url="https://example.com/openai",
                publishedAt="2026-08-23",
                lang="en",
                source=NewsSource(
                    id="test",
                    name="Test",
                    url="https://example.com",
                ),
            ),

            NewsArticle(
                source_id="news:anthropic-test",
                id="anthropic-test",
                title="Anthropic announces new model",
                description="Anthropic released a new model.",
                content="Anthropic announced a new model.",
                url="https://example.com/anthropic",
                publishedAt="2026-08-23",
                lang="en",
                source=NewsSource(
                    id="test",
                    name="Test",
                    url="https://example.com",
                ),
            ),
        ],
    )

    filtered = filter_company_data(
        data,
        "OpenAI"
    )

    # GitHub
    assert len(filtered.github) == 1
    assert (
        filtered.github[0].name
        == "openai/openai-python"
    )

    # News
    assert len(filtered.news) == 1
    assert (
        filtered.news[0].id
        == "openai-test"
    )

    print(
        "✅ Test 8 PASSED: "
        "OpenAI 公司过滤成功"
    )

def test_valid_relevance():

    report = {
        "company_overview": {
            "summary": "OpenAI test",
            "evidence_ids": []
        },

        "github_signals": [
            {
                "project": "openai/openai-python",
                "language": "Python",
                "technical_direction": "OpenAI API",
                "activity": "active",
                "evidence": "Official OpenAI repository",

                "relevance": "direct",

                "source_id": "github:openai/openai-python"
            }
        ],

        "news_signals": [
            {
                "title": "OpenAI API pricing",
                "source": "Test",
                "published_at": "2026-08-23",
                "summary": "test",
                "implication": "test",

                "relevance": "direct",

                "source_id": "news:openai-test"
            }
        ],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    }

    try:

        JobAnalysisReport.model_validate(
            report
        )

        print(
            "✅ Test 9 PASSED: "
            "合法 relevance 正常通过"
        )

    except Exception as e:

        print(
            "❌ Test 9 FAILED: "
            "合法 relevance 被错误拦截"
        )

        print(e)

def test_openai_filter_is_strict():

    data = CompanyIntelligence(
        github=[
            GitHubRepository(
                name="openai/openai-python",
                description="Official OpenAI Python library",
                stars=100,
                forks=10,
                language="Python",
                updated="2026-08-23",
                url="https://github.com/openai/openai-python",
            ),

            GitHubRepository(
                name="Significant-Gravitas/AutoGPT",
                description="Autonomous AI agent",
                stars=100,
                forks=10,
                language="Python",
                updated="2026-08-23",
                url="https://github.com/Significant-Gravitas/AutoGPT",
            ),
        ],

        news=[
            NewsArticle(
                source_id="news:openai-test",
                id="openai-test",
                title="OpenAI announces new API pricing",
                description="OpenAI changes API pricing.",
                content="OpenAI announced new pricing.",
                url="https://example.com/openai",
                publishedAt="2026-08-23",
                lang="en",
                source=NewsSource(
                    id="test",
                    name="Test",
                    url="https://example.com",
                ),
            ),

            NewsArticle(
                source_id="news:anthropic-test",
                id="anthropic-test",
                title="Anthropic announces new model",
                description="Anthropic released a new model.",
                content="Anthropic announced a new model.",
                url="https://example.com/anthropic",
                publishedAt="2026-08-23",
                lang="en",
                source=NewsSource(
                    id="test",
                    name="Test",
                    url="https://example.com",
                ),
            ),
        ],
    )

    filtered = filter_company_data(
        data,
        "OpenAI"
    )

    # -----------------------------
    # GitHub
    # -----------------------------

    github_names = [
        repo.name
        for repo in filtered.github
    ]

    assert "openai/openai-python" in github_names

    assert "Significant-Gravitas/AutoGPT" not in github_names

    # -----------------------------
    # News
    # -----------------------------

    news_ids = [
        article.id
        for article in filtered.news
    ]

    assert "openai-test" in news_ids

    assert "anthropic-test" not in news_ids

    print(
        "✅ Test 11 PASSED: "
        "OpenAI filter 严格过滤第三方 GitHub / News"
    )

def test_relevance_validation():

    evidence = [
        Evidence(
            source_id="github:openai/openai-python",
            source_type="github",
            title="openai/openai-python",
            url="https://github.com/openai/openai-python",
        ),
        Evidence(
            source_id="news:openai-test",
            source_type="news",
            title="OpenAI Test News",
            url="https://example.com/openai",
        ),
    ]

    fake_report = {
        "company_overview": {
            "summary": "test",
            "evidence_ids": [
                "news:openai-test"
            ]
        },

        "github_signals": [
            {
                "project": "openai/openai-python",
                "language": "Python",
                "technical_direction": "OpenAI API",
                "activity": "active",
                "evidence": "test",

                # 故意制造非法 relevance
                "relevance": "maybe",

                "source_id": "github:openai/openai-python"
            }
        ],

        "news_signals": [],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [
            {
                "source_id": "github:openai/openai-python",
                "source_type": "github",
                "title": "openai/openai-python",
                "url": "https://github.com/openai/openai-python",
            },
            {
                "source_id": "news:openai-test",
                "source_type": "news",
                "title": "OpenAI Test News",
                "url": "https://example.com/openai",
            },
        ],
    }

    try:

        JobAnalysisReport.model_validate(
            fake_report
        )

        print(
            "❌ Test 12 FAILED: "
            "非法 relevance 没有被拦截"
        )

    except Exception:

        print(
            "✅ Test 12 PASSED: "
            "成功拦截非法 relevance"
        )





def test_target_consistency():

    # =========================================
    # Test 13-A: 合法 OpenAI direct GitHub
    # =========================================

    valid_report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI develops AI models.",
            "evidence_ids": [],
        },

        github_signals=[
            {
                "project": "openai/openai-python",
                "language": "Python",
                "technical_direction": "OpenAI API SDK",
                "activity": "active",
                "evidence": "Official OpenAI Python library.",
                "relevance": "direct",
                "source_id": "github:openai/openai-python",
            }
        ],

        news_signals=[],

        cross_analysis=[],

        job_value={
            "job_directions": [],
            "technical_skills": [],
            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    try:

        validate_target_consistency(
            valid_report,
            "OpenAI"
        )

        print(
            "✅ Test 13-A PASSED: "
            "合法 OpenAI direct GitHub 正常通过"
        )

    except ValueError as e:

        print(
            "❌ Test 13-A FAILED: "
            "合法 OpenAI GitHub 被错误拦截"
        )

        print(f"   {e}")

    # =========================================
    # Test 13-B: 非法 direct GitHub
    # AutoGPT 冒充 OpenAI
    # =========================================

    invalid_github_report = valid_report.model_copy(
        deep=True
    )

    invalid_github_report.github_signals[0].source_id = (
        "github:Significant-Gravitas/AutoGPT"
    )

    invalid_github_report.github_signals[0].project = (
        "Significant-Gravitas/AutoGPT"
    )

    try:

        validate_target_consistency(
            invalid_github_report,
            "OpenAI"
        )

        print(
            "❌ Test 13-B FAILED: "
            "没有拦截非 OpenAI direct GitHub"
        )

    except ValueError as e:

        print(
            "✅ Test 13-B PASSED: "
            "成功拦截非 OpenAI direct GitHub"
        )

        print(f"   {e}")

    # =========================================
    # Test 13-C: 合法 OpenAI direct News
    # =========================================

    valid_news_report = valid_report.model_copy(
        deep=True
    )

    valid_news_report.github_signals = []

    valid_news_report.news_signals = [
        NewsSignal(
            title="OpenAI announces new API pricing",
            source="Test News",
            published_at="2026-08-23",
            summary="OpenAI changes API pricing.",
            implication="OpenAI is adjusting its pricing strategy.",
            evidence="OpenAI changes API pricing.",
            relevance="direct",
            source_id="news:openai-test",
        )
    ]
    try:

        validate_target_consistency(
            valid_news_report,
            "OpenAI"
        )

        print(
            "✅ Test 13-C PASSED: "
            "合法 OpenAI direct News 正常通过"
        )

    except ValueError as e:

        print(
            "❌ Test 13-C FAILED: "
            "合法 OpenAI News 被错误拦截"
        )

        print(f"   {e}")

    # =========================================
    # Test 13-D: 非法 direct News
    # Anthropic 冒充 OpenAI
    # =========================================

    invalid_news_report = valid_news_report.model_copy(
        deep=True
    )

    invalid_news_report.news_signals[0].title = (
        "Anthropic announces new model"
    )

    invalid_news_report.news_signals[0].summary = (
        "Anthropic released a new model."
    )

    invalid_news_report.news_signals[0].implication = (
        "Anthropic is expanding its AI capabilities."
    )

    try:

        validate_target_consistency(
            invalid_news_report,
            "OpenAI"
        )

        print(
            "❌ Test 13-D FAILED: "
            "没有拦截非 OpenAI direct News"
        )

    except ValueError as e:

        print(
            "✅ Test 13-D PASSED: "
            "成功拦截非 OpenAI direct News"
        )

        print(f"   {e}")

    # =========================================
    # Test 13-E: competitive News
    # Anthropic 可以存在
    # =========================================

    competitive_report = valid_report.model_copy(
        deep=True
    )

    competitive_report.github_signals = []

    competitive_report.news_signals = [
        NewsSignal(
            title="Anthropic announces new model",
            source="Test News",
            published_at="2026-08-23",
            summary="Anthropic released a new model.",
            implication="Anthropic competes with OpenAI.",
            evidence="Anthropic announces new model.",
            relevance="competitive",
            source_id="news:anthropic-test",
        )
    ]

    try:

        validate_target_consistency(
            competitive_report,
            "OpenAI"
        )

        print(
            "✅ Test 13-E PASSED: "
            "competitive News 正常允许"
        )

    except ValueError as e:

        print(
            "❌ Test 13-E FAILED: "
            "competitive News 被错误拦截"
        )

        print(f"   {e}")

    print(
        "\n🎉 Test 13 target consistency tests completed."
    )

def test_company_overview_evidence_relevance():

    # =========================================
    # Base report
    # =========================================

    base_report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI develops AI models.",
            "evidence_ids": [],
        },

        github_signals=[],

        news_signals=[],

        cross_analysis=[],

        job_value={
            "job_directions": [],
            "technical_skills": [],
            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    # =========================================
    # Test 15-A
    # Direct News should be allowed
    # =========================================

    direct_report = base_report.model_copy(
        deep=True
    )

    direct_report.news_signals = [
        NewsSignal(
            title="OpenAI announces new API pricing",
            source="Test News",
            published_at="2026-08-23",
            summary="OpenAI changes API pricing.",
            implication="OpenAI is adjusting its pricing strategy.",
            relevance="direct",
            source_id="news:openai-direct",
        )
    ]

    direct_report.company_overview.evidence_ids = [
        "news:openai-direct"
    ]

    try:

        validate_target_consistency(
            direct_report,
            "OpenAI"
        )

        print(
            "✅ Test 15-A PASSED: "
            "direct evidence 可以用于 company_overview"
        )

    except ValueError as e:

        print(
            "❌ Test 15-A FAILED: "
            "合法 direct evidence 被错误拦截"
        )

        print(f"   {e}")

    # =========================================
    # Test 15-B
    # Competitive News should be rejected
    # =========================================

    competitive_report = base_report.model_copy(
        deep=True
    )

    competitive_report.news_signals = [
        NewsSignal(
            title="Anthropic announces new model",
            source="Test News",
            published_at="2026-08-23",
            summary="Anthropic released a new model.",
            implication="Anthropic competes with OpenAI.",
            evidence="Anthropic announces new model.",
            relevance="competitive",
            source_id="news:anthropic-test",
        )
    ]

    competitive_report.company_overview.evidence_ids = [
        "news:anthropic-test"
    ]

    try:

        validate_target_consistency(
            competitive_report,
            "OpenAI"
        )

        print(
            "❌ Test 15-B FAILED: "
            "competitive evidence 没有被拦截"
        )

    except ValueError as e:

        print(
            "✅ Test 15-B PASSED: "
            "成功阻止 competitive evidence "
            "作为 company_overview 直接事实"
        )

        print(f"   {e}")

    # =========================================
    # Test 15-C
    # Industry News should be rejected
    # =========================================

    industry_report = base_report.model_copy(
        deep=True
    )

    industry_report.news_signals = [
        NewsSignal(
            title="AI industry investment reaches record level",
            source="Test News",
            published_at="2026-08-23",
            summary="Global AI investment continues to grow.",
            implication="The AI industry is expanding rapidly.",
            relevance="industry",
            source_id="news:industry-test",
        )
    ]

    industry_report.company_overview.evidence_ids = [
        "news:industry-test"
    ]

    try:

        validate_target_consistency(
            industry_report,
            "OpenAI"
        )

        print(
            "❌ Test 15-C FAILED: "
            "industry evidence 没有被拦截"
        )

    except ValueError as e:

        print(
            "✅ Test 15-C PASSED: "
            "成功阻止 industry evidence "
            "作为 company_overview 直接事实"
        )

        print(f"   {e}")

    print(
        "\n🎉 Test 15 company overview evidence "
        "relevance tests completed."
    )

def test_cross_analysis_evidence_relevance():

    evidence = [
        Evidence(
            source_id="news:openai-test",
            source_type="news",
            title="OpenAI announces new API pricing",
            url="https://example.com/openai",
        ),

        Evidence(
            source_id="news:anthropic-test",
            source_type="news",
            title="Anthropic announces new model",
            url="https://example.com/anthropic",
        ),
    ]

    # =========================================
    # Test 16-A
    # direct evidence -> company conclusion
    # =========================================

    valid_report = {
        "company_overview": {
            "summary": "OpenAI changed API pricing.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [
            {
                "title": "OpenAI announces new API pricing",
                "source": "Test News",
                "published_at": "2026-08-23",
                "summary": "OpenAI changes API pricing.",
                "implication": "OpenAI is changing pricing.",
                "relevance": "direct",
                "source_id": "news:openai-test",
            }
        ],

        "cross_analysis": [
            {
                "relationship": (
                    "OpenAI's pricing strategy "
                    "is changing."
                ),

                "evidence": (
                    "OpenAI announced new API pricing."
                ),

                "evidence_ids": [
                    "news:openai-test"
                ]
            }
        ],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        }
    }

    try:

        validate_cross_analysis_evidence(
            valid_report,
            evidence,
            "OpenAI"
        )

        print(
            "✅ Test 16-A PASSED: "
            "direct evidence 可以用于 cross_analysis"
        )

    except ValueError as e:

        print(
            "❌ Test 16-A FAILED"
        )

        print(f"   {e}")

    # =========================================
    # Test 16-B
    # OpenAI + competitor
    # -> competitive conclusion
    # =========================================

    competitive_report = {
        "company_overview": {
            "summary": "OpenAI faces competition.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [
            {
                "title": "OpenAI announces new API pricing",
                "source": "Test News",
                "published_at": "2026-08-23",
                "summary": "OpenAI changes API pricing.",
                "implication": "OpenAI is changing pricing.",
                "relevance": "direct",
                "source_id": "news:openai-test",
            },

            {
                "title": "Anthropic announces new model",
                "source": "Test News",
                "published_at": "2026-08-23",
                "summary": "Anthropic released a competing model.",
                "implication": (
                    "Anthropic competes with OpenAI."
                ),
                "relevance": "competitive",
                "source_id": "news:anthropic-test",
            }
        ],

        "cross_analysis": [
            {
                "relationship": (
                    "OpenAI faces competitive pressure "
                    "from Anthropic."
                ),

                "evidence": (
                    "OpenAI changed pricing while "
                    "Anthropic launched a competing model."
                ),

                "evidence_ids": [
                    "news:openai-test",
                    "news:anthropic-test"
                ]
            }
        ],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        }
    }

    try:

        validate_cross_analysis_evidence(
            competitive_report,
            evidence,
            "OpenAI"
        )

        print(
            "✅ Test 16-B PASSED: "
            "OpenAI + competitive evidence "
            "可以形成竞争分析"
        )

    except ValueError as e:

        print(
            "❌ Test 16-B FAILED"
        )

        print(f"   {e}")

    # =========================================
    # Test 16-C
    # competitor-only evidence
    # -> false OpenAI company fact
    # =========================================

    invalid_report = {
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [
            {
                "title": "Anthropic announces new model",
                "source": "Test News",
                "published_at": "2026-08-23",
                "summary": "Anthropic released a new model.",
                "implication": (
                    "Anthropic is expanding "
                    "its AI capabilities."
                ),
                "relevance": "competitive",
                "source_id": "news:anthropic-test",
            }
        ],

        "cross_analysis": [
            {
                "relationship": (
                    "OpenAI's valuation is increasing "
                    "because Anthropic is preparing "
                    "for an IPO."
                ),

                "evidence": (
                    "Anthropic's IPO could break records."
                ),

                "evidence_ids": [
                    "news:anthropic-test"
                ]
            }
        ],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        }
    }

    try:

        validate_cross_analysis_evidence(
            invalid_report,
            evidence,
            "OpenAI"
        )

        print(
            "❌ Test 16-C FAILED: "
            "竞争对手 evidence 被错误推导成 "
            "OpenAI 事实"
        )

    except ValueError as e:

        print(
            "✅ Test 16-C PASSED: "
            "成功阻止竞争对手 evidence "
            "被推导成 OpenAI 直接事实"
        )

        print(f"   {e}")

    print(
        "\n🎉 Test 16 cross analysis evidence "
        "relevance tests completed."
    )



def test_cross_analysis_causality():

    evidence = [
        Evidence(
            source_id="news:openai-test",
            source_type="news",
            title="OpenAI announces new API pricing",
            url="https://example.com/openai",
        ),

        Evidence(
            source_id="news:anthropic-test",
            source_type="news",
            title="Anthropic announces new model",
            url="https://example.com/anthropic",
        ),

        Evidence(
            source_id="news:industry-test",
            source_type="news",
            title="AI industry growth accelerates",
            url="https://example.com/industry",
        ),
    ]

    # =========================================
    # Test 17-A
    #
    # Direct evidence + simple relationship
    #
    # OpenAI changed pricing.
    #
    # No unsupported causal claim.
    # =========================================

    report_a = {
        "news_signals": [
            {
                "source_id": "news:openai-test",
                "relevance": "direct",
            }
        ],

        "github_signals": [],

        "cross_analysis": [
            {
                "relationship": (
                    "OpenAI changed its API pricing."
                ),

                "evidence": (
                    "OpenAI announced new API pricing."
                ),

                "evidence_ids": [
                    "news:openai-test"
                ]
            }
        ]
    }

    try:

        validate_cross_analysis_causality(
            report_a,
            evidence,
            "OpenAI"
        )

        print(
            "✅ Test 17-A PASSED: "
            "direct evidence 的事实关系正常通过"
        )

    except ValueError as e:

        print(
            "❌ Test 17-A FAILED"
        )

        print(f"   {e}")

    # =========================================
    # Test 17-B
    #
    # Direct + competitive evidence
    #
    # "OpenAI faces competitive pressure."
    #
    # 这是允许的竞争分析。
    # =========================================

    report_b = {
        "news_signals": [

            {
                "source_id": "news:openai-test",
                "relevance": "direct",
            },

            {
                "source_id": "news:anthropic-test",
                "relevance": "competitive",
            },
        ],

        "github_signals": [],

        "cross_analysis": [
            {
                "relationship": (
                    "OpenAI faces competitive pressure "
                    "from Anthropic."
                ),

                "evidence": (
                    "OpenAI changed its pricing while "
                    "Anthropic launched a competing model."
                ),

                "evidence_ids": [
                    "news:openai-test",
                    "news:anthropic-test",
                ]
            }
        ]
    }

    try:

        validate_cross_analysis_causality(
            report_b,
            evidence,
            "OpenAI"
        )

        print(
            "✅ Test 17-B PASSED: "
            "竞争关系分析正常通过"
        )

    except ValueError as e:

        print(
            "❌ Test 17-B FAILED"
        )

        print(f"   {e}")

    # =========================================
    # Test 17-C
    #
    # Direct + competitive evidence
    #
    # BUT claims:
    #
    # "OpenAI lowered prices because Anthropic
    # launched a new model."
    #
    # This is causal.
    #
    # We do NOT have explicit causal evidence.
    #
    # Must be rejected.
    # =========================================

    report_c = {
        "news_signals": [

            {
                "source_id": "news:openai-test",
                "relevance": "direct",
            },

            {
                "source_id": "news:anthropic-test",
                "relevance": "competitive",
            },
        ],

        "github_signals": [],

        "cross_analysis": [
            {
                "relationship": (
                    "OpenAI lowered its API prices "
                    "because Anthropic launched "
                    "a new model."
                ),

                "evidence": (
                    "OpenAI changed pricing and "
                    "Anthropic launched a new model."
                ),

                "evidence_ids": [
                    "news:openai-test",
                    "news:anthropic-test",
                ]
            }
        ]
    }

    try:

        validate_cross_analysis_causality(
            report_c,
            evidence,
            "OpenAI"
        )

        print(
            "❌ Test 17-C FAILED: "
            "未经证实的竞争因果关系没有被拦截"
        )

    except ValueError as e:

        print(
            "✅ Test 17-C PASSED: "
            "成功拦截未经证实的竞争因果关系"
        )

        print(f"   {e}")

    # =========================================
    # Test 17-D
    #
    # Competitor-only evidence
    #
    # "OpenAI changed pricing because Anthropic..."
    #
    # No OpenAI direct evidence.
    #
    # Must be rejected.
    # =========================================

    report_d = {
        "news_signals": [

            {
                "source_id": "news:anthropic-test",
                "relevance": "competitive",
            }
        ],

        "github_signals": [],

        "cross_analysis": [
            {
                "relationship": (
                    "OpenAI changed its API pricing "
                    "because Anthropic launched "
                    "a new model."
                ),

                "evidence": (
                    "Anthropic launched a new model."
                ),

                "evidence_ids": [
                    "news:anthropic-test"
                ]
            }
        ]
    }

    try:

        validate_cross_analysis_causality(
            report_d,
            evidence,
            "OpenAI"
        )

        print(
            "❌ Test 17-D FAILED: "
            "竞争对手 evidence 被错误用于 "
            "建立 OpenAI 因果事实"
        )

    except ValueError as e:

        print(
            "✅ Test 17-D PASSED: "
            "成功拦截竞争对手 evidence "
            "建立 OpenAI 因果事实"
        )

        print(f"   {e}")

    print(
        "\n🎉 Test 17 cross analysis causality "
        "tests completed."
    )

def test_evidence_content_consistency():

    # =========================================
    # Authoritative evidence
    # =========================================

    evidence = [
        Evidence(
            source_id="github:openai/openai-python",
            source_type="github",
            title="openai/openai-python",
            url="https://github.com/openai/openai-python",
        ),

        Evidence(
            source_id="github:langgenius/dify",
            source_type="github",
            title="langgenius/dify",
            url="https://github.com/langgenius/dify",
        ),

        Evidence(
            source_id="news:openai-test",
            source_type="news",
            title="OpenAI announces new API pricing",
            url="https://example.com/openai",
        ),

        Evidence(
            source_id="news:anthropic-test",
            source_type="news",
            title="Anthropic announces new model",
            url="https://example.com/anthropic",
        ),
    ]

    # =========================================
    # Test 19-A
    # Valid GitHub signal
    # =========================================

    report_a = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI test",
            "evidence_ids": [],
        },

        github_signals=[
            {
                "project": "openai/openai-python",
                "language": "Python",
                "technical_direction": "OpenAI API SDK",
                "activity": "active",
                "evidence": "Official OpenAI repository.",
                "relevance": "direct",
                "source_id": "github:openai/openai-python",
            }
        ],

        news_signals=[],

        cross_analysis=[],

        job_value={
            "job_directions": [],
            "technical_skills": [],
            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    try:

        validate_evidence_content_consistency(
            report_a,
            evidence
        )

        print(
            "✅ Test 19-A PASSED: "
            "GitHub project 与 Evidence 一致"
        )

    except ValueError as e:

        print(
            "❌ Test 19-A FAILED"
        )

        print(f"   {e}")

    # =========================================
    # Test 19-B
    # Same source_id but fabricated project
    # =========================================

    report_b = report_a.model_copy(
        deep=True
    )

    report_b.github_signals[0].project = (
        "langgenius/dify"
    )

    try:

        validate_evidence_content_consistency(
            report_b,
            evidence
        )

        print(
            "❌ Test 19-B FAILED: "
            "没有拦截伪造 GitHub project"
        )

    except ValueError as e:

        print(
            "✅ Test 19-B PASSED: "
            "成功拦截 GitHub project 与 Evidence 不一致"
        )

        print(f"   {e}")

    # =========================================
    # Test 19-C
    # Valid News signal
    # =========================================

    report_c = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI test",
            "evidence_ids": [],
        },

        github_signals=[],

        news_signals=[
            {
                "title": "OpenAI announces new API pricing",
                "source": "Test News",
                "published_at": "2026-08-23",
                "summary": "OpenAI changed API pricing.",
                "implication": "Pricing changed.",
                "evidence": "OpenAI announced new API pricing.",
                "relevance": "direct",
                "source_id": "news:openai-test",
            }
        ],

        cross_analysis=[],

        job_value={
            "job_directions": [],
            "technical_skills": [],
            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    try:

        validate_evidence_content_consistency(
            report_c,
            evidence
        )

        print(
            "✅ Test 19-C PASSED: "
            "News title 与 Evidence 一致"
        )

    except ValueError as e:

        print(
            "❌ Test 19-C FAILED"
        )

        print(f"   {e}")

    # =========================================
    # Test 19-D
    # Same source_id but fabricated News title
    # =========================================

    report_d = report_c.model_copy(
        deep=True
    )

    report_d.news_signals[0].title = (
        "OpenAI launches new safety framework"
    )

    try:

        validate_evidence_content_consistency(
            report_d,
            evidence
        )

        print(
            "❌ Test 19-D FAILED: "
            "没有拦截伪造 News title"
        )

    except ValueError as e:

        print(
            "✅ Test 19-D PASSED: "
            "成功拦截 News title 与 Evidence 不一致"
        )

        print(f"   {e}")

    print(
        "\n🎉 Test 19 evidence content "
        "consistency tests completed."
    )




def test_evidence_content_support():

    evidence = [
        Evidence(
            source_id="news:openai-test",
            source_type="news",
            title="OpenAI announces new API pricing",
            url="https://example.com/openai",
            description=(
                "OpenAI reduced API pricing "
                "for developers."
            ),
            snippet=(
                "OpenAI announced a reduction "
                "in API pricing."
            ),
        ),

        Evidence(
            source_id="news:anthropic-test",
            source_type="news",
            title="Anthropic announces new model",
            url="https://example.com/anthropic",
            description=(
                "Anthropic announced a new model."
            ),
            snippet=(
                "Anthropic released a new model "
                "to compete in the AI market."
            ),
        ),
    ]

    # =========================================
    # 20-A
    # Valid evidence
    # =========================================

    report_a = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI pricing changed.",
            "evidence_ids": [],
        },

        github_signals=[],

        news_signals=[
            {
                "title": "OpenAI announces new API pricing",
                "source": "Test",
                "published_at": "2026-08-23",
                "summary": "OpenAI reduced API pricing.",
                "implication": "API access became cheaper.",
                "evidence": (
                    "OpenAI announced a reduction "
                    "in API pricing."
                ),
                "relevance": "direct",
                "source_id": "news:openai-test",
            }
        ],

        cross_analysis=[],

        job_value={
            "job_directions": [],
            "technical_skills": [],
            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    # 这个模型字段叫 evidence，
    # 如果你的 NewsSignal 已经包含 evidence 字段，
    # 这里填写：
    report_a.news_signals[0].evidence = (
        "OpenAI announced a reduction "
        "in API pricing."
    )

    try:

        validate_evidence_content_support(
            report_a,
            evidence
        )

        print(
            "✅ Test 20-A PASSED: "
            "evidence 内容得到原始证据支持"
        )

    except ValueError as e:

        print(
            "❌ Test 20-A FAILED"
        )

        print(f"   {e}")

    # =========================================
    # 20-B
    # Fabricated evidence
    # =========================================

    report_b = report_a.model_copy(
        deep=True
    )

    report_b.news_signals[0].evidence = (
        "OpenAI launched a completely new "
        "quantum computing division."
    )

    try:

        validate_evidence_content_support(
            report_b,
            evidence
        )

        print(
            "❌ Test 20-B FAILED: "
            "没有拦截伪造 evidence"
        )

    except ValueError as e:

        print(
            "✅ Test 20-B PASSED: "
            "成功拦截未经原始证据支持的 evidence"
        )

        print(f"   {e}")

    # =========================================
    # 20-C
    # Valid Anthropic evidence
    # =========================================

    report_c = JobAnalysisReport(
        company_overview={
            "summary": "Competition test.",
            "evidence_ids": [],
        },

        github_signals=[],

        news_signals=[
            {
                "title": "Anthropic announces new model",
                "source": "Test",
                "published_at": "2026-08-23",
                "summary": "Anthropic announced a new model.",
                "implication": "Anthropic is competing.",
                "evidence": "Anthropic announced a new model.",
                "relevance": "competitive",
                "source_id": "news:anthropic-test",
            }
        ],

        cross_analysis=[],

        job_value={
            "job_directions": [],
            "technical_skills": [],
            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    report_c.news_signals[0].evidence = (
        "Anthropic announced a new model."
    )

    try:

        validate_evidence_content_support(
            report_c,
            evidence
        )

        print(
            "✅ Test 20-C PASSED: "
            "竞争对手 evidence 得到原始证据支持"
        )

    except ValueError as e:

        print(
            "❌ Test 20-C FAILED"
        )

        print(f"   {e}")

    # =========================================
    # 20-D
    # Fabricated causal evidence
    # =========================================

    report_d = report_c.model_copy(
        deep=True
    )

    report_d.news_signals[0].evidence = (
        "Anthropic's IPO directly caused "
        "OpenAI to cut its API prices."
    )

    try:

        validate_evidence_content_support(
            report_d,
            evidence
        )

        print(
            "❌ Test 20-D FAILED: "
            "没有拦截伪造因果 evidence"
        )

    except ValueError as e:

        print(
            "✅ Test 20-D PASSED: "
            "成功拦截原始 evidence 不支持的因果关系"
        )

        print(f"   {e}")


    print(
        "\n🎉 Test 20 evidence content "
        "support tests completed."
    )


def test_claim_validation():

    evidence = [
        Evidence(
            source_id="news:openai-test",
            source_type="news",
            title="OpenAI announces new API pricing",
            url="https://example.com/openai",
            description="OpenAI reduced API pricing.",
            snippet="OpenAI announced a reduction in API pricing.",
        )
    ]

    report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI pricing changed.",
            "evidence_ids": [
                "news:openai-test"
            ]
        },

        github_signals=[],

        news_signals=[
            {
                "title": "OpenAI announces new API pricing",
                "source": "Test",
                "published_at": "2026-08-23",
                "summary": "OpenAI reduced API pricing.",
                "implication": "Pricing changed.",
                "evidence": "OpenAI announced a reduction in API pricing.",
                "relevance": "direct",
                "source_id": "news:openai-test",
            }
        ],

        cross_analysis=[
            {
                "relationship": "OpenAI pricing changed.",
                "evidence": "OpenAI reduced API pricing.",
                "evidence_ids": [
                    "news:openai-test"
                ],
                "claims": [
                    {
                        "statement": "OpenAI reduced API pricing.",
                        "claim_type": "fact",
                        "evidence_ids": [
                            "news:openai-test"
                        ],
                        "confidence": 0.95,
                    }
                ],
            }
        ],

        job_value={
            "job_directions": [],
            "technical_skills": [],
            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    print("✅ Test 21-A PASSED: 合法 Claim 正常通过")
    invalid_report = report.model_copy(
        deep=True
    )

    invalid_report.cross_analysis[0].claims[0].evidence_ids = []

    try:

        validate_claims(
            invalid_report,
            evidence
        )

        print(
            "❌ Test 21-B FAILED: "
            "没有拦截无 Evidence Claim"
        )

    except ValueError as e:

        print(
            "✅ Test 21-B PASSED: "
            "成功拦截无 Evidence Claim"
        )

        print(f"   {e}")



    invalid_report = report.model_copy(
        deep=True
    )

    invalid_report.cross_analysis[0].claims[0].evidence_ids = [
        "news:not-real"
    ]

    try:

        validate_claims(
            invalid_report,
            evidence
        )

        print(
            "❌ Test 21-C FAILED: "
            "非法 Claim source_id 没有被拦截"
        )

    except ValueError as e:

        print(
            "✅ Test 21-C PASSED: "
            "成功拦截非法 Claim source_id"
        )

        print(f"   {e}")

    # =========================================
    # Test 21-D
    # confidence 越界
    # =========================================

    invalid_claim = {
        "statement": "OpenAI reduced API pricing.",
        "claim_type": "fact",
        "evidence_ids": [
            "news:openai-test"
        ],
        "confidence": 1.5,
    }

    try:

        Claim.model_validate(
            invalid_claim
        )

        print(
            "❌ Test 21-D FAILED: "
            "非法 confidence 没有被拦截"
        )

    except Exception as e:

        print(
            "✅ Test 21-D PASSED: "
            "成功拦截 confidence > 1"
        )

        print(
            f"   {e}"
        )

    invalid_claim_low = {
        "statement": "OpenAI reduced API pricing.",
        "claim_type": "fact",
        "evidence_ids": [
            "news:openai-test"
        ],
        "confidence": -0.1,
    }

    try:

        Claim.model_validate(
            invalid_claim_low
        )

        print(
            "❌ Test 21-E FAILED: "
            "非法 confidence < 0 没有被拦截"
        )

    except Exception as e:

        print(
            "✅ Test 21-E PASSED: "
            "成功拦截 confidence < 0"
        )

        print(
            f"   {e}"
        )

def calculate_claim_confidence(
    claim: Claim,
    evidence,
    signal_relevance_map: dict[str, str],
) -> float:

    score = 0.0

    # =========================================
    # Evidence existence
    # =========================================

    if claim.evidence_ids:

        score += 0.30

    # =========================================
    # Multiple evidence
    # =========================================

    if len(claim.evidence_ids) >= 2:

        score += 0.15

    # =========================================
    # Relevance
    # =========================================

    relevances = []

    for source_id in claim.evidence_ids:

        relevance = signal_relevance_map.get(
            source_id
        )

        if relevance:

            relevances.append(
                relevance
            )

    # Direct evidence
    if "direct" in relevances:

        score += 0.25

    # Competitive evidence
    elif "competitive" in relevances:

        score += 0.10

    # =========================================
    # Claim type
    # =========================================

    if claim.claim_type == ClaimType.FACT:

        score += 0.15

    elif claim.claim_type == ClaimType.INFERENCE:

        score -= 0.10

    elif claim.claim_type == ClaimType.CAUSAL:

        score -= 0.15

    elif claim.claim_type == ClaimType.INDUSTRY:

        score -= 0.10

    # =========================================
    # Clamp
    # =========================================

    return max(
        0.0,
        min(
            1.0,
            round(score, 2)
        )
    )

def build_signal_relevance_map(
    report: JobAnalysisReport
) -> dict[str, str]:

    relevance_map = {}

    for signal in report.github_signals:

        relevance_map[
            signal.source_id
        ] = signal.relevance

    for signal in report.news_signals:

        relevance_map[
            signal.source_id
        ] = signal.relevance

    return relevance_map

def apply_confidence_scores(
    report: JobAnalysisReport,
    evidence,
):
    relevance_map = (
        build_signal_relevance_map(report)
    )

    for analysis in report.cross_analysis:

        for claim in analysis.claims:

            calculated = (
                calculate_claim_confidence(
                    claim,
                    evidence,
                    relevance_map,
                )
            )

            claim.confidence = calculated

    return report

def test_confidence_scoring():
    evidence_b = [
        Evidence(
            source_id="news:openai-test",
            source_type="news",
            title="OpenAI announces new API pricing",
            url="https://example.com/openai",
            description="OpenAI reduced API pricing.",
            snippet="OpenAI announced a reduction in API pricing.",
        ),
        Evidence(
            source_id="news:openai-test-2",
            source_type="news",
            title="OpenAI confirms pricing changes",
            url="https://example.com/openai-2",
            description="OpenAI confirmed the pricing reduction.",
            snippet="OpenAI confirmed reduced API pricing for developers.",
        ),
    ]

    report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI pricing changed.",
            "evidence_ids": [
                "news:openai-test"
            ]
        },

        github_signals=[],

        news_signals=[
            {
                "title": "OpenAI announces new API pricing",
                "source": "Test",
                "published_at": "2026-08-23",
                "summary": "OpenAI reduced API pricing.",
                "implication": "Pricing changed.",
                "evidence": "OpenAI announced a reduction in API pricing.",
                "relevance": "direct",
                "source_id": "news:openai-test",
            },
            {
                "title": "OpenAI confirms pricing changes",
                "source": "Test",
                "published_at": "2026-08-23",
                "summary": "OpenAI confirmed the pricing reduction.",
                "implication": "The pricing change was confirmed.",
                "evidence": "OpenAI confirmed reduced API pricing.",
                "relevance": "direct",
                "source_id": "news:openai-test-2",
            },
        ],

        cross_analysis=[
            {
                "relationship": "OpenAI pricing changed.",
                "evidence": "OpenAI reduced API pricing.",
                "evidence_ids": [
                    "news:openai-test"
                ],
                "claims": [
                    {
                        "statement": "OpenAI reduced API pricing.",
                        "claim_type": "fact",
                        "evidence_ids": [
                            "news:openai-test",
                            "news:openai-test-2",
                        ],
                        # LLM deliberately lies
                        "confidence": 0.01,
                    }
                ],
            }
        ],

        job_value={
            "job_directions": [],
            "technical_skills": [],
            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    apply_confidence_scores(
        report,
        evidence_b
    )

    confidence = (
        report.cross_analysis[0]
        .claims[0]
        .confidence
    )

    print(
        f"Test 22-B actual confidence = {confidence}"
    )
    assert confidence == 0.85

    print(
        "✅ Test 22-B PASSED: "
        "后端 confidence 覆盖 LLM 初始值"
    )

    print(
        f"   calculated confidence = {confidence}"
    )





def test_recommendation_evidence_support():

    evidence = [
        Evidence(
            source_id="news:openai-test",
            source_type="news",
            title="OpenAI improves AI safety",
            url="https://example.com/openai",
        )
    ]

    report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI test",
            "evidence_ids": []
        },

        github_signals=[],

        news_signals=[],

        cross_analysis=[
            {
                "relationship": "OpenAI safety focus",
                "evidence": "OpenAI focuses on AI safety.",
                "evidence_ids": [
                    "news:openai-test"
                ],
                "claims": [
                    {
                        "statement": (
                            "OpenAI is focusing "
                            "on AI safety."
                        ),
                        "claim_type": "fact",
                        "evidence_ids": [
                            "news:openai-test"
                        ],
                        "confidence": 0.9
                    }
                ]
            }
        ],

        job_value={
            "job_directions": [
                {
                    "name": "AI Safety Engineer",
                    "rationale": (
                        "The evidence shows "
                        "a direct focus on AI safety."
                    ),
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:openai-test"
                    ],
                    "confidence": 0.85
                }
            ],
            "technical_skills": [],
            "important_areas": []
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": []
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        evidence=[]
    )

    try:

        validate_recommendation_evidence_support(
            report,
            evidence
        )

        print(
            "✅ Test 23-A PASSED: "
            "合法 Recommendation 正常通过"
        )

    except ValueError as e:

        print(
            "❌ Test 23-A FAILED"
        )

        print(e)

    # =========================================
    # Test 23-B
    # Recommendation 没有 claim_refs
    # =========================================

    report_b = report.model_copy(deep=True)

    report_b.job_value.job_directions[0].claim_refs = []

    try:

        validate_recommendation_evidence_support(
            report_b,
            evidence
        )

        print(
            "❌ Test 23-B FAILED: "
            "没有拦截无 claim_refs 的 Recommendation"
        )

    except ValueError as e:

        print(
            "✅ Test 23-B PASSED: "
            "成功拦截无 claim_refs 的 Recommendation"
        )

        print(f"   {e}")

    # =========================================
    # Test 23-C
    # 非法 claim_ref
    # =========================================

    report_c = report.model_copy(deep=True)

    report_c.job_value.job_directions[0].claim_refs = [
        "cross_analysis:99.claims:99"
    ]

    try:

        validate_recommendation_evidence_support(
            report_c,
            evidence
        )

        print(
            "❌ Test 23-C FAILED: "
            "没有拦截非法 claim_ref"
        )

    except ValueError as e:

        print(
            "✅ Test 23-C PASSED: "
            "成功拦截非法 claim_ref"
        )

        print(f"   {e}")

    # =========================================
    # Test 23-D
    # Recommendation.evidence_ids 被恶意篡改
    # =========================================

    evidence.append(
        Evidence(
            source_id="news:unrelated-test",
            source_type="news",
            title="Unrelated Evidence",
            url="https://example.com/unrelated",
        )
    )

    report_d = report.model_copy(deep=True)

    # 模拟 LLM / 外部输入篡改
    report_d.job_value.job_directions[0].evidence_ids = [
        "news:unrelated-test"
    ]

    validate_recommendation_evidence_support(
        report_d,
        evidence
    )

    actual_evidence_ids = (
        report_d
        .job_value
        .job_directions[0]
        .evidence_ids
    )

    expected_evidence_ids = [
        "news:openai-test"
    ]

    assert actual_evidence_ids == expected_evidence_ids

    print(
        "✅ Test 23-D PASSED: "
        "成功覆盖被篡改的 Recommendation evidence_ids"
    )

    print(
        f"   actual={actual_evidence_ids}"
    )

    print(
        f"   expected={expected_evidence_ids}"
    )

def calculate_recommendation_confidence(
    recommendation: Recommendation,
    claim_map: dict[str, Claim],
) -> float:
    """
    Recommendation confidence is calculated from
    the confidence of the Claims it references.

    Rule:
        average(all referenced Claim confidence)

    Raises:
        ValueError:
            - no claim_refs
            - invalid claim_ref
    """

    if not recommendation.claim_refs:
        raise ValueError(
            "Recommendation has no claim_refs: "
            f"{recommendation.name}"
        )

    claim_confidences = []

    for claim_ref in recommendation.claim_refs:

        if claim_ref not in claim_map:
            raise ValueError(
                "Recommendation references invalid "
                f"claim_ref: {claim_ref}"
            )

        claim = claim_map[claim_ref]

        claim_confidences.append(
            float(claim.confidence)
        )

    if not claim_confidences:
        raise ValueError(
            "Recommendation has no valid Claims: "
            f"{recommendation.name}"
        )

    confidence = (
            sum(claim_confidences)
            / len(claim_confidences)
    )

    confidence = round(
        confidence,
        2
    )

    return max(
        0.0,
        min(
            1.0,
            confidence
        )
    )

def apply_recommendation_confidence_scores(
    report: JobAnalysisReport,
) -> None:
    """
    Recalculate confidence for every Recommendation.

    LLM-provided Recommendation confidence is ignored.
    """

    claim_map = {}

    for analysis_index, analysis in enumerate(
        report.cross_analysis
    ):

        for claim_index, claim in enumerate(
            analysis.claims
        ):

            claim_ref = (
                f"cross_analysis:"
                f"{analysis_index}"
                f".claims:"
                f"{claim_index}"
            )

            claim_map[claim_ref] = claim

    recommendation_groups = [
        report.job_value.job_directions,
        report.job_value.technical_skills,
        report.job_value.important_areas,
        report.interview_preparation.topics,
        report.interview_preparation.practical_tasks,
    ]

    for group in recommendation_groups:

        for recommendation in group:

            recommendation.confidence = (
                calculate_recommendation_confidence(
                    recommendation,
                    claim_map,
                )
            )

def test_recommendation_confidence_scoring():

    # =========================================
    # Test 24-A
    # 单 Claim
    # =========================================

    report_a = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI test",
            "evidence_ids": [],
        },

        github_signals=[],

        news_signals=[],

        cross_analysis=[
            {
                "relationship": "AI safety",
                "evidence": "OpenAI focuses on safety.",
                "evidence_ids": [
                    "news:openai-test"
                ],
                "claims": [
                    {
                        "statement": (
                            "OpenAI focuses "
                            "on AI safety."
                        ),
                        "claim_type": "fact",
                        "evidence_ids": [
                            "news:openai-test"
                        ],
                        "confidence": 0.90,
                    }
                ],
            }
        ],

        job_value={
            "job_directions": [
                {
                    "name": "AI Safety Engineer",
                    "rationale": (
                        "Safety evidence supports "
                        "this direction."
                    ),
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:openai-test"
                    ],
                    # 故意给错误值
                    "confidence": 0.10,
                }
            ],
            "technical_skills": [],
            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    apply_recommendation_confidence_scores(
        report_a
    )

    confidence = (
        report_a
        .job_value
        .job_directions[0]
        .confidence
    )

    print(
        f"Test 24-A actual confidence = "
        f"{confidence}"
    )

    assert confidence == 0.90

    print(
        "✅ Test 24-A PASSED: "
        "单 Claim Recommendation confidence 正确计算"
    )

    # =========================================
    # Test 24-B
    # 多 Claim → 平均 confidence
    # =========================================

    report_b = report_a.model_copy(
        deep=True
    )

    report_b.cross_analysis[0].claims.append(
        Claim(
            statement=(
                "OpenAI monitors AI security."
            ),
            claim_type="fact",
            evidence_ids=[
                "news:openai-test"
            ],
            confidence=0.80,
        )
    )

    report_b.job_value.job_directions[0].claim_refs = [
        "cross_analysis:0.claims:0",
        "cross_analysis:0.claims:1",
    ]

    report_b.job_value.job_directions[0].confidence = 0.01

    apply_recommendation_confidence_scores(
        report_b
    )

    confidence = (
        report_b
        .job_value
        .job_directions[0]
        .confidence
    )

    print(
        f"Test 24-B actual confidence = "
        f"{confidence}"
    )

    assert confidence == 0.85

    print(
        "✅ Test 24-B PASSED: "
        "多 Claim Recommendation confidence "
        "正确取平均值"
    )

    # =========================================
    # Test 24-C
    # LLM confidence 必须被后端覆盖
    # =========================================

    report_c = report_a.model_copy(
        deep=True
    )

    report_c.job_value.job_directions[0].confidence = (
        0.99
    )

    apply_recommendation_confidence_scores(
        report_c
    )

    confidence = (
        report_c
        .job_value
        .job_directions[0]
        .confidence
    )

    print(
        f"Test 24-C actual confidence = "
        f"{confidence}"
    )

    assert confidence == 0.90

    print(
        "✅ Test 24-C PASSED: "
        "后端 confidence 覆盖 LLM 初始值"
    )

    # =========================================
    # Test 24-D
    # 无有效 Claim
    # =========================================

    report_d = report_a.model_copy(
        deep=True
    )

    report_d.job_value.job_directions[0].claim_refs = [
        "cross_analysis:99.claims:99"
    ]

    try:

        apply_recommendation_confidence_scores(
            report_d
        )

        print(
            "❌ Test 24-D FAILED: "
            "没有拦截无效 Claim"
        )

    except ValueError as e:

        print(
            "✅ Test 24-D PASSED: "
            "成功拦截无效 Claim"
        )

        print(
            f"   {e}"
        )

    # =========================================
    # Test 24-E
    # confidence 始终在 0~1
    # =========================================

    for value in [0.0, 0.35, 0.75, 1.0]:

        report_e = report_a.model_copy(
            deep=True
        )

        report_e.cross_analysis[0].claims[0].confidence = (
            value
        )

        apply_recommendation_confidence_scores(
            report_e
        )

        confidence = (
            report_e
            .job_value
            .job_directions[0]
            .confidence
        )

        assert 0.0 <= confidence <= 1.0

    print(
        "✅ Test 24-E PASSED: "
        "Recommendation confidence 始终在 0~1"
    )

    print(
        "\n🎉 Test 24 Recommendation confidence "
        "tests completed."
    )

def calculate_recommendation_priority(
    recommendation: Recommendation,
    claim_map: dict[str, Claim],
) -> float:
    """
    Calculate recommendation priority.

    Formula:

        50% recommendation confidence
        30% average claim confidence
        20% evidence coverage

    Returns:
        float between 0 and 1
    """

    # -----------------------------------------
    # Recommendation confidence
    # -----------------------------------------

    recommendation_confidence = (
        float(recommendation.confidence)
    )

    # -----------------------------------------
    # Claim confidence
    # -----------------------------------------

    claim_confidences = []

    for claim_ref in recommendation.claim_refs:

        if claim_ref not in claim_map:
            raise ValueError(
                "Recommendation references invalid "
                f"claim_ref: {claim_ref}"
            )

        claim_confidences.append(
            float(
                claim_map[
                    claim_ref
                ].confidence
            )
        )

    if not claim_confidences:

        raise ValueError(
            "Recommendation has no valid claims: "
            f"{recommendation.name}"
        )

    average_claim_confidence = (
        sum(claim_confidences)
        / len(claim_confidences)
    )

    # -----------------------------------------
    # Evidence coverage
    # -----------------------------------------

    supported_evidence_ids = set()

    for claim_ref in recommendation.claim_refs:

        supported_evidence_ids.update(
            claim_map[
                claim_ref
            ].evidence_ids
        )

    recommendation_evidence_ids = set(
        recommendation.evidence_ids
    )

    if not recommendation_evidence_ids:

        evidence_coverage = 0.0

    else:

        supported_count = len(
            recommendation_evidence_ids
            & supported_evidence_ids
        )

        evidence_coverage = (
            supported_count
            / len(
                recommendation_evidence_ids
            )
        )

    # -----------------------------------------
    # Final score
    # -----------------------------------------

    score = (
        0.50 * recommendation_confidence
        + 0.30 * average_claim_confidence
        + 0.20 * evidence_coverage
    )

    return round(
        max(
            0.0,
            min(
                1.0,
                score
            )
        ),
        2
    )



def get_priority_level(
    priority_score: float,
) -> str:

    if priority_score >= 0.80:
        return "P1"

    if priority_score >= 0.60:
        return "P2"

    return "P3"

def apply_recommendation_priority(
    report: JobAnalysisReport,
) -> None:

    claim_map = {}

    for analysis_index, analysis in enumerate(
        report.cross_analysis
    ):

        for claim_index, claim in enumerate(
            analysis.claims
        ):

            claim_ref = (
                f"cross_analysis:"
                f"{analysis_index}"
                f".claims:"
                f"{claim_index}"
            )

            claim_map[claim_ref] = claim

    recommendation_groups = [
        report.job_value.job_directions,
        report.job_value.technical_skills,
        report.job_value.important_areas,
        report.interview_preparation.topics,
        report.interview_preparation.practical_tasks,
    ]

    for group in recommendation_groups:

        for recommendation in group:

            score = (
                calculate_recommendation_priority(
                    recommendation,
                    claim_map,
                )
            )

            recommendation.priority_score = (
                score
            )

            recommendation.priority_level = (
                get_priority_level(score)
            )

def test_recommendation_priority():

    report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI test",
            "evidence_ids": [],
        },

        github_signals=[],

        news_signals=[],

        cross_analysis=[
            {
                "relationship": "AI safety",
                "evidence": (
                    "OpenAI focuses on AI safety."
                ),
                "evidence_ids": [
                    "news:openai-test"
                ],
                "claims": [
                    {
                        "statement": (
                            "OpenAI focuses "
                            "on AI safety."
                        ),
                        "claim_type": "fact",
                        "evidence_ids": [
                            "news:openai-test"
                        ],
                        "confidence": 0.90,
                    }
                ],
            }
        ],

        job_value={
            "job_directions": [
                {
                    "name": "AI Safety Engineer",
                    "rationale": (
                        "Direct evidence "
                        "supports the direction."
                    ),
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:openai-test"
                    ],
                    "confidence": 0.90,
                    "priority_score": 0.0,
                    "priority_level": "P3",
                }
            ],

            "technical_skills": [],

            "important_areas": [],
        },

        interview_preparation={
            "topics": [],

            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    apply_recommendation_confidence_scores(
        report
    )

    apply_recommendation_priority(
        report
    )

    recommendation = (
        report
        .job_value
        .job_directions[0]
    )

    print(
        "Test 25-A score =",
        recommendation.priority_score
    )

    print(
        "Test 25-A level =",
        recommendation.priority_level
    )

    assert recommendation.priority_score == 0.92

    assert recommendation.priority_level == "P1"

    print(
        "✅ Test 25-A PASSED: "
        "高置信度 Recommendation 正确获得 P1"
    )

    # =========================================
    # Test 25-B
    # 中等置信度 Recommendation → P2
    # =========================================

    report_b = report.model_copy(
        deep=True
    )

    report_b.cross_analysis[0].claims[0].confidence = 0.60

    report_b.job_value.job_directions[0].confidence = 0.65

    report_b.job_value.job_directions[0].priority_score = 0.0
    report_b.job_value.job_directions[0].priority_level = "P3"

    apply_recommendation_confidence_scores(
        report_b
    )

    apply_recommendation_priority(
        report_b
    )

    recommendation_b = (
        report_b
        .job_value
        .job_directions[0]
    )

    print(
        "Test 25-B score =",
        recommendation_b.priority_score
    )

    print(
        "Test 25-B level =",
        recommendation_b.priority_level
    )

    assert recommendation_b.priority_score == 0.68

    assert recommendation_b.priority_level == "P2"

    print(
        "✅ Test 25-B PASSED: "
        "中等置信度 Recommendation 正确获得 P2"
    )

    # =========================================
    # Test 25-C
    # 低置信度 Recommendation → P3
    # =========================================

    report_c = report.model_copy(
        deep=True
    )

    report_c.cross_analysis[0].claims[0].confidence = 0.30

    report_c.job_value.job_directions[0].confidence = 0.30

    report_c.job_value.job_directions[0].priority_score = 0.99
    report_c.job_value.job_directions[0].priority_level = "P1"

    apply_recommendation_confidence_scores(
        report_c
    )

    apply_recommendation_priority(
        report_c
    )

    recommendation_c = (
        report_c
        .job_value
        .job_directions[0]
    )

    print(
        "Test 25-C score =",
        recommendation_c.priority_score
    )

    print(
        "Test 25-C level =",
        recommendation_c.priority_level
    )

    assert recommendation_c.priority_score == 0.44

    assert recommendation_c.priority_level == "P3"

    print(
        "✅ Test 25-C PASSED: "
        "低置信度 Recommendation 正确获得 P3"
    )

    # =========================================
    # Test 25-D
    # LLM 给出错误 Priority
    # Backend 必须覆盖
    # =========================================

    report_d = report.model_copy(
        deep=True
    )

    # Claim 真实置信度较低
    report_d.cross_analysis[0].claims[0].confidence = 0.30

    # LLM 故意伪造一个很高的 Recommendation confidence
    report_d.job_value.job_directions[0].confidence = 0.99

    # LLM 故意伪造 Priority
    report_d.job_value.job_directions[0].priority_score = 0.99
    report_d.job_value.job_directions[0].priority_level = "P1"

    # Backend 重新计算 Recommendation confidence
    apply_recommendation_confidence_scores(
        report_d
    )

    # Backend 重新计算 Priority
    apply_recommendation_priority(
        report_d
    )

    recommendation_d = (
        report_d
        .job_value
        .job_directions[0]
    )

    print(
        "Test 25-D actual recommendation confidence =",
        recommendation_d.confidence
    )

    print(
        "Test 25-D actual priority score =",
        recommendation_d.priority_score
    )

    print(
        "Test 25-D actual priority level =",
        recommendation_d.priority_level
    )

    # LLM 的 0.99 / P1 必须被覆盖
    assert recommendation_d.confidence == 0.30

    assert recommendation_d.priority_score == 0.44

    assert recommendation_d.priority_level == "P3"

    print(
        "✅ Test 25-D PASSED: "
        "后端成功覆盖 LLM 提供的错误 Priority"
    )



def normalize_recommendation_name(
    name: str,
) -> str:
    """
    Normalize recommendation names for
    deduplication.

    Examples:

        "AI Safety and Governance"
        "AI Safety & Governance"

    become approximately:

        "ai safety governance"
    """

    text = name.strip().lower()

    # & -> and
    text = text.replace("&", " and ")

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9\u4e00-\u9fff]+",
        " ",
        text
    )

    # Normalize "and"
    text = re.sub(
        r"\band\b",
        " ",
        text
    )

    # Collapse whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()



def recommendation_name_similarity(
    name_a: str,
    name_b: str,
) -> float:

    normalized_a = (
        normalize_recommendation_name(
            name_a
        )
    )

    normalized_b = (
        normalize_recommendation_name(
            name_b
        )
    )

    if not normalized_a or not normalized_b:
        return 0.0

    return SequenceMatcher(
        None,
        normalized_a,
        normalized_b,
    ).ratio()

def deduplicate_recommendations(
    recommendations: list[Recommendation],
    similarity_threshold: float = 0.85,
) -> list[Recommendation]:
    """
    Deduplicate highly similar recommendations.

    Conservative strategy:
    - Keep the first representative.
    - Merge claim_refs.
    - Merge evidence_ids.
    - Keep the stronger confidence.
    """

    result = []

    for recommendation in recommendations:

        duplicate = None

        for existing in result:

            similarity = (
                recommendation_name_similarity(
                    recommendation.name,
                    existing.name,
                )
            )

            if similarity >= similarity_threshold:

                duplicate = existing
                break

        if duplicate is None:

            result.append(
                recommendation.model_copy(
                    deep=True
                )
            )

            continue

        # -----------------------------------------
        # Merge claim_refs
        # -----------------------------------------

        duplicate.claim_refs = list(
            dict.fromkeys(
                duplicate.claim_refs
                + recommendation.claim_refs
            )
        )

        # -----------------------------------------
        # Merge evidence_ids
        # -----------------------------------------

        duplicate.evidence_ids = list(
            dict.fromkeys(
                duplicate.evidence_ids
                + recommendation.evidence_ids
            )
        )

        # -----------------------------------------
        # Keep stronger confidence
        # -----------------------------------------

        duplicate.confidence = max(
            duplicate.confidence,
            recommendation.confidence,
        )

    return result

def get_top_recommendations(
    recommendations: list[Recommendation],
    top_k: int = 5,
) -> list[Recommendation]:
    """
    Return the top-K recommendations ranked by:

    1. priority_score descending
    2. confidence descending

    Does not mutate the original list.
    """

    if top_k <= 0:
        return []

    sorted_recommendations = sorted(
        recommendations,
        key=lambda item: (
            float(item.priority_score),
            float(item.confidence),
        ),
        reverse=True,
    )

    return sorted_recommendations[:top_k]

def test_recommendation_deduplication():

    recommendations = [

        Recommendation(
            name="AI Safety and Governance",
            rationale="Safety and governance focus.",
            claim_refs=[
                "cross_analysis:0.claims:0"
            ],
            evidence_ids=[
                "news:openai-safety"
            ],
            confidence=0.70,
            priority_score=0.76,
            priority_level="P2",
        ),

        Recommendation(
            name="AI Safety & Governance",
            rationale="Governance and safety focus.",
            claim_refs=[
                "cross_analysis:1.claims:0"
            ],
            evidence_ids=[
                "news:openai-policy"
            ],
            confidence=0.80,
            priority_score=0.82,
            priority_level="P1",
        ),
    ]

    deduplicated = (
        deduplicate_recommendations(
            recommendations
        )
    )

    print(
        "Test 26-A count =",
        len(deduplicated)
    )

    assert len(
        deduplicated
    ) == 1

    merged = deduplicated[0]

    assert set(
        merged.claim_refs
    ) == {
        "cross_analysis:0.claims:0",
        "cross_analysis:1.claims:0",
    }

    assert set(
        merged.evidence_ids
    ) == {
        "news:openai-safety",
        "news:openai-policy",
    }

    assert (
        merged.confidence
        == 0.80
    )

    print(
        "✅ Test 26-A PASSED: "
        "高度相似 Recommendation 成功去重并合并证据链"
    )

    # =========================================
    # Test 26-B
    # 不相似 Recommendation 不应合并
    # =========================================

    recommendations_b = [

        Recommendation(
            name="AI Safety and Governance",
            rationale="Safety.",
            claim_refs=[
                "cross_analysis:0.claims:0"
            ],
            evidence_ids=[
                "news:safety"
            ],
            confidence=0.80,
            priority_score=0.80,
            priority_level="P1",
        ),

        Recommendation(
            name="Mobile App Development",
            rationale="Mobile development.",
            claim_refs=[
                "cross_analysis:1.claims:0"
            ],
            evidence_ids=[
                "news:mobile"
            ],
            confidence=0.70,
            priority_score=0.70,
            priority_level="P2",
        ),
    ]

    deduplicated_b = (
        deduplicate_recommendations(
            recommendations_b
        )
    )

    print(
        "Test 26-B count =",
        len(deduplicated_b)
    )

    assert len(
        deduplicated_b
    ) == 2

    print(
        "✅ Test 26-B PASSED: "
        "不相关 Recommendation 未被错误合并"
    )


    # =========================================
    # Test 26-C
    # 多个高度相似 Recommendation
    # =========================================

    recommendations_c = [

        Recommendation(
            name="AI Safety and Governance",
            rationale="A",
            claim_refs=[
                "cross_analysis:0.claims:0"
            ],
            evidence_ids=[
                "news:a"
            ],
            confidence=0.70,
            priority_score=0.70,
            priority_level="P2",
        ),

        Recommendation(
            name="AI Safety & Governance",
            rationale="B",
            claim_refs=[
                "cross_analysis:1.claims:0"
            ],
            evidence_ids=[
                "news:b"
            ],
            confidence=0.80,
            priority_score=0.80,
            priority_level="P1",
        ),

        Recommendation(
            name="Mobile App Development",
            rationale="C",
            claim_refs=[
                "cross_analysis:2.claims:0"
            ],
            evidence_ids=[
                "news:c"
            ],
            confidence=0.60,
            priority_score=0.60,
            priority_level="P2",
        ),
    ]

    deduplicated_c = (
        deduplicate_recommendations(
            recommendations_c
        )
    )

    print(
        "Test 26-C count =",
        len(deduplicated_c)
    )

    assert len(
        deduplicated_c
    ) == 2

    print(
        "✅ Test 26-C PASSED: "
        "多个重复 Recommendation 正常压缩"
    )

    # =========================================
    # Test 26-D
    # Deduplication 后重新计算
    # Recommendation confidence + priority
    # =========================================

    deduplicated_d = (
        deduplicate_recommendations(
            recommendations_c
        )
    )

    # -----------------------------------------
    # 模拟最终 Claim map
    # -----------------------------------------

    claim_map = {

        "cross_analysis:0.claims:0": Claim(
            statement="Safety claim A",
            claim_type="fact",
            evidence_ids=[
                "news:a"
            ],
            confidence=0.70,
        ),

        "cross_analysis:1.claims:0": Claim(
            statement="Safety claim B",
            claim_type="fact",
            evidence_ids=[
                "news:b"
            ],
            confidence=0.80,
        ),

        "cross_analysis:2.claims:0": Claim(
            statement="Mobile claim",
            claim_type="fact",
            evidence_ids=[
                "news:c"
            ],
            confidence=0.60,
        ),
    }

    # -----------------------------------------
    # Recalculate confidence
    # -----------------------------------------

    for recommendation in deduplicated_d:

        recommendation.confidence = (
            calculate_recommendation_confidence(
                recommendation,
                claim_map,
            )
        )

    # -----------------------------------------
    # Recalculate priority
    # -----------------------------------------

    for recommendation in deduplicated_d:

        recommendation.priority_score = (
            calculate_recommendation_priority(
                recommendation,
                claim_map,
            )
        )

        recommendation.priority_level = (
            get_priority_level(
                recommendation.priority_score
            )
        )

    # -----------------------------------------
    # Find merged recommendation
    # -----------------------------------------

    merged = next(
        item
        for item in deduplicated_d
        if "AI Safety"
        in item.name
    )

    print(
        "Test 26-D merged confidence =",
        merged.confidence
    )

    print(
        "Test 26-D merged priority =",
        merged.priority_score
    )

    print(
        "Test 26-D merged level =",
        merged.priority_level
    )

    assert merged.confidence == 0.75

    assert merged.priority_score == 0.80

    assert merged.priority_level == "P1"

    print(
        "✅ Test 26-D PASSED: "
        "去重后正确重新计算 Confidence 和 Priority"
    )

def test_top_recommendations():

    recommendations = [

        Recommendation(
            name="AI Safety",
            rationale="A",
            claim_refs=[
                "cross_analysis:0.claims:0"
            ],
            evidence_ids=[
                "news:a"
            ],
            confidence=0.90,
            priority_score=0.92,
            priority_level="P1",
        ),

        Recommendation(
            name="Product Security",
            rationale="B",
            claim_refs=[
                "cross_analysis:1.claims:0"
            ],
            evidence_ids=[
                "news:b"
            ],
            confidence=0.80,
            priority_score=0.84,
            priority_level="P1",
        ),

        Recommendation(
            name="AI Policy",
            rationale="C",
            claim_refs=[
                "cross_analysis:2.claims:0"
            ],
            evidence_ids=[
                "news:c"
            ],
            confidence=0.70,
            priority_score=0.75,
            priority_level="P1",
        ),

        Recommendation(
            name="Mobile Security",
            rationale="D",
            claim_refs=[
                "cross_analysis:3.claims:0"
            ],
            evidence_ids=[
                "news:d"
            ],
            confidence=0.70,
            priority_score=0.68,
            priority_level="P2",
        ),

        Recommendation(
            name="AI Product",
            rationale="E",
            claim_refs=[
                "cross_analysis:4.claims:0"
            ],
            evidence_ids=[
                "news:e"
            ],
            confidence=0.60,
            priority_score=0.62,
            priority_level="P2",
        ),

        Recommendation(
            name="AI Ethics",
            rationale="F",
            claim_refs=[
                "cross_analysis:5.claims:0"
            ],
            evidence_ids=[
                "news:f"
            ],
            confidence=0.50,
            priority_score=0.51,
            priority_level="P3",
        ),
    ]

    top = get_top_recommendations(
        recommendations,
        top_k=5,
    )

    print(
        "Test 27-A count =",
        len(top)
    )

    assert len(top) == 5

    assert top[0].name == "AI Safety"
    assert top[1].name == "Product Security"
    assert top[2].name == "AI Policy"
    assert top[3].name == "Mobile Security"
    assert top[4].name == "AI Product"

    assert all(
        item.name != "AI Ethics"
        for item in top
    )

    print(
        "✅ Test 27-A PASSED: "
        "成功选出 Top 5 Recommendations"
    )

    # =========================================
    # Test 27-B
    # Recommendation 少于 5 个
    # =========================================

    small_set = recommendations[:3]

    top_small = get_top_recommendations(
        small_set,
        top_k=5,
    )

    print(
        "Test 27-B count =",
        len(top_small)
    )

    assert len(top_small) == 3

    print(
        "✅ Test 27-B PASSED: "
        "Recommendation 不足 Top 5 时保留全部有效结果"
    )

    # =========================================
    # Test 27-C
    # 相同 priority_score
    # 使用 confidence 决胜
    # =========================================

    tie_recommendations = [

        Recommendation(
            name="Lower Confidence",
            rationale="A",
            claim_refs=[
                "cross_analysis:0.claims:0"
            ],
            evidence_ids=[
                "news:a"
            ],
            confidence=0.60,
            priority_score=0.80,
            priority_level="P1",
        ),

        Recommendation(
            name="Higher Confidence",
            rationale="B",
            claim_refs=[
                "cross_analysis:1.claims:0"
            ],
            evidence_ids=[
                "news:b"
            ],
            confidence=0.90,
            priority_score=0.80,
            priority_level="P1",
        ),
    ]

    top_tie = get_top_recommendations(
        tie_recommendations,
        top_k=2,
    )

    print(
        "Test 27-C order =",
        [
            item.name
            for item in top_tie
        ]
    )

    assert (
        top_tie[0].name
        == "Higher Confidence"
    )

    assert (
        top_tie[1].name
        == "Lower Confidence"
    )

    print(
        "✅ Test 27-C PASSED: "
        "同分时正确使用 confidence 排序"
    )

    print(
        "\n🎉 Test 27 Top Recommendations tests completed."
    )

def collect_recommendations_with_category(
    report: JobAnalysisReport,
) -> list[tuple[Recommendation, RecommendationCategory]]:
    """
    Collect all recommendations together with their category.
    """

    result = []

    for item in report.job_value.job_directions:
        result.append(
            (
                item,
                RecommendationCategory.JOB_DIRECTION
            )
        )

    for item in report.job_value.technical_skills:
        result.append(
            (
                item,
                RecommendationCategory.TECHNICAL_SKILL
            )
        )

    for item in report.job_value.important_areas:
        result.append(
            (
                item,
                RecommendationCategory.IMPORTANT_AREA
            )
        )

    for item in report.interview_preparation.topics:
        result.append(
            (
                item,
                RecommendationCategory.INTERVIEW_TOPIC
            )
        )

    for item in report.interview_preparation.practical_tasks:
        result.append(
            (
                item,
                RecommendationCategory.PRACTICAL_TASK
            )
        )

    return result

def build_top_recommendations(
    report: JobAnalysisReport,
    top_k: int = 5,
) -> list[TopRecommendation]:
    """
    Build Category-aware Top-K recommendations.

    Selection strategy:
    1. Rank all recommendations by priority_score and confidence.
    2. First pass: prefer category diversity.
       Each category contributes at most one recommendation.
    3. Second pass: if fewer than top_k recommendations were selected,
       fill remaining slots according to priority.
    4. Final result is ranked again by priority_score and confidence.

    The function does not recalculate confidence or priority.
    It only selects and converts existing Recommendations.
    """

    items = collect_recommendations_with_category(
        report
    )

    # =========================================
    # Step 0
    # Nothing to select
    # =========================================

    if not items:
        return []

    if top_k <= 0:
        return []

    # =========================================
    # Step 1
    # Global ranking
    #
    # Priority remains the primary ranking signal.
    # Confidence is only the tie-breaker.
    # =========================================

    sorted_items = sorted(
        items,
        key=lambda pair: (
            float(pair[0].priority_score),
            float(pair[0].confidence),
        ),
        reverse=True,
    )

    selected = []

    selected_categories = set()

    selected_names = set()

    # =========================================
    # Step 2
    # Category-aware selection
    #
    # Prefer the first/highest-priority
    # recommendation from each category.
    #
    # A normalized Recommendation name may
    # appear only once in Top-K.
    # =========================================

    for recommendation, category in sorted_items:

        normalized_name = (
            recommendation.name.strip().lower()
        )

        if category in selected_categories:
            continue

        if normalized_name in selected_names:
            continue

        selected.append(
            (
                recommendation,
                category,
            )
        )

        selected_categories.add(
            category
        )

        selected_names.add(
            normalized_name
        )

        if len(selected) >= top_k:
            break

    # =========================================
    # Step 3
    # Priority fallback
    #
    # If the available categories are fewer
    # than top_k, fill remaining slots using
    # the remaining highest-priority items.
    # =========================================

    if len(selected) < top_k:

        for recommendation, category in sorted_items:

            normalized_name = (
                recommendation.name.strip().lower()
            )

            if normalized_name in selected_names:
                continue

            selected.append(
                (
                    recommendation,
                    category,
                )
            )

            selected_names.add(
                normalized_name
            )

            if len(selected) >= top_k:
                break

    # =========================================
    # Step 4
    # Final ranking
    #
    # Category awareness decides which items
    # enter Top-K.
    #
    # Priority decides their final display order.
    # =========================================

    selected = sorted(
        selected,
        key=lambda pair: (
            float(pair[0].priority_score),
            float(pair[0].confidence),
        ),
        reverse=True,
    )

    # =========================================
    # Step 5
    # Convert Recommendation
    # → TopRecommendation
    #
    # Preserve Claim → Evidence chain.
    # =========================================

    result = []

    for recommendation, category in selected:

        result.append(
            TopRecommendation(
                name=recommendation.name,
                rationale=recommendation.rationale,
                category=category,
                claim_refs=list(
                    recommendation.claim_refs
                ),
                evidence_ids=list(
                    recommendation.evidence_ids
                ),
                confidence=recommendation.confidence,
                priority_score=recommendation.priority_score,
                priority_level=recommendation.priority_level,
            )
        )

    return result






def test_top_recommendation_integrity():

    # =========================================
    # Build test report
    # =========================================

    report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI",
            "evidence_ids": [],
        },

        github_signals=[],

        news_signals=[],

        cross_analysis=[
            {
                "relationship": "OpenAI safety focus",

                "evidence": (
                    "OpenAI focuses on AI safety."
                ),

                "evidence_ids": [
                    "news:openai"
                ],

                "claims": [
                    {
                        "statement": (
                            "OpenAI focuses on AI safety."
                        ),

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai"
                        ],

                        "confidence": 0.90,
                    }
                ],
            }
        ],

        job_value={
            "job_directions": [
                {
                    "name": "AI Safety Engineer",

                    "rationale": (
                        "AI safety is an important "
                        "technical direction."
                    ),

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [
                        "news:openai"
                    ],

                    "confidence": 0.90,

                    "priority_score": 0.90,

                    "priority_level": "P1",
                }
            ],

            "technical_skills": [],

            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[
            Evidence(
                source_id="news:openai",
                source_type="news",
                title="OpenAI safety",
                url="https://example.com/openai",
            )
        ],
    )

    # =========================================
    # Build Top Recommendation
    # =========================================

    report.top_recommendations = (
        build_top_recommendations(
            report,
            top_k=5,
        )
    )

    # =========================================
    # Test 32-A
    # =========================================

    validate_top_recommendation_integrity(
        report
    )

    print(
        "✅ Test 32-A PASSED: "
        "合法 TopRecommendation 正确对应 Recommendation"
    )

    # =========================================
    # Test 32-B
    # claim_refs
    # =========================================

    top = report.top_recommendations[0]

    assert (
        top.claim_refs
        == report.job_value.job_directions[0].claim_refs
    )

    print(
        "✅ Test 32-B PASSED: "
        "TopRecommendation.claim_refs 与 Recommendation 一致"
    )

    # =========================================
    # Test 32-C
    # evidence_ids
    # =========================================

    assert (
        top.evidence_ids
        == report.job_value.job_directions[0].evidence_ids
    )

    print(
        "✅ Test 32-C PASSED: "
        "TopRecommendation.evidence_ids 与 Recommendation 一致"
    )

    # =========================================
    # Test 32-D
    # confidence
    # =========================================

    assert (
        top.confidence
        == report.job_value.job_directions[0].confidence
    )

    print(
        "✅ Test 32-D PASSED: "
        "TopRecommendation.confidence 与 Recommendation 一致"
    )

    # =========================================
    # Test 32-E
    # priority_score
    # =========================================

    assert (
        top.priority_score
        == report.job_value.job_directions[0].priority_score
    )

    print(
        "✅ Test 32-E PASSED: "
        "TopRecommendation.priority_score 与 Recommendation 一致"
    )

    # =========================================
    # Test 32-F
    # priority_level
    # =========================================

    assert (
        top.priority_level
        == report.job_value.job_directions[0].priority_level
    )

    print(
        "✅ Test 32-F PASSED: "
        "TopRecommendation.priority_level 与 Recommendation 一致"
    )

    # =========================================
    # Test 32-G
    # Fake TopRecommendation
    # =========================================

    fake = top.model_copy(
        deep=True
    )

    fake.name = "Fake Recommendation"

    report.top_recommendations = [
        fake
    ]

    try:

        validate_top_recommendation_integrity(
            report
        )

        print(
            "❌ Test 32-G FAILED: "
            "伪造 TopRecommendation 没有被拦截"
        )

    except ValueError as e:

        print(
            "✅ Test 32-G PASSED: "
            "成功拦截不存在的 TopRecommendation"
        )

        print(
            f"   {e}"
        )

    print(
        "\n🎉 Test 32 Top Recommendation "
        "integrity tests completed."
    )



def test_top_recommendation_categories():

    report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI",
            "evidence_ids": [],
        },

        github_signals=[],

        news_signals=[],

        cross_analysis=[],

        job_value={
            "job_directions": [
                {
                    "name": "AI Safety",
                    "rationale": "Safety",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:a"
                    ],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [
                {
                    "name": "Model Evaluation",
                    "rationale": "Evaluation",
                    "claim_refs": [
                        "cross_analysis:1.claims:0"
                    ],
                    "evidence_ids": [
                        "news:b"
                    ],
                    "confidence": 0.80,
                    "priority_score": 0.80,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [],

        },

        interview_preparation={
            "topics": [
                {
                    "name": "AI Safety Interview",
                    "rationale": "Interview topic",
                    "claim_refs": [
                        "cross_analysis:2.claims:0"
                    ],
                    "evidence_ids": [
                        "news:c"
                    ],
                    "confidence": 0.70,
                    "priority_score": 0.70,
                    "priority_level": "P2",
                }
            ],

            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    top = build_top_recommendations(
        report,
        top_k=3,
    )

    print(
        "Test 28-A categories =",
        [
            item.category.value
            for item in top
        ]
    )

    assert top[0].category == (
        RecommendationCategory.JOB_DIRECTION
    )

    assert top[1].category == (
        RecommendationCategory.TECHNICAL_SKILL
    )

    assert top[2].category == (
        RecommendationCategory.INTERVIEW_TOPIC
    )

    print(
        "✅ Test 28-A PASSED: "
        "Top Recommendation category 正确映射"
    )

    # =========================================
    # Test 28-B
    # Top-K 保留 category / claim / evidence
    # =========================================

    assert top[0].name == "AI Safety"

    assert top[0].claim_refs == [
        "cross_analysis:0.claims:0"
    ]

    assert top[0].evidence_ids == [
        "news:a"
    ]

    assert top[0].priority_score == 0.90

    assert top[0].priority_level == "P1"

    print(
        "✅ Test 28-B PASSED: "
        "Top Recommendation 保留完整证据链"
    )

    # =========================================
    # Test 28-C
    # Recommendation 少于 Top 5
    # =========================================

    top_small = build_top_recommendations(
        report,
        top_k=5,
    )

    assert len(top_small) == 3

    print(
        "✅ Test 28-C PASSED: "
        "不足 Top 5 时保留全部有效 Recommendation"
    )

    print(
        "\n🎉 Test 28 Top Recommendation "
        "category tests completed."
    )

def test_category_aware_top_recommendations():
    """
    V2.6.2 Test 31

    Test 31-A:
    Category diversity should be preferred
    when enough categories are available.
    """

    report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI",
            "evidence_ids": [],
        },

        github_signals=[],

        news_signals=[],

        cross_analysis=[],

        job_value={
            "job_directions": [
                {
                    "name": "AI Safety",
                    "rationale": "Safety direction.",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:a"
                    ],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                },

                {
                    "name": "Developer Relations",
                    "rationale": "Developer ecosystem.",
                    "claim_refs": [
                        "cross_analysis:1.claims:0"
                    ],
                    "evidence_ids": [
                        "news:b"
                    ],
                    "confidence": 0.89,
                    "priority_score": 0.89,
                    "priority_level": "P1",
                },
            ],

            "technical_skills": [
                {
                    "name": "API Development",
                    "rationale": "API engineering.",
                    "claim_refs": [
                        "cross_analysis:2.claims:0"
                    ],
                    "evidence_ids": [
                        "news:c"
                    ],
                    "confidence": 0.88,
                    "priority_score": 0.88,
                    "priority_level": "P1",
                },
            ],

            "important_areas": [
                {
                    "name": "AI Ecosystem",
                    "rationale": "Ecosystem growth.",
                    "claim_refs": [
                        "cross_analysis:3.claims:0"
                    ],
                    "evidence_ids": [
                        "news:d"
                    ],
                    "confidence": 0.87,
                    "priority_score": 0.87,
                    "priority_level": "P1",
                },
            ],
        },

        interview_preparation={
            "topics": [
                {
                    "name": "AI Interview",
                    "rationale": "Interview preparation.",
                    "claim_refs": [
                        "cross_analysis:4.claims:0"
                    ],
                    "evidence_ids": [
                        "news:e"
                    ],
                    "confidence": 0.86,
                    "priority_score": 0.86,
                    "priority_level": "P2",
                },
            ],

            "practical_tasks": [
                {
                    "name": "Build AI Application",
                    "rationale": "Practical task.",
                    "claim_refs": [
                        "cross_analysis:5.claims:0"
                    ],
                    "evidence_ids": [
                        "news:f"
                    ],
                    "confidence": 0.85,
                    "priority_score": 0.85,
                    "priority_level": "P2",
                },
            ],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    top = build_top_recommendations(
        report,
        top_k=5,
    )

    print(
        "\n========== Test 31-A =========="
    )

    print(
        "Top count =",
        len(top)
    )

    for index, item in enumerate(top, start=1):
        print(
            f"#{index}",
            item.name,
            "| category =",
            item.category.value,
            "| priority =",
            item.priority_score,
        )

    # =========================================
    # Count
    # =========================================

    assert len(top) == 5

    # =========================================
    # Category diversity
    # =========================================

    categories = {
        item.category
        for item in top
    }

    assert categories == {
        RecommendationCategory.JOB_DIRECTION,
        RecommendationCategory.TECHNICAL_SKILL,
        RecommendationCategory.IMPORTANT_AREA,
        RecommendationCategory.INTERVIEW_TOPIC,
        RecommendationCategory.PRACTICAL_TASK,
    }

    # =========================================
    # Duplicate category with lower priority
    # should not enter Top 5
    # =========================================

    assert all(
        item.name != "Developer Relations"
        for item in top
    )

    # =========================================
    # Final priority ordering
    # =========================================

    scores = [
        float(item.priority_score)
        for item in top
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    # =========================================
    # Claim → Evidence chain
    # =========================================

    for item in top:

        assert item.claim_refs
        assert item.evidence_ids

    print(
        "✅ Test 31-A PASSED: "
        "Category-aware Top 5 正确覆盖五类 Category"
    )

def test_category_aware_top_recommendations_edge_cases():

    # =========================================
    # Test 31-B
    # Only three categories available
    # =========================================

    report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI",
            "evidence_ids": [],
        },

        github_signals=[],
        news_signals=[],
        cross_analysis=[],

        job_value={
            "job_directions": [
                {
                    "name": "Job Direction A",
                    "rationale": "A",
                    "claim_refs": ["claim:a"],
                    "evidence_ids": ["news:a"],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                },
                {
                    "name": "Job Direction B",
                    "rationale": "B",
                    "claim_refs": ["claim:b"],
                    "evidence_ids": ["news:b"],
                    "confidence": 0.89,
                    "priority_score": 0.89,
                    "priority_level": "P1",
                },
            ],

            "technical_skills": [
                {
                    "name": "Technical Skill A",
                    "rationale": "C",
                    "claim_refs": ["claim:c"],
                    "evidence_ids": ["news:c"],
                    "confidence": 0.88,
                    "priority_score": 0.88,
                    "priority_level": "P1",
                },
                {
                    "name": "Technical Skill B",
                    "rationale": "D",
                    "claim_refs": ["claim:d"],
                    "evidence_ids": ["news:d"],
                    "confidence": 0.87,
                    "priority_score": 0.87,
                    "priority_level": "P1",
                },
            ],

            "important_areas": [
                {
                    "name": "Important Area",
                    "rationale": "E",
                    "claim_refs": ["claim:e"],
                    "evidence_ids": ["news:e"],
                    "confidence": 0.86,
                    "priority_score": 0.86,
                    "priority_level": "P2",
                },
            ],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    top = build_top_recommendations(
        report,
        top_k=5,
    )

    print(
        "\n========== Test 31-B =========="
    )

    print(
        "Top count =",
        len(top)
    )

    for index, item in enumerate(top, start=1):
        print(
            f"#{index}",
            item.name,
            "| category =",
            item.category.value,
            "| priority =",
            item.priority_score,
        )

    # Must still return five recommendations
    assert len(top) == 5

    # Only three real categories exist
    categories = {
        item.category
        for item in top
    }

    assert categories == {
        RecommendationCategory.JOB_DIRECTION,
        RecommendationCategory.TECHNICAL_SKILL,
        RecommendationCategory.IMPORTANT_AREA,
    }

    # No fake category
    assert all(
        item.category != RecommendationCategory.INTERVIEW_TOPIC
        for item in top
    )

    assert all(
        item.category != RecommendationCategory.PRACTICAL_TASK
        for item in top
    )

    # Final ranking remains priority-descending
    scores = [
        float(item.priority_score)
        for item in top
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    print(
        "✅ Test 31-B PASSED: "
        "Category 不足时正确使用 Priority fallback"
    )

def test_category_aware_top_recommendations_single_category():

    # =========================================
    # Test 31-C
    # Only one category exists
    # =========================================

    report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI",
            "evidence_ids": [],
        },

        github_signals=[],
        news_signals=[],
        cross_analysis=[],

        job_value={
            "job_directions": [
                {
                    "name": "Direction A",
                    "rationale": "A",
                    "claim_refs": ["claim:a"],
                    "evidence_ids": ["news:a"],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                },
                {
                    "name": "Direction B",
                    "rationale": "B",
                    "claim_refs": ["claim:b"],
                    "evidence_ids": ["news:b"],
                    "confidence": 0.89,
                    "priority_score": 0.89,
                    "priority_level": "P1",
                },
                {
                    "name": "Direction C",
                    "rationale": "C",
                    "claim_refs": ["claim:c"],
                    "evidence_ids": ["news:c"],
                    "confidence": 0.88,
                    "priority_score": 0.88,
                    "priority_level": "P1",
                },
                {
                    "name": "Direction D",
                    "rationale": "D",
                    "claim_refs": ["claim:d"],
                    "evidence_ids": ["news:d"],
                    "confidence": 0.87,
                    "priority_score": 0.87,
                    "priority_level": "P2",
                },
                {
                    "name": "Direction E",
                    "rationale": "E",
                    "claim_refs": ["claim:e"],
                    "evidence_ids": ["news:e"],
                    "confidence": 0.86,
                    "priority_score": 0.86,
                    "priority_level": "P2",
                },
            ],

            "technical_skills": [],
            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    top = build_top_recommendations(
        report,
        top_k=5,
    )

    print(
        "\n========== Test 31-C =========="
    )

    print(
        "Top count =",
        len(top)
    )

    for index, item in enumerate(top, start=1):
        print(
            f"#{index}",
            item.name,
            "| category =",
            item.category.value,
            "| priority =",
            item.priority_score,
        )

    assert len(top) == 5

    assert all(
        item.category
        == RecommendationCategory.JOB_DIRECTION
        for item in top
    )

    expected_names = [
        "Direction A",
        "Direction B",
        "Direction C",
        "Direction D",
        "Direction E",
    ]

    assert [
        item.name
        for item in top
    ] == expected_names

    scores = [
        float(item.priority_score)
        for item in top
    ]

    assert scores == [
        0.90,
        0.89,
        0.88,
        0.87,
        0.86,
    ]

    # Evidence chain must survive
    for item in top:
        assert item.claim_refs
        assert item.evidence_ids

    print(
        "✅ Test 31-C PASSED: "
        "单一 Category 时正确回退到 Priority Top 5"
    )

def apply_recommendation_deduplication(
    report: JobAnalysisReport,
) -> None:
    """
    Deduplicate recommendations in-place
    across each recommendation category.
    """

    report.job_value.job_directions = (
        deduplicate_recommendations(
            report.job_value.job_directions
        )
    )

    report.job_value.technical_skills = (
        deduplicate_recommendations(
            report.job_value.technical_skills
        )
    )

    report.job_value.important_areas = (
        deduplicate_recommendations(
            report.job_value.important_areas
        )
    )

    report.interview_preparation.topics = (
        deduplicate_recommendations(
            report.interview_preparation.topics
        )
    )

    report.interview_preparation.practical_tasks = (
        deduplicate_recommendations(
            report.interview_preparation.practical_tasks
        )
    )

def test_build_top_recommendations():

    report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI",
            "evidence_ids": [],
        },

        github_signals=[],

        news_signals=[],

        cross_analysis=[
            {
                "relationship": "Safety",
                "evidence": "Safety evidence",
                "evidence_ids": [
                    "news:a"
                ],
                "claims": [
                    {
                        "statement": "Safety claim",
                        "claim_type": "fact",
                        "evidence_ids": [
                            "news:a"
                        ],
                        "confidence": 0.90,
                    }
                ],
            }
        ],

        job_value={
            "job_directions": [
                {
                    "name": "AI Safety and Governance",
                    "rationale": "Safety.",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:a"
                    ],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                },

                {
                    "name": "AI Safety & Governance",
                    "rationale": "Duplicate safety.",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:a"
                    ],
                    "confidence": 0.80,
                    "priority_score": 0.80,
                    "priority_level": "P1",
                },
            ],

            "technical_skills": [
                {
                    "name": "Model Evaluation",
                    "rationale": "Evaluation.",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:a"
                    ],
                    "confidence": 0.70,
                    "priority_score": 0.70,
                    "priority_level": "P2",
                }
            ],

            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    # -----------------------------------------
    # Dedup
    # -----------------------------------------

    apply_recommendation_deduplication(
        report
    )

    # -----------------------------------------
    # Confidence
    # -----------------------------------------

    apply_recommendation_confidence_scores(
        report
    )

    # -----------------------------------------
    # Priority
    # -----------------------------------------

    apply_recommendation_priority(
        report
    )

    # -----------------------------------------
    # Top 5
    # -----------------------------------------

    report.top_recommendations = (
        build_top_recommendations(
            report,
            top_k=5
        )
    )

    print(
        "Test 29 count =",
        len(
            report.top_recommendations
        )
    )

    print(
        "Test 29 names =",
        [
            item.name
            for item in (
                report.top_recommendations
            )
        ]
    )

    # Duplicate should be removed
    assert len(
        report.top_recommendations
    ) == 2

    assert (
        report.top_recommendations[0].name
        == "AI Safety and Governance"
    )

    assert (
        report.top_recommendations[1].name
        == "Model Evaluation"
    )

    # Category check
    assert (
        report.top_recommendations[0].category
        == RecommendationCategory.JOB_DIRECTION
    )

    assert (
        report.top_recommendations[1].category
        == RecommendationCategory.TECHNICAL_SKILL
    )

    print(
        "✅ Test 29 PASSED: "
        "Dedup → Confidence → Priority → Top Recommendations "
        "链路正常"
    )

def test_top_recommendation_consistency():

    report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI",
            "evidence_ids": [],
        },

        github_signals=[],
        news_signals=[],
        cross_analysis=[],

        job_value={
            "job_directions": [
                {
                    "name": "AI Safety",
                    "rationale": "Safety",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:safety"
                    ],
                    "confidence": 0.72,
                    "priority_score": 0.78,
                    "priority_level": "P2",
                }
            ],
            "technical_skills": [],
            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        top_recommendations=[],

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    report.top_recommendations = (
        build_top_recommendations(
            report,
            top_k=5
        )
    )

    source = (
        report.job_value.job_directions[0]
    )

    top = (
        report.top_recommendations[0]
    )

    print(
        "Source confidence =",
        source.confidence
    )

    print(
        "Top confidence =",
        top.confidence
    )

    print(
        "Source priority =",
        source.priority_score
    )

    print(
        "Top priority =",
        top.priority_score
    )

    print(
        "Source level =",
        source.priority_level
    )

    print(
        "Top level =",
        top.priority_level
    )

    assert (
        top.name
        == source.name
    )

    assert (
        top.confidence
        == source.confidence
    )

    assert (
        top.priority_score
        == source.priority_score
    )

    assert (
        top.priority_level
        == source.priority_level
    )

    assert (
        top.claim_refs
        == source.claim_refs
    )

    assert (
        top.evidence_ids
        == source.evidence_ids
    )

    print(
        "✅ Test 30 PASSED: "
        "Top Recommendation 与最终 Recommendation 保持一致"
    )

def test_recommendation_evidence_derivation():

    evidence = [
        Evidence(
            source_id="news:openai",
            source_type="news",
            title="OpenAI pricing",
            url="https://example.com/openai",
        ),
        Evidence(
            source_id="news:anthropic",
            source_type="news",
            title="Anthropic model",
            url="https://example.com/anthropic",
        ),
    ]

    report = JobAnalysisReport(

        evidence=[
            Evidence(
                source_id="news:openai",
                source_type="news",
                title="OpenAI pricing change",
                url="https://example.com/openai",
            ),

            Evidence(
                source_id="news:anthropic",
                source_type="news",
                title="Anthropic launches new model",
                url="https://example.com/anthropic",
            ),
        ],

        company_overview={
            "summary": "OpenAI faces competitive pressure.",
            "evidence_ids": [
                "news:openai"
            ],
        },

        github_signals=[],

        news_signals=[],

        cross_analysis=[
            {
                "relationship": (
                    "OpenAI faces competitive pressure."
                ),

                "evidence": (
                    "OpenAI pricing and Anthropic "
                    "model signals."
                ),

                "evidence_ids": [
                    "news:openai",
                    "news:anthropic",
                ],

                "claims": [
                    {
                        "statement": (
                            "OpenAI faces competitive "
                            "pressure."
                        ),

                        "claim_type": "inference",

                        "evidence_ids": [
                            "news:openai",
                            "news:anthropic",
                        ],

                        "confidence": 0.9,
                    }
                ],
            }
        ],

        job_value={
            "job_directions": [
                {
                    "name": "AI Engineer",

                    "rationale": (
                        "Competitive AI development "
                        "is important."
                    ),

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [
                        "news:openai",
                        "news:anthropic",
                    ],

                    "confidence": 0.9,

                    "priority_score": 0.9,

                    "priority_level": "P1",
                }
            ],

            "technical_skills": [],

            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },
    )

    # =========================================
    # Build Claim map
    # =========================================

    claim_map = {}

    for analysis_index, analysis in enumerate(
        report.cross_analysis
    ):

        for claim_index, claim in enumerate(
            analysis.claims
        ):

            claim_ref = (
                f"cross_analysis:"
                f"{analysis_index}"
                f".claims:"
                f"{claim_index}"
            )

            claim_map[claim_ref] = claim

    recommendation = (
        report.job_value.job_directions[0]
    )

    # =========================================
    # Test 18-A
    #
    # Derive evidence from claim_refs
    # =========================================

    derived = (
        derive_recommendation_evidence_ids(
            recommendation,
            claim_map,
        )
    )

    assert set(derived) == {
        "news:openai",
        "news:anthropic",
    }

    print(
        "✅ Test 18-A PASSED: "
        "Recommendation evidence_ids "
        "可以从 claim_refs 自动派生"
    )

    # =========================================
    # Test 18-B
    #
    # Recommendation 自己少写一个 evidence
    # validator 应该发现
    # =========================================

    recommendation.evidence_ids = [
        "news:openai"
    ]

    derived = (
        derive_recommendation_evidence_ids(
            recommendation,
            claim_map,
        )
    )

    assert set(
        recommendation.evidence_ids
    ) != set(derived)

    print(
        "✅ Test 18-B PASSED: "
        "Recommendation 自己声明的 "
        "evidence_ids 无法改变派生结果"
    )

    # =========================================
    # Test 18-C
    #
    # Fake Recommendation evidence_ids
    # must not affect the derived result.
    # =========================================

    report.job_value.job_directions[0].evidence_ids = [
        "news:fake"
    ]

    validate_recommendation_evidence_support(
        report,
        evidence,
    )

    actual_evidence_ids = (
        report.job_value.job_directions[0].evidence_ids
    )

    expected_evidence_ids = [
        "news:openai",
        "news:anthropic",
    ]

    if actual_evidence_ids == expected_evidence_ids:

        print(
            "✅ Test 18-C PASSED: "
            "伪造 Recommendation evidence_ids "
            "被自动覆盖，无法影响最终结果"
        )

    else:

        print(
            "❌ Test 18-C FAILED: "
            "Recommendation evidence_ids "
            "没有被正确重新派生"
        )

        print(
            f"   actual={actual_evidence_ids}"
        )

        print(
            f"   expected={expected_evidence_ids}"
        )

    # =========================================
    # Test 18-D
    #
    # invalid claim_ref
    # =========================================

    recommendation.claim_refs = [
        "cross_analysis:0.claims:999"
    ]

    try:

        derive_recommendation_evidence_ids(
            recommendation,
            claim_map,
        )

        print(
            "❌ Test 18-D FAILED: "
            "invalid claim_ref 没有被拦截"
        )

    except ValueError as e:

        print(
            "✅ Test 18-D PASSED: "
            "invalid claim_ref 被成功拦截"
        )

        print(f"   {e}")

    print(
        "\n🎉 Test 18 recommendation evidence "
        "derivation tests completed."
    )

def test_full_recommendation_pipeline_integrity():

    evidence = [
        Evidence(
            source_id="news:openai",
            source_type="news",
            title="OpenAI improves AI safety",
            url="https://example.com/openai",
            description="OpenAI announced improvements in AI safety.",
            snippet="OpenAI focuses on AI safety.",
        ),
        Evidence(
            source_id="news:openai-api",
            source_type="news",
            title="OpenAI improves API",
            url="https://example.com/openai-api",
            description="OpenAI improved its API platform.",
            snippet="OpenAI improved API capabilities.",
        ),
    ]

    report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI focuses on AI safety and API development.",
            "evidence_ids": [
                "news:openai",
                "news:openai-api",
            ],
        },

        github_signals=[],

        news_signals=[],

        cross_analysis=[
            {
                "relationship": "OpenAI AI safety focus",

                "evidence": (
                    "OpenAI focuses on AI safety."
                ),

                "evidence_ids": [
                    "news:openai"
                ],

                "claims": [
                    {
                        "statement": (
                            "OpenAI focuses on AI safety."
                        ),

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai"
                        ],

                        "confidence": 0.10,
                    }
                ],
            },

            {
                "relationship": "OpenAI API development",

                "evidence": (
                    "OpenAI improved API capabilities."
                ),

                "evidence_ids": [
                    "news:openai-api"
                ],

                "claims": [
                    {
                        "statement": (
                            "OpenAI improved API capabilities."
                        ),

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-api"
                        ],

                        "confidence": 0.20,
                    }
                ],
            },
        ],

        job_value={
            "job_directions": [
                {
                    "name": "AI Safety Engineer",

                    "rationale": (
                        "AI safety is a relevant "
                        "technical direction."
                    ),

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [
                        "news:openai"
                    ],

                    "confidence": 0.01,

                    "priority_score": 0.01,

                    "priority_level": "P3",
                }
            ],

            "technical_skills": [
                {
                    "name": "API Development",

                    "rationale": (
                        "API development is an important "
                        "technical skill."
                    ),

                    "claim_refs": [
                        "cross_analysis:1.claims:0"
                    ],

                    "evidence_ids": [
                        "news:openai-api"
                    ],

                    "confidence": 0.01,

                    "priority_score": 0.01,

                    "priority_level": "P3",
                }
            ],

            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=evidence,
    )

    # =========================================
    # 1. Validate Claims
    # =========================================

    validate_claims(
        report,
        evidence
    )

    # =========================================
    # 2. Validate Evidence Support
    # =========================================

    validate_evidence_content_support(
        report,
        evidence
    )

    # =========================================
    # 3. Derive Recommendation Evidence
    # =========================================

    validate_recommendation_evidence_support(
        report,
        evidence
    )

    # =========================================
    # 4. Calculate Claim Confidence
    # =========================================

    apply_confidence_scores(
        report,
        evidence
    )

    # =========================================
    # 5. Deduplicate
    # =========================================

    apply_recommendation_deduplication(
        report
    )

    # =========================================
    # 6. Recommendation Confidence
    # =========================================

    apply_recommendation_confidence_scores(
        report
    )

    # =========================================
    # 7. Recommendation Priority
    # =========================================

    apply_recommendation_priority(
        report
    )

    # =========================================
    # 8. Build Top 5
    # =========================================

    report.top_recommendations = (
        build_top_recommendations(
            report,
            top_k=5
        )
    )

    # =========================================
    # 9. Integrity
    # =========================================

    validate_top_recommendation_integrity(
        report
    )

    print(
        "\n========== Test 33 =========="
    )

    print(
        "Recommendations:"
    )

    for group_name, group in [
        (
            "job_directions",
            report.job_value.job_directions
        ),
        (
            "technical_skills",
            report.job_value.technical_skills
        ),
        (
            "important_areas",
            report.job_value.important_areas
        ),
        (
            "interview_topics",
            report.interview_preparation.topics
        ),
        (
            "practical_tasks",
            report.interview_preparation.practical_tasks
        ),
    ]:

        for item in group:

            print(
                group_name,
                "|",
                item.name,
                "| confidence =",
                item.confidence,
                "| priority =",
                item.priority_score,
                "| level =",
                item.priority_level,
                "| evidence =",
                item.evidence_ids,
            )

    print(
        "\nTop Recommendations:"
    )

    for index, item in enumerate(
        report.top_recommendations,
        start=1
    ):

        print(
            f"#{index}",
            item.name,
            "| category =",
            item.category.value,
            "| confidence =",
            item.confidence,
            "| priority =",
            item.priority_score,
            "| level =",
            item.priority_level,
            "| claims =",
            item.claim_refs,
            "| evidence =",
            item.evidence_ids,
        )

        print(
            "✅ Test 33 PASSED: "
            "Full Recommendation Pipeline Integrity"
        )



def test_32_a_ecosystem_cannot_be_company_fact():

    report = {
        "github_signals": [
            {
                "project": "AutoGPT",
                "language": "Python",
                "technical_direction": "AI Agent",
                "activity": "active",
                "evidence": "AutoGPT is an open-source autonomous agent project.",
                "relevance": "ecosystem",
                "source_id": "github:Significant-Gravitas/AutoGPT",
            }
        ],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "OpenAI technical direction",
                "evidence": "OpenAI focuses on autonomous AI agents.",
                "evidence_ids": [
                    "github:Significant-Gravitas/AutoGPT"
                ],
                "claims": [
                    {
                        "statement": "OpenAI develops autonomous AI agents.",
                        "claim_type": "fact",
                        "evidence_ids": [
                            "github:Significant-Gravitas/AutoGPT"
                        ],
                        "confidence": 0.9,
                    }
                ],
            }
        ],
    }

    evidence = [
        type(
            "Evidence",
            (),
            {
                "source_id":
                    "github:Significant-Gravitas/AutoGPT"
            },
        )()
    ]

    try:

        validate_cross_analysis_evidence(
            report,
            evidence,
            "OpenAI",
        )

    except ValueError as e:

        print(
            "TEST 32-A PASS"
        )

        print(
            "Expected validation error:"
        )

        print(e)

        return

    raise AssertionError(
        "TEST 32-A FAILED: "
        "ecosystem evidence was incorrectly "
        "accepted as direct company fact."
    )

def test_32_b_ecosystem_industry_claim_allowed():

    report = {
        "github_signals": [
            {
                "project": "AutoGPT",
                "language": "Python",
                "technical_direction": "AI Agent",
                "activity": "active",
                "evidence": "AutoGPT is an open-source autonomous agent project.",
                "relevance": "ecosystem",
                "source_id": "github:Significant-Gravitas/AutoGPT",
            }
        ],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship":
                    "Broader AI ecosystem shows activity "
                    "in autonomous-agent frameworks",

                "evidence":
                    "The broader AI ecosystem shows activity "
                    "in autonomous-agent frameworks.",

                "evidence_ids": [
                    "github:Significant-Gravitas/AutoGPT"
                ],

                "claims": [
                    {
                        "statement":
                            "The broader AI ecosystem shows "
                            "activity in autonomous-agent frameworks.",

                        "claim_type": "industry",

                        "evidence_ids": [
                            "github:Significant-Gravitas/AutoGPT"
                        ],

                        "confidence": 0.8,
                    }
                ],
            }
        ],
    }

    evidence = [
        type(
            "Evidence",
            (),
            {
                "source_id":
                    "github:Significant-Gravitas/AutoGPT"
            },
        )()
    ]

    result = validate_cross_analysis_evidence(
        report,
        evidence,
        "OpenAI",
    )

    assert result is True

    print(
        "TEST 32-B PASS"
    )

def test_32_c_direct_company_fact_allowed():

    report = {
        "github_signals": [],

        "news_signals": [
            {
                "title": "OpenAI example",
                "source": "Example",
                "published_at": "2026-08-25",
                "summary": "OpenAI announced a new product.",
                "implication": "Direct company activity.",
                "evidence":
                    "OpenAI announced a new product.",
                "relevance": "direct",
                "source_id": "news:openai-test",
            }
        ],

        "cross_analysis": [
            {
                "relationship":
                    "OpenAI product activity",

                "evidence":
                    "OpenAI announced a new product.",

                "evidence_ids": [
                    "news:openai-test"
                ],

                "claims": [
                    {
                        "statement":
                            "OpenAI announced a new product.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-test"
                        ],

                        "confidence": 0.95,
                    }
                ],
            }
        ],
    }

    evidence = [
        type(
            "Evidence",
            (),
            {
                "source_id":
                    "news:openai-test"
            },
        )()
    ]

    result = validate_cross_analysis_evidence(
        report,
        evidence,
        "OpenAI",
    )

    assert result is True

    print(
        "TEST 32-C PASS"
    )

def test_33_a_valid_recommendation_chain():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI announced a new product.",
            "evidence_ids": [
                "news:openai-test"
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "OpenAI product activity",

                "evidence": "OpenAI announced a new product.",

                "evidence_ids": [
                    "news:openai-test"
                ],

                "claims": [
                    {
                        "statement":
                            "OpenAI announced a new product.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-test"
                        ],

                        "confidence": 0.95
                    }
                ]
            }
        ],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Product Development",

                    "rationale":
                        "The product activity provides evidence "
                        "for AI product development relevance.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [],

                    "confidence": 0.9
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

                "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:openai-test",
                "source_type": "news",
                "title": "OpenAI Test",
                "url": "https://example.com/openai-test"
            }
        ]
    })

    evidence = [
        Evidence(
            source_id="news:openai-test",
            source_type="news",
            title="OpenAI Test",
            url="https://example.com/openai-test"
        )
    ]

    result = validate_recommendation_evidence_support(
        report,
        evidence
    )

    assert result is True

    assert (
        report.job_value
        .job_directions[0]
        .evidence_ids
        == ["news:openai-test"]
    )

    print("TEST 33-A PASS")

def test_33_b_invalid_claim_ref():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "OpenAI activity",

                "evidence": "OpenAI activity.",

                "evidence_ids": [
                    "news:openai-test"
                ],

                "claims": [
                    {
                        "statement": "OpenAI activity.",
                        "claim_type": "fact",
                        "evidence_ids": [
                            "news:openai-test"
                        ],
                        "confidence": 0.9
                    }
                ]
            }
        ],

        "job_value": {
            "job_directions": [
                {
                    "name": "Invalid Recommendation",

                    "rationale": "Test.",

                    "claim_refs": [
                        "cross_analysis:99.claims:99"
                    ],

                    "evidence_ids": [],

                    "confidence": 0.8
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

                "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:openai-test",
                "source_type": "news",
                "title": "OpenAI Test",
                "url": "https://example.com/openai-test"
            }
        ]
    })

    evidence = [
        Evidence(
            source_id="news:openai-test",
            source_type="news",
            title="OpenAI Test",
            url="https://example.com/openai-test"
        )
    ]

    try:

        validate_recommendation_evidence_support(
            report,
            evidence
        )

    except ValueError as e:

        print("TEST 33-B PASS")
        print("Expected validation error:")
        print(e)

        return

    raise AssertionError(
        "TEST 33-B FAILED: "
        "invalid claim_ref was accepted."
    )

def test_33_c_claim_without_evidence():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "OpenAI activity",

                "evidence": "OpenAI activity.",

                "evidence_ids": [],

                "claims": [
                    {
                        "statement":
                            "OpenAI has a certain technical direction.",

                        "claim_type": "inference",

                        "evidence_ids": [],

                        "confidence": 0.8
                    }
                ]
            }
        ],

        "job_value": {
            "job_directions": [
                {
                    "name": "Unsupported Recommendation",

                    "rationale":
                        "This should not be accepted.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [],

                    "confidence": 0.8
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    evidence = []

    try:

        validate_recommendation_evidence_support(
            report,
            evidence
        )

    except ValueError as e:

        print("TEST 33-C PASS")
        print("Expected validation error:")
        print(e)

        return

    raise AssertionError(
        "TEST 33-C FAILED: "
        "Recommendation was accepted without "
        "Claim evidence."
    )

def test_34_a_valid_recommendation_claim_support():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI develops AI products.",
            "evidence_ids": [
                "news:openai-product"
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship":
                    "OpenAI product development activity",

                "evidence":
                    "OpenAI announced a new AI product.",

                "evidence_ids": [
                    "news:openai-product"
                ],

                "claims": [
                    {
                        "statement":
                            "OpenAI announced a new AI product.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-product"
                        ],

                        "confidence": 0.95
                    }
                ]
            }
        ],

        "job_value": {
            "job_directions": [
                {
                    "name":
                        "AI Product Development",

                    "rationale":
                        "The product activity provides "
                        "evidence for AI product development.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [],

                    "confidence": 0.9
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:openai-product",
                "source_type": "news",
                "title": "OpenAI Product Announcement",
                "url": "https://example.com"
            }
        ]
    })

    result = validate_recommendation_claim_support(
        report
    )

    assert result is True

    print("TEST 34-A PASS")


def test_34_b_unrelated_claim_rejected():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI develops AI products.",
            "evidence_ids": [
                "news:openai-product"
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship":
                    "OpenAI product activity",

                "evidence":
                    "OpenAI announced a new AI product.",

                "evidence_ids": [
                    "news:openai-product"
                ],

                "claims": [
                    {
                        "statement":
                            "OpenAI announced a new AI product.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-product"
                        ],

                        "confidence": 0.95
                    }
                ]
            }
        ],

        "job_value": {
            "job_directions": [
                {
                    "name":
                        "GPU Hardware Engineering",

                    "rationale":
                        "The recommendation concerns "
                        "GPU hardware engineering.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [],

                    "confidence": 0.8
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:openai-product",
                "source_type": "news",
                "title": "OpenAI Product Announcement",
                "url": "https://example.com"
            }
        ]
    })

    try:

        validate_recommendation_claim_support(
            report
        )

    except ValueError as e:

        print("TEST 34-B PASS")
        print("Expected validation error:")
        print(e)

        return

    raise AssertionError(
        "TEST 34-B FAILED: "
        "unrelated Claim was accepted."
    )

def test_34_c_multi_claim_recommendation_allowed():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary":
                "OpenAI is active in AI product development.",
            "evidence_ids": [
                "news:openai-product",
                "news:openai-engineering"
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship":
                    "OpenAI product and engineering activity",

                "evidence":
                    "OpenAI announced a new AI product "
                    "and expanded engineering activity.",

                "evidence_ids": [
                    "news:openai-product",
                    "news:openai-engineering"
                ],

                "claims": [
                    {
                        "statement":
                            "OpenAI announced a new AI product.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-product"
                        ],

                        "confidence": 0.95
                    },
                    {
                        "statement":
                            "OpenAI expanded engineering activity.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-engineering"
                        ],

                        "confidence": 0.9
                    }
                ]
            }
        ],

        "job_value": {
            "job_directions": [
                {
                    "name":
                        "AI Product Engineering",

                    "rationale":
                        "Product development and engineering "
                        "activity both support this direction.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0",
                        "cross_analysis:0.claims:1"
                    ],

                    "evidence_ids": [],

                    "confidence": 0.9
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:openai-product",
                "source_type": "news",
                "title": "OpenAI Product Announcement",
                "url": "https://example.com/product"
            },
            {
                "source_id": "news:openai-engineering",
                "source_type": "news",
                "title": "OpenAI Engineering Activity",
                "url": "https://example.com/engineering"
            }
        ]
    })

    result = validate_recommendation_claim_support(
        report
    )

    assert result is True

    print("TEST 34-C PASS")


def test_35_a_valid_top_recommendation_integrity():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "OpenAI AI activity",

                "evidence": "OpenAI develops AI systems.",

                "evidence_ids": [
                    "news:openai-test"
                ],

                "claims": [
                    {
                        "statement":
                            "OpenAI develops AI systems.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-test"
                        ],

                        "confidence": 0.95
                    }
                ]
            }
        ],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",

                    "rationale":
                        "AI system development is relevant "
                        "to AI engineering.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [
                        "news:openai-test"
                    ],

                    "confidence": 0.9,

                    "priority_score": 0.95,

                    "priority_level": "P1"
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "top_recommendations": [],

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:openai-test",
                "source_type": "news",
                "title": "OpenAI Test",
                "url": "https://example.com/openai-test"
            }
        ]
    })

    report.top_recommendations = build_top_recommendations(
        report
    )

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    assert len(report.top_recommendations) == 1

    top = report.top_recommendations[0]

    assert top.name == "AI Engineering"

    assert top.category == (
        RecommendationCategory.JOB_DIRECTION
    )

    assert top.claim_refs == [
        "cross_analysis:0.claims:0"
    ]

    assert top.evidence_ids == [
        "news:openai-test"
    ]

    assert top.confidence == 0.9

    assert top.priority_score == 0.95

    assert top.priority_level == "P1"

    print("TEST 35-A PASS")


def test_35_b_tampered_evidence_ids_rejected():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "OpenAI AI activity",

                "evidence": "OpenAI develops AI systems.",

                "evidence_ids": [
                    "news:openai-test"
                ],

                "claims": [
                    {
                        "statement":
                            "OpenAI develops AI systems.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-test"
                        ],

                        "confidence": 0.95
                    }
                ]
            }
        ],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",

                    "rationale": "AI engineering relevance.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [
                        "news:openai-test"
                    ],

                    "confidence": 0.9,

                    "priority_score": 0.95,

                    "priority_level": "P1"
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "top_recommendations": [],

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:openai-test",
                "source_type": "news",
                "title": "OpenAI Test",
                "url": "https://example.com/openai-test"
            }
        ]
    })

    report.top_recommendations = build_top_recommendations(
        report
    )

    # =========================================
    # Tamper TopRecommendation
    # =========================================

    report.top_recommendations[0].evidence_ids = [
        "github:fake-repository"
    ]

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print("TEST 35-B PASS")
        print("Expected validation error:")
        print(e)

        return

    raise AssertionError(
        "TEST 35-B FAILED: "
        "tampered evidence_ids was accepted."
    )


def test_35_c_tampered_claim_refs_rejected():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "OpenAI AI activity",

                "evidence": "OpenAI develops AI systems.",

                "evidence_ids": [
                    "news:openai-test"
                ],

                "claims": [
                    {
                        "statement":
                            "OpenAI develops AI systems.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-test"
                        ],

                        "confidence": 0.95
                    }
                ]
            }
        ],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",

                    "rationale": "AI engineering relevance.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [
                        "news:openai-test"
                    ],

                    "confidence": 0.9,

                    "priority_score": 0.95,

                    "priority_level": "P1"
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "top_recommendations": [],

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:openai-test",
                "source_type": "news",
                "title": "OpenAI Test",
                "url": "https://example.com/openai-test"
            }
        ]
    })

    report.top_recommendations = build_top_recommendations(
        report
    )

    # =========================================
    # Tamper claim_refs
    # =========================================

    report.top_recommendations[0].claim_refs = [
        "cross_analysis:99.claims:99"
    ]

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print("TEST 35-C PASS")
        print("Expected validation error:")
        print(e)

        return

    raise AssertionError(
        "TEST 35-C FAILED: "
        "tampered claim_refs was accepted."
    )

def test_36_a_valid_top_recommendation_category():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI develops AI products.",
            "evidence_ids": [
                "news:openai-test"
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "OpenAI AI product activity",

                "evidence": "OpenAI develops AI products.",

                "evidence_ids": [
                    "news:openai-test"
                ],

                "claims": [
                    {
                        "statement":
                            "OpenAI develops AI products.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-test"
                        ],

                        "confidence": 0.95
                    }
                ]
            }
        ],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",

                    "rationale":
                        "AI product activity supports "
                        "AI engineering relevance.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [
                        "news:openai-test"
                    ],

                    "confidence": 0.90
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:openai-test",
                "source_type": "news",
                "title": "OpenAI Test",
                "url": "https://example.com/openai-test"
            }
        ],

        "top_recommendations": [
            {
                "name": "AI Engineering",

                "rationale":
                    "AI product activity supports "
                    "AI engineering relevance.",

                "category": "job_direction",

                "claim_refs": [
                    "cross_analysis:0.claims:0"
                ],

                "evidence_ids": [
                    "news:openai-test"
                ],

                "confidence": 0.90,

                "priority_score": 0.0,

                "priority_level": "P3"
            }
        ]
    })

    evidence = [
        Evidence(
            source_id="news:openai-test",
            source_type="news",
            title="OpenAI Test",
            url="https://example.com/openai-test"
        )
    ]

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    print("TEST 36-A PASS")


def test_36_b_tampered_category_rejected():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI develops AI products.",
            "evidence_ids": [
                "news:openai-test"
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "OpenAI AI product activity",

                "evidence": "OpenAI develops AI products.",

                "evidence_ids": [
                    "news:openai-test"
                ],

                "claims": [
                    {
                        "statement":
                            "OpenAI develops AI products.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-test"
                        ],

                        "confidence": 0.95
                    }
                ]
            }
        ],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",

                    "rationale":
                        "AI product activity supports "
                        "AI engineering relevance.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [],

                    "confidence": 0.90
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:openai-test",
                "source_type": "news",
                "title": "OpenAI Test",
                "url": "https://example.com/openai-test"
            }
        ],

        "top_recommendations": [
            {
                "name": "AI Engineering",

                "rationale":
                    "AI product activity supports "
                    "AI engineering relevance.",

                # 故意篡改
                "category": "technical_skill",

                "claim_refs": [
                    "cross_analysis:0.claims:0"
                ],

                "evidence_ids": [
                    "news:openai-test"
                ],

                "confidence": 0.90,

                "priority_score": 0.0,

                "priority_level": "P3"
            }
        ]
    })

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print("TEST 36-B PASS")

        print("Expected validation error:")

        print(e)

        return

    raise AssertionError(
        "TEST 36-B FAILED: "
        "tampered TopRecommendation category "
        "was incorrectly accepted."
    )

def test_36_c_technical_skill_category_allowed():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI uses advanced AI systems.",
            "evidence_ids": [
                "news:openai-test"
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "OpenAI AI systems",

                "evidence":
                    "OpenAI uses advanced AI systems.",

                "evidence_ids": [
                    "news:openai-test"
                ],

                "claims": [
                    {
                        "statement":
                            "OpenAI uses advanced AI systems.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-test"
                        ],

                        "confidence": 0.95
                    }
                ]
            }
        ],

        "job_value": {

            "job_directions": [],

            "technical_skills": [
                {
                    "name": "Python",

                    "rationale":
                        "Python is relevant to AI system "
                        "development.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [
                        "news:openai-test"
                    ],

                    "confidence": 0.85
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:openai-test",
                "source_type": "news",
                "title": "OpenAI Test",
                "url": "https://example.com/openai-test"
            }
        ],

        "top_recommendations": [
            {
                "name": "Python",

                "rationale":
                    "Python is relevant to AI system "
                    "development.",

                "category": "technical_skill",

                "claim_refs": [
                    "cross_analysis:0.claims:0"
                ],

                "evidence_ids": [
                    "news:openai-test"
                ],

                "confidence": 0.85,

                "priority_score": 0.0,

                "priority_level": "P3"
            }
        ]
    })

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    print("TEST 36-C PASS")

def test_37_a_category_diversity_top_5():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI develops AI systems.",
            "evidence_ids": [
                "news:test"
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "OpenAI AI activity",

                "evidence": "OpenAI AI activity.",

                "evidence_ids": [
                    "news:test"
                ],

                "claims": [
                    {
                        "statement": "OpenAI AI activity.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:test"
                        ],

                        "confidence": 0.9
                    }
                ]
            }
        ],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:test"
                    ],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "B",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:test"
                    ],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "C",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:test"
                    ],
                    "confidence": 0.85,
                    "priority_score": 0.85,
                    "priority_level": "P1"
                }
            ]
        },

        "interview_preparation": {

            "topics": [
                {
                    "name": "System Design",
                    "rationale": "D",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:test"
                    ],
                    "confidence": 0.80,
                    "priority_score": 0.80,
                    "priority_level": "P2"
                }
            ],

            "practical_tasks": [
                {
                    "name": "Build AI Pipeline",
                    "rationale": "E",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:test"
                    ],
                    "confidence": 0.75,
                    "priority_score": 0.75,
                    "priority_level": "P2"
                }
            ]
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:test",
                "source_type": "news",
                "title": "Test",
                "url": "https://example.com"
            }
        ],

        "top_recommendations": []
    })

    top = build_top_recommendations(
        report,
        top_k=5
    )

    assert len(top) == 5

    categories = [
        item.category
        for item in top
    ]

    assert set(categories) == {
        RecommendationCategory.JOB_DIRECTION,
        RecommendationCategory.TECHNICAL_SKILL,
        RecommendationCategory.IMPORTANT_AREA,
        RecommendationCategory.INTERVIEW_TOPIC,
        RecommendationCategory.PRACTICAL_TASK,
    }

    print("TEST 37-A PASS")

def test_37_b_category_shortage_priority_fallback():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test company.",
            "evidence_ids": [
                "news:test"
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "Test",

                "evidence": "Test.",

                "evidence_ids": [
                    "news:test"
                ],

                "claims": [
                    {
                        "statement": "Test.",
                        "claim_type": "fact",
                        "evidence_ids": [
                            "news:test"
                        ],
                        "confidence": 0.9
                    }
                ]
            }
        ],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:test"
                    ],
                    "confidence": 0.9,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                },
                {
                    "name": "AI Research",
                    "rationale": "B",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:test"
                    ],
                    "confidence": 0.8,
                    "priority_score": 0.80,
                    "priority_level": "P1"
                },
                {
                    "name": "AI Infrastructure",
                    "rationale": "C",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:test"
                    ],
                    "confidence": 0.7,
                    "priority_score": 0.70,
                    "priority_level": "P2"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "D",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:test"
                    ],
                    "confidence": 0.85,
                    "priority_score": 0.85,
                    "priority_level": "P1"
                },
                {
                    "name": "PyTorch",
                    "rationale": "E",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:test"
                    ],
                    "confidence": 0.75,
                    "priority_score": 0.75,
                    "priority_level": "P2"
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:test",
                "source_type": "news",
                "title": "Test",
                "url": "https://example.com"
            }
        ],

        "top_recommendations": []
    })

    top = build_top_recommendations(
        report,
        top_k=5
    )

    assert len(top) == 5

    names = [
        item.name
        for item in top
    ]

    expected = [
        "AI Engineering",
        "Python",
        "AI Research",
        "PyTorch",
        "AI Infrastructure",
    ]

    assert names == expected

    print("TEST 37-B PASS")

def test_37_c_same_category_highest_priority_selected():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test company.",
            "evidence_ids": [
                "news:test"
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "Test",

                "evidence": "Test.",

                "evidence_ids": [
                    "news:test"
                ],

                "claims": [
                    {
                        "statement": "Test.",
                        "claim_type": "fact",
                        "evidence_ids": [
                            "news:test"
                        ],
                        "confidence": 0.9
                    }
                ]
            }
        ],

        "job_value": {

            "job_directions": [
                {
                    "name": "Low Priority Direction",
                    "rationale": "A",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:test"
                    ],
                    "confidence": 0.95,
                    "priority_score": 0.60,
                    "priority_level": "P2"
                },
                {
                    "name": "High Priority Direction",
                    "rationale": "B",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:test"
                    ],
                    "confidence": 0.80,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "C",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:test"
                    ],
                    "confidence": 0.90,
                    "priority_score": 0.80,
                    "priority_level": "P1"
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:test",
                "source_type": "news",
                "title": "Test",
                "url": "https://example.com"
            }
        ],

        "top_recommendations": []
    })

    top = build_top_recommendations(
        report,
        top_k=2
    )

    assert len(top) == 2

    names = [
        item.name
        for item in top
    ]

    assert "High Priority Direction" in names

    assert "Low Priority Direction" not in names

    assert "Python" in names

    print("TEST 37-C PASS")

def test_38_a_final_top_recommendations_sorted_by_priority():
    """
    Test 38-A

    Category-aware selection may introduce recommendations
    from different categories, but the final Top-K result
    must still be sorted by priority_score descending.
    """

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.60,
                    "priority_level": "P2"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "C",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.80,
                    "priority_level": "P1"
                }
            ]
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    top = build_top_recommendations(
        report,
        top_k=3
    )

    assert len(top) == 3

    assert top[0].name == "Python"
    assert top[1].name == "AI Safety"
    assert top[2].name == "AI Engineering"

    assert (
        top[0].priority_score
        >= top[1].priority_score
        >= top[2].priority_score
    )

    print("TEST 38-A PASS")


def test_38_b_same_priority_sorted_by_confidence():
    """
    Test 38-B

    When priority_score is identical,
    confidence becomes the tie-breaker.
    """

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.60,
                    "priority_score": 0.80,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.80,
                    "priority_level": "P1"
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "C",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.75,
                    "priority_score": 0.70,
                    "priority_level": "P2"
                }
            ]
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    top = build_top_recommendations(
        report,
        top_k=3
    )

    assert len(top) == 3

    assert top[0].name == "Python"
    assert top[1].name == "AI Engineering"
    assert top[2].name == "AI Safety"

    assert (
        top[0].priority_score
        == top[1].priority_score
    )

    assert (
        top[0].confidence
        > top[1].confidence
    )

    print("TEST 38-B PASS")


def test_38_c_category_diversity_does_not_break_final_order():
    """
    Test 38-C

    Category diversity determines which recommendations
    enter Top-K.

    It must NOT determine their final display order.

    Final order must still follow:
        priority_score
        confidence
    """

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "Job Direction",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.70,
                    "priority_score": 0.65,
                    "priority_level": "P2"
                }
            ],

            "technical_skills": [
                {
                    "name": "Technical Skill",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "important_areas": [
                {
                    "name": "Important Area",
                    "rationale": "C",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.85,
                    "priority_level": "P1"
                }
            ]
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    top = build_top_recommendations(
        report,
        top_k=3
    )

    assert len(top) == 3

    # Category diversity should select all three categories.
    categories = {
        item.category
        for item in top
    }

    assert len(categories) == 3

    # But final display order must be priority-based.
    assert top[0].name == "Technical Skill"
    assert top[1].name == "Important Area"
    assert top[2].name == "Job Direction"

    assert (
        top[0].priority_score
        > top[1].priority_score
        > top[2].priority_score
    )

    print("TEST 38-C PASS")

def test_39_a_full_recommendation_pipeline():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI develops AI systems.",
            "evidence_ids": [
                "news:openai"
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "OpenAI AI engineering activity",
                "evidence": "OpenAI develops AI systems.",
                "evidence_ids": [
                    "news:openai"
                ],
                "claims": [
                    {
                        "statement": "OpenAI develops AI systems.",
                        "claim_type": "fact",
                        "evidence_ids": [
                            "news:openai"
                        ],
                        "confidence": 0.95,
                    }
                ],
            }
        ],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "Relevant to AI engineering.",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:openai"
                    ],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python is relevant to AI engineering.",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:openai"
                    ],
                    "confidence": 0.88,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "AI safety is important.",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:openai"
                    ],
                    "confidence": 0.85,
                    "priority_score": 0.88,
                    "priority_level": "P1",
                }
            ],
        },

        "interview_preparation": {
            "topics": [
                {
                    "name": "System Design",
                    "rationale": "System design is relevant.",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:openai"
                    ],
                    "confidence": 0.80,
                    "priority_score": 0.82,
                    "priority_level": "P2",
                }
            ],

            "practical_tasks": [
                {
                    "name": "Build AI Agent",
                    "rationale": "Agent implementation is relevant.",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:openai"
                    ],
                    "confidence": 0.78,
                    "priority_score": 0.78,
                    "priority_level": "P2",
                }
            ],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [
            {
                "source_id": "news:openai",
                "source_type": "news",
                "title": "OpenAI Test",
                "url": "https://example.com/openai",
            }
        ],
    })

    report.top_recommendations = build_top_recommendations(
        report,
        top_k=5,
    )

    assert len(report.top_recommendations) == 5

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    print("TEST 39-A PASS")

def test_39_b_full_pipeline_invalid_claim_rejected():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [
            {
                "title": "OpenAI Test",
                "source": "Example",
                "published_at": "2026-08-25",
                "summary": "OpenAI announced a new product.",
                "implication": "Direct company activity.",
                "evidence":
                    "OpenAI announced a new product.",
                "relevance": "direct",
                "source_id": "news:openai-test"
            }
        ],

        "cross_analysis": [
            {
                "relationship":
                    "OpenAI product activity",

                "evidence":
                    "OpenAI announced a new product.",

                "evidence_ids": [
                    "news:openai-test"
                ],

                "claims": [
                    {
                        "statement":
                            "OpenAI announced a new product.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-test"
                        ],

                        "confidence": 0.95
                    }
                ]
            }
        ],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",

                    "rationale":
                        "The product activity is relevant "
                        "to AI engineering.",

                    "claim_refs": [
                        "cross_analysis:99.claims:99"
                    ],

                    "evidence_ids": [],

                    "confidence": 0.90
                }
            ],

            "technical_skills": [],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:openai-test",
                "source_type": "news",
                "title": "OpenAI Test",
                "url": "https://example.com/openai-test"
            }
        ]
    })

    evidence = [
        Evidence(
            source_id="news:openai-test",
            source_type="news",
            title="OpenAI Test",
            url="https://example.com/openai-test"
        )
    ]

    try:

        validate_recommendation_evidence_support(
            report,
            evidence
        )

    except ValueError as e:

        print("TEST 39-B PASS")

        print("Expected validation error:")

        print(e)

        return

    raise AssertionError(
        "TEST 39-B FAILED: "
        "invalid Claim reference was incorrectly accepted."
    )

def test_39_c_full_pipeline_top_recommendation_integrity():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship":
                    "OpenAI AI engineering activity",

                "evidence":
                    "OpenAI develops AI engineering systems.",

                "evidence_ids": [
                    "news:ai-engineering"
                ],

                "claims": [
                    {
                        "statement":
                            "OpenAI develops AI engineering systems.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:ai-engineering"
                        ],

                        "confidence": 0.95
                    }
                ]
            }
        ],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",

                    "rationale":
                        "This is relevant to AI engineering.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [
                        "news:ai-engineering"
                    ],

                    "confidence": 0.90,

                    "priority_score": 0.95,

                    "priority_level": "P1"
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:ai-engineering",
                "source_type": "news",
                "title": "AI Engineering",
                "url": "https://example.com/ai-engineering"
            }
        ],

        "top_recommendations": [
            {
                "name": "AI Engineering",

                "rationale":
                    "This is relevant to AI engineering.",

                "category":
                    "job_direction",

                "claim_refs": [
                    "cross_analysis:0.claims:0"
                ],

                "evidence_ids": [
                    "news:ai-engineering"
                ],

                "confidence": 0.90,

                "priority_score": 0.95,

                "priority_level": "P1"
            }
        ]
    })

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    print("TEST 39-C PASS")

def test_40_a_top_k_one():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                },
                {
                    "name": "AI Product",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.80,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "C",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    top = build_top_recommendations(
        report,
        top_k=1
    )

    assert len(top) == 1

    assert top[0].name == "AI Engineering"

    assert top[0].priority_score == 0.95

    print("TEST 40-A PASS")

def test_40_b_top_k_exceeds_recommendation_count():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "C",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.85,
                    "priority_level": "P2"
                }
            ]
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    top = build_top_recommendations(
        report,
        top_k=10
    )

    assert len(top) == 3

    assert {
        item.name
        for item in top
    } == {
        "AI Engineering",
        "Python",
        "AI Safety"
    }

    assert top[0].name == "AI Engineering"
    assert top[1].name == "Python"
    assert top[2].name == "AI Safety"

    print("TEST 40-B PASS")

def test_40_c_top_k_zero():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    top = build_top_recommendations(
        report,
        top_k=0
    )

    assert top == []

    print("TEST 40-C PASS")

def test_41_a_duplicate_recommendation_name_rejected():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "Job direction.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "AI Engineering",
                    "rationale": "Technical skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print("TEST 41-A PASS")

        print("Expected validation error:")

        print(e)

        return

    raise AssertionError(
        "TEST 41-A FAILED: "
        "duplicate Recommendation name "
        "was incorrectly accepted."
    )



def test_41_b_normalized_duplicate_recommendation_name_rejected():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "Job direction.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "  ai engineering  ",
                    "rationale": "Technical skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print("TEST 41-B PASS")

        print("Expected validation error:")

        print(e)

        return

    raise AssertionError(
        "TEST 41-B FAILED: "
        "normalized duplicate Recommendation name "
        "was incorrectly accepted."
    )

def test_41_c_distinct_recommendation_names_allowed():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "Job direction.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Technical skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    print("TEST 41-C PASS")

def test_42_a_top_recommendations_unique():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "C",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.85,
                    "priority_level": "P2"
                }
            ]
        },

        "interview_preparation": {
            "topics": [
                {
                    "name": "System Design",
                    "rationale": "D",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.75,
                    "priority_score": 0.80,
                    "priority_level": "P2"
                }
            ],

            "practical_tasks": [
                {
                    "name": "Build AI Agent",
                    "rationale": "E",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.70,
                    "priority_score": 0.75,
                    "priority_level": "P3"
                }
            ]
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    top = build_top_recommendations(
        report,
        top_k=5
    )

    assert len(top) == 5

    names = [
        item.name.strip().lower()
        for item in top
    ]

    assert len(names) == len(set(names))

    print("TEST 42-A PASS")

def test_42_b_normalized_duplicate_top_recommendation_rejected():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.9,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [
                {
                    "name": "  ai engineering  ",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.8,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [],

        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=5,
    )

    normalized_names = [
        item.name.strip().lower()
        for item in top
    ]

    assert len(normalized_names) == len(
        set(normalized_names)
    )

    assert len(top) == 1

    assert top[0].name == "AI Engineering"

    print("TEST 42-B PASS")

def test_42_c_distinct_normalized_top_recommendations_allowed():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": [
            {
                "name": "AI Engineering",
                "rationale": "A",
                "category": "job_direction",
                "claim_refs": [],
                "evidence_ids": [],
                "confidence": 0.9,
                "priority_score": 0.9,
                "priority_level": "P1"
            },
            {
                "name": "Python Engineering",
                "rationale": "B",
                "category": "technical_skill",
                "claim_refs": [],
                "evidence_ids": [],
                "confidence": 0.8,
                "priority_score": 0.8,
                "priority_level": "P1"
            }
        ]
    })

    names = [
        item.name.strip().lower()
        for item in report.top_recommendations
    ]

    assert len(names) == len(set(names))

    print("TEST 42-C PASS")

def test_43_a_top_recommendation_source_identity_allowed():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI engineering direction.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.9,
                    "priority_score": 0.9,
                    "priority_level": "P1"
                }
            ],
            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.8,
                    "priority_score": 0.8,
                    "priority_level": "P1"
                }
            ],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": [
            {
                "name": "AI Engineering",
                "rationale": "AI engineering direction.",
                "category": "job_direction",
                "claim_refs": [],
                "evidence_ids": [],
                "confidence": 0.9,
                "priority_score": 0.9,
                "priority_level": "P1"
            },
            {
                "name": "Python",
                "rationale": "Python skill.",
                "category": "technical_skill",
                "claim_refs": [],
                "evidence_ids": [],
                "confidence": 0.8,
                "priority_score": 0.8,
                "priority_level": "P1"
            }
        ]
    })

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    print("TEST 43-A PASS")

def test_43_b_top_recommendation_wrong_source_identity_rejected():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI Engineering rationale.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": [
            {
                "name": "AI Engineering",

                "rationale":
                    "AI Engineering rationale.",

                # 故意使用错误的 category
                "category":
                    RecommendationCategory.TECHNICAL_SKILL,

                "claim_refs": [],

                "evidence_ids": [],

                "confidence": 0.90,

                "priority_score": 0.90,

                "priority_level": "P1",
            }
        ]
    })

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print(
            "TEST 43-B PASS"
        )

        print(
            "Expected validation error:"
        )

        print(e)

        return

    raise AssertionError(
        "TEST 43-B FAILED: "
        "TopRecommendation with wrong source "
        "identity was not rejected."
    )

def test_43_c_top_recommendation_correct_source_identity_allowed():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI develops AI systems.",
            "evidence_ids": [
                "news:openai-test"
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "OpenAI AI engineering activity",

                "evidence": "OpenAI develops AI systems.",

                "evidence_ids": [
                    "news:openai-test"
                ],

                "claims": [
                    {
                        "statement":
                            "OpenAI develops AI systems.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-test"
                        ],

                        "confidence": 0.95
                    }
                ]
            }
        ],

        "job_value": {

            "job_directions": [],

            "technical_skills": [
                {
                    "name": "AI Engineering",

                    "rationale":
                        "AI engineering is directly relevant "
                        "to the company's AI system development.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [
                        "news:openai-test"
                    ],

                    "confidence": 0.9,

                    "priority_score": 0.92,

                    "priority_level": "P1"
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {

            "topics": [],

            "practical_tasks": []
        },

        "top_recommendations": [

            {
                "name": "AI Engineering",

                "rationale":
                    "AI engineering is directly relevant "
                    "to the company's AI system development.",

                "category":
                    RecommendationCategory.TECHNICAL_SKILL,

                "claim_refs": [
                    "cross_analysis:0.claims:0"
                ],

                "evidence_ids": [
                    "news:openai-test"
                ],

                "confidence": 0.9,

                "priority_score": 0.92,

                "priority_level": "P1"
            }
        ],

        "reliability": {

            "supported_conclusions": [],

            "uncertain_conclusions": [],

            "limitations": []
        },

        "evidence": [

            {
                "source_id": "news:openai-test",

                "source_type": "news",

                "title": "OpenAI Test",

                "url":
                    "https://example.com/openai-test"
            }
        ]
    })

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    print(
        "TEST 43-C PASS"
    )

def test_44_a_valid_top_recommendation_count():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": [
            {
                "name": "AI Engineering",
                "rationale": "A",
                "category": "job_direction",
                "claim_refs": [],
                "evidence_ids": [],
                "confidence": 0.9,
                "priority_score": 0.9,
                "priority_level": "P1"
            },
            {
                "name": "Python",
                "rationale": "B",
                "category": "technical_skill",
                "claim_refs": [],
                "evidence_ids": [],
                "confidence": 0.8,
                "priority_score": 0.8,
                "priority_level": "P1"
            },
            {
                "name": "AI Safety",
                "rationale": "C",
                "category": "important_area",
                "claim_refs": [],
                "evidence_ids": [],
                "confidence": 0.8,
                "priority_score": 0.7,
                "priority_level": "P2"
            },
            {
                "name": "System Design",
                "rationale": "D",
                "category": "interview_topic",
                "claim_refs": [],
                "evidence_ids": [],
                "confidence": 0.7,
                "priority_score": 0.6,
                "priority_level": "P2"
            },
            {
                "name": "Build an AI Agent",
                "rationale": "E",
                "category": "practical_task",
                "claim_refs": [],
                "evidence_ids": [],
                "confidence": 0.7,
                "priority_score": 0.5,
                "priority_level": "P3"
            }
        ]
    })

    result = validate_top_recommendation_result_constraints(
        report,
        top_k=5,
    )

    assert result is True

    assert len(
        report.top_recommendations
    ) == 5

    print("TEST 44-A PASS")

def test_44_b_top_recommendation_count_exceeds_top_k_rejected():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": [
            {
                "name": f"Recommendation {i}",
                "rationale": f"Rationale {i}",
                "category": "job_direction",
                "claim_refs": [],
                "evidence_ids": [],
                "confidence": 0.8,
                "priority_score": 0.8,
                "priority_level": "P1"
            }
            for i in range(6)
        ]
    })

    try:

        validate_top_recommendation_result_constraints(
            report,
            top_k=5,
        )

    except ValueError as e:

        print("TEST 44-B PASS")

        print(
            "Expected validation error:"
        )

        print(e)

        return

    raise AssertionError(
        "TEST 44-B FAILED: "
        "TopRecommendation count exceeding "
        "top_k was not rejected."
    )

def test_44_c_empty_top_recommendations_allowed():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": []
    })

    result = validate_top_recommendation_result_constraints(
        report,
        top_k=5,
    )

    assert result is True

    assert (
        len(report.top_recommendations) == 0
    )

    print("TEST 44-C PASS")

def test_45_a_built_top_recommendations_pass_full_validation():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "C",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.85,
                    "priority_level": "P2"
                }
            ]
        },

        "interview_preparation": {
            "topics": [
                {
                    "name": "System Design",
                    "rationale": "D",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.80,
                    "priority_level": "P2"
                }
            ],

            "practical_tasks": [
                {
                    "name": "Build AI Agent",
                    "rationale": "E",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.75,
                    "priority_score": 0.75,
                    "priority_level": "P3"
                }
            ]
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    report.top_recommendations = build_top_recommendations(
        report,
        top_k=5
    )

    result_integrity = validate_top_recommendation_integrity(
        report
    )

    result_constraints = validate_top_recommendation_result_constraints(
        report,
        top_k=5
    )

    assert result_integrity is True

    assert result_constraints is True

    assert len(report.top_recommendations) == 5

    print("TEST 45-A PASS")

def test_45_b_tampered_built_top_recommendation_rejected():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    # =========================================
    # Build legitimate TopRecommendations
    # =========================================

    report.top_recommendations = build_top_recommendations(
        report,
        top_k=5
    )

    assert len(
        report.top_recommendations
    ) == 2

    # =========================================
    # Tamper with derived result
    # =========================================

    report.top_recommendations[0].rationale = (
        "TAMPERED RATIONALE"
    )

    # =========================================
    # Full validation must reject
    # =========================================

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print("TEST 45-B PASS")

        print(
            "Expected validation error:"
        )

        print(e)

        return

    raise AssertionError(
        "TEST 45-B FAILED: "
        "Tampered TopRecommendation was not rejected."
    )

def test_45_c_tampered_built_top_recommendation_priority_rejected():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": [
                "news:openai-test"
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "OpenAI product activity",

                "evidence": "OpenAI announced a new product.",

                "evidence_ids": [
                    "news:openai-test"
                ],

                "claims": [
                    {
                        "statement":
                            "OpenAI announced a new product.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-test"
                        ],

                        "confidence": 0.95
                    }
                ]
            }
        ],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",

                    "rationale":
                        "The evidence supports AI engineering relevance.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [
                        "news:openai-test"
                    ],

                    "confidence": 0.90,

                    "priority_score": 0.92,

                    "priority_level": "P1"
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:openai-test",
                "source_type": "news",
                "title": "OpenAI Test",
                "url": "https://example.com/openai-test"
            }
        ],

        "top_recommendations": []
    })

    # =========================================
    # Build TopRecommendation
    # =========================================

    report.top_recommendations = build_top_recommendations(
        report,
        top_k=5
    )

    assert len(report.top_recommendations) == 1

    # =========================================
    # Tamper priority_score
    # =========================================

    report.top_recommendations[0].priority_score = 0.10

    # =========================================
    # Validation must reject
    # =========================================

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print("TEST 45-C PASS")

        print(
            "Expected validation error:"
        )

        print(e)

        return

    raise AssertionError(
        "TEST 45-C FAILED: "
        "tampered TopRecommendation priority_score "
        "was not rejected."
    )

def test_46_a_priority_order_preserved():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.65,
                    "priority_level": "P2"
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "C",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ]
        },

        "interview_preparation": {

            "topics": [
                {
                    "name": "System Design",
                    "rationale": "D",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.85,
                    "priority_level": "P2"
                }
            ],

            "practical_tasks": [
                {
                    "name": "RAG Implementation",
                    "rationale": "E",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.75,
                    "priority_score": 0.70,
                    "priority_level": "P2"
                }
            ]
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": []
    })

    # =========================================
    # Build Top Recommendations
    # =========================================

    top = build_top_recommendations(
        report,
        top_k=5
    )

    # =========================================
    # Verify count
    # =========================================

    assert len(top) == 5

    # =========================================
    # Verify final order
    #
    # priority_score:
    #
    # AI Engineering      0.95
    # AI Safety            0.90
    # System Design        0.85
    # RAG Implementation   0.70
    # Python                0.65
    # =========================================

    assert top[0].name == "AI Engineering"
    assert top[1].name == "AI Safety"
    assert top[2].name == "System Design"
    assert top[3].name == "RAG Implementation"
    assert top[4].name == "Python"

    # =========================================
    # Verify numeric priority order
    # =========================================

    assert (
        top[0].priority_score
        >= top[1].priority_score
        >= top[2].priority_score
        >= top[3].priority_score
        >= top[4].priority_score
    )

    print("TEST 46-A PASS")

def test_46_b_confidence_tiebreaker():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "Lower Confidence",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.60,
                    "priority_score": 0.80,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Higher Confidence",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.80,
                    "priority_level": "P1"
                }
            ],

            "important_areas": [
                {
                    "name": "Medium Confidence",
                    "rationale": "C",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.75,
                    "priority_score": 0.80,
                    "priority_level": "P1"
                }
            ]
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": []
    })

    # =========================================
    # Build Top Recommendations
    # =========================================

    top = build_top_recommendations(
        report,
        top_k=3
    )

    # =========================================
    # Verify count
    # =========================================

    assert len(top) == 3

    # =========================================
    # All priority_score are equal:
    #
    # 0.80
    # 0.80
    # 0.80
    #
    # Therefore confidence decides order.
    # =========================================

    assert top[0].name == "Higher Confidence"
    assert top[1].name == "Medium Confidence"
    assert top[2].name == "Lower Confidence"

    # =========================================
    # Verify confidence descending
    # =========================================

    assert (
        top[0].confidence
        >= top[1].confidence
        >= top[2].confidence
    )

    # =========================================
    # Verify priority_score remains unchanged
    # =========================================

    assert all(
        item.priority_score == 0.80
        for item in top
    )

    print("TEST 46-B PASS")

def test_46_c_category_diversity_does_not_break_priority_order():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.80,
                    "priority_level": "P2"
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "C",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ]
        },

        "interview_preparation": {

            "topics": [
                {
                    "name": "System Design",
                    "rationale": "D",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.85,
                    "priority_level": "P2"
                }
            ],

            "practical_tasks": [
                {
                    "name": "RAG Implementation",
                    "rationale": "E",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.75,
                    "priority_score": 0.70,
                    "priority_level": "P3"
                }
            ]
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": []
    })

    # =========================================
    # Build Category-aware Top 5
    # =========================================

    top = build_top_recommendations(
        report,
        top_k=5
    )

    # =========================================
    # Verify exactly five recommendations
    # =========================================

    assert len(top) == 5

    # =========================================
    # Verify category diversity
    #
    # Five recommendations come from five
    # different categories.
    # =========================================

    categories = [
        item.category
        for item in top
    ]

    assert len(categories) == len(set(categories))

    # =========================================
    # Verify final priority order
    #
    # AI Engineering       0.95
    # AI Safety             0.90
    # System Design         0.85
    # Python                0.80
    # RAG Implementation    0.70
    # =========================================

    assert top[0].name == "AI Engineering"
    assert top[1].name == "AI Safety"
    assert top[2].name == "System Design"
    assert top[3].name == "Python"
    assert top[4].name == "RAG Implementation"

    # =========================================
    # Verify priority_score descending
    # =========================================

    assert (
        top[0].priority_score
        >= top[1].priority_score
        >= top[2].priority_score
        >= top[3].priority_score
        >= top[4].priority_score
    )

    print("TEST 46-C PASS")

def test_47_a_top_recommendation_fields_preserved():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "Important AI engineering direction.",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "github:test"
                    ],
                    "confidence": 0.91,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": []
    })

    top = build_top_recommendations(
        report,
        top_k=5
    )

    assert len(top) == 1

    item = top[0]

    # =========================================
    # Verify name
    # =========================================

    assert item.name == "AI Engineering"

    # =========================================
    # Verify rationale
    # =========================================

    assert (
        item.rationale
        == "Important AI engineering direction."
    )

    print("TEST 47-A PASS")

def test_47_b_top_recommendation_support_refs_preserved():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",

                    "rationale":
                        "AI engineering is strongly relevant.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0",
                        "cross_analysis:1.claims:0"
                    ],

                    "evidence_ids": [
                        "github:openai/test",
                        "news:openai-test"
                    ],

                    "confidence": 0.92,

                    "priority_score": 0.94,

                    "priority_level": "P1"
                }
            ],

            "technical_skills": [],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": []
    })

    # =========================================
    # Build TopRecommendation
    # =========================================

    top = build_top_recommendations(
        report,
        top_k=5
    )

    assert len(top) == 1

    item = top[0]

    # =========================================
    # Verify claim_refs
    # =========================================

    assert item.claim_refs == [
        "cross_analysis:0.claims:0",
        "cross_analysis:1.claims:0"
    ]

    # =========================================
    # Verify evidence_ids
    # =========================================

    assert item.evidence_ids == [
        "github:openai/test",
        "news:openai-test"
    ]

    # =========================================
    # Verify list contents and order
    # =========================================

    assert len(item.claim_refs) == 2
    assert len(item.evidence_ids) == 2

    assert (
        item.claim_refs[0]
        == "cross_analysis:0.claims:0"
    )

    assert (
        item.claim_refs[1]
        == "cross_analysis:1.claims:0"
    )

    assert (
        item.evidence_ids[0]
        == "github:openai/test"
    )

    assert (
        item.evidence_ids[1]
        == "news:openai-test"
    )

    print("TEST 47-B PASS")

def test_47_c_top_recommendation_priority_level_preserved():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI develops AI systems.",
            "evidence_ids": [
                "news:openai-test"
            ]
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [
            {
                "relationship": "OpenAI AI activity",

                "evidence": "OpenAI develops AI systems.",

                "evidence_ids": [
                    "news:openai-test"
                ],

                "claims": [
                    {
                        "statement": "OpenAI develops AI systems.",

                        "claim_type": "fact",

                        "evidence_ids": [
                            "news:openai-test"
                        ],

                        "confidence": 0.95
                    }
                ]
            }
        ],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",

                    "rationale":
                        "Relevant to AI engineering.",

                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],

                    "evidence_ids": [
                        "news:openai-test"
                    ],

                    "confidence": 0.92,

                    "priority_score": 0.95,

                    "priority_level": "P1"
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [
            {
                "source_id": "news:openai-test",
                "source_type": "news",
                "title": "OpenAI Test",
                "url": "https://example.com/openai-test"
            }
        ],

        "top_recommendations": []
    })

    top = build_top_recommendations(
        report,
        top_k=1
    )

    assert len(top) == 1

    assert top[0].name == "AI Engineering"

    assert top[0].priority_level == "P1"

    assert (
        top[0].priority_level
        ==
        report.job_value
        .job_directions[0]
        .priority_level
    )

    print("TEST 47-C PASS")

def test_48_a_cross_category_duplicate_recommendation_rejected():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "Job direction.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "AI Engineering",
                    "rationale": "Technical skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": []
    })

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print("TEST 48-A PASS")

        print("Expected validation error:")
        print(e)

        return

    raise AssertionError(
        "TEST 48-A FAILED: "
        "cross-category duplicate Recommendation "
        "name was not rejected."
    )

def test_48_b_cross_category_distinct_recommendations_allowed():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "Job direction.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Technical skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "Important area.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.85,
                    "priority_level": "P1"
                }
            ]
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": []
    })

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    print("TEST 48-B PASS")

def test_48_c_cross_category_normalized_duplicate_rejected():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "Job direction.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "  ai engineering  ",
                    "rationale": "Technical skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": []
    })

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print("TEST 48-C PASS")

        print("Expected validation error:")
        print(e)

        return

    raise AssertionError(
        "TEST 48-C FAILED: "
        "normalized cross-category duplicate "
        "Recommendation was not rejected."
    )

def test_49_a_built_top_recommendations_integrity():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI engineering direction.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python is an important skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "AI safety is an important area.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.85,
                    "priority_level": "P1"
                }
            ]
        },

        "interview_preparation": {
            "topics": [
                {
                    "name": "LLM Systems",
                    "rationale": "LLM systems are relevant.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.75,
                    "priority_score": 0.80,
                    "priority_level": "P2"
                }
            ],

            "practical_tasks": [
                {
                    "name": "Build an AI Agent",
                    "rationale": "Practical agent implementation.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.70,
                    "priority_score": 0.75,
                    "priority_level": "P2"
                }
            ]
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": []
    })

    # =========================================
    # Build Top-K automatically
    # =========================================

    report.top_recommendations = build_top_recommendations(
        report,
        top_k=5,
    )

    # =========================================
    # Validate automatically built result
    # =========================================

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    assert len(
        report.top_recommendations
    ) == 5

    print("TEST 49-A PASS")

def test_49_b_built_top_recommendations_respect_top_k():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI engineering relevance.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                },
                {
                    "name": "ML Engineering",
                    "rationale": "ML engineering relevance.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python skill relevance.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.88,
                    "priority_level": "P1"
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "AI safety relevance.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.82,
                    "priority_level": "P2"
                }
            ]
        },

        "interview_preparation": {
            "topics": [
                {
                    "name": "System Design",
                    "rationale": "System design interview relevance.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.78,
                    "priority_level": "P2"
                }
            ],

            "practical_tasks": [
                {
                    "name": "Build an AI Agent",
                    "rationale": "Practical AI task relevance.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.75,
                    "priority_score": 0.70,
                    "priority_level": "P3"
                }
            ]
        },

        "top_recommendations": [],

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    top = build_top_recommendations(
        report,
        top_k=3,
    )

    assert len(top) == 3

    assert len(top) <= 3

    assert validate_top_recommendation_integrity(
        report.model_copy(
            update={
                "top_recommendations": top
            }
        )
    ) is True

    print("TEST 49-B PASS")

def test_49_c_built_top_recommendations_category_identity_preserved():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI engineering relevance.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python skill relevance.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "top_recommendations": [],

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    top = build_top_recommendations(
        report,
        top_k=2,
    )

    assert len(top) == 2

    categories = {
        item.name: item.category
        for item in top
    }

    assert (
        categories["AI Engineering"]
        == RecommendationCategory.JOB_DIRECTION
    )

    assert (
        categories["Python"]
        == RecommendationCategory.TECHNICAL_SKILL
    )

    report_with_top = report.model_copy(
        update={
            "top_recommendations": top
        }
    )

    assert validate_top_recommendation_integrity(
        report_with_top
    ) is True

    print("TEST 49-C PASS")

def test_50_a_full_report_top_recommendation_consistency():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI develops advanced AI systems.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale":
                        "Relevant to AI engineering roles.",

                    "claim_refs": [],

                    "evidence_ids": [],

                    "confidence": 0.92,

                    "priority_score": 0.95,

                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale":
                        "Python is relevant to AI engineering.",

                    "claim_refs": [],

                    "evidence_ids": [],

                    "confidence": 0.88,

                    "priority_score": 0.90,

                    "priority_level": "P1"
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale":
                        "AI safety is an important area.",

                    "claim_refs": [],

                    "evidence_ids": [],

                    "confidence": 0.84,

                    "priority_score": 0.85,

                    "priority_level": "P2"
                }
            ]
        },

        "interview_preparation": {

            "topics": [
                {
                    "name": "System Design",
                    "rationale":
                        "System design is relevant to interviews.",

                    "claim_refs": [],

                    "evidence_ids": [],

                    "confidence": 0.80,

                    "priority_score": 0.78,

                    "priority_level": "P2"
                }
            ],

            "practical_tasks": [
                {
                    "name": "Build an AI Agent",
                    "rationale":
                        "Agent implementation is a useful practical task.",

                    "claim_refs": [],

                    "evidence_ids": [],

                    "confidence": 0.76,

                    "priority_score": 0.72,

                    "priority_level": "P3"
                }
            ]
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": [],

        "top_recommendations": []
    })

    # =========================================
    # Step 1
    # Build Top-K
    # =========================================

    top = build_top_recommendations(
        report,
        top_k=5,
    )

    assert len(top) == 5

    # =========================================
    # Step 2
    # Put generated TopRecommendations
    # back into the Report
    # =========================================

    report.top_recommendations = top

    # =========================================
    # Step 3
    # Validate complete identity chain
    # =========================================

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    # =========================================
    # Step 4
    # Validate uniqueness
    # =========================================

    names = [
        item.name.strip().lower()
        for item in report.top_recommendations
    ]

    assert len(names) == len(set(names))

    # =========================================
    # Step 5
    # Validate final ranking
    # =========================================

    scores = [
        float(item.priority_score)
        for item in report.top_recommendations
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    print("TEST 50-A PASS")

def test_50_b_missing_source_recommendation_rejected():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale":
                        "AI engineering relevance.",

                    "claim_refs": [],

                    "evidence_ids": [],

                    "confidence": 0.90,

                    "priority_score": 0.95,

                    "priority_level": "P1"
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "top_recommendations": [

            {
                "name": "Nonexistent Recommendation",

                "rationale":
                    "This recommendation does not exist.",

                "category":
                    RecommendationCategory.JOB_DIRECTION,

                "claim_refs": [],

                "evidence_ids": [],

                "confidence": 0.90,

                "priority_score": 0.90,

                "priority_level": "P1"
            }
        ],

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print("TEST 50-B PASS")

        print(
            "Expected validation error:"
        )

        print(e)

        return

    raise AssertionError(
        "TEST 50-B FAILED: "
        "TopRecommendation referencing a "
        "missing Recommendation was accepted."
    )

def test_50_c_tampered_category_rejected():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI engineering relevance.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "top_recommendations": [
            {
                "name": "AI Engineering",
                "rationale": "AI engineering relevance.",

                # 故意篡改 category
                "category":
                    RecommendationCategory.TECHNICAL_SKILL,

                "claim_refs": [],
                "evidence_ids": [],
                "confidence": 0.90,
                "priority_score": 0.95,
                "priority_level": "P1"
            }
        ],

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print("TEST 50-C PASS")

        print(
            "Expected validation error:"
        )

        print(e)

        return

    raise AssertionError(
        "TEST 50-C FAILED: "
        "Tampered TopRecommendation category "
        "was accepted."
    )

def test_51_a_top_k_one_category_identity():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI engineering.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.85,
                    "priority_level": "P1"
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "top_recommendations": [],

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    top = build_top_recommendations(
        report,
        top_k=1,
    )

    assert len(top) == 1

    assert top[0].name == "AI Engineering"

    assert (
        top[0].category
        == RecommendationCategory.JOB_DIRECTION
    )

    report.top_recommendations = top

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    print("TEST 51-A PASS")

def test_51_b_top_k_two_category_identity():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI engineering.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.85,
                    "priority_level": "P1"
                }
            ],

            "important_areas": [],

        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "top_recommendations": [],

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    top = build_top_recommendations(
        report,
        top_k=2,
    )

    # -----------------------------------------
    # Count
    # -----------------------------------------

    assert len(top) == 2

    # -----------------------------------------
    # Priority order
    # -----------------------------------------

    assert top[0].name == "AI Engineering"
    assert top[1].name == "Python"

    # -----------------------------------------
    # Category identity
    # -----------------------------------------

    assert (
        top[0].category
        == RecommendationCategory.JOB_DIRECTION
    )

    assert (
        top[1].category
        == RecommendationCategory.TECHNICAL_SKILL
    )

    # -----------------------------------------
    # Attach generated TopRecommendations
    # -----------------------------------------

    report.top_recommendations = top

    # -----------------------------------------
    # Full integrity validation
    # -----------------------------------------

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    print("TEST 51-B PASS")

def test_51_c_full_category_identity_top_k():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI engineering.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "AI safety.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.85,
                    "priority_level": "P1"
                }
            ],
        },

        "interview_preparation": {

            "topics": [
                {
                    "name": "AI Interview",
                    "rationale": "Interview topic.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.80,
                    "priority_level": "P2"
                }
            ],

            "practical_tasks": [
                {
                    "name": "Build AI Agent",
                    "rationale": "Practical task.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.75,
                    "priority_score": 0.75,
                    "priority_level": "P2"
                }
            ]
        },

        "top_recommendations": [],

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    top = build_top_recommendations(
        report,
        top_k=5,
    )

    # -----------------------------------------
    # Count
    # -----------------------------------------

    assert len(top) == 5

    # -----------------------------------------
    # Final priority order
    # -----------------------------------------

    assert [
        item.name
        for item in top
    ] == [
        "AI Engineering",
        "Python",
        "AI Safety",
        "AI Interview",
        "Build AI Agent",
    ]

    # -----------------------------------------
    # Category identity
    # -----------------------------------------

    assert [
        item.category
        for item in top
    ] == [
        RecommendationCategory.JOB_DIRECTION,
        RecommendationCategory.TECHNICAL_SKILL,
        RecommendationCategory.IMPORTANT_AREA,
        RecommendationCategory.INTERVIEW_TOPIC,
        RecommendationCategory.PRACTICAL_TASK,
    ]

    # -----------------------------------------
    # Every category appears exactly once
    # -----------------------------------------

    categories = [
        item.category
        for item in top
    ]

    assert len(categories) == len(set(categories))

    # -----------------------------------------
    # Attach generated TopRecommendations
    # -----------------------------------------

    report.top_recommendations = top

    # -----------------------------------------
    # Full integrity validation
    # -----------------------------------------

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    print("TEST 51-C PASS")

def test_52_a_all_top_recommendation_fields_consistent():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI develops AI systems.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "Strong relevance to AI engineering.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.92,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    report.top_recommendations = build_top_recommendations(
        report,
        top_k=5
    )

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    source = report.job_value.job_directions[0]
    top = report.top_recommendations[0]

    assert top.name == source.name
    assert top.rationale == source.rationale
    assert top.category == RecommendationCategory.JOB_DIRECTION
    assert top.claim_refs == source.claim_refs
    assert top.evidence_ids == source.evidence_ids
    assert top.confidence == source.confidence
    assert top.priority_score == source.priority_score
    assert top.priority_level == source.priority_level

    print("TEST 52-A PASS")

def test_52_b_tampered_confidence_rejected():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI develops AI systems.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "Strong relevance to AI engineering.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.92,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    report.top_recommendations = build_top_recommendations(
        report,
        top_k=5
    )

    report.top_recommendations[0].confidence = 0.50

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print("TEST 52-B PASS")
        print("Expected validation error:")
        print(e)
        return

    raise AssertionError(
        "TEST 52-B FAILED: "
        "tampered confidence was not rejected."
    )

def test_52_c_tampered_priority_level_rejected():

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "OpenAI develops AI systems.",
            "evidence_ids": []
        },

        "github_signals": [],

        "news_signals": [],

        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "Strong relevance to AI engineering.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.92,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [],
            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    report.top_recommendations = build_top_recommendations(
        report,
        top_k=5
    )

    report.top_recommendations[0].priority_level = "P3"

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print("TEST 52-C PASS")
        print("Expected validation error:")
        print(e)
        return

    raise AssertionError(
        "TEST 52-C FAILED: "
        "tampered priority_level was not rejected."
    )

def test_53_a_same_category_low_priority_filtered():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "High priority AI engineering.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                },
                {
                    "name": "AI Research",
                    "rationale": "Lower priority AI research.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.60,
                    "priority_score": 0.50,
                    "priority_level": "P3"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1"
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "top_recommendations": [],

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    top = build_top_recommendations(
        report,
        top_k=2
    )

    assert len(top) == 2

    assert top[0].name == "AI Engineering"
    assert top[1].name == "Python"

    assert (
        top[0].category
        == RecommendationCategory.JOB_DIRECTION
    )

    assert (
        top[1].category
        == RecommendationCategory.TECHNICAL_SKILL
    )

    assert all(
        item.name != "AI Research"
        for item in top
    )

    report.top_recommendations = top

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    print("TEST 53-A PASS")

def test_53_b_category_diversity_over_same_category_priority():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "OpenAI test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "High priority AI engineering.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1"
                },
                {
                    "name": "AI Research",
                    "rationale": "Second job direction.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.89,
                    "priority_level": "P1"
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Important technical skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.88,
                    "priority_level": "P1"
                }
            ],

            "important_areas": []
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": []
        },

        "top_recommendations": [],

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": []
        },

        "evidence": []
    })

    top = build_top_recommendations(
        report,
        top_k=2
    )

    # -----------------------------------------
    # Exactly two recommendations
    # -----------------------------------------

    assert len(top) == 2

    # -----------------------------------------
    # Category-aware selection
    #
    # AI Engineering wins JOB_DIRECTION.
    # Python wins TECHNICAL_SKILL.
    #
    # AI Research must NOT enter the first pass.
    # -----------------------------------------

    assert top[0].name == "AI Engineering"
    assert top[1].name == "Python"

    assert all(
        item.name != "AI Research"
        for item in top
    )

    # -----------------------------------------
    # Category identity
    # -----------------------------------------

    assert (
        top[0].category
        == RecommendationCategory.JOB_DIRECTION
    )

    assert (
        top[1].category
        == RecommendationCategory.TECHNICAL_SKILL
    )

    # -----------------------------------------
    # Final priority order
    # -----------------------------------------

    assert (
        top[0].priority_score
        > top[1].priority_score
    )

    # -----------------------------------------
    # Integrity validation
    # -----------------------------------------

    report.top_recommendations = top

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    print("TEST 53-B PASS")

def test_53_c_category_shortage_fallback_preserves_priority():
    """
    Test 53-C

    当可用类别不足 Top-K 时：
    1. 首先选择不同类别的最高优先级 Recommendation
    2. 再通过 priority fallback 补足剩余数量
    3. 最终结果仍按 priority_score + confidence 排序
    """

    report = JobAnalysisReport.model_validate({
        "company_overview": {
            "summary": "Test company.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                },
                {
                    "name": "Backend Engineering",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.80,
                    "priority_level": "P2",
                },
            ],

            "technical_skills": [],

            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=3,
    )

    print(
        "Test 53-C names =",
        [item.name for item in top]
    )

    # 只有一个类别，因此第二个推荐必须通过 fallback 进入
    assert len(top) == 2

    assert top[0].name == "AI Engineering"
    assert top[1].name == "Backend Engineering"

    # 最终仍然保持 priority 顺序
    assert (
        top[0].priority_score
        >= top[1].priority_score
    )

    # 两个 Recommendation 都来自同一个合法类别
    assert (
        top[0].category
        == RecommendationCategory.JOB_DIRECTION
    )

    assert (
        top[1].category
        == RecommendationCategory.JOB_DIRECTION
    )

    print(
        "TEST 53-C PASS"
    )

def test_54_a_duplicate_names_not_duplicated_in_built_top():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test company.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [
                {
                    "name": "AI Engineering",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.80,
                    "priority_level": "P2",
                }
            ],

            "important_areas": [],

        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=5,
    )

    print(
        "Test 54-A names =",
        [item.name for item in top]
    )

    # -----------------------------------------
    # Duplicate name must not appear twice
    # -----------------------------------------

    names = [
        item.name.strip().lower()
        for item in top
    ]

    assert len(names) == len(set(names))

    # -----------------------------------------
    # Only one AI Engineering should survive
    # -----------------------------------------

    assert len(top) == 1

    assert top[0].name == "AI Engineering"

    print("TEST 54-A PASS")

def test_54_b_normalized_duplicate_names_not_duplicated_in_built_top():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test company.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "A",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [
                {
                    "name": "  ai engineering  ",
                    "rationale": "B",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [],

        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=5,
    )

    print(
        "Test 54-B names =",
        [item.name for item in top]
    )

    # -----------------------------------------
    # Normalized identity
    # -----------------------------------------

    names = [
        item.name.strip().lower()
        for item in top
    ]

    assert len(names) == len(set(names))

    # -----------------------------------------
    # Only one normalized identity should survive
    # -----------------------------------------

    assert len(top) == 1

    assert (
        top[0].name.strip().lower()
        == "ai engineering"
    )

    # -----------------------------------------
    # Higher-priority Recommendation survives
    # -----------------------------------------

    assert top[0].priority_score == 0.95

    print("TEST 54-B PASS")

def test_54_c_cross_category_normalized_duplicate_not_duplicated_in_built_top():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test company.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "Job direction.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [
                {
                    "name": "  ai engineering  ",
                    "rationale": "Technical skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [],

        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=5,
    )

    print(
        "Test 54-C names =",
        [item.name for item in top]
    )

    print(
        "Test 54-C categories =",
        [item.category for item in top]
    )

    # =========================================
    # Step 1
    # Normalize names
    # =========================================

    names = [
        item.name.strip().lower()
        for item in top
    ]

    # =========================================
    # Step 2
    # No normalized duplicate allowed
    # =========================================

    assert len(names) == len(set(names))

    # =========================================
    # Step 3
    # Only one Recommendation survives
    # =========================================

    assert len(top) == 1

    # =========================================
    # Step 4
    # Highest-priority Recommendation survives
    # =========================================

    assert top[0].priority_score == 0.95

    # =========================================
    # Step 5
    # Original source identity is preserved
    # =========================================

    assert top[0].name == "AI Engineering"

    assert (
        top[0].category
        == RecommendationCategory.JOB_DIRECTION
    )

    print("TEST 54-C PASS")

# =========================================================
# Test 55-A
# Final Top-K ranking must preserve priority_score order
# =========================================================

def test_55_a_final_priority_order_preserved():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test company.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "Backend Engineering",
                    "rationale": "Backend.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.70,
                    "priority_score": 0.70,
                    "priority_level": "P2",
                },
                {
                    "name": "AI Engineering",
                    "rationale": "AI.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                },
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.85,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [],

        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=3,
    )

    names = [
        item.name
        for item in top
    ]

    scores = [
        item.priority_score
        for item in top
    ]

    print(
        "Test 55-A names =",
        names
    )

    print(
        "Test 55-A priority_scores =",
        scores
    )

    assert names == [
        "AI Engineering",
        "Python",
        "Backend Engineering",
    ]

    assert scores == [
        0.95,
        0.85,
        0.70,
    ]

    assert all(
        scores[index] >= scores[index + 1]
        for index in range(len(scores) - 1)
    )

    print("TEST 55-A PASS")


# =========================================================
# Test 55-B
# Same priority_score must use confidence as tie-breaker
# =========================================================

def test_55_b_final_confidence_tiebreaker_preserved():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test company.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "Lower Confidence",
                    "rationale": "Lower confidence.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.60,
                    "priority_score": 0.80,
                    "priority_level": "P1",
                },
                {
                    "name": "Higher Confidence",
                    "rationale": "Higher confidence.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.80,
                    "priority_level": "P1",
                },
            ],

            "technical_skills": [
                {
                    "name": "Highest Priority",
                    "rationale": "Highest priority.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.70,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [],

        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=3,
    )

    names = [
        item.name
        for item in top
    ]

    print(
        "Test 55-B names =",
        names
    )

    assert names == [
        "Highest Priority",
        "Higher Confidence",
        "Lower Confidence",
    ]

    assert top[1].priority_score == 0.80
    assert top[2].priority_score == 0.80

    assert top[1].confidence > top[2].confidence

    print("TEST 55-B PASS")


# =========================================================
# Test 55-C
# Category diversity selection must not break
# final priority ordering
# =========================================================

def test_55_c_category_diversity_then_final_priority_order():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test company.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI direction.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                },
                {
                    "name": "Backend Engineering",
                    "rationale": "Backend direction.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                },
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.85,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "Safety area.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.80,
                    "priority_level": "P2",
                }
            ],

        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=3,
    )

    names = [
        item.name
        for item in top
    ]

    categories = [
        item.category
        for item in top
    ]

    scores = [
        item.priority_score
        for item in top
    ]

    print(
        "Test 55-C names =",
        names
    )

    print(
        "Test 55-C categories =",
        categories
    )

    print(
        "Test 55-C priority_scores =",
        scores
    )

    # First pass selects one from each category:
    #
    # AI Engineering       0.95  job_direction
    # Python               0.85  technical_skill
    # AI Safety            0.80  important_area
    #
    # Backend Engineering  0.90 is initially skipped
    # because job_direction already contributed one item.
    #
    # Therefore final Top-3 must remain:
    # 0.95 -> 0.85 -> 0.80

    assert len(top) == 3

    assert names == [
        "AI Engineering",
        "Python",
        "AI Safety",
    ]

    assert categories == [
        RecommendationCategory.JOB_DIRECTION,
        RecommendationCategory.TECHNICAL_SKILL,
        RecommendationCategory.IMPORTANT_AREA,
    ]

    assert scores == [
        0.95,
        0.85,
        0.80,
    ]

    assert all(
        scores[index] >= scores[index + 1]
        for index in range(len(scores) - 1)
    )

    print("TEST 55-C PASS")

# =========================================================
# Test 56-A
# Empty report must return empty TopRecommendations
# =========================================================

def test_56_a_empty_recommendations_return_empty():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Empty test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {
            "job_directions": [],
            "technical_skills": [],
            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=5,
    )

    print(
        "Test 56-A top =",
        top
    )

    assert top == []

    print("TEST 56-A PASS")


# =========================================================
# Test 56-B
# One Recommendation with top_k greater than count
# =========================================================

def test_56_b_single_recommendation_preserved():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Single recommendation test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI engineering relevance.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [],
            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=5,
    )

    print(
        "Test 56-B names =",
        [item.name for item in top]
    )

    assert len(top) == 1

    assert top[0].name == "AI Engineering"

    assert top[0].priority_score == 0.95

    assert top[0].confidence == 0.90

    assert (
        top[0].category
        == RecommendationCategory.JOB_DIRECTION
    )

    print("TEST 56-B PASS")


# =========================================================
# Test 56-C
# top_k exactly equals recommendation count
# =========================================================

def test_56_c_top_k_equals_recommendation_count():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Exact top_k test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.85,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "Safety.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.80,
                    "priority_level": "P2",
                }
            ],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=3,
    )

    names = [
        item.name
        for item in top
    ]

    print(
        "Test 56-C names =",
        names
    )

    assert len(top) == 3

    assert names == [
        "AI Engineering",
        "Python",
        "AI Safety",
    ]

    assert [
        item.priority_score
        for item in top
    ] == [
        0.95,
        0.85,
        0.80,
    ]

    print("TEST 56-C PASS")

# =========================================================
# Test 57-A
# Different categories should produce unique TopRecommendations
# =========================================================

def test_57_a_different_categories_unique():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test company.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "Safety.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.85,
                    "priority_level": "P1",
                }
            ],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=3,
    )

    names = [
        item.name.strip().lower()
        for item in top
    ]

    print(
        "Test 57-A names =",
        [item.name for item in top]
    )

    assert len(names) == len(set(names))

    assert len(top) == 3

    print("TEST 57-A PASS")


# =========================================================
# Test 57-B
# Same-category duplicates should not occupy multiple slots
# =========================================================

def test_57_b_same_category_duplicate_filtered():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test company.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI high priority.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                },
                {
                    "name": "Backend Engineering",
                    "rationale": "Backend.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.70,
                    "priority_level": "P2",
                },
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=2,
    )

    names = [
        item.name.strip().lower()
        for item in top
    ]

    print(
        "Test 57-B names =",
        [item.name for item in top]
    )

    assert len(top) == 2

    assert len(names) == len(set(names))

    assert top[0].name == "AI Engineering"
    assert top[1].name == "Python"

    print("TEST 57-B PASS")


# =========================================================
# Test 57-C
# Normalized duplicate names across categories
# must remain unique
# =========================================================

def test_57_c_cross_category_normalized_duplicate_unique():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test company.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "Job direction.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [
                {
                    "name": "  ai engineering  ",
                    "rationale": "Technical skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                },
                {
                    "name": "Python",
                    "rationale": "Python.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.80,
                    "priority_level": "P2",
                }
            ],

            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=3,
    )

    normalized_names = [
        item.name.strip().lower()
        for item in top
    ]

    print(
        "Test 57-C names =",
        [item.name for item in top]
    )

    print(
        "Test 57-C normalized names =",
        normalized_names
    )

    assert len(normalized_names) == len(
        set(normalized_names)
    )

    assert "ai engineering" in normalized_names

    assert "python" in normalized_names

    assert len(top) == 2

    print("TEST 57-C PASS")

def test_58_a_top_k_one_highest_priority():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.99,
                    "priority_score": 0.80,
                    "priority_level": "P2",
                }
            ],

            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=1,
    )

    print(
        "Test 58-A names =",
        [item.name for item in top]
    )

    assert len(top) == 1

    assert top[0].name == "AI Engineering"

    assert top[0].priority_score == 0.95

    print("TEST 58-A PASS")

def test_58_b_top_k_two_category_diversity():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                },
                {
                    "name": "Backend Engineering",
                    "rationale": "Backend.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                },
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.85,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=2,
    )

    names = [
        item.name
        for item in top
    ]

    categories = [
        item.category
        for item in top
    ]

    print(
        "Test 58-B names =",
        names
    )

    print(
        "Test 58-B categories =",
        categories
    )

    assert len(top) == 2

    assert names == [
        "AI Engineering",
        "Python",
    ]

    assert categories == [
        RecommendationCategory.JOB_DIRECTION,
        RecommendationCategory.TECHNICAL_SKILL,
    ]

    print("TEST 58-B PASS")

def test_58_c_larger_top_k_allows_same_category_fallback():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                },
                {
                    "name": "Backend Engineering",
                    "rationale": "Backend.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                },
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.85,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=3,
    )

    names = [
        item.name
        for item in top
    ]

    print(
        "Test 58-C names =",
        names
    )

    assert len(top) == 3

    assert names == [
        "AI Engineering",
        "Backend Engineering",
        "Python",
    ]

    assert [
        item.priority_score
        for item in top
    ] == [
        0.95,
        0.90,
        0.85,
    ]

    print("TEST 58-C PASS")

def test_59_a_build_top_recommendations_deterministic():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                },
                {
                    "name": "Backend Engineering",
                    "rationale": "Backend.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                },
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.85,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "Safety.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.88,
                    "priority_score": 0.80,
                    "priority_level": "P2",
                }
            ],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top_1 = build_top_recommendations(
        report,
        top_k=4,
    )

    top_2 = build_top_recommendations(
        report,
        top_k=4,
    )

    result_1 = [
        (
            item.name,
            item.category,
            item.claim_refs,
            item.evidence_ids,
            item.confidence,
            item.priority_score,
            item.priority_level,
        )
        for item in top_1
    ]

    result_2 = [
        (
            item.name,
            item.category,
            item.claim_refs,
            item.evidence_ids,
            item.confidence,
            item.priority_score,
            item.priority_level,
        )
        for item in top_2
    ]

    print("Test 59-A result 1 =", result_1)
    print("Test 59-A result 2 =", result_2)

    assert result_1 == result_2

    print("TEST 59-A PASS")

def test_59_b_top_k_prefix_consistency():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                },
                {
                    "name": "Backend Engineering",
                    "rationale": "Backend.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                },
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.85,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "Safety.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.75,
                    "priority_level": "P2",
                }
            ],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top_2 = build_top_recommendations(
        report,
        top_k=2,
    )

    top_3 = build_top_recommendations(
        report,
        top_k=3,
    )

    names_2 = [
        item.name
        for item in top_2
    ]

    names_3 = [
        item.name
        for item in top_3
    ]

    print(
        "Test 59-B top_k=2 =",
        names_2
    )

    print(
        "Test 59-B top_k=3 =",
        names_3
    )

    assert names_2 == [
        "AI Engineering",
        "Python",
    ]

    assert names_3[:2] == names_2

    assert names_3 == [
        "AI Engineering",
        "Python",
        "AI Safety",
    ]

    print("TEST 59-B PASS")

def test_59_c_top_k_never_exceeds_available():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.85,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    top = build_top_recommendations(
        report,
        top_k=10,
    )

    print(
        "Test 59-C names =",
        [item.name for item in top]
    )

    assert len(top) == 2

    assert len(top) <= 10

    print("TEST 59-C PASS")

def test_60_a_built_top_recommendations_validate():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI engineering relevance.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                },
                {
                    "name": "Backend Engineering",
                    "rationale": "Backend relevance.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.80,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                },
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.85,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [
                {
                    "name": "AI Safety",
                    "rationale": "AI safety relevance.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.85,
                    "priority_score": 0.80,
                    "priority_level": "P2",
                }
            ],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    report.top_recommendations = build_top_recommendations(
        report,
        top_k=4,
    )

    result = validate_top_recommendation_integrity(
        report
    )

    print(
        "Test 60-A top names =",
        [item.name for item in report.top_recommendations]
    )

    assert result is True

    print("TEST 60-A PASS")

def test_60_b_tampered_built_category_rejected():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI engineering relevance.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.85,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    report.top_recommendations = build_top_recommendations(
        report,
        top_k=2,
    )

    assert len(report.top_recommendations) == 2

    # -------------------------------------------------
    # Tamper category
    # -------------------------------------------------

    original_category = (
        report.top_recommendations[0].category
    )

    report.top_recommendations[0].category = (
        RecommendationCategory.TECHNICAL_SKILL
    )

    assert (
        report.top_recommendations[0].category
        != original_category
    )

    # -------------------------------------------------
    # Validation must reject the tampered result
    # -------------------------------------------------

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print(
            "TEST 60-B PASS"
        )

        print(
            "Expected validation error:"
        )

        print(e)

        return

    raise AssertionError(
        "TEST 60-B FAILED: "
        "tampered built TopRecommendation category "
        "was not rejected."
    )

def test_60_c_tampered_built_priority_rejected():

    report = JobAnalysisReport.model_validate({

        "company_overview": {
            "summary": "Test.",
            "evidence_ids": []
        },

        "github_signals": [],
        "news_signals": [],
        "cross_analysis": [],

        "job_value": {

            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI engineering relevance.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.95,
                    "priority_score": 0.95,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [
                {
                    "name": "Python",
                    "rationale": "Python skill.",
                    "claim_refs": [],
                    "evidence_ids": [],
                    "confidence": 0.90,
                    "priority_score": 0.85,
                    "priority_level": "P1",
                }
            ],

            "important_areas": [],
        },

        "interview_preparation": {
            "topics": [],
            "practical_tasks": [],
        },

        "reliability": {
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        "evidence": [],
    })

    # -------------------------------------------------
    # Build TopRecommendation
    # -------------------------------------------------

    report.top_recommendations = build_top_recommendations(
        report,
        top_k=2,
    )

    assert len(report.top_recommendations) == 2

    # -------------------------------------------------
    # Tamper priority_score
    # -------------------------------------------------

    original_priority = (
        report.top_recommendations[0].priority_score
    )

    report.top_recommendations[0].priority_score = 0.10

    assert (
        report.top_recommendations[0].priority_score
        != original_priority
    )

    # -------------------------------------------------
    # Validation must reject
    # -------------------------------------------------

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print(
            "TEST 60-C PASS"
        )

        print(
            "Expected validation error:"
        )

        print(e)

        return

    raise AssertionError(
        "TEST 60-C FAILED: "
        "tampered built TopRecommendation "
        "priority_score was not rejected."
    )

def test_61_a_built_claim_and_evidence_refs_preserved():

    report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI",
            "evidence_ids": [],
        },

        github_signals=[],

        news_signals=[],

        cross_analysis=[
            {
                "relationship": "AI Engineering",
                "evidence": "AI engineering evidence",
                "evidence_ids": [
                    "news:a"
                ],
                "claims": [
                    {
                        "statement": "AI engineering claim",
                        "claim_type": "fact",
                        "evidence_ids": [
                            "news:a"
                        ],
                        "confidence": 0.90,
                    }
                ],
            }
        ],

        job_value={
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI engineering.",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:a"
                    ],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [],

            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    report.top_recommendations = (
        build_top_recommendations(
            report,
            top_k=1,
        )
    )

    assert len(
        report.top_recommendations
    ) == 1

    top = report.top_recommendations[0]

    print(
        "Test 61-A claim_refs =",
        top.claim_refs
    )

    print(
        "Test 61-A evidence_ids =",
        top.evidence_ids
    )

    assert top.claim_refs == [
        "cross_analysis:0.claims:0"
    ]

    assert top.evidence_ids == [
        "news:a"
    ]

    result = validate_top_recommendation_integrity(
        report
    )

    assert result is True

    print("TEST 61-A PASS")

def test_61_b_tampered_claim_refs_rejected():

    report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI",
            "evidence_ids": [],
        },

        github_signals=[],
        news_signals=[],

        cross_analysis=[
            {
                "relationship": "AI Engineering",
                "evidence": "AI engineering evidence",
                "evidence_ids": [
                    "news:a"
                ],
                "claims": [
                    {
                        "statement": "AI engineering claim",
                        "claim_type": "fact",
                        "evidence_ids": [
                            "news:a"
                        ],
                        "confidence": 0.90,
                    }
                ],
            }
        ],

        job_value={
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI engineering.",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:a"
                    ],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [],
            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    # -----------------------------------------
    # Build
    # -----------------------------------------

    report.top_recommendations = (
        build_top_recommendations(
            report,
            top_k=1,
        )
    )

    assert len(
        report.top_recommendations
    ) == 1

    # -----------------------------------------
    # Tamper claim_refs
    # -----------------------------------------

    report.top_recommendations[0].claim_refs = [
        "cross_analysis:999.claims:999"
    ]

    # -----------------------------------------
    # Validation must reject
    # -----------------------------------------

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print("TEST 61-B PASS")

        print(
            "Expected validation error:"
        )

        print(e)

        return

    raise AssertionError(
        "TEST 61-B FAILED: "
        "tampered TopRecommendation claim_refs "
        "was not rejected."
    )

def test_61_c_tampered_evidence_ids_rejected():

    report = JobAnalysisReport(
        company_overview={
            "summary": "OpenAI",
            "evidence_ids": [],
        },

        github_signals=[],
        news_signals=[],

        cross_analysis=[
            {
                "relationship": "AI Engineering",
                "evidence": "AI engineering evidence",
                "evidence_ids": [
                    "news:a"
                ],
                "claims": [
                    {
                        "statement": "AI engineering claim",
                        "claim_type": "fact",
                        "evidence_ids": [
                            "news:a"
                        ],
                        "confidence": 0.90,
                    }
                ],
            }
        ],

        job_value={
            "job_directions": [
                {
                    "name": "AI Engineering",
                    "rationale": "AI engineering.",
                    "claim_refs": [
                        "cross_analysis:0.claims:0"
                    ],
                    "evidence_ids": [
                        "news:a"
                    ],
                    "confidence": 0.90,
                    "priority_score": 0.90,
                    "priority_level": "P1",
                }
            ],

            "technical_skills": [],
            "important_areas": [],
        },

        interview_preparation={
            "topics": [],
            "practical_tasks": [],
        },

        reliability={
            "supported_conclusions": [],
            "uncertain_conclusions": [],
            "limitations": [],
        },

        evidence=[],
    )

    # -----------------------------------------
    # Build
    # -----------------------------------------

    report.top_recommendations = (
        build_top_recommendations(
            report,
            top_k=1,
        )
    )

    assert len(
        report.top_recommendations
    ) == 1

    # -----------------------------------------
    # Tamper evidence_ids
    # -----------------------------------------

    report.top_recommendations[0].evidence_ids = [
        "news:tampered"
    ]

    # -----------------------------------------
    # Validation must reject
    # -----------------------------------------

    try:

        validate_top_recommendation_integrity(
            report
        )

    except ValueError as e:

        print("TEST 61-C PASS")

        print(
            "Expected validation error:"
        )

        print(e)

        return

    raise AssertionError(
        "TEST 61-C FAILED: "
        "tampered TopRecommendation evidence_ids "
        "was not rejected."
    )



if __name__ == "__main__":
    test_category_aware_top_recommendations()
    test_category_aware_top_recommendations_edge_cases()
    test_category_aware_top_recommendations_single_category()
