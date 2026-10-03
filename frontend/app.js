const API_BASE = "http://127.0.0.1:8010";

const form = document.getElementById("searchForm");
const keywordInput = document.getElementById("keyword");
const analyzeButton = document.getElementById("analyzeButton");

const loadingState = document.getElementById("loadingState");
const errorState = document.getElementById("errorState");
const errorText = document.getElementById("errorText");
const dashboard = document.getElementById("dashboard");

const statusText = document.getElementById("statusText");
const durationText = document.getElementById("durationText");

const retryButton = document.getElementById("retryButton");

let lastKeyword = keywordInput.value.trim();

let currentReport = null;

let currentNewsFilter = "all";
let currentGithubFilter = "all";


function setLoading(isLoading) {

  analyzeButton.disabled = isLoading;

  analyzeButton.textContent =
    isLoading
      ? "Analyzing"
      : "Analyze";

  loadingState.classList.toggle(
    "hidden",
    !isLoading
  );
}


function showError(message) {

  errorText.textContent = message;

  errorState.classList.remove(
    "hidden"
  );
}


function clearError() {

  errorState.classList.add(
    "hidden"
  );

  errorText.textContent = "";
}


function formatDate(value) {

  if (!value) {
    return "";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}


function escapeHtml(value) {

  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}


/* =========================================================
   NEWS
   ========================================================= */

function renderNews(
  items = [],
  filter = currentNewsFilter
) {

  const root =
    document.getElementById(
      "newsList"
    );

  const filteredItems =
    filter === "all"
      ? items
      : items.filter(
          item =>
            item.relevance === filter
        );

  if (!filteredItems.length) {

    root.innerHTML = `
      <div class="empty">
        No ${escapeHtml(filter)}
        news signals.
      </div>
    `;

    return;
  }

  root.innerHTML =
    filteredItems
      .map(item => {

        const relevance =
          item.relevance || "industry";

        return `
          <article
            class="signal-card"
          >

            <div class="signal-top">

              <div>

                <h3
                  class="signal-title"
                >
                  ${escapeHtml(
                    item.title
                  )}
                </h3>

                <div
                  class="signal-meta"
                >
                  ${escapeHtml(
                    item.source
                  )}
                  路
                  ${escapeHtml(
                    formatDate(
                      item.published_at
                    )
                  )}
                </div>

              </div>

              <span
                class="badge ${escapeHtml(
                  relevance
                )}"
              >
                ${escapeHtml(
                  relevance
                )}
              </span>

            </div>

            <p
              class="signal-summary"
            >
              ${escapeHtml(
                item.summary
              )}
            </p>

            ${
              item.implication
                ? `
                  <p
                    class="signal-summary"
                  >
                    <strong>
                      Implication:
                    </strong>

                    ${escapeHtml(
                      item.implication
                    )}
                  </p>
                `
                : ""
            }

            ${
              item.evidence
                ? `
                  <p
                    class="signal-summary"
                  >
                    <strong>
                      Evidence:
                    </strong>

                    ${escapeHtml(
                      item.evidence
                    )}
                  </p>
                `
                : ""
            }

            ${
              item.source_id
                ? `
                  <button
                    type="button"
                    class="evidence-jump-button"
                    data-source-id="${escapeHtml(
                      item.source_id
                    )}"
                  >
                    View evidence
                  </button>
                `
                : ""
            }

          </article>
        `;
      })
      .join("");
}


/* =========================================================
   GITHUB
   ========================================================= */

function renderGithub(
  items = [],
  filter = currentGithubFilter
) {

  const root =
    document.getElementById(
      "githubList"
    );

  const filteredItems =
    filter === "all"
      ? items
      : items.filter(
          item =>
            item.relevance === filter
        );

  if (!filteredItems.length) {

    root.innerHTML = `
      <div class="empty">
        No ${escapeHtml(filter)}
        GitHub signals.
      </div>
    `;

    return;
  }

  root.innerHTML =
    filteredItems
      .map(item => {

        const relevance =
          item.relevance || "ecosystem";

        return `
          <article
            class="signal-card"
          >

            <div class="signal-top">

              <div>

                <h3
                  class="signal-title"
                >
                  ${escapeHtml(
                    item.project
                  )}
                </h3>

                <div
                  class="signal-meta"
                >
                  ${escapeHtml(
                    item.language ||
                    "Unknown language"
                  )}
                </div>

              </div>

              <span
                class="badge ${escapeHtml(
                  relevance
                )}"
              >
                ${escapeHtml(
                  relevance
                )}
              </span>

            </div>

            <p
              class="signal-summary"
            >
              <strong>
                Direction:
              </strong>

              ${escapeHtml(
                item.technical_direction
              )}
            </p>

            <p
              class="signal-summary"
            >
              <strong>
                Activity:
              </strong>

              ${escapeHtml(
                item.activity
              )}
            </p>

            <p
              class="signal-summary"
            >
              <strong>
                Evidence:
              </strong>

              ${escapeHtml(
                item.evidence
              )}
            </p>

            ${
              item.source_id
                ? `
                  <button
                    type="button"
                    class="evidence-jump-button"
                    data-source-id="${escapeHtml(
                      item.source_id
                    )}"
                  >
                    View evidence
                  </button>
                `
                : ""
            }

          </article>
        `;
      })
      .join("");
}


/* =========================================================
   CLAIMS
   ========================================================= */

function flattenClaims(
  crossAnalysis = []
) {

  return crossAnalysis.flatMap(
    (
      analysis,
      analysisIndex
    ) =>

      (analysis.claims || [])
        .map(
          (
            claim,
            claimIndex
          ) => ({
            ...claim,

            relationship:
              analysis.relationship,

            analysisIndex,

            claimIndex,
          })
        )
  );
}

function getConfidenceLevel(confidence) {

  if (confidence >= 0.80) {
    return {
      label: "High",
      className: "high",
    };
  }

  if (confidence >= 0.50) {
    return {
      label: "Medium",
      className: "medium",
    };
  }

  return {
    label: "Low",
    className: "low",
  };
}

function renderClaims(crossAnalysis = []) {

  const claims =
    flattenClaims(crossAnalysis);

  const root =
    document.getElementById("claimsList");

  if (!claims.length) {

    root.innerHTML = `
      <div class="empty">
        No claims returned.
      </div>
    `;

    return;
  }

  root.innerHTML =
    claims.map(claim => {

      const confidence =
        Math.max(
          0,
          Math.min(
            1,
            Number(claim.confidence) || 0
          )
        );

      const pct =
        Math.round(
          confidence * 100
        );

      const level =
        getConfidenceLevel(
          confidence
        );

      const sources =
        (claim.evidence_ids || [])
          .map(
            id => `
              <button
                type="button"
                class="source-chip source-chip-button"
                data-source-id="${escapeHtml(id)}"
              >
                ${escapeHtml(id)}
              </button>
            `
          )
          .join("");

      return `
        <article class="claim-card">

          <div class="claim-top">

            <p class="claim-statement">
              ${escapeHtml(
                claim.statement
              )}
            </p>

            <span class="claim-type">
              ${escapeHtml(
                claim.claim_type
              )}
            </span>

          </div>

          ${
            claim.relationship
              ? `
                <p class="signal-meta">
                  ${escapeHtml(
                    claim.relationship
                  )}
                </p>
              `
              : ""
          }

          <div class="confidence">

            <div class="confidence-row">

              <div class="confidence-label">
                <span>Confidence</span>

                <span
                  class="confidence-level ${level.className}"
                >
                  ${level.label}
                </span>
              </div>

              <strong>
                ${pct}%
              </strong>

            </div>

            <div class="progress">
              <span
                class="${level.className}"
                style="width:${pct}%"
              ></span>
            </div>

          </div>

          <div class="claim-sources">

            ${
              sources ||
              `
                <span class="source-chip">
                  No evidence IDs
                </span>
              `
            }

          </div>

        </article>
      `;

    }).join("");
}

function renderRecommendationItem(item = {}) {

  // =========================================
  // Recommendation confidence
  // =========================================

  const confidence =
    Math.max(
      0,
      Math.min(
        1,
        Number(item.confidence) || 0
      )
    );

  const pct =
    Math.round(
      confidence * 100
    );

  const level =
    getConfidenceLevel(
      confidence
    );


  // =========================================
  // Recommendation priority
  // =========================================

  const priorityScore =
    Math.max(
      0,
      Math.min(
        1,
        Number(item.priority_score) || 0
      )
    );

  const priorityLevel =
    item.priority_level || "P3";

  const priorityPct =
    Math.round(
      priorityScore * 100
    );


  // =========================================
  // Claim references
  // =========================================

  const claimButtons =
    (item.claim_refs || [])
      .map(
        ref => `
          <button
            type="button"
            class="recommendation-ref claim-ref-button"
            data-claim-ref="${escapeHtml(ref)}"
          >
            ${escapeHtml(ref)}
          </button>
        `
      )
      .join("");


  // =========================================
  // Evidence references
  // =========================================

  const evidenceButtons =
    (item.evidence_ids || [])
      .map(
        id => `
          <button
            type="button"
            class="recommendation-ref evidence-ref-button"
            data-source-id="${escapeHtml(id)}"
          >
            ${escapeHtml(id)}
          </button>
        `
      )
      .join("");


  // =========================================
  // Render
  // =========================================

  return `
    <article class="recommendation-card">

      <div class="recommendation-header">

        <div>

          <h3 class="recommendation-title">
            ${escapeHtml(
              item.name
            )}
          </h3>

          <p class="recommendation-rationale">
            ${escapeHtml(
              item.rationale
            )}
          </p>

        </div>


        <div class="recommendation-badges">

          <span
            class="priority-badge ${escapeHtml(
              priorityLevel.toLowerCase()
            )}"
          >
            ${escapeHtml(
              priorityLevel
            )}
          </span>

          <span
            class="confidence-level ${level.className}"
          >
            ${level.label}
          </span>

        </div>

      </div>


      <!-- Priority -->

      <div class="recommendation-priority">

        <div class="confidence-row">

          <span>
            Priority
          </span>

          <strong>
            ${priorityPct}%
          </strong>

        </div>

        <div class="progress">

          <span
            class="priority-progress ${escapeHtml(
              priorityLevel.toLowerCase()
            )}"
            style="width:${priorityPct}%"
          ></span>

        </div>

      </div>


      <!-- Recommendation confidence -->

      <div class="recommendation-confidence">

        <div class="confidence-row">

          <span>
            Recommendation confidence
          </span>

          <strong>
            ${pct}%
          </strong>

        </div>

        <div class="progress">

          <span
            class="${level.className}"
            style="width:${pct}%"
          ></span>

        </div>

      </div>


      <!-- Claims -->

      ${
        claimButtons
          ? `
            <div class="recommendation-block">

              <div class="recommendation-block-label">
                CLAIMS
              </div>

              <div class="recommendation-refs">
                ${claimButtons}
              </div>

            </div>
          `
          : ""
      }


      <!-- Evidence -->

      ${
        evidenceButtons
          ? `
            <div class="recommendation-block">

              <div class="recommendation-block-label">
                EVIDENCE
              </div>

              <div class="recommendation-refs">
                ${evidenceButtons}
              </div>

            </div>
          `
          : ""
      }

    </article>
  `;
}


function renderRecommendationList(
  items = [],
  rootId
) {

  const root =
    document.getElementById(
      rootId
    );

  if (!root) {
    return;
  }

  if (!items.length) {

    root.innerHTML = `
      <div class="empty">
        No recommendations available.
      </div>
    `;

    return;
  }

  const sortedItems =
    [...items].sort(
      (a, b) => {

        const priorityDiff =
          (
            Number(
              b.priority_score
            ) || 0
          )
          -
          (
            Number(
              a.priority_score
            ) || 0
          );

        if (priorityDiff !== 0) {
          return priorityDiff;
        }

        return (
          (
            Number(
              b.confidence
            ) || 0
          )
          -
          (
            Number(
              a.confidence
            ) || 0
          )
        );
      }
    );

  root.innerHTML =
    sortedItems
      .map(
        item =>
          renderRecommendationItem(item)
      )
      .join("");
}


function renderJobValue(jobValue = {}) {

  renderRecommendationList(
    jobValue.job_directions || [],
    "jobDirections"
  );

  renderRecommendationList(
    jobValue.technical_skills || [],
    "technicalSkills"
  );

  renderRecommendationList(
    jobValue.important_areas || [],
    "importantAreas"
  );
}


function renderInterviewPreparation(
  preparation = {}
) {

  renderRecommendationList(
    preparation.topics || [],
    "interviewTopics"
  );

  renderRecommendationList(
    preparation.practical_tasks || [],
    "interviewTasks"
  );
}


/* =========================================================
   EVIDENCE
   ========================================================= */

function renderEvidence(
  items = []
) {

  const root =
    document.getElementById(
      "evidenceList"
    );

  if (!items.length) {

    root.innerHTML = `
      <div class="empty">
        No evidence sources.
      </div>
    `;

    return;
  }

  root.innerHTML =
    items.map(item => {

      const sourceId =
        escapeHtml(
          item.source_id
        );

      return `
        <article
          id="evidence-${sourceId}"
          class="evidence-item"
          data-source-id="${sourceId}"
        >

          <div
            class="evidence-header"
          >

            <div>

              <h3
                class="evidence-title"
              >
                ${escapeHtml(
                  item.title
                )}
              </h3>

              <div
                class="signal-meta"
              >
                ${escapeHtml(
                  item.source_type
                )}

                路

                ${escapeHtml(
                  item.source_id
                )}
              </div>

            </div>

          </div>

          <a
            class="evidence-link"
            href="${escapeHtml(
              item.url
            )}"
            target="_blank"
            rel="noopener noreferrer"
          >
            ${escapeHtml(
              item.url
            )}
          </a>

          ${
            item.description
              ? `
                <p
                  class="evidence-snippet"
                >
                  ${escapeHtml(
                    item.description
                  )}
                </p>
              `
              : ""
          }

          ${
            item.snippet
              ? `
                <p
                  class="evidence-snippet"
                >
                  <strong>
                    Snippet:
                  </strong>

                  ${escapeHtml(
                    item.snippet
                  )}
                </p>
              `
              : ""
          }

        </article>
      `;
    }).join("");
}



/* =========================================================
   CLAIM 鈫?EVIDENCE
   ========================================================= */

function focusEvidence(sourceId) {

  if (!sourceId) {
    return;
  }

  const element =
    document.getElementById(
      `evidence-${sourceId}`
    );

  if (!element) {

    console.warn(
      "Evidence not found:",
      sourceId
    );

    return;
  }

  document
    .querySelectorAll(
      ".evidence-item.selected"
    )
    .forEach(item => {
      item.classList.remove(
        "selected"
      );
    });

  element.classList.add(
    "selected"
  );

  element.scrollIntoView({
    behavior: "smooth",
    block: "center",
  });

  window.setTimeout(() => {

    element.classList.remove(
      "selected"
    );

  }, 2500);
}

/* =========================================================
   DASHBOARD
   ========================================================= */

function renderDashboard(
  data,
  keyword
) {

  const report =
    data?.analysis ?? data;

  const claims =
    flattenClaims(
      report.cross_analysis || []
    );

  currentReport = report;

  const recommendations = [
  ...(report.job_value?.job_directions || []),

  ...(report.job_value?.technical_skills || []),

  ...(report.job_value?.important_areas || []),

  ...(report.interview_preparation?.topics || []),

  ...(report.interview_preparation?.practical_tasks || [])
];

const firstRecommendation =
  recommendations[0];

const priorityDebug =
  firstRecommendation
    ? `score=${firstRecommendation.priority_score ?? "missing"} | level=${firstRecommendation.priority_level ?? "missing"}`
    : "no recommendations";

const p1Count =
  recommendations.filter(
    item =>
      item.priority_level === "P1"
  ).length;

  document.getElementById(
    "companyName"
  ).textContent =
    keyword;

  document.getElementById(
    "overviewSummary"
  ).textContent =
    report.company_overview
      ?.summary ||
    "No overview available.";

  document.getElementById(
    "newsCount"
  ).textContent =
    (
      report.news_signals || []
    ).length;

  document.getElementById(
    "githubCount"
  ).textContent =
    (
      report.github_signals || []
    ).length;

  document.getElementById(
    "claimCount"
  ).textContent =
    claims.length;

  document.getElementById(
    "p1Count"
  ).textContent =
    p1Count;

  document.getElementById(
    "priorityDebug"
  ).textContent =
    priorityDebug;

  document.getElementById(
    "evidenceCount"
  ).textContent =
    `${
      (
        report.evidence || []
      ).length
    } evidence`;

  renderNews(
  report.news_signals || [],
  currentNewsFilter
);

renderGithub(
  report.github_signals || [],
  currentGithubFilter
);

renderClaims(
  report.cross_analysis || []
);

renderTopRecommendations(
  report.top_recommendations || []
);

renderJobValue(
  report.job_value || {}
);

renderInterviewPreparation(
  report.interview_preparation || {}
);



renderEvidence(
  report.evidence || []
);

  dashboard.classList.remove(
    "hidden"
  );

  document.getElementById(
    "lastUpdated"
  ).textContent =
    `Updated ${
      new Date().toLocaleTimeString()
    }`;
}


/* =========================================================
   API
   ========================================================= */

async function analyzeKeyword(
  keyword
) {

  const cleanKeyword =
    keyword.trim();

  if (!cleanKeyword) {
    return;
  }

  lastKeyword =
    cleanKeyword;

  clearError();

  dashboard.classList.add(
    "hidden"
  );

  setLoading(true);

  statusText.textContent =
    "Running analysis";

  durationText.textContent =
    "";

  const started =
    performance.now();

  try {

    const url =
      `${API_BASE}/analyze?keyword=${
        encodeURIComponent(
          cleanKeyword
        )
      }`;

    const response =
      await fetch(
        url,
        {
          method: "POST",

          headers: {
            "Accept":
              "application/json",
          },
        }
      );

    const raw =
      await response.text();

    let payload;

    try {

      payload =
        raw
          ? JSON.parse(raw)
          : null;

    } catch {

      throw new Error(
        `Backend returned invalid JSON (${response.status}).`
      );
    }

    if (!response.ok) {

      const detail =
        payload?.detail ||
        payload?.message ||
        raw ||
        `HTTP ${response.status}`;

      throw new Error(
        String(detail)
      );
    }

    renderDashboard(
      payload,
      cleanKeyword
    );

    const seconds =
      (
        (
          performance.now() -
          started
        ) / 1000
      ).toFixed(1);

    statusText.textContent =
      "Analysis complete";

    durationText.textContent =
      `${seconds}s`;

  } catch (error) {

    const message =
      error instanceof Error
        ? error.message
        : String(error);

    showError(message);

    statusText.textContent =
      "Request failed";

    durationText.textContent =
      "";

  } finally {

    setLoading(false);
  }
}


/* =========================================================
   SEARCH
   ========================================================= */

form.addEventListener(
  "submit",
  event => {

    event.preventDefault();

    analyzeKeyword(
      keywordInput.value
    );
  }
);


retryButton.addEventListener(
  "click",
  () => {

    analyzeKeyword(
      lastKeyword ||
      keywordInput.value
    );

  }
);


/* =========================================================
   FILTERS
   ========================================================= */

document
  .getElementById(
    "newsFilters"
  )
  .addEventListener(
    "click",
    event => {

      const button =
        event.target.closest(
          ".filter-button"
        );

      if (!button) {
        return;
      }

      currentNewsFilter =
        button.dataset.filter;

      document
        .querySelectorAll(
          "#newsFilters .filter-button"
        )
        .forEach(
          item =>
            item.classList.toggle(
              "active",
              item === button
            )
        );

      renderNews(
        currentReport?.news_signals ||
          [],
        currentNewsFilter
      );
    }
  );


document
  .getElementById(
    "githubFilters"
  )
  .addEventListener(
    "click",
    event => {

      const button =
        event.target.closest(
          ".filter-button"
        );

      if (!button) {
        return;
      }

      currentGithubFilter =
        button.dataset.filter;

      document
        .querySelectorAll(
          "#githubFilters .filter-button"
        )
        .forEach(
          item =>
            item.classList.toggle(
              "active",
              item === button
            )
        );

      renderGithub(
        currentReport?.github_signals ||
          [],
        currentGithubFilter
      );
    }
  );


/* =========================================================
   EVIDENCE CLICK HANDLERS
   ========================================================= */

document.addEventListener(
  "click",
  event => {

    const claimButton =
      event.target.closest(
        "[data-claim-ref]"
      );

    if (claimButton) {

      const claimRef =
        claimButton.dataset.claimRef;

      focusClaim(
        claimRef
      );

      return;
    }


    const evidenceButton =
      event.target.closest(
        "[data-source-id]"
      );

    if (!evidenceButton) {
      return;
    }

    if (
      evidenceButton.classList.contains(
        "filter-button"
      )
    ) {
      return;
    }

    const sourceId =
      evidenceButton.dataset.sourceId;

    if (!sourceId) {
      return;
    }

    focusEvidence(
      sourceId
    );
  }
);


document.getElementById(
  "apiBadge"
).textContent =
  `API: ${API_BASE.replace(
    "http://",
    ""
  )}`;

function findClaimByRef(
  claimRef
) {

  const match =
    /^cross_analysis:(\d+)\.claims:(\d+)$/
      .exec(claimRef);

  if (!match) {
    return null;
  }

  const analysisIndex =
    Number(match[1]);

  const claimIndex =
    Number(match[2]);

  const analysis =
    currentReport?.cross_analysis?.[
      analysisIndex
    ];

  if (!analysis) {
    return null;
  }

  return analysis.claims?.[
    claimIndex
  ] || null;
}

function focusClaim(
  claimRef
) {

  if (!claimRef) {
    return;
  }

  const claims =
    flattenClaims(
      currentReport?.cross_analysis || []
    );

  const claim =
    claims.find(item =>
      item.analysisIndex === Number(
        claimRef.match(
          /^cross_analysis:(\d+)\.claims:(\d+)$/
        )?.[1]
      )
      &&
      item.claimIndex === Number(
        claimRef.match(
          /^cross_analysis:(\d+)\.claims:(\d+)$/
        )?.[2]
      )
    );

  if (!claim) {

    console.warn(
      "Claim not found:",
      claimRef
    );

    return;
  }

  const root =
    document.getElementById(
      "claimsList"
    );

  const cards =
    root.querySelectorAll(
      ".claim-card"
    );

  const index =
    claims.indexOf(claim);

  const card =
    cards[index];

  if (!card) {
    return;
  }

  document
    .querySelectorAll(
      ".claim-card.selected"
    )
    .forEach(item =>
      item.classList.remove(
        "selected"
      )
    );

  card.classList.add(
    "selected"
  );

  card.scrollIntoView({
    behavior: "smooth",
    block: "center"
  });

  window.setTimeout(
    () => {
      card.classList.remove(
        "selected"
      );
    },
    2500
  );
}

function formatRecommendationCategory(
  category
) {

  const labels = {
    job_direction: "Job Direction",
    technical_skill: "Technical Skill",
    important_area: "Important Area",
    interview_topic: "Interview Topic",
    practical_task: "Practical Task",
  };

  return (
    labels[category] ||
    category ||
    "Recommendation"
  );
}

function renderTopRecommendations(
  items = []
) {

  const root =
    document.getElementById(
      "topRecommendationsList"
    );

  if (!root) {
    return;
  }

  if (!items.length) {

    root.innerHTML = `
      <div class="empty">
        No top recommendations available.
      </div>
    `;

    return;
  }

  root.innerHTML =
    items
      .map(
        (item, index) => {

          const priorityScore =
            Math.max(
              0,
              Math.min(
                1,
                Number(
                  item.priority_score
                ) || 0
              )
            );

          const priorityPct =
            Math.round(
              priorityScore * 100
            );

          const priorityLevel =
            item.priority_level ||
            "P3";

          const confidence =
            Math.max(
              0,
              Math.min(
                1,
                Number(
                  item.confidence
                ) || 0
              )
            );

          const confidencePct =
            Math.round(
              confidence * 100
            );

          const category =
            formatRecommendationCategory(
              item.category
            );

          const claimButtons =
            (
              item.claim_refs || []
            )
              .map(
                ref => `
                  <button
                    type="button"
                    class="recommendation-ref claim-ref-button"
                    data-claim-ref="${escapeHtml(ref)}"
                  >
                    ${escapeHtml(ref)}
                  </button>
                `
              )
              .join("");

          const evidenceButtons =
            (
              item.evidence_ids || []
            )
              .map(
                id => `
                  <button
                    type="button"
                    class="recommendation-ref evidence-ref-button"
                    data-source-id="${escapeHtml(id)}"
                  >
                    ${escapeHtml(id)}
                  </button>
                `
              )
              .join("");

          return `
            <article
              class="top-recommendation-card"
            >

              <div
                class="top-recommendation-rank"
              >
                #${index + 1}
              </div>

              <div
                class="top-recommendation-main"
              >

                <div
                  class="top-recommendation-header"
                >

                  <div>

                    <h3
                      class="recommendation-title"
                    >
                      ${escapeHtml(
                        item.name
                      )}
                    </h3>

                    <div
                      class="top-recommendation-meta"
                    >

                      <span
                        class="category-badge"
                      >
                        ${escapeHtml(
                          category
                        )}
                      </span>

                      <span
                        class="priority-badge ${escapeHtml(
                          priorityLevel.toLowerCase()
                        )}"
                      >
                        ${escapeHtml(
                          priorityLevel
                        )}
                      </span>

                    </div>

                  </div>

                  <strong
                    class="top-recommendation-score"
                  >
                    ${priorityPct}%
                  </strong>

                </div>

                <p
                  class="recommendation-rationale"
                >
                  ${escapeHtml(
                    item.rationale
                  )}
                </p>

                <div
                  class="recommendation-confidence"
                >

                  <div
                    class="confidence-row"
                  >
                    <span>
                      Priority
                    </span>

                    <strong>
                      ${priorityPct}%
                    </strong>
                  </div>

                  <div
                    class="progress"
                  >
                    <span
                      class="priority-progress ${escapeHtml(
                        priorityLevel.toLowerCase()
                      )}"
                      style="width:${priorityPct}%"
                    ></span>
                  </div>

                </div>

                <div
                  class="confidence-row"
                  style="margin-top:10px;"
                >
                  <span>
                    Recommendation confidence
                  </span>

                  <strong>
                    ${confidencePct}%
                  </strong>
                </div>

                ${
                  claimButtons
                    ? `
                      <div
                        class="recommendation-block"
                      >

                        <div
                          class="recommendation-block-label"
                        >
                          CLAIMS
                        </div>

                        <div
                          class="recommendation-refs"
                        >
                          ${claimButtons}
                        </div>

                      </div>
                    `
                    : ""
                }

                ${
                  evidenceButtons
                    ? `
                      <div
                        class="recommendation-block"
                      >

                        <div
                          class="recommendation-block-label"
                        >
                          EVIDENCE
                        </div>

                        <div
                          class="recommendation-refs"
                        >
                          ${evidenceButtons}
                        </div>

                      </div>
                    `
                    : ""
                }

              </div>

            </article>
          `;
        }
      )
      .join("");
}


/* =========================================================
   V2.9-3.1 API HEALTH STATUS
   ========================================================= */

async function refreshApiStatus() {
  const badge = document.getElementById("apiBadge");

  if (!badge) {
    return;
  }

  if (window.location.protocol === "file:") {
    badge.textContent = `API: ${API_BASE.replace("http://", "")}`;
    return;
  }

  badge.classList.remove("online", "offline");
  badge.textContent = "API CHECKING?";

  try {
    const response = await fetch(`${API_BASE}/health`, {
      method: "GET",
      cache: "no-store",
    });

    const health = await response.json();

    if (!response.ok || health.status !== "ok") {
      throw new Error("Health check failed");
    }

    badge.textContent = `API ONLINE ? v${health.version || "2.9.0"}`;
    badge.classList.add("online");
    badge.title = `${health.service || "AI-OSINT-Agent"} is running`;

    if (statusText.textContent === "Ready") {
      statusText.textContent = "API connected ? Ready";
    }
  } catch (error) {
    badge.textContent = "API OFFLINE";
    badge.classList.add("offline");
    badge.title = "Backend API is not reachable";
  }
}

refreshApiStatus();
