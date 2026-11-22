# Smart Beneficiary Targeting Validation — Updated Project Plan

**Status:** Phases 1–2 complete; proceed with Phases 3–6  
**Data source:** `global_cbpf_project_summary.csv` only  
**Goal:** Flag unusually high or low beneficiary-to-budget ratios relative to comparable humanitarian projects, recommend peers for benchmarking, and visualize historical changes in the ratio.

> **Interpretation:** This is a project-level statistical screening and decision-support tool, **not** a way to determine whether a project was truly inefficient or whether its beneficiary targets matched humanitarian needs. The source lacks verified crisis-severity measures and an explicit humanitarian cluster field.

## Primary metric

$$
\text{Beneficiaries per \$1,000} = \frac{\text{Men} + \text{Women} + \text{Boys} + \text{Girls}}{\text{Budget}}\times 1000
$$

A result of **25** means that the project's reported beneficiary count corresponds to **25 beneficiaries per $1,000 of budget**. Higher values represent more reported beneficiaries per budget dollar; they do **not** automatically imply better targeting. Confirm whether the count fields represent targeted versus reached beneficiaries, and whether these categories can be added without overlap.

## Overall pipeline

1. **Phase 1 — EDA** (completed)
2. **Phase 2 — Cleaning and feature engineering** (completed and verified)
3. **Phase 3 — Sector audit and comparable-project matching** (next)
4. **Phase 4 — Peer-relative anomaly scoring**
5. **Phase 5 — Model evaluation and sensitivity testing**
6. **Phase 6 — Interactive product, testing and deployment**

---

## Phase 1 — Exploratory Data Analysis (COMPLETE)

**Objective:** Understand project granularity, feature distributions, missingness, and the suitability of candidate similarity variables.

Review and retain the completed EDA findings, particularly:
- Budget, beneficiary count, and beneficiaries-per-$1,000 distributions; investigate heavy tails and possible data errors.
- Differences across `PooledFundName`, allocation/submission year, `OrganizationType`, and duration.
- Uniqueness of `ChfProjectCode`; whether non-direct-service projects need separate treatment.
- Missing explicit humanitarian cluster, event type, and crisis-severity labels.
- Meaning of `DateSubmitted`: submission timing, **not** disaster timing or time of service delivery.

**Deliverable:** Reproducible Phase 1 EDA and documented limitations. Do not redo the phase unless downstream audits uncover issues.

---

## Phase 2 — Cleaning, Features and Trend Aggregation (COMPLETE AND VERIFIED)

**Notebook:** `Phase2_Cleaning_Features/Phase2_cleaning_features.ipynb` (9 cells executed successfully; all programmed validations passed).

### Saved outputs (`Phase2_Cleaning_Features/outputs/`)

| Output | Contents | Intended downstream use |
|---|---|---|
| `project_level_clean.parquet` / `.csv` | 17,581 rows × 51 columns; one row per unique project, with flags and derived features | Source of truth for matching and anomaly scoring |
| `country_period_trend.parquet` / `.csv` | 1,204 fund-months across 37 funds; counts, summed budgets/beneficiaries, pooled ratio, median project ratio, IQR, sample sufficiency | Dashboard historical trend view |
| `data_dictionary.csv` | 16 documented derived fields with definitions/caveats | Model documentation, transparency and dashboard glossary |

### Verified findings

- **17,084** projects eligible for benchmarking; **497** excluded from scoring, retained with reason flags. Two records have non-positive budgets. Do not label excluded cases as anomalous.
- Primary `beneficiaries_per_1000` median: **34.6**; maximum approximately **70,000**, indicating a substantial right tail requiring an extreme-value audit.
- `duration_months` populated for **17,564** records; median **8 months**.
- `DateSubmitted` parsed successfully for all records and used to engineer month, quarter, and year features.
- Sector classifications are **inferred from titles**, not official cluster labels; their counts and reliability still need inspection.
- The country-period pooled ratio is calculated as the **ratio of sums**, not the unweighted average of project ratios:

$$
R_{c,t}=\frac{\sum_{i\in(c,t)}\text{Beneficiaries}_i}{\sum_{i\in(c,t)}\text{Budget}_i}\times 1000
$$

- Verification example: **South Sudan, January 2019**, 166 projects: pooled ratio **82.2**, versus unweighted project mean **98.9**. These answer different questions.
- **699 of 1,204 fund-months (58%)** have at least five eligible projects; 42% are sparse, so the dashboard must show warnings and offer coarser aggregation.

**Deliverable:** Phase 2 artifacts above. Preserve the notebook and source outputs; add new model outputs separately.

---

## Phase 3 — Audit Sector Labels and Match Comparable Projects (NEXT)

**Objective:** Build a simple, interpretable peer-retrieval system before deciding whether a project's ratio is unusual.

### 3.1 Audit inferred sector labels (do this first)

1. Tabulate counts and percentages by inferred `sector_label` (including `Unknown`, `Multisector`, and missing labels if applicable).
2. Cross-tabulate sectors by fund and year to identify small or imbalanced groups.
3. Manually review approximately **50–100** stratified project titles, including multilingual, ambiguous and multisector examples.
4. Record evident mismatches and whether title-derived classifications are sufficiently reliable for **hard filtering**, **soft preference**, or **display only**.
5. Never present title-inferred sectors as verified OCHA cluster labels. If sector reliability is weak, omit it from hard matching and disclose the limitation.

**Decision gate:** Choose the treatment of sectors only after the audit. Classification confidence is not a verified probability unless calibrated.

### 3.2 Candidate features and matching rules

| Feature | Proposed use |
|---|---|
| `PooledFundName` | Prefer the same fund/country; relax if peer set is too small |
| Sector inferred from `ProjectTitle` | Prefer same sector **only if** audit supports its use |
| Submission/allocation year | Prefer ±2 years; use one consistent date basis and disclose it |
| `log_budget` | Numeric distance for financial scale |
| `duration_months` | Numeric distance for comparable project scope |
| `OrganizationType` | Optional categorical match/penalty |

**Exclude from peer similarity:** `beneficiaries_per_1000`, `cost_per_beneficiary`, `log_ratio`, and `total_beneficiaries`—these are the outcomes under examination or encode them. Do not choose peers because their ratios look similar.

### 3.3 Implement rule-based filtering + nearest neighbors

1. Restrict to the **17,084** eligible projects with usable feature values (handle missing duration without silently dropping large groups).
2. Apply a small set of sensible constraints (same fund if possible; acceptable year window; sector preference if validated).
3. Encode/scale distances for numerical and categorical features; keep distance weights simple and documented.
4. Use `sklearn.neighbors.NearestNeighbors` or a transparent weighted-distance ranking to find **K = 20** nearest projects; **exclude the target project itself**.
5. If fewer than the required number of credible peers exist, broaden the year window and/or fund restriction in a defined order; label the fallback that was used. Do not relax a reliable, substantively essential sector distinction merely to reach 20.
6. If the result remains inadequate, report **insufficient comparable projects** rather than forcing a score.

**Avoid:** K-means, neural networks and additional models without a demonstrated need. Nearest-neighbor *retrieval* is enough here; we are not training a KNN classifier to predict known anomaly labels.

### 3.4 Phase 3 checks and deliverables

- Measure coverage: fraction of eligible projects with at least 10, 20, and 30 credible peers.
- Summarize peer distance and differences in year, budget, duration, fund, and inferred sector.
- Manually inspect nearest-neighbor lists for a sample of representative and extreme projects.
- Save `peer_matches` (target ID, peer ID, rank, distance, constraints/fallback) and documented matching function.

**Deliverable:** Audited sector-label report; validated, reproducible comparable-project retrieval pipeline.

---

## Phase 4 — Anomaly Detection and Benchmarking

**Objective:** Flag projects that have unusually high or low beneficiaries-per-$1,000 ratios relative to their matched peers.

### 4.1 Primary scoring: robust Z-score on log ratios

For each eligible project *i*, use the log ratios of **its peers** (not a global distribution):

$$
z_i = \frac{\log(r_i)-\operatorname{median}\!\left(\log(r_{\mathrm{peers}})\right)}{1.4826\,\operatorname{MAD}\!\left(\log(r_{\mathrm{peers}})\right)}
$$

where `r_i = beneficiaries_per_1000` and `MAD` is the median absolute deviation of the logged peer ratios. Start with **|z| > 3** as a *candidate* flag threshold and calibrate through Phase 5. A negative score corresponds to fewer beneficiaries per $1,000 than peers, and a positive score to more.

- Require a credible minimum peer-group size (initial target: **20**, examine 10/30 in sensitivity tests).
- Handle near-zero MAD and insufficient neighbors explicitly; avoid division by zero or misleading huge scores.
- Do **not** automatically remove extremely high ratios. Audit whether small budgets, unusually large beneficiary totals, or coding errors explain them; mark questionable data separately.
- Keep scores for eligible projects only and preserve the precise peers used to generate each score.

### 4.2 Simple baseline: peer IQR / percentiles

Compute peer median, Q1, Q3 and IQR. Compare the robust-score flags with a simple peer-IQR rule and percentile ranks. Recognize that a skewed, nonnegative ratio distribution can produce an unhelpful negative IQR lower bound; consider log-scale IQR for a fairer baseline.

### 4.3 Save model output

At a minimum, generate `project_anomaly_results.parquet` / `.csv` with:

- Project identifier and eligibility/exclusion reason.
- Primary ratio and optional secondary cost-per-beneficiary value.
- Peer count and matching/fallback method.
- Peer median ratio, quartiles and percentile rank.
- Robust Z-score; flag (`unusually_low`, `typical`, `unusually_high`, `insufficient_data`).
- Sector-inference limitations and data-quality warnings.

### 4.4 Keep time series as a separate, descriptive analysis

Use the **existing** `country_period_trend` table for fund-month trends rather than recomputing the monthly ratios. Plot `DateSubmitted` period on the x-axis and the **pooled beneficiaries per $1,000** on the y-axis. A median project-ratio line can be shown for context.

- Support **monthly, quarterly, annual** aggregation; derive quarter/year totals from summed budgets and beneficiaries (do **not** average monthly ratios).
- Show project count on hover and a visible warning for periods with fewer than **five** eligible projects; optional filtering to sufficient periods.
- Check concentrations of submission dates (e.g., the 166-project South Sudan month) for batch submission artifacts.
- Do not interpret trend spikes as crisis severity changes or automatically assign project-level anomaly flags based on country-month behavior.
- More sophisticated change-point detection is **optional**, not required for the MVP.

**Deliverable:** Reproducible scoring notebook, anomaly results and optional model-based flag summary; existing trend output integrated with appropriate aggregation rules.

---

## Phase 5 — Evaluation and Validation

**Objective:** Show that peers are credible and flags are reasonably robust, while being explicit that the dataset has **no verified ground-truth targeting-error labels**.

| Test | What to do | What to report |
|---|---|---|
| **Sector audit** | Review stratified project titles and inferred sector labels | Label quality, unknown/multisector proportion, rules for sector use |
| **Peer relevance** | Inspect matched projects and feature distances | Similarity by fund, year, log-budget, duration and sector |
| **Coverage** | Count projects with adequate peers | % scored, % insufficient peers, % requiring relaxed rules |
| **Flag distribution** | Count high/low/typical flags | Total rates and rates by fund/year/sector (where credible) |
| **K sensitivity** | Compare K = 10, 20 and 30 | Agreement/stability of flags and peer medians |
| **Threshold sensitivity** | Compare candidate robust-score thresholds and baseline IQR | Flag counts and agreement; borderline cases |
| **Manual case review** | Inspect selected high, low, typical and questionable records | Plausibility, data issues, interpretability (not definitive correctness) |
| **Temporal validation** | Compare retrospective peers with historical-only peers | Differences in flags and coverage; possible look-ahead leakage |
| **Trend robustness** | Compare monthly vs quarterly/annual, pooled vs median, sparse vs sufficient periods | Which trends persist and why |

### Historical-only benchmark test

For example, when reviewing a project submitted in 2024, run a separate test using **only projects submitted before that date** as reference peers. Compare with retrospective matching over all available records. Clarify which mode the final app uses: retrospective project review, or prospective validation at submission time.

### Evaluation caveats

- **Do not claim accuracy, recall or precision for true inefficient projects** without adjudicated labels.
- Manual review can identify implausible peer matches and data-entry concerns, not certify actual humanitarian targeting quality.
- Separate sector classification quality, peer matching quality and anomaly-scoring stability; they are different questions.

**Deliverable:** Evaluation notebook/report, example cases, sensitivity tables, and documented final thresholds/matching decisions.

---

## Phase 6 — Dashboard, Usability Testing and Deployment

**Objective:** Deliver a simple, usable application that allows human reviewers to explore statistical flags and their project-level context.

### Dashboard page 1 — Project Outlier Explorer

- Search/filter by project code, title, fund, year, organization and inferred sector (clearly labeled).
- Show `beneficiaries_per_1000`, budget, total reported beneficiaries, eligibility, and anomaly category.
- Summarize flagged high/low projects; avoid efficiency or performance judgments.
- Provide a drill-down for excluded, extreme, or potentially erroneous records.

### Dashboard page 2 — Comparable-project Benchmarking

- Select a target project; show its K nearest peers, matching features, distance/criteria and any broadened fallback rules.
- Compare target ratio with peer median, quartiles, and robust-score or percentile status.
- Display a sortable peer table including budget, beneficiaries, duration, fund, year, organization and inferred sector.
- Warn clearly when peer support or sector classification is insufficient.

### Dashboard page 3 — Historical Submission Trends

- X-axis: month/quarter/year from `DateSubmitted`.
- Y-axis: **aggregate beneficiaries per $1,000**, calculated from total beneficiaries divided by total budget within each period.
- Filter by `PooledFundName`; optionally show multiple fund series if chart readability permits.
- Display count and sufficiency warnings for each period; drill into the contributing projects.
- Explain that the chart reflects submission composition and **does not measure event severity, crisis onset, realized delivery or actual spending**.

### Product implementation

- **Recommended stack:** Python preprocessing and scoring + **Streamlit** for MVP dashboard; Parquet/CSV outputs for quick loading.
- Build once from reproducible model outputs rather than rerunning every neighbor search during UI interactions.
- Include a methods/about section explaining the ratio, peer selection, labels, outlier thresholds and all limitations.
- Test on a few complete workflows: select a high outlier, select a low outlier, select an excluded project, select a project with insufficient peers, and inspect a sparse country-month.
- Gather feedback from reviewers on whether recommended peers and anomaly explanations make sense; fix confusing interactions before deployment.
- Package code, notebooks, saved outputs, README, data dictionary, and deployment instructions.

**Deliverable:** Deployed interactive MVP, walkthrough/demo, and documented reproducibility instructions.

---

## Revised work schedule (remaining phases)

| Work block | Tasks | Output |
|---|---|---|
| **Block 1 — Phase 3** | Audit sector labels; inspect candidate pool sizes; implement rule-based filtering and KNN; inspect peer examples | Peer matching pipeline and sector audit |
| **Block 2 — Phase 4** | Implement robust log-ratio Z-scores and peer-IQR baseline; handle extreme cases; save project-level results | Scored project table and flag explanations |
| **Block 3 — Phase 5** | Evaluate relevance, coverage, K/threshold sensitivity, temporal-only peers, and trend stability | Evaluation report and final method decisions |
| **Block 4 — Phase 6** | Build three dashboard views, test with example projects, document and deploy | Working web app and README |

Dashboard UI can begin in parallel with scoring using placeholder outputs, but replace these with validated model results before delivery. No need to redo Phase 2's verified country-month aggregation.

## Minimum success criteria

1. **Reliable project-level base:** 17,581 retained records with 17,084 eligible ratios and auditable exclusion reasons.
2. **Trustworthy peer matching:** Every scored project has a defensible peer set and explainable fallback rules; otherwise it is marked insufficient.
3. **Meaningful statistical flags:** A reproducible, peer-relative high/low label, with robust-score diagnostics and no claim of proven inefficiency.
4. **Usable historical context:** Country/fund monthly, quarterly, and annual trends display properly weighted ratios and sample-size warnings.
5. **Validated and usable app:** Users can search a project, understand its ratio, inspect recommended peers, review the anomaly explanation and explore historical trends.
6. **Transparent limitations:** Title-based sectors are inferred; the file does not verify crisis severity, event type, humanitarian needs, or actual beneficiary outcomes.

## Immediate next steps

1. Open Phase 2's cleaned project table and **inspect inferred-sector label counts**.
2. Audit a stratified sample of project titles; decide whether sectors should be hard or soft matching criteria.
3. Run a peer-availability analysis under same-fund, ±2-year, sector and budget/duration restrictions; quantify coverage.
4. Build rule-based candidate filtering and **K = 20** nearest-neighbor retrieval, with fallback and inadequate-peer handling.
5. Proceed to robust Z-score benchmarking only after project-peer lists look credible.
