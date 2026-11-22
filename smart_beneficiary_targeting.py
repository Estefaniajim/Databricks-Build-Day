# Databricks notebook source
# MAGIC %md
# MAGIC # Smart Beneficiary Targeting Validation
# MAGIC Run all cells on serverless compute. The notebook reads the CSV from `/Workspace/Hackathon/Data/global_cbpf_project_summary.csv`.
# MAGIC
# MAGIC Each point represents one project. The vertical axis shows **budget per 1,000 reported direct beneficiaries**, calculated as `1,000 × Budget / (Men + Women + Boys + Girls)`. Values use the source budget units and assume a common currency. Set **fund** to narrow the review queue.
# MAGIC
# MAGIC A project is screened against other projects in the same pooled fund, allocation source, and activity category within two allocation years. The notebook needs at least 12 eligible peers and flags values outside the 1.5× interquartile fences on the log scale. This is a retrospective review, so the window can include later projects. Missing activity categories are compared together with lower confidence. HFU management costs go to a separate review status because their budgets cover fund administration.
# MAGIC
# MAGIC Allocation source and activity category describe funding context. The CSV lacks cluster, event type, and crisis severity. If a later extract includes verified project-level cluster labels, set **cluster_column** to that field to require a matching cluster. Use verified event links or independently sourced geography-period severity for crisis analysis. Flags support manual review of scope, service intensity, duration, and beneficiary counting before any targeting decision.

# COMMAND ----------

import numpy as np
import pandas as pd

dbutils.widgets.text("fund", "", "Pooled fund name (blank = all)")
dbutils.widgets.text("project_code", "", "Benchmark project code (blank = top flag)")
dbutils.widgets.text("cluster_column", "", "Verified cluster column (optional)")

csv_path = "/Workspace/Hackathon/Data/global_cbpf_project_summary.csv"
projects = pd.read_csv(csv_path, dtype="string", keep_default_na=False, encoding="utf-8-sig")
cluster_column = dbutils.widgets.get("cluster_column").strip()
population_columns = ["Men", "Women", "Boys", "Girls"]
required = {
    "ChfProjectCode", "PooledFundId", "PooledFundName", "AllocationYear",
    "AllocationSourceName", "AllocationTypeCategory", "ProjectTitle", "Budget",
    *population_columns,
}
if cluster_column:
    required.add(cluster_column)
missing = required.difference(projects.columns)
if missing:
    raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")

for column in required:
    projects[column] = projects[column].str.strip()
codes = projects["ChfProjectCode"]
if codes.eq("").any() or codes.duplicated().any():
    raise ValueError("Expected one row per nonblank ChfProjectCode.")

projects["Budget"] = pd.to_numeric(projects["Budget"], errors="coerce").astype(float)
projects["AllocationYear"] = pd.to_numeric(projects["AllocationYear"], errors="coerce").astype(float)
population = projects[population_columns].apply(pd.to_numeric, errors="coerce").astype(float)
valid_counts = np.isfinite(population) & population.ge(0) & population.mod(1).eq(0)
projects["DirectBeneficiaries"] = population.where(valid_counts).sum(axis=1, min_count=4)
valid_budget = np.isfinite(projects["Budget"]) & projects["Budget"].gt(0)
valid_denominator = projects["DirectBeneficiaries"].gt(0)
valid_year = projects["AllocationYear"].between(1900, 2100) & projects["AllocationYear"].mod(1).eq(0)

ratio = 1000 * projects["Budget"].where(valid_budget) / projects["DirectBeneficiaries"].where(valid_denominator)
valid_ratio = np.isfinite(ratio) & ratio.gt(0)
projects["BudgetPer1000"] = ratio.where(valid_ratio)
projects["Category"] = projects["AllocationTypeCategory"].str.casefold().replace(
    {"": "unclassified", "core pipelines": "core pipeline"}
)

scope_fields = ["PooledFundId", "AllocationSourceName"] + ([cluster_column] if cluster_column else [])
missing_scope = projects[scope_fields].eq("").any(axis=1)
projects["Status"] = np.select(
    [
        ~valid_budget, projects["DirectBeneficiaries"].isna(), ~valid_denominator,
        ~valid_ratio, ~valid_year, missing_scope, projects["Category"].eq("hfu management cost"),
    ],
    [
        "Invalid budget", "Missing or invalid counts", "Zero direct beneficiaries",
        "Invalid ratio", "Invalid allocation year", "Missing comparison fields", "HFU management",
    ],
    default="Too few peers",
)

# COMMAND ----------

cohort_fields = [*scope_fields, "Category"]
eligible = projects.loc[projects["Status"].eq("Too few peers")].copy()
eligible["LogRatio"] = np.log(eligible["BudgetPer1000"])
projects["PeerCount"] = 0
for column in ["PeerMedian", "LowerFence", "UpperFence", "FlagDistance"]:
    projects[column] = np.nan

for _, cohort in eligible.groupby(cohort_fields, sort=False):
    years = cohort["AllocationYear"].to_numpy()
    log_ratios = cohort["LogRatio"].to_numpy()
    indices = cohort.index.to_numpy()
    for index, year, log_ratio in zip(indices, years, log_ratios):
        peers = log_ratios[(np.abs(years - year) <= 2) & (indices != index)]
        projects.at[index, "PeerCount"] = len(peers)
        if len(peers) < 12:
            continue
        q1, median, q3 = np.quantile(peers, [0.25, 0.5, 0.75])
        iqr = q3 - q1
        projects.at[index, "PeerMedian"] = np.exp(median)
        if iqr <= 0:
            projects.at[index, "Status"] = "No peer spread"
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        projects.loc[index, ["LowerFence", "UpperFence"]] = np.exp([lower, upper])
        if log_ratio < lower:
            projects.at[index, "Status"] = "Low ratio"
            projects.at[index, "FlagDistance"] = lower - log_ratio
        elif log_ratio > upper:
            projects.at[index, "Status"] = "High ratio"
            projects.at[index, "FlagDistance"] = log_ratio - upper
        else:
            projects.at[index, "Status"] = "Within peer range"

fund = dbutils.widgets.get("fund").strip()
view = projects.loc[projects["PooledFundName"].str.casefold().eq(fund.casefold())] if fund else projects
if view.empty:
    raise ValueError(f"No projects for pooled fund {fund!r}. Use an exact fund name or clear the fund widget.")

display(view.groupby("Status").size().reset_index(name="Projects"))

# COMMAND ----------

review_columns = list(dict.fromkeys([
    "ChfProjectCode", "ProjectTitle", "PooledFundName", "AllocationYear", "AllocationSourceName",
    "Category", *([cluster_column] if cluster_column else []), "Status", "Budget", "DirectBeneficiaries",
    "BudgetPer1000", "PeerCount", "PeerMedian", "LowerFence", "UpperFence",
]))
flags = view.loc[view["Status"].isin(["High ratio", "Low ratio"])].sort_values(
    ["FlagDistance", "ChfProjectCode"], ascending=[False, True]
)
display(flags[review_columns])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Benchmark a project
# MAGIC Enter a project code or leave it blank to review the largest departure from a peer fence in the current fund view. The shortlist favors the closest allocation year, then the closest beneficiary count on a proportional scale. Review the project titles and budgets before treating the matches as benchmarks.

# COMMAND ----------

code = dbutils.widgets.get("project_code").strip()
if not code and not flags.empty:
    code = flags.iloc[0]["ChfProjectCode"]

if not code:
    print("No ratio flags in this view. Enter a project code to look up peers.")
else:
    matches = projects.loc[projects["ChfProjectCode"].eq(code)]
    if matches.empty:
        raise ValueError(f"Unknown project code: {code}")
    target = matches.iloc[0]
    display(matches[review_columns])
    if target.name not in eligible.index:
        print(f"No benchmark for this project. Review status is {target['Status']}.")
    else:
        same_cohort = eligible[cohort_fields].eq(target[cohort_fields], axis="columns").all(axis=1)
        year_gap = eligible["AllocationYear"].sub(target["AllocationYear"]).abs()
        candidates = projects.loc[eligible.index[same_cohort & year_gap.le(2) & (eligible.index != target.name)]]
        shortlist = candidates.assign(
            YearGap=year_gap,
            SizeGap=np.log(candidates["DirectBeneficiaries"] / target["DirectBeneficiaries"]).abs(),
        ).sort_values(["YearGap", "SizeGap", "ChfProjectCode"]).head(5)
        if shortlist.empty:
            print("No eligible projects match this cohort and time window.")
        else:
            display(shortlist[["YearGap", *review_columns]])

# COMMAND ----------

import plotly.express as px

chart_data = view.loc[valid_year & view["BudgetPer1000"].notna()]
if not chart_data.empty:
    chart = px.scatter(
        chart_data, x="AllocationYear", y="BudgetPer1000", color="Status",
        symbol="AllocationSourceName", log_y=True,
        hover_data=["ChfProjectCode", "PooledFundName", "Category", "Budget", "DirectBeneficiaries", "PeerCount"],
        labels={"AllocationYear": "Allocation year", "BudgetPer1000": "Budget per 1,000 direct beneficiaries",
                "AllocationSourceName": "Allocation source"},
        color_discrete_map={"High ratio": "#b13924", "Low ratio": "#1465ac", "Within peer range": "#a8adb5"},
    )
    chart.update_xaxes(dtick=1)
    chart.show()
