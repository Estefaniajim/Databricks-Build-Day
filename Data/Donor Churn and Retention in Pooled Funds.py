# Databricks notebook source
# DBTITLE 1,Overview & Methodology
# MAGIC %md
# MAGIC # Donor Churn & Retention in Humanitarian Pooled Funds
# MAGIC
# MAGIC **Research question:** Which humanitarian pooled funds experience the highest donor churn rates, and how has donor retention changed over time?
# MAGIC
# MAGIC ## Methodology
# MAGIC * **Active donor (fund-year):** a unique `GMSDonorID` with total `PaidAmt > 0` for that pooled fund in that fiscal year. `IsTransfer = 1` rows are excluded per spec (none exist in this extract), and donors with multiple contributions in a year are deduplicated by aggregation.
# MAGIC * **Churned (base year N):** active for a fund in year N but not active for the *same fund* in N+1. Churn means inactivity within that particular fund, not withdrawal from humanitarian funding altogether.
# MAGIC * **New donor:** first-ever active year with that fund. **Returning donor:** active in N after contributing in an earlier year but not N−1.
# MAGIC
# MAGIC ## Data-quality exclusions (verified during inspection)
# MAGIC * 85 rows (~US$5.4M) belonging to the pseudo-donor **"Private Contributions"** lack a `GMSDonorID` and cannot be tracked as donors → excluded from donor-level analysis.
# MAGIC * FY2027–2029 rows carry zero `PaidAmt` (future pledges) → contribute nothing under the active-donor rule.
# MAGIC * Payments for a fiscal year keep arriving until ~April of the following year; FY2026 is still open (max `PaidDate` = 2026-11-20) → **churn is computed only for base years 2006–2024**, which have a complete year N+1.
# MAGIC * Funds with no activity after 2024 (Colombia, Haiti, Iraq, Jordan) have terminal-year churn excluded — that is fund closure, not donor attrition.

# COMMAND ----------

# DBTITLE 1,Install plotly
# MAGIC %pip install plotly

# COMMAND ----------

# DBTITLE 1,Load, Clean & Data Quality
import pandas as pd
import numpy as np

PATH = "/Workspace/Hackathon/Data/global_cbpf_contributions.csv"
raw = pd.read_csv(PATH)

# --- Data quality & coverage ---
print(f"Rows: {len(raw):,} | Pooled funds: {raw['PooledFundId'].nunique()} | Unique GMSDonorID: {raw['GMSDonorID'].nunique()}")
print(f"Fiscal years present: {raw['FiscalYear'].min()}-{raw['FiscalYear'].max()}")
print(f"Rows with IsTransfer = 1 (excluded per spec): {(raw['IsTransfer'] == 1).sum()}")
missing_id = raw["GMSDonorID"].isna()
print(f"Rows missing GMSDonorID (excluded): {missing_id.sum()} — all pseudo-donor 'Private Contributions' (${raw.loc[missing_id, 'PaidAmt'].sum() / 1e6:.1f}M)")
print(f"Rows with PaidAmt <= 0 (never active): {(raw['PaidAmt'] <= 0).sum()} — includes all FY2027-2029 future-pledge rows")

# --- Cleaning filters: donor-identifiable, non-transfer, positive paid amounts ---
df = raw[raw["GMSDonorID"].notna() & (raw["IsTransfer"] != 1) & (raw["PaidAmt"] > 0)].copy()
df["GMSDonorID"] = df["GMSDonorID"].astype(int)

paid_dates = pd.to_datetime(df["PaidDate"], errors="coerce")
print(f"\nMax PaidDate: {paid_dates.max().date()} → FY2026 still open; a fiscal year's payments trail into the following April")
LAST_COMPLETE_YEAR = 2025
print(f"Last complete fiscal year: {LAST_COMPLETE_YEAR} → churn computed for base years 2006-{LAST_COMPLETE_YEAR - 1}")

# --- Deduplicate multi-contribution donors: one row per fund-donor-year ---
donor_year = (
    df.groupby(["PooledFundId", "PooledFundName", "FiscalYear", "GMSDonorID"], as_index=False)
    .agg(PaidAmt=("PaidAmt", "sum"))
)
donor_year = donor_year[donor_year["PaidAmt"] > 0]
print(f"\nFund-donor-year records: {len(donor_year):,} | funds: {donor_year['PooledFundId'].nunique()} | unique donors: {donor_year['GMSDonorID'].nunique()}")
display(donor_year.head(10))

# COMMAND ----------

# DBTITLE 1,Churn & Retention Metrics per Fund-Year
# Last active year per fund (to exclude terminal years of closed/dormant funds)
fund_last = donor_year.groupby("PooledFundId")["FiscalYear"].max()

closed = donor_year.groupby("PooledFundName")["FiscalYear"].max()
print("Funds with no activity after 2024 (terminal-year churn excluded = fund closure, not donor attrition):")
print(closed[closed < 2025].to_string())

records = []
for (pid, fname), g in donor_year.groupby(["PooledFundId", "PooledFundName"]):
    donors_by_year = {y: set(d) for y, d in g.groupby("FiscalYear")["GMSDonorID"]}
    seen_ever = set()
    for y in sorted(donors_by_year):
        cur = donors_by_year[y]
        prev = donors_by_year.get(y - 1, set())
        rec = {
            "PooledFundId": pid,
            "PooledFundName": fname,
            "Year": y,
            "ActiveDonors": len(cur),
            "NewDonors": len(cur - seen_ever),                # first-ever year with this fund
            "ReturningDonors": len((cur & seen_ever) - prev), # gave earlier, skipped N-1
        }
        nxt = donors_by_year.get(y + 1, set())
        # Churn needs a COMPLETE next year and the fund must not have ended after year y
        if y + 1 <= LAST_COMPLETE_YEAR and fund_last[pid] > y:
            rec["RetainedDonors"] = len(cur & nxt)
            rec["ChurnedDonors"] = len(cur - nxt)
            rec["ChurnRate"] = len(cur - nxt) / len(cur) * 100
            rec["RetentionRate"] = len(cur & nxt) / len(cur) * 100
        seen_ever |= cur
        records.append(rec)

metrics = pd.DataFrame(records)
churn_metrics = metrics.dropna(subset=["ChurnRate"]).copy()
print(f"\nFund-year churn metrics: {len(churn_metrics)} rows (base years {int(churn_metrics['Year'].min())}-{int(churn_metrics['Year'].max())})")
display(churn_metrics.head(12))

# COMMAND ----------

# DBTITLE 1,Churn & Retention Over Time
# Donor-weighted overall churn/retention by base year (pooled across all funds)
overall = (
    churn_metrics.groupby("Year")
    .apply(lambda g: pd.Series({
        "Funds": len(g),
        "PriorYearDonors": int(g["ActiveDonors"].sum()),
        "Retained": int(g["RetainedDonors"].sum()),
        "Churned": int(g["ChurnedDonors"].sum()),
        "ChurnRate%": g["ChurnedDonors"].sum() / g["ActiveDonors"].sum() * 100,
        "RetentionRate%": g["RetainedDonors"].sum() / g["ActiveDonors"].sum() * 100,
    }), include_groups=False)
    .reset_index()
)
display(overall.round(1))

early = overall[overall["Year"] <= 2015]["ChurnRate%"].mean()
stable = overall[(overall["Year"] >= 2015) & (overall["Year"] <= 2023)]["ChurnRate%"].mean()
y2024 = overall.loc[overall["Year"] == 2024, "ChurnRate%"].iloc[0]
best = overall.loc[overall["ChurnRate%"].idxmin()]
print(f"Mean churn 2006-2015: {early:.1f}% | 2015-2023: {stable:.1f}% | 2024: {y2024:.1f}%")
print(f"Best year: {int(best['Year'])} ({best['ChurnRate%']:.1f}% churn) | 2024 churned {int(overall.loc[overall['Year'] == 2024, 'Churned'].iloc[0])} of {int(overall.loc[overall['Year'] == 2024, 'PriorYearDonors'].iloc[0])} prior-year donors")

# COMMAND ----------

# DBTITLE 1,Top 10 Funds by Donor Churn
# Only compare fund-years with at least 5 donors in the base year
elig = churn_metrics[churn_metrics["ActiveDonors"] >= 5]

fund_summary = (
    elig.groupby("PooledFundName")
    .agg(BaseYears=("ChurnRate", "size"),
         AvgDonors=("ActiveDonors", "mean"),
         TotalChurned=("ChurnedDonors", "sum"),
         TotalPriorActive=("ActiveDonors", "sum"))
)
fund_summary["PooledChurnRate%"] = fund_summary["TotalChurned"] / fund_summary["TotalPriorActive"] * 100
top10 = fund_summary.sort_values("PooledChurnRate%", ascending=False).head(10)
print("Top 10 pooled funds by donor churn rate (base years 2006-2024 with >= 5 donors):")
display(top10.round(1))

print("Worst single fund-years (>= 5 donors):")
display(elig.nlargest(8, "ChurnRate")[["PooledFundName", "Year", "ActiveDonors", "ChurnedDonors", "ChurnRate"]].round(1))

# COMMAND ----------

# DBTITLE 1,Churn vs Year-over-Year Funding Change
fund_amt = (
    donor_year.groupby(["PooledFundId", "FiscalYear"], as_index=False)["PaidAmt"]
    .sum()
    .rename(columns={"FiscalYear": "Year", "PaidAmt": "FundPaidAmt"})
)
fm = metrics.merge(fund_amt, on=["PooledFundId", "Year"], how="left").sort_values(["PooledFundId", "Year"])
fm["NextYearPaidAmt"] = fm.groupby("PooledFundId")["FundPaidAmt"].shift(-1)
fm["YoYChange%"] = (fm["NextYearPaidAmt"] - fm["FundPaidAmt"]) / fm["FundPaidAmt"] * 100
cf = fm.dropna(subset=["ChurnRate", "YoYChange%"]).copy()

from scipy import stats
r_p, p_p = stats.pearsonr(cf["ChurnRate"], cf["YoYChange%"])
r_s, p_s = stats.spearmanr(cf["ChurnRate"], cf["YoYChange%"])
print(f"Fund-years with churn + funding change: {len(cf)}")
print(f"Pearson r = {r_p:.3f} (p = {p_p:.4f}) | Spearman rho = {r_s:.3f} (p = {p_s:.6f})")
print("Spearman (rank-based) is robust to the extreme YoY % swings of small funds; Pearson is outlier-sensitive.")

cf["ChurnBand"] = pd.cut(cf["ChurnRate"], [0, 10, 25, 50, 100], right=False)
print("\nYoY funding change (%) by churn-rate band (medians are robust to outliers):")
display(cf.groupby("ChurnBand", observed=True)["YoYChange%"].agg(["mean", "median", "count"]).round(1))

# COMMAND ----------

# DBTITLE 1,Major vs Small Donor Churn
# Donor share of the fund-year's total paid contributions
dy = donor_year.copy()
dy["FundYearTotal"] = dy.groupby(["PooledFundId", "FiscalYear"])["PaidAmt"].transform("sum")
dy["Share%"] = dy["PaidAmt"] / dy["FundYearTotal"] * 100

# Retention outcome for each fund-donor-year (base year <= 2024, fund not ended)
rows = []
for (pid, fname), g in donor_year.groupby(["PooledFundId", "PooledFundName"]):
    donors_by_year = {y: set(d) for y, d in g.groupby("FiscalYear")["GMSDonorID"]}
    for y, dset in donors_by_year.items():
        if y + 1 > LAST_COMPLETE_YEAR - 1 or fund_last[pid] <= y:
            continue
        nxt = donors_by_year.get(y + 1, set())
        for d in dset:
            rows.append((pid, fname, y, d, int(d in nxt)))

outcomes = pd.DataFrame(rows, columns=["PooledFundId", "PooledFundName", "Year", "GMSDonorID", "RetainedNext"])
outcomes = outcomes.merge(
    dy[["PooledFundId", "FiscalYear", "GMSDonorID", "Share%"]],
    left_on=["PooledFundId", "Year", "GMSDonorID"],
    right_on=["PooledFundId", "FiscalYear", "GMSDonorID"],
).drop(columns=["FiscalYear"])
outcomes["Major"] = outcomes["Share%"] >= 10  # >= 10% of the fund-year's paid contributions

print(f"Donor-year outcomes: {len(outcomes):,}")
grp = outcomes.groupby("Major")["RetainedNext"].agg(Outcomes="count")
grp["ChurnRate%"] = outcomes.groupby("Major")["RetainedNext"].apply(lambda s: (1 - s.mean()) * 100)
print("\nMajor donors (>= 10% of fund-year contributions) vs minor donors:")
display(grp.round(1))

outcomes["ShareQuartile"] = pd.qcut(outcomes["Share%"], 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"])
q = outcomes.groupby("ShareQuartile", observed=True)["RetainedNext"].agg(Outcomes="count")
q["ChurnRate%"] = outcomes.groupby("ShareQuartile", observed=True)["RetainedNext"].apply(lambda s: (1 - s.mean()) * 100)
print("Churn rate by donor-share quartile (within fund-year):")
display(q.round(1))

# COMMAND ----------

# DBTITLE 1,Visualizations
import plotly.graph_objects as go
import plotly.express as px

# 1) Line: annual donor churn & retention rates
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=overall["Year"], y=overall["ChurnRate%"], mode="lines+markers",
                          name="Churn rate", line=dict(color="firebrick")))
fig1.add_trace(go.Scatter(x=overall["Year"], y=overall["RetentionRate%"], mode="lines+markers",
                          name="Retention rate", line=dict(color="seagreen")))
fig1.update_layout(
    title="Annual donor churn vs retention across pooled funds (base year N, outcome in N+1)",
    xaxis_title="Base fiscal year", yaxis_title="% of prior-year active donors",
    yaxis_range=[0, 100], height=450)
fig1.show()

# 2) Bar: top 10 funds by churn rate
t10 = top10.reset_index().sort_values("PooledChurnRate%", ascending=True)
fig2 = px.bar(t10, x="PooledChurnRate%", y="PooledFundName", orientation="h",
              color="AvgDonors", color_continuous_scale="OrRd",
              text=[f"{v:.0f}%  ({int(c)}/{int(a)})" for v, c, a in
                    zip(t10["PooledChurnRate%"], t10["TotalChurned"], t10["TotalPriorActive"])],
              title="Top 10 pooled funds by donor churn rate (base years with >= 5 donors, 2006-2024)<br><sup>label = churned / prior-active donors</sup>")
fig2.update_layout(xaxis_title="Pooled churn rate (%)", yaxis_title="", height=500)
fig2.show()

# 3) Heatmap: retention rate by fund and base year (funds sorted by average retention, worst first)
hm = churn_metrics.pivot_table(index="PooledFundName", columns="Year", values="RetentionRate")
hm = hm.loc[hm.mean(axis=1).sort_values().index]
fig3 = px.imshow(hm, color_continuous_scale="RdYlGn", aspect="auto",
                 title="Donor retention rate (%) by pooled fund and base year",
                 labels=dict(color="Retention %"))
fig3.update_layout(height=950)
fig3.show()

# 4) Scatter: churn rate vs YoY funding change (y-axis clipped for readability)
sc = cf.copy()
sc["YoY_clipped%"] = sc["YoYChange%"].clip(-100, 300)
fig4 = px.scatter(sc, x="ChurnRate", y="YoY_clipped%", opacity=0.55,
                  hover_data=["PooledFundName", "Year", "ActiveDonors", "YoYChange%"],
                  title="Donor churn rate vs year-over-year funding change (y clipped to [-100%, +300%])")
fig4.add_hline(y=0, line_dash="dash", line_color="gray")
fig4.update_layout(xaxis_title="Churn rate (%) in base year",
                   yaxis_title="YoY change in fund PaidAmt (%)", height=450)
fig4.show()

# COMMAND ----------

# DBTITLE 1,Key Findings
# MAGIC %md
# MAGIC ## Key findings
# MAGIC
# MAGIC 1. **Retention improved for a decade, then reversed.** Donor churn collapsed from a 56% peak in 2010 (the system's scale-up year) to a low of 9.2% in 2015 (2007's 0% rate covered only 8 donors), then held a 13–20% band through 2023. In base year 2024 it jumped to **26.8%** (80 of 299 prior-year donors lapsed across funds) — the highest since 2011.
# MAGIC 2. **Surge-response funds churn worst.** Haiti (91%: 2010 earthquake one-off donors) and Pakistan (90%: 2010 floods) top the list; regional hosted funds (Colombia RhPF-LAC 40%, Haiti RhPF-LAC 33%, Niger RhPF-WCA 26%, Burkina Faso RhPF-WCA 22%) and chronically volatile country funds (Jordan 29%, Iraq 23%, oPt 22%, Syria Cross-border 22%) follow. oPt and Syria Cross-border combine high churn with the largest at-risk donor bases (139 and 153 prior-active donors pooled).
# MAGIC 3. **Churn tracks funding decline.** Median YoY funding change falls monotonically with churn band: +12.2% (churn <10%) → −3.6% (10–25%) → −8.5% (25–50%) → −66.0% (≥50%). Rank correlation: Spearman ρ = −0.26 (p < 0.001).
# MAGIC 4. **Major donors are the most loyal, not the most flighty.** Donors contributing ≥10% of a fund-year's total churn at 11.3% vs 21.8% for smaller donors; the smallest-share quartile churns 36.7% vs ~11% for the two largest quartiles. Attrition risk concentrates in the long tail of small donors.
# MAGIC 5. **New-donor acquisition is fragile.** The worst fund-years are surge years (Haiti 2010, Pakistan 2010, Syria Cross-border 2024 at 61% churn), where newly acquired donors do not return — retention-focused stewardship right after a surge year yields the largest payoff.