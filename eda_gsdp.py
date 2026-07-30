"""
EDA on State-wise GSDP (Current Prices) 2011-12 to 2021-22
Libraries: pandas, numpy, matplotlib, seaborn
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.bbox'] = 'tight'

OUT = "/home/claude/gsdp_eda/outputs"

# ---------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------
df = pd.read_csv("/home/claude/gsdp_eda/GSDP_Current_2011-12_State_wise.csv")
df.columns = [c.strip() for c in df.columns]
df['State/Uts'] = df['State/Uts'].str.strip()

gdp_cols = [c for c in df.columns if c.startswith("GSDP")]
growth_cols = [c for c in df.columns if c.startswith("% Growth")]
years = [c.split("(")[0].replace("GSDP-Curr-", "") for c in gdp_cols]

# Replace 0s in Jammu & Kashmir-U.T. (newly formed UT, no data before 2019-20) with NaN
jk_ut_mask = df['State/Uts'] == 'Jammu & Kashmir-U.T.'
for c in gdp_cols[:8]:  # years before UT formation
    df.loc[jk_ut_mask, c] = np.nan

# Convert to numeric (NA strings -> NaN)
for c in gdp_cols + growth_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Exclude UTs / special cases from "state" rankings where relevant
non_states = ['Andaman & Nicobar Islands', 'Chandigarh', 'Delhi', 'Puducherry',
              'Jammu & Kashmir-U.T.', 'Jammu & Kashmir*']
states_df = df[~df['State/Uts'].isin(non_states)].copy()

print("="*70)
print("DATASET OVERVIEW")
print("="*70)
print(f"Shape: {df.shape}")
print(f"Entities (states/UTs): {df['State/Uts'].nunique()}")
print(f"Years covered: {years[0]} to {years[-1]} ({len(years)} years)")
print(f"\nMissing values per GDP year column:\n{df[gdp_cols].isna().sum()}")
print(f"\nTotal missing cells: {df[gdp_cols+growth_cols].isna().sum().sum()}")
df.to_csv(f"{OUT}/cleaned_data.csv", index=False)

# ---------------------------------------------------------------
# 2. NATIONAL TOTAL GDP PER YEAR (sum across states, excluding UTs to avoid double count issues optional)
# ---------------------------------------------------------------
yearly_total = df[gdp_cols].sum(skipna=True)
yearly_total.index = years

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(years, yearly_total.values, marker='o', linewidth=2.5, color='#2b6a99')
ax.fill_between(years, yearly_total.values, alpha=0.15, color='#2b6a99')
ax.set_title("Total GSDP (All States & UTs) by Year", fontsize=14, fontweight='bold')
ax.set_ylabel("GSDP (₹ Crore)")
ax.set_xlabel("Year")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
ax.annotate('2021-22 dip is a data\nreporting gap, not a real\neconomic decline\n(15 states/UTs not yet reported)',
            xy=(len(years)-1, yearly_total.values[-1]), xytext=(-160, 40),
            textcoords='offset points', fontsize=8.5, color='#b3541e',
            arrowprops=dict(arrowstyle='->', color='#b3541e'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/01_yearwise_total_gdp.png")
plt.close()

# ---------------------------------------------------------------
# NOTE ON DATA COMPLETENESS:
# 2021-22 is missing for 15 entities (incl. Maharashtra, Gujarat, West Bengal,
# Bihar) - likely not yet reported at source. For "current snapshot" analyses
# (top 10, distribution, outliers, contribution, regional ranking) we use
# 2020-21, the most recent YEAR WITH NEAR-COMPLETE DATA (only 2 missing: J&K*
# and A&N Islands, both small economies). Trend lines still use all years.
# ---------------------------------------------------------------
# 3. TOP 10 STATES BY LATEST-COMPLETE YEAR GDP (2020-21)
# ---------------------------------------------------------------
latest_col = 'GSDP-Curr-2020-21(Cr)'   # most recent near-complete year
final_col = gdp_cols[-1]               # 2021-22, partial data (trend only)
snapshot_year = "2020-21"
top10 = states_df[['State/Uts', latest_col]].dropna().sort_values(latest_col, ascending=False).head(10)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=top10, y='State/Uts', x=latest_col, hue='State/Uts', palette='viridis', legend=False, ax=ax)
ax.set_title(f"Top 10 States by GSDP ({snapshot_year})", fontsize=14, fontweight='bold')
ax.set_xlabel("GSDP (₹ Crore)")
ax.set_ylabel("")
for i, v in enumerate(top10[latest_col]):
    ax.text(v, i, f' {v:,.0f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/02_top10_states.png")
plt.close()

# High / Low GDP state (latest year)
high_state = top10.iloc[0]
low_state_row = states_df[['State/Uts', latest_col]].dropna().sort_values(latest_col).iloc[0]

# ---------------------------------------------------------------
# 4. GDP TREND OVER TIME - Top 6 states line chart
# ---------------------------------------------------------------
top6_states = top10['State/Uts'].head(6).tolist()
fig, ax = plt.subplots(figsize=(12, 6.5))
for s in top6_states:
    row = df[df['State/Uts'] == s][gdp_cols].values.flatten()
    ax.plot(years, row, marker='o', label=s, linewidth=2)
ax.set_title("GSDP Trend Over Time — Top 6 States", fontsize=14, fontweight='bold')
ax.set_ylabel("GSDP (₹ Crore)")
ax.set_xlabel("Year")
ax.legend(loc='upper left', fontsize=9)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/03_gdp_trend_top6.png")
plt.close()

# ---------------------------------------------------------------
# 5. STATE-WISE CAGR (Compound Annual Growth Rate) using first & last available data points
# ---------------------------------------------------------------
def cagr(row):
    vals = row[gdp_cols].values.astype(float)
    valid = np.where(~np.isnan(vals))[0]
    if len(valid) < 2:
        return np.nan
    first_idx, last_idx = valid[0], valid[-1]
    n_years = last_idx - first_idx
    if n_years == 0 or vals[first_idx] <= 0:
        return np.nan
    return ((vals[last_idx] / vals[first_idx]) ** (1 / n_years) - 1) * 100

states_df['CAGR_%'] = states_df.apply(cagr, axis=1)
cagr_sorted = states_df[['State/Uts', 'CAGR_%']].dropna().sort_values('CAGR_%', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
top_growth = cagr_sorted.head(10)
sns.barplot(data=top_growth, y='State/Uts', x='CAGR_%', hue='State/Uts', palette='Greens_r', legend=False, ax=axes[0])
axes[0].set_title("Top 10 Fastest-Growing States (CAGR %)", fontweight='bold')
axes[0].set_xlabel("CAGR (%)")
axes[0].set_ylabel("")

slow_growth = cagr_sorted.tail(10).sort_values('CAGR_%')
sns.barplot(data=slow_growth, y='State/Uts', x='CAGR_%', hue='State/Uts', palette='Reds', legend=False, ax=axes[1])
axes[1].set_title("Bottom 10 Slowest-Growing States (CAGR %)", fontweight='bold')
axes[1].set_xlabel("CAGR (%)")
axes[1].set_ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/04_growth_rate_top_bottom.png")
plt.close()

# ---------------------------------------------------------------
# 6. GDP DISTRIBUTION (latest year) - histogram + boxplot
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sns.histplot(states_df[latest_col].dropna(), bins=15, kde=True, color='#3a7ca5', ax=axes[0])
axes[0].set_title(f"Distribution of GSDP across States ({snapshot_year})", fontweight='bold')
axes[0].set_xlabel("GSDP (₹ Crore)")

sns.boxplot(x=states_df[latest_col].dropna(), color='#f4a261', ax=axes[1])
axes[1].set_title(f"GSDP Boxplot — Outlier Detection ({snapshot_year})", fontweight='bold')
axes[1].set_xlabel("GSDP (₹ Crore)")
plt.tight_layout()
plt.savefig(f"{OUT}/05_gdp_distribution_outliers.png")
plt.close()

# Outlier detection via IQR
q1, q3 = states_df[latest_col].quantile([0.25, 0.75])
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
lower_bound = q1 - 1.5 * iqr
outliers = states_df[(states_df[latest_col] > upper_bound) | (states_df[latest_col] < lower_bound)][['State/Uts', latest_col]]

# ---------------------------------------------------------------
# 7. CORRELATION HEATMAP BETWEEN YEARS
# ---------------------------------------------------------------
corr = df[gdp_cols].corr()
corr.index = years
corr.columns = years
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=0.9, vmax=1, ax=ax, square=True)
ax.set_title("Correlation Between Years (GSDP across states)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/06_year_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------------
# 8. STATE CONTRIBUTION TO NATIONAL GDP (latest year) - pie/treemap style bar
# ---------------------------------------------------------------
contrib = states_df[['State/Uts', latest_col]].dropna().sort_values(latest_col, ascending=False)
contrib['share_%'] = contrib[latest_col] / contrib[latest_col].sum() * 100
top_n = contrib.head(10).copy()
others_share = contrib['share_%'].iloc[10:].sum()
pie_data = pd.concat([top_n[['State/Uts','share_%']],
                       pd.DataFrame({'State/Uts':['Others'], 'share_%':[others_share]})])

fig, ax = plt.subplots(figsize=(9, 9))
colors = sns.color_palette('viridis', len(pie_data))
ax.pie(pie_data['share_%'], labels=pie_data['State/Uts'], autopct='%1.1f%%',
       colors=colors, startangle=90, pctdistance=0.8,
       wedgeprops={'edgecolor': 'white', 'linewidth': 1})
ax.set_title(f"Contribution of Each State to National GSDP ({snapshot_year})", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/07_state_contribution_pie.png")
plt.close()

# ---------------------------------------------------------------
# 9. REGIONAL RANKING / STANDARD (grouping states into regions)
# ---------------------------------------------------------------
region_map = {
    'Andhra Pradesh':'South','Karnataka':'South','Kerala':'South','Tamil Nadu':'South','Telangana':'South','Puducherry':'South',
    'Bihar':'East','Jharkhand':'East','Odisha':'East','West Bengal':'East',
    'Chhattisgarh':'Central','Madhya Pradesh':'Central','Uttar Pradesh':'Central','Uttarakhand':'Central',
    'Goa':'West','Gujarat':'West','Maharashtra':'West',
    'Haryana':'North','Himachal Pradesh':'North','Punjab':'North','Rajasthan':'North','Delhi':'North','Chandigarh':'North','Jammu & Kashmir*':'North','Jammu & Kashmir-U.T.':'North',
    'Arunachal Pradesh':'Northeast','Assam':'Northeast','Manipur':'Northeast','Meghalaya':'Northeast','Mizoram':'Northeast','Nagaland':'Northeast','Sikkim':'Northeast','Tripura':'Northeast',
    'Andaman & Nicobar Islands':'Island'
}
df['Region'] = df['State/Uts'].map(region_map)
region_gdp = df.groupby('Region')[latest_col].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 5.5))
sns.barplot(x=region_gdp.values, y=region_gdp.index, hue=region_gdp.index, palette='mako', legend=False, ax=ax)
ax.set_title(f"Regional GSDP Standing ({snapshot_year})", fontsize=14, fontweight='bold')
ax.set_xlabel("Total GSDP (₹ Crore)")
for i, v in enumerate(region_gdp.values):
    ax.text(v, i, f' {v:,.0f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/08_regional_ranking.png")
plt.close()

# ---------------------------------------------------------------
# 10. INCREMENT OVER 10 YEARS (2011-12 -> 2021-22, or last available)
# ---------------------------------------------------------------
first_col = gdp_cols[0]
inc_df = states_df[['State/Uts', first_col, latest_col]].dropna().copy()
inc_df['Increase_Cr'] = inc_df[latest_col] - inc_df[first_col]
inc_df['Increase_%'] = (inc_df['Increase_Cr'] / inc_df[first_col]) * 100
inc_sorted = inc_df.sort_values('Increase_%', ascending=False)

fig, ax = plt.subplots(figsize=(11, 8))
sns.barplot(data=inc_sorted, y='State/Uts', x='Increase_%', hue='State/Uts', palette='coolwarm', legend=False, ax=ax)
ax.set_title(f"% Increase in GSDP: {years[0]} → {snapshot_year}", fontsize=14, fontweight='bold')
ax.set_xlabel("% Increase over the period")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/09_decade_increment.png")
plt.close()

# ---------------------------------------------------------------
# 11. GDP GROWTH RATE HEATMAP (State x Year % growth)
# ---------------------------------------------------------------
growth_matrix = states_df.set_index('State/Uts')[growth_cols]
growth_matrix.columns = [c.replace('% Growth', '') for c in growth_cols]

fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(growth_matrix.astype(float), annot=True, fmt=".1f", cmap="RdYlGn",
            center=10, ax=ax, linewidths=0.4, linecolor='white',
            cbar_kws={'label': '% Growth YoY'})
ax.set_title("YoY % Growth Rate Heatmap — State x Year", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/10_growth_rate_heatmap.png")
plt.close()

# ---------------------------------------------------------------
# 12. COVID-19 IMPACT: Growth rate before vs during vs after COVID
# ---------------------------------------------------------------
covid_cols = {
    'Pre-COVID (2018-19)': '% Growth2018-19',
    'Pre-COVID (2019-20)': '% Growth2019-20',
    'COVID Year (2020-21)': '% Growth2020-21',
    'Recovery (2021-22)': '% Growth2021-22'
}
covid_avg = {}
for label, col in covid_cols.items():
    covid_avg[label] = states_df[col].mean(skipna=True)

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(covid_avg.keys(), covid_avg.values(), color=['#4c956c','#4c956c','#e63946','#2a9d8f'])
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title("Average State GSDP Growth Rate: Before, During & After COVID-19", fontsize=13, fontweight='bold')
ax.set_ylabel("Average % Growth (across states)")
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h + (0.3 if h>=0 else -0.8), f'{h:.2f}%', ha='center', fontweight='bold')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{OUT}/11_covid_impact.png")
plt.close()

# States most negatively affected by COVID (2020-21 growth)
covid_impact_states = states_df[['State/Uts', '% Growth2020-21']].dropna().sort_values('% Growth2020-21')

# ---------------------------------------------------------------
# 13. Save summary insights to text file
# ---------------------------------------------------------------
summary = f"""
GSDP EDA — KEY INSIGHTS SUMMARY
================================
Dataset: {df.shape[0]} states/UTs, {len(years)} years ({years[0]} to {years[-1]})

HIGH / LOW GDP ({snapshot_year}):
  Highest GSDP: {high_state['State/Uts']} — ₹{high_state[latest_col]:,.0f} Cr
  Lowest GSDP (states): {low_state_row['State/Uts']} — ₹{low_state_row[latest_col]:,.0f} Cr

TOP 10 STATES BY GSDP ({snapshot_year}):
{top10[['State/Uts', latest_col]].to_string(index=False)}

FASTEST GROWING STATES (CAGR over full period):
{cagr_sorted.head(5).to_string(index=False)}

SLOWEST GROWING STATES (CAGR over full period):
{cagr_sorted.tail(5).to_string(index=False)}

OUTLIERS (IQR method, {snapshot_year} GSDP):
{outliers.to_string(index=False) if len(outliers) else "None detected"}

REGIONAL RANKING ({snapshot_year} total GSDP):
{region_gdp.to_string()}

DECADE INCREMENT ({years[0]} -> {snapshot_year}), Top 5 by % increase:
{inc_sorted.head(5)[['State/Uts','Increase_%']].to_string(index=False)}

COVID-19 IMPACT (avg growth rate across states):
{pd.Series(covid_avg).to_string()}

Most negatively hit by COVID (2020-21 growth rate), bottom 5:
{covid_impact_states.head(5).to_string(index=False)}

TOTAL NATIONAL GSDP BY YEAR:
{yearly_total.to_string()}
"""
with open(f"{OUT}/insights_summary.txt", "w") as f:
    f.write(summary)

print(summary)
print("\nAll charts and summary saved to:", OUT)
