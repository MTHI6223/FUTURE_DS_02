# Telco Customer Churn Analysis
# Future Interns – Data Science & Analytics Task 2
# pip install pandas matplotlib numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# Load the data and fix the obvious issues
# =============================================================================

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)


df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)

total_customers = len(df)
total_churned   = df["Churn_bin"].sum()
churn_pct       = df["Churn_bin"].mean() * 100

print(f"Loaded {total_customers:,} customers — {total_churned:,} churned ({churn_pct:.1f}%)")
print("Data looks clean. Moving on...\n")


# =============================================================================
# Engineer a couple of useful features
# =============================================================================


df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[0, 6, 12, 24, 48, 72],
    labels=["0–6m", "7–12m", "13–24m", "25–48m", "49–72m"]
)


df["high_risk"] = (
    (df["Contract"] == "Month-to-month") &
    (df["InternetService"] == "Fiber optic") &
    (df["tenure"] < 12)
).astype(int)


# =============================================================================
# Calculate all the metrics we'll need for charts and the summary
# =============================================================================

# Top-level KPIs
overall_churn  = df["Churn_bin"].mean() * 100
avg_tenure_ch  = df.loc[df["Churn_bin"] == 1, "tenure"].mean()
avg_tenure_ret = df.loc[df["Churn_bin"] == 0, "tenure"].mean()
avg_charge_ch  = df.loc[df["Churn_bin"] == 1, "MonthlyCharges"].mean()
avg_charge_ret = df.loc[df["Churn_bin"] == 0, "MonthlyCharges"].mean()

# High-risk segment
high_risk_churn = df.loc[df["high_risk"] == 1, "Churn_bin"].mean() * 100
high_risk_count = df["high_risk"].sum()

# Demographics
senior_churn   = df.loc[df["SeniorCitizen"] == 1, "Churn_bin"].mean() * 100
non_senior_ch  = df.loc[df["SeniorCitizen"] == 0, "Churn_bin"].mean() * 100
paper_churn    = df.loc[df["PaperlessBilling"] == "Yes", "Churn_bin"].mean() * 100
no_paper_churn = df.loc[df["PaperlessBilling"] == "No",  "Churn_bin"].mean() * 100

# Group breakdowns — churn rate per category
contract_churn = df.groupby("Contract")["Churn_bin"].mean() * 100
tenure_churn   = df.groupby("tenure_group", observed=True)["Churn_bin"].mean() * 100
internet_churn = df.groupby("InternetService")["Churn_bin"].mean() * 100
payment_churn  = df.groupby("PaymentMethod")["Churn_bin"].mean() * 100


addon_services = [
    "OnlineSecurity", "TechSupport", "OnlineBackup",
    "DeviceProtection", "StreamingTV", "StreamingMovies"
]

service_with    = []
service_without = []

for service in addon_services:
    rate_with    = df.loc[df[service] == "Yes", "Churn_bin"].mean() * 100
    rate_without = df.loc[df[service] == "No",  "Churn_bin"].mean() * 100
    service_with.append(rate_with)
    service_without.append(rate_without)


# =============================================================================
# Set up the dark theme styling
# =============================================================================

# Colour palette — dark background
BG      = "#1a1a1a"   
PANEL   = "#242424"   
TEXT    = "#e8e8e8"   
SUBTEXT = "#aaaaaa"   
RED     = "#e84545"   
AMBER   = "#c49a3c"   
GREEN   = "#4caf73"   
BLUE    = "#4a9fd4"   
LGRAY   = "#3a3a3a"   

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   LGRAY,
    "axes.labelcolor":  SUBTEXT,
    "xtick.color":      SUBTEXT,
    "ytick.color":      SUBTEXT,
    "text.color":       TEXT,
    "grid.color":       LGRAY,
    "grid.linewidth":   0.5,
    "font.family":      "DejaVu Sans",
    "font.size":        9,
})


def traffic_light_colors(values, medium_threshold=15, high_threshold=30):
    """
    Colour each bar red/amber/green based on how bad the churn rate is.
    Anything above 30% is red (danger zone), 15-30% is amber, below 15% is green.
    """
    colors = []
    for v in values:
        if v >= high_threshold:
            colors.append(RED)
        elif v >= medium_threshold:
            colors.append(AMBER)
        else:
            colors.append(GREEN)
    return colors


def add_bar_labels(ax, bars, pad=1.5):
    """
    Put the percentage value just above each bar.
    Much easier to read than trying to eyeball the y-axis.
    """
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + pad,
            f"{height:.1f}%",
            ha="center", va="bottom",
            fontsize=8, color=TEXT, fontweight="bold"
        )


# =============================================================================
# Build the main dashboard (Figure 1)
# =============================================================================

fig = plt.figure(figsize=(16, 14), facecolor=BG)
fig.suptitle(
    "Telco Customer Churn Analysis Dashboard",
    fontsize=18, fontweight="bold", color=TEXT, y=0.98
)


gs = GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
              left=0.06, right=0.97, top=0.93, bottom=0.04)


# KPI strip 

ax_kpi = fig.add_subplot(gs[0, :])
ax_kpi.set_facecolor(PANEL)
ax_kpi.axis("off")

kpi_cards = [
    (
        "overall churn rate",
        f"{overall_churn:.1f}%",
        RED,
        f"{df['Churn_bin'].sum():,} of {len(df):,} customers"
    ),
    (
        "avg. tenure (churned)",
        f"{avg_tenure_ch:.0f} mo",
        TEXT,
        f"vs {avg_tenure_ret:.0f} mo for retained"
    ),
    (
        "avg. monthly charges (churned)",
        f"${avg_charge_ch:.2f}",
        TEXT,
        f"vs ${avg_charge_ret:.2f} for retained"
    ),
    (
        "highest-risk segment",
        f"{high_risk_churn:.1f}%",
        RED,
        f"M2M + Fiber + <12m tenure\n{high_risk_count} customers"
    ),
]

for i, (label, value, value_color, subtitle) in enumerate(kpi_cards):
    x_pos = 0.02 + i * 0.25
    ax_kpi.text(x_pos, 0.90, label,    transform=ax_kpi.transAxes, fontsize=8,   color=SUBTEXT)
    ax_kpi.text(x_pos, 0.52, value,    transform=ax_kpi.transAxes, fontsize=20,  color=value_color, fontweight="bold")
    ax_kpi.text(x_pos, 0.10, subtitle, transform=ax_kpi.transAxes, fontsize=7.5, color=SUBTEXT, linespacing=1.4)


    if i < 3:
        ax_kpi.axvline(x=0.25 * (i + 1), color=LGRAY, linewidth=0.8, ymin=0.05, ymax=0.95)

ax_kpi.set_xlim(0, 1)
ax_kpi.set_ylim(0, 1)
for spine in ax_kpi.spines.values():
    spine.set_edgecolor(LGRAY)
    spine.set_linewidth(0.8)



ax1 = fig.add_subplot(gs[1, 0])

contract_order = ["Month-to-month", "One year", "Two year"]
contract_vals  = [contract_churn.get(c, 0) for c in contract_order]

bars = ax1.bar(contract_order, contract_vals,
               color=traffic_light_colors(contract_vals), width=0.5, zorder=2)
add_bar_labels(ax1, bars)

ax1.set_title("CHURN BY CONTRACT TYPE", fontsize=9, color=SUBTEXT, loc="left", pad=8, fontweight="bold")
ax1.set_ylabel("Churn rate (%)", fontsize=8)
ax1.set_ylim(0, 55)
ax1.yaxis.grid(True, zorder=1)
ax1.set_axisbelow(True)
ax1.tick_params(axis="x", labelsize=8)




ax2 = fig.add_subplot(gs[1, 1])

tenure_order = ["0–6m", "7–12m", "13–24m", "25–48m", "49–72m"]
tenure_vals  = [tenure_churn.get(t, 0) for t in tenure_order]

bars2 = ax2.bar(tenure_order, tenure_vals,
                color=traffic_light_colors(tenure_vals), width=0.55, zorder=2)
add_bar_labels(ax2, bars2)

ax2.set_title("CHURN BY TENURE COHORT", fontsize=9, color=SUBTEXT, loc="left", pad=8, fontweight="bold")
ax2.set_ylabel("Churn rate (%)", fontsize=8)
ax2.set_ylim(0, 70)
ax2.yaxis.grid(True, zorder=1)
ax2.set_axisbelow(True)


# --- Internet service + Payment method 

ax3 = fig.add_subplot(gs[2, 0])
ax3.axis("off")

# Internet service breakdown
internet_order = ["Fiber optic", "DSL", "No internet"]
int_vals       = [internet_churn.get(k, 0) for k in internet_order]
int_colors     = traffic_light_colors(int_vals)

ax3.text(0.0, 0.98, "CHURN BY INTERNET SERVICE",
         transform=ax3.transAxes, fontsize=9, fontweight="bold", color=SUBTEXT)

for idx, (label, val, col) in enumerate(zip(internet_order, int_vals, int_colors)):
    y = 0.82 - idx * 0.16
    ax3.text(0.0, y, label, transform=ax3.transAxes, fontsize=8.5, color=TEXT)

    
    bar_width = val / 60
    rect = mpatches.FancyBboxPatch(
        (0.0, y - 0.07), bar_width, 0.06,
        boxstyle="round,pad=0.005",
        facecolor=col, transform=ax3.transAxes, clip_on=False
    )
    ax3.add_patch(rect)
    ax3.text(bar_width + 0.02, y - 0.04, f"{val:.1f}%",
             transform=ax3.transAxes, fontsize=8.5, color=col, fontweight="bold")

# Payment method breakdown
pay_order  = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
pay_labels = ["Electronic check", "Mailed check", "Bank transfer (auto)", "Credit card (auto)"]
pay_vals   = [payment_churn.get(k, 0) for k in pay_order]
pay_colors = traffic_light_colors(pay_vals)

ax3.text(0.0, 0.38, "CHURN BY PAYMENT METHOD",
         transform=ax3.transAxes, fontsize=9, fontweight="bold", color=SUBTEXT)

for idx, (label, val, col) in enumerate(zip(pay_labels, pay_vals, pay_colors)):
    y = 0.26 - idx * 0.13
    ax3.text(0.0, y, label, transform=ax3.transAxes, fontsize=8.5, color=TEXT)

    bar_width = val / 60
    rect = mpatches.FancyBboxPatch(
        (0.0, y - 0.065), bar_width, 0.055,
        boxstyle="round,pad=0.005",
        facecolor=col, transform=ax3.transAxes, clip_on=False
    )
    ax3.add_patch(rect)
    ax3.text(bar_width + 0.02, y - 0.04, f"{val:.1f}%",
             transform=ax3.transAxes, fontsize=8.5, color=col, fontweight="bold")


# Protective effect of add-on services 


ax4 = fig.add_subplot(gs[2, 1])

x_positions = np.arange(len(addon_services))
bar_width    = 0.38

ax4.bar(x_positions - bar_width / 2, service_with,    width=bar_width, color=BLUE,  zorder=2)
ax4.bar(x_positions + bar_width / 2, service_without, width=bar_width, color=LGRAY, zorder=2)

ax4.set_title("PROTECTIVE EFFECT OF ADD-ON SERVICES",
              fontsize=9, color=SUBTEXT, loc="left", pad=8, fontweight="bold")
ax4.set_xticks(x_positions)
ax4.set_xticklabels(
    ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtect.", "StreamingTV", "StreamingMovies"],
    rotation=30, ha="right", fontsize=7.5
)
ax4.set_ylabel("Churn rate (%)", fontsize=8)
ax4.set_ylim(0, 55)
ax4.yaxis.grid(True, zorder=1)
ax4.set_axisbelow(True)

legend_handles = [
    mpatches.Patch(color=BLUE,  label="with service"),
    mpatches.Patch(color=LGRAY, label="without service"),
]
ax4.legend(handles=legend_handles, fontsize=8, loc="upper right",
           framealpha=0.3, facecolor=PANEL, edgecolor=LGRAY)


# Critical risk segments + recommendations 

ax5 = fig.add_subplot(gs[3, :])
ax5.axis("off")
ax5.set_facecolor(PANEL)
for spine in ax5.spines.values():
    spine.set_edgecolor(RED)
    spine.set_linewidth(1)

# Three headline
risk_stats = [
    (f"{high_risk_churn:.1f}%", RED,   f"M2M + Fiber + <12m tenure\n{high_risk_count} customers"),
    (f"{senior_churn:.1f}%",   AMBER,  f"senior citizens\nvs {non_senior_ch:.1f}% non-seniors"),
    (f"{paper_churn:.1f}%",    AMBER,  f"paperless billing\nvs {no_paper_churn:.1f}% paper billing"),
]

ax5.text(0.01, 0.97, "CRITICAL RISK SEGMENT BREAKDOWN",
         transform=ax5.transAxes, fontsize=9, fontweight="bold", color=RED)

for i, (stat_value, stat_color, stat_label) in enumerate(risk_stats):
    x_pos = 0.01 + i * 0.13
    ax5.text(x_pos, 0.75, stat_value, transform=ax5.transAxes,
             fontsize=18, fontweight="bold", color=stat_color)
    ax5.text(x_pos, 0.40, stat_label, transform=ax5.transAxes,
             fontsize=7.5, color=SUBTEXT, linespacing=1.5)

# Five actionable recommendations on the right side
recommendations = [
    (
        "1", "Offer contract upgrade incentives",
        "Customers on M2M churn at 42.7% vs 2.8% on two-year plans. "
        "A targeted discount to lock in annual contracts in the first 12 months could significantly reduce churn."
    ),
    (
        "2", "Investigate Fiber Optic quality issues",
        "Despite being a premium service, Fiber customers churn at twice the DSL rate. "
        "This may signal a service quality, pricing, or expectation-gap problem worth investigating."
    ),
    (
        "3", "Promote security & support add-ons early",
        "OnlineSecurity and TechSupport customers churn at ~15% vs ~42% without them. "
        "Bundle these into onboarding for new customers."
    ),
    (
        "4", "Target the first 6 months aggressively",
        "Over half of customers who churn do so within 6 months. "
        "A structured 90-day onboarding program and check-in calls could substantially improve early retention."
    ),
    (
        "5", "Nudge electronic check users to auto-pay",
        "Electronic check has the highest churn (45.3%) while auto-pay sits below 17%. "
        "A small bill credit for switching to autopay could reduce both churn and payment risk."
    ),
]

ax5.text(0.40, 0.97, "KEY RECOMMENDATIONS",
         transform=ax5.transAxes, fontsize=9, fontweight="bold", color=SUBTEXT)

for i, (number, title, body) in enumerate(recommendations):
    y_pos = 0.82 - i * 0.185
    ax5.text(0.40, y_pos, number, transform=ax5.transAxes,
             fontsize=10, fontweight="bold", color=AMBER)
    ax5.text(0.43, y_pos, f"{title}  —  {body}",
             transform=ax5.transAxes, fontsize=7.8, color=TEXT,
             wrap=True, va="top",
             bbox=dict(boxstyle="round,pad=0.0", fc="none", ec="none"))

plt.savefig("churn_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.show()
print("Dashboard saved → churn_dashboard.png")


# =============================================================================
# Demographic deep-dive (Figure 2)
# =============================================================================


fig2, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=BG)
fig2.suptitle("Churn Deep-Dive: Demographic & Behavioural Segments",
              fontsize=14, fontweight="bold", color=TEXT, y=1.01)

demographic_panels = [
    ("gender",           "Churn by Gender"),
    ("Partner",          "Churn by Partner Status"),
    ("Dependents",       "Churn by Dependents"),
    ("SeniorCitizen",    "Churn: Senior vs Non-Senior"),
    ("PaperlessBilling", "Churn by Paperless Billing"),
    ("MultipleLines",    "Churn by Multiple Lines"),
]

for ax, (column, title) in zip(axes.flat, demographic_panels):
    group_churn = df.groupby(column)["Churn_bin"].mean() * 100
    group_churn = group_churn.sort_values(ascending=False)

    bars = ax.bar(
        group_churn.index.astype(str),
        group_churn.values,
        color=traffic_light_colors(group_churn.values),
        width=0.5,
        zorder=2
    )
    add_bar_labels(ax, bars, pad=1)

    ax.set_title(title, fontsize=9, color=SUBTEXT, fontweight="bold")
    ax.set_ylabel("Churn rate (%)", fontsize=8)
    ax.set_ylim(0, group_churn.max() * 1.35)
    ax.yaxis.grid(True, zorder=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)

plt.tight_layout(pad=2.5)
plt.savefig("churn_deepdive.png", dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.show()
print("Deep-dive chart saved → churn_deepdive.png")


# =============================================================================
# Print a quick summary to the terminal
# =============================================================================
# 

print("\n" + "=" * 60)
print("  TELCO CUSTOMER CHURN — EXECUTIVE SUMMARY")
print("=" * 60)

print(f"\n  Total customers   : {len(df):,}")
print(f"  Churned           : {df['Churn_bin'].sum():,}  ({overall_churn:.1f}%)")
print(f"  Retained          : {(df['Churn_bin']==0).sum():,}  ({100 - overall_churn:.1f}%)")

print(f"\n  Avg tenure  — churned  : {avg_tenure_ch:.1f} months")
print(f"\n  Avg tenure  — retained : {avg_tenure_ret:.1f} months")
print(f"  Avg charges — churned  : ${avg_charge_ch:.2f}/mo")
print(f"  Avg charges — retained : ${avg_charge_ret:.2f}/mo")

print("\n  Churn by contract type:")
for contract_type, rate in contract_churn.sort_values(ascending=False).items():
    print(f"    {contract_type:<22} {rate:.1f}%")

print("\n  Churn by internet service:")
for service_type, rate in internet_churn.sort_values(ascending=False).items():
    print(f"    {service_type:<22} {rate:.1f}%")

print("\n  Churn by payment method:")
for method, rate in payment_churn.sort_values(ascending=False).items():
    print(f"    {method:<35} {rate:.1f}%")

print("\n  Churn by tenure cohort:")
for cohort, rate in tenure_churn.items():
    print(f"    {str(cohort):<12} {rate:.1f}%")

print(f"\n  High-risk segment (M2M + Fiber + <12m) : {high_risk_churn:.1f}%  ({high_risk_count} customers)")
print(f"  Senior citizen churn rate              : {senior_churn:.1f}%")
print(f"  Paperless billing churn rate           : {paper_churn:.1f}%")
print("=" * 60)
print("\nDone. Both charts have been saved to your working directory.")