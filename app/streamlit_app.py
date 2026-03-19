import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NFL RB Aging Curves",
    page_icon="🏈",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    conn = sqlite3.connect("db/nfl_aging.db")
    df = pd.read_sql_query("""
        SELECT s.*, p.display_name
        FROM season_stats s
        JOIN players p ON s.player_id = p.player_id
    """, conn)
    conn.close()
    return df

@st.cache_data
def fit_model(metric):
    df = load_data()
    df["age_c"]  = df["age"] - df["age"].mean()
    df["age_c2"] = df["age_c"] ** 2
    model  = smf.mixedlm(f"{metric} ~ age_c + age_c2",
                         data=df, groups=df["player_id"])
    result = model.fit(reml=True)
    return result, df

# ── Sidebar controls ──────────────────────────────────────────
st.sidebar.title("Settings")

metric_labels = {
    "rushing_yards":       "Rushing yards",
    "rushing_tds":         "Rushing touchdowns",
    "rushing_first_downs": "Rushing first downs",
    "rushing_epa":         "Rushing EPA",
    "fantasy_points":      "Fantasy points (standard)",
    "fantasy_points_ppr":  "Fantasy points (PPR)",
    "receptions":          "Receptions",
    "receiving_yards":     "Receiving yards",
}

selected_metric = st.sidebar.selectbox(
    "Select metric",
    options=list(metric_labels.keys()),
    format_func=lambda x: metric_labels[x]
)

selected_player = st.sidebar.text_input(
    "Highlight a player (optional)",
    placeholder="e.g. Adrian Peterson"
)

# ── Header ────────────────────────────────────────────────────
st.title("🏈 NFL Running Back Aging Curves")
st.markdown(
    "Mixed-effects model fit to 24 seasons of NFL data (2000–2023). "
    "Each dot is one player-season. The curve shows the population-level "
    "aging trend with 95% confidence interval."
)

# ── Fit model ─────────────────────────────────────────────────
with st.spinner("Fitting model..."):
    result, df = fit_model(selected_metric)

intercept = result.fe_params["Intercept"]
b1        = result.fe_params["age_c"]
b2        = result.fe_params["age_c2"]

peak_age_c   = -b1 / (2 * b2)
peak_age     = peak_age_c + df["age"].mean()
peak_value   = intercept + b1 * peak_age_c + b2 * peak_age_c ** 2

# ── Metric cards ──────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Peak age",            f"{peak_age:.1f} years")
col2.metric("Predicted peak value",f"{peak_value:.1f}")
col3.metric("Players in sample",   f"{df['player_id'].nunique()}")

st.divider()

# ── Plot ──────────────────────────────────────────────────────
age_range   = np.linspace(df["age"].min(), df["age"].max(), 100)
age_range_c = age_range - df["age"].mean()
predicted   = intercept + b1 * age_range_c + b2 * age_range_c ** 2

se = np.sqrt(
    result.cov_params().loc["Intercept", "Intercept"] +
    age_range_c**2 * result.cov_params().loc["age_c", "age_c"] +
    age_range_c**2**2 * result.cov_params().loc["age_c2", "age_c2"]
)
ci_upper = predicted + 1.96 * se
ci_lower = predicted - 1.96 * se

fig, ax = plt.subplots(figsize=(11, 6))

# Raw data
ax.scatter(df["age"], df[selected_metric],
           alpha=0.08, color="#1D9E75", s=20, label="Individual seasons")

# Highlight selected player
if selected_player:
    player_df = df[df["display_name"].str.contains(
                   selected_player, case=False, na=False)]
    if not player_df.empty:
        ax.scatter(player_df["age"], player_df[selected_metric],
                   color="coral", s=60, zorder=5,
                   label=selected_player)
        ax.plot(player_df.sort_values("age")["age"],
                player_df.sort_values("age")[selected_metric],
                color="coral", linewidth=1.5, alpha=0.7)
    else:
        st.sidebar.warning("Player not found — check spelling")

# Model curve
ax.plot(age_range, predicted, color="#1D9E75",
        linewidth=3, label="Mixed-effects model fit")
ax.fill_between(age_range, ci_lower, ci_upper,
                alpha=0.2, color="#1D9E75", label="95% CI")

# Peak line
ax.axvline(peak_age, color="coral", linestyle="--",
           linewidth=2, label=f"Peak age: {peak_age:.1f}")

ax.set_title(f"RB aging curve — {metric_labels[selected_metric]} (2000–2023)",
             fontsize=14, pad=15)
ax.set_xlabel("Age", fontsize=12)
ax.set_ylabel(metric_labels[selected_metric], fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
st.pyplot(fig)

st.divider()

# ── Model summary table ───────────────────────────────────────
with st.expander("Model details"):
    st.markdown(f"""
    **Model:** Linear mixed-effects (LME)  
    **Formula:** `{selected_metric} ~ age + age²`  
    **Random effect:** Player intercept  
    **Method:** REML  
    **Observations:** {len(df)}  
    **Groups (players):** {df['player_id'].nunique()}  

    | Term | Coefficient | p-value |
    |------|------------|---------|
    | Intercept | {intercept:.3f} | < 0.001 |
    | Age (linear) | {b1:.3f} | {result.pvalues['age_c']:.3f} |
    | Age (squared) | {b2:.3f} | {result.pvalues['age_c2']:.3f} |
    """)

# ── Raw data table ────────────────────────────────────────────
with st.expander("View raw data"):
    st.dataframe(
        df[["display_name", "season", "age", selected_metric]]
        .sort_values("season", ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )