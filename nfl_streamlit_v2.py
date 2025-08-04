import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# --- File paths ---
REG_PRED_PATH = "data/predictions_2025_regression.csv"
CLASS_PRED_PATH = "data/predictions_2025_classification.csv"
TEAM_METRICS_PATH = "data/team_season_week_metrics.csv"
SCHED_PATH = "data/nflsched_2025.xlsx"
NEAREST_MODEL_PATH = "data/nearest_games_model.pkl"



@st.cache_resource
def load_nearest_model():
    return joblib.load(NEAREST_MODEL_PATH)

def find_similar_games(spread_line, total_line, n=7):
    model_bundle = load_nearest_model()
    nn_model = model_bundle['model']
    scaler = model_bundle['scaler']
    data = model_bundle['data']

    query = scaler.transform([[spread_line, total_line]])
    distances, indices = nn_model.kneighbors(query, n_neighbors=n)
    return data.iloc[indices[0]].copy()

# --- Cached Data Loaders ---
@st.cache_data
def load_regression():
    df = pd.read_csv(REG_PRED_PATH)
    df['home_team_covers'] = df['home_team_margin'] > 0
    return df

@st.cache_data
def load_classification():
    return pd.read_csv(CLASS_PRED_PATH)

@st.cache_data
def load_metrics():
    return pd.read_csv(TEAM_METRICS_PATH)

@st.cache_data
def load_schedule():
    return pd.read_excel(SCHED_PATH)
# --- Load Data ---
reg_df = pd.read_csv(REG_PRED_PATH)
cls_df = pd.read_csv(CLASS_PRED_PATH)
metrics_df = pd.read_csv(TEAM_METRICS_PATH)
sched_df = pd.read_excel(SCHED_PATH)


# Add regression interpretation column
reg_df['home_team_covers'] = reg_df['home_team_margin'] > 0

# Complete team list (32 teams)
nfl_teams = sorted(set(sched_df['home_team'].unique()) | set(sched_df['away_team'].unique()))

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("2025 NFL Game Prediction Dashboard")
tabs = st.tabs(["📘 Overview", "📊 Regression Predictions", "📈 Classification Predictions", "🔍 Nearest Neighbors Search"])

# --- Tab 1: Overview ---
with tabs[0]:
    st.header("Model Overview")
    st.markdown("""
    This dashboard uses machine learning models trained on historical NFL play-by-play data (2020–2024) to forecast:

    - **Regression Predictions:** Final score predictions for both teams using XGBoost regressors.
    - **Classification Predictions:** Whether the home team will cover the spread using a classification model.
    - **Nearest Neighbors Search:** Finds similar historical games (2010-2024) based on spread and total line using a KNN model. 
      - ⚠️ *Spread input is from the **away team's** perspective: -2.5 = away is favorite; +2.5 = away is underdog.*

    All predictions are updated at the start of a new NFL week (Tuesday).
    """)

    st.subheader("\U0001F4CA Explore Team Metrics")
    metrics_rounded = metrics_df.copy()
    numeric_cols = metrics_rounded.select_dtypes(include=["float", "float64", "int"]).columns
    metrics_rounded[numeric_cols] = metrics_rounded[numeric_cols].round(3)

    with st.expander("🔎 Filter Team Metrics Table", expanded=True):
        selected_years = st.multiselect(
            "Filter by Season",
            sorted(metrics_rounded['season'].unique()),
            default=sorted(metrics_rounded['season'].unique())
        )
    
        selected_teams = st.multiselect(
            "Filter by Team",
            sorted(metrics_rounded['team'].unique()),
            default=sorted(metrics_rounded['team'].unique())
        )
    
        selected_weeks = st.multiselect(
            "Filter by Week",
            sorted(metrics_rounded['week'].unique()),
            default=sorted(metrics_rounded['week'].unique())
        )
    
        col_filter = st.multiselect(
            "Columns to Display",
            metrics_rounded.columns.tolist(),
            default=metrics_rounded.columns.tolist()
        )


    filtered_metrics = metrics_rounded[
        metrics_rounded['season'].isin(selected_years) &
        metrics_rounded['team'].isin(selected_teams) &
        metrics_rounded['week'].isin(selected_weeks)
    ][col_filter]

    st.dataframe(filtered_metrics, use_container_width=True)

    st.subheader("\U0001F4C8 Season Metrics Visualization (Scatter)")
    season_metrics = [
        col for col in metrics_rounded.columns 
        if col not in ['season', 'week', 'team'] and 'week' not in col.lower() and metrics_rounded[col].dtype != 'object'
    ]
    with st.expander("📊 Customize Season Scatter Plot", expanded=True):
        metric_y = st.selectbox("Select Metric for Y-axis", season_metrics)
        x_axis = st.radio("Select X-axis", ["season", "team"], horizontal=True)

        if x_axis == "season":
            selectable_seasons = sorted(metrics_rounded['season'].unique())
            selected_x_vals = st.multiselect("Choose Seasons to Display", selectable_seasons, default=[max(selectable_seasons)])
            plot_df = metrics_rounded[metrics_rounded['season'].isin(selected_x_vals)]
        else:
            most_recent_season = metrics_rounded['season'].max()
            selectable_teams = sorted(metrics_rounded['team'].unique())
            selected_x_vals = st.multiselect("Choose Teams to Display", selectable_teams, default=selectable_teams)
            plot_df = metrics_rounded[(metrics_rounded['team'].isin(selected_x_vals)) & (metrics_rounded['season'] == most_recent_season)]


    fig = px.scatter(
        plot_df,
        x=x_axis,
        y=metric_y,
        color='team',
        hover_data=['season', 'week'],
        title=f"{metric_y} vs {x_axis.title()} (Season View)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("\U0001F4C9 Weekly Metric Trends (Scatter + Trendline)")
    week_metrics = [
        "cpoe", "total_epa", "rush_attempt_off", "pass_attempt_off", "rush_epa", "pass_epa",
        "rush_count", "pass_count", "total_count", "rush_epa_per_play", "pass_epa_per_play",
        "off_total_epa_per_play", "avg_ryoe", "avg_pyoe", "avg_yacoe", "success_rate",
        "expl_rush_rate", "expl_pass_rate", "allowed_epa", "rush_attempt_def",
        "pass_attempt_def", "allowed_rush_epa", "allowed_pass_epa", "opp_rush_count",
        "opp_pass_count", "opp_total_count", "allowed_rush_epa_per_play", "allowed_pass_epa_per_play",
        "def_total_epa_per_play", "allowed_avg_ryoe", "allowed_avg_pyoe", "allowed_avg_yacoe",
        "opp_success_rate", "allowed_expl_rush_rate", "allowed_expl_pass_rate",
        "off_turnover_rate", "def_turnover_forced_rate", "off_voa_week", "def_voa_week",
        "off_voa_cum", "def_voa_cum", "off_dvoa_cum", "def_dvoa_cum"
    ]
    with st.expander("📈 Customize Weekly Trend Chart", expanded=True):
        week_metric_y = st.selectbox("Select Weekly Metric for Y-axis", week_metrics)
        selected_team_line = st.selectbox("Choose Team to Display", sorted(metrics_rounded['team'].unique()), index=sorted(metrics_rounded['team'].unique()).index("KC"))
        selected_seasons_line = st.multiselect("Choose Season(s)", sorted(metrics_rounded['season'].unique()), default=[metrics_rounded['season'].max()])


    line_df = metrics_rounded[
        (metrics_rounded['team'] == selected_team_line) &
        (metrics_rounded['season'].isin(selected_seasons_line))
    ]

    fig_line = px.scatter(
        line_df,
        x='week',
        y=week_metric_y,
        color='team',
        trendline="ols",
        trendline_color_override="red",
        hover_data=['season']
    )
    st.plotly_chart(fig_line, use_container_width=True)
with tabs[1]:
    st.header("Regression Model Predictions")
    st.markdown("""
    These predictions use a multi-output XGBoost regressor trained on 2020–2024 data.
    The model predicts:
    - Final score for home team
    - Final score for away team
    - Total score
    - Home team margin (model-derived spread)

    **How to interpret:**
    - `home_team_margin = home_score_prediction - away_score_prediction`
    - If `home_team_margin > 0`, the model believes the **home team will cover the spread**

    Use this to identify differences between Vegas spreads and model expectations.
    ***ALL SPREAD_LINE ARE FROM THE AWAY TEAM POINT OF VIEW***
    """)
    st.dataframe(reg_df.sort_values(by="home_team_margin", ascending=False))

# --- Tab 3: Classification Results ---
with tabs[2]:
    st.header("Classification Model Predictions")
    st.markdown("""
    These predictions use an XGBoost classifier to estimate whether the **home team will cover the spread**.

    - `home_team_covers = 1` means model predicts home team will cover
    - `prob_cover` is the model's confidence in that outcome

    This model was trained on 2020–2024 historical data with a season/week split to avoid leakage.
    ***ALL SPREAD_LINE ARE FROM THE AWAY TEAM POINT OF VIEW***
    """)
    st.dataframe(cls_df)

# --- Tab 4: Nearest Neighbors Search ---
with tabs[3]:
    st.header("Find Similar Historical Games")
    st.markdown("""
    This tool uses a K-Nearest Neighbors model trained on 2010–2024 historical games.
    It finds past games with similar **spread line** and **total line** to the one you input.

    ⚠️ **Spread is from the away team's perspective**:
    - `-2.5` → away team is favored by 2.5
    - `+2.5` → away team is 2.5-point underdog
    """)

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", nfl_teams)
        spread_line = st.number_input("Spread Line (away team perspective)", value=0.0, step=0.5)
    with col2:
        away_team = st.selectbox("Away Team", nfl_teams)
        total_line = st.number_input("Total Line", value=44.5, step=0.5)

    if st.button("Find Similar Games"):
        similar_games = find_similar_games(spread_line=spread_line, total_line=total_line)
        st.subheader("Most Similar Historical Matchups")
        st.dataframe(similar_games[['season', 'week', 'home_team', 'away_team', 'spread_line', 'total_line',
                                    'total_home_score', 'total_away_score', 'total_score', 'over_under_result']])

        # Prediction logic based on majority class
        if not similar_games.empty:
            over_count = sum("over" in str(x).lower() for x in similar_games['over_under_result'])
            under_count = sum("under" in str(x).lower() for x in similar_games['over_under_result'])

            if over_count > under_count:
                model_pick = "Over"
            elif under_count > over_count:
                model_pick = "Under"
            else:
                model_pick = "Push / No Clear Lean"

            st.markdown(f"### 📊 Model Prediction Based on Similar Games: **{model_pick}**")
            st.markdown(f"- Over: {over_count} of 7")
            st.markdown(f"- Under: {under_count} of 7")







