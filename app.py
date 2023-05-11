# from dotenv import load_dotenv, find_dotenv
import pandas as pd
import plotly.express as px
from plotly.colors import n_colors
import streamlit as st
from glob import glob
from utils import run_analysis, run_forecast
from jmspack.NLTSA import fluctuation_intensity
from jmspack.utils import apply_scaling

# CONFIG
st.set_page_config(
    page_title="Facial Expression Analysis Dashboard",
    page_icon=":sparkles:",
    layout="wide",
)


# CREATE CACHE DATA FUNCTION
# @st.cache_data(ttl=3600)
def get_data(filename="data/face_expressions.csv"):
    df = pd.read_csv(filename)
    df.columns = [x.replace(" ", "") for x in df.columns.tolist()]
    return df


# CREATE ANALYSIS CACHE FUNCTION
# @st.cache_data(ttl=3600)
def get_analysis_output(df, outcome, feature_list):
    df_pred_test, shap_df, gini_df = run_analysis(
        data=df, outcome=outcome, feature_list=feature_list
    )
    return df_pred_test, shap_df, gini_df


# CREATE FORECAST CACHE FUNCTION
# @st.cache_data(ttl=3600)
# def get_forecast_output(df, outcome, feature_list):
#     plot_df = run_forecast(data=df, outcome=outcome, feature_list=feature_list)
#     return plot_df

# SIDEBAR - TITLE AND DATA SOURCE
file_options = glob("data/*.csv")
st.sidebar.header("Please filter here:")
data_source = st.sidebar.selectbox(
    "Select a data source:",
    options=file_options,
)

# READ DATA
df = get_data(filename=data_source)

# SIDEBAR - OUTCOME AND FEATURE LIST
outcome = st.sidebar.selectbox(
    "Select an outcome measure of interest:", options=df.columns.tolist()
)

feature_list = st.sidebar.multiselect(
    "Select features to include in the analysis:",
    options=df.filter(regex="AU[\d\D][\d\D]_r").columns.tolist(),
    default=df.filter(regex="AU[\d\D][\d\D]_r").columns.tolist()[0:10],
)

# feature_list = df.drop(columns=[outcome]).columns.tolist()

# SIDEBAR - LOGO AND CREDITS
st.sidebar.markdown("---")
st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div style="text-align: center; padding-right: 10px;">
        <img alt="logo" src="https://services.jms.rocks/img/logo.png" width="100">
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    """
    <div style="text-align: center; color: #E8C003; margin-top: 40px; margin-bottom: 40px;">
        <a href="https://services.jms.rocks" style="color: #E8C003;">Created by James Twose</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# RUN MAIN ANALYSIS
df_pred_test, shap_df, gini_df = get_analysis_output(
    df=df, outcome=outcome, feature_list=feature_list
)

# RUN FORECAST ANALYSIS
# top 3 important features based on SHAP variance
top_3_features = (
    shap_df.groupby("variable")
    .var()["shap_data"]
    .sort_values(ascending=False)
    .head(3)
    .index.tolist()
)
# forecast_plot_df = get_forecast_output(df=df, outcome=outcome, feature_list=top_3_features)
# forecast_plot_df = get_forecast_output(df=df, outcome=outcome, feature_list=feature_list)

pred_test_corr = df_pred_test.corr().iloc[0, 1].round(3)
likelihood_overfit = "Yes" if pred_test_corr > 0.8 else "No"
shap_feature_importance_sorted = (
    shap_df.groupby("variable")
    .var()["shap_data"]
    .sort_values(ascending=True)
    .index.tolist()
)
shap_best_feature = shap_feature_importance_sorted[-1]

gini_best_feature = gini_df.head(1)["feature"].tolist()[0]

# MAINPAGE
st.markdown("<h1>Facial Expression Analysis Dashboard</h1>", unsafe_allow_html=True)
left_column, middle_column, right_column = st.columns(3)
with left_column:
    st.subheader("Chosen Outcome")
    st.subheader(outcome)
with middle_column:
    st.subheader("Mean of chosen outcome")
    st.subheader(f"{df[outcome].mean():,.3f}")
with right_column:
    st.subheader("Variance of chosen outcome")
    st.subheader(f"{df[outcome].var():,.3f}")
st.markdown("---")

st.header("Main Report")
if likelihood_overfit == "Yes":
    st.markdown(
        f"""There is a likelihood that the model will not generalize (i.e.
        it is overfitting the current data). This is based on the Pearson
        Correlation between the predicted and actual :green[{outcome}] being so
        high (r-value=:green[{pred_test_corr}])."""
    )
st.markdown(
    f"""Based on the variance in SHAP values, the following feature
    is the most important: :green[{shap_best_feature}]"""
)
st.markdown(
    f"""Based on the gini importance, the following feature
    is the most important: :green[{gini_best_feature}]"""
)
st.markdown("---")

st.header("Selected Dataframe")
st.dataframe(df)
st.markdown("---")
st.header("Descriptive Statistics")
st.dataframe(df.describe())
st.markdown("---")

st.header("Pearson Correlations between all columns")
plot_df = df[feature_list].corr().round(3)
# correlation plot
corr_heat = px.imshow(plot_df, text_auto=True, color_continuous_scale="RdBu")
st.plotly_chart(corr_heat)
st.markdown("---")

# SHOW MAIN ANALYSIS OUTPUT
st.header("Model Creation and Feature Importance Calculation")

# SHAP PLOT
st.markdown(
    """
    <div>
        <p>For more information on SHAP Values, please see the following: 
            <a href="https://christophm.github.io/interpretable-ml-book/shap.html"
            style="color: #E8C003;">SHAP Values Explanation</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
shap_fig = px.strip(
    shap_df.sort_values("actual_data"),
    x="shap_data",
    y="variable",
    color="actual_data",
    color_discrete_sequence=n_colors(
        "rgb(143, 15, 212)", "rgb(252, 221, 20)", shap_df.shape[0], colortype="rgb"
    ),
    title="Feature Importance Based on SHAP Values",
)
shap_fig.update_layout(showlegend=False, coloraxis_showscale=True)
shap_fig.update_yaxes(
    categoryorder="array", categoryarray=shap_feature_importance_sorted
)
st.plotly_chart(shap_fig)

# GINI PLOT
st.markdown(
    """
    <div>
        <p>For more information on Gini Gain, please see the following:
            <a href="https://www.codecademy.com/article/fe-feature-importance-final"
            style="color: #E8C003;">Gini Gain Explanation</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
gini_fig = px.bar(
    gini_df,
    x="feature_importance",
    y="feature",
    color="feature",
    title="Feature Importance Based on Gini Gain",
)
st.plotly_chart(gini_fig)
st.markdown("---")

# SHOW MAIN FORECAST OUTPUT
st.header("Forecasting time series")
# TIME SERIES PLOT
df_melt = df[["timestamp"] + feature_list].melt(id_vars="timestamp")
time_series_fig = px.line(
    df_melt,
    x="timestamp",
    y="value",
    color="variable",
    title="Time Series Plot of all variables",
)
st.plotly_chart(time_series_fig)

# NLTSA PLOTS
st.markdown(
    """Using Nonlinear Time Series Analysis (NLTSA) to visualize change over time."""
)

# TS LEVELS PLOT
fi_df = fluctuation_intensity(
    df=df[feature_list].pipe(apply_scaling),
    win=60,
    xmin=0,
    xmax=1,
    col_first=1,
    col_last=df[feature_list].shape[1],
)

fi_heat = px.imshow(
    fi_df.T,
    text_auto=False,
    color_continuous_scale="RdBu",
    title="Fluctuation Intensity",
)
st.plotly_chart(fi_heat)
st.markdown("---")

# FORECAST TIME SERIES PLOT
# st.markdown(
#     f"""The forecast is based on the top 3 important features as defined
#     by the variance in SHAP Values. These are: :green[{top_3_features}]"""
# )
# forecast_fig = px.line(
#     forecast_plot_df,
#     x="timestamp",
#     y=outcome,
#     color="type",
#     markers=True,
#     title=f"Forecasted outcome == {outcome}",
# )
# st.plotly_chart(forecast_fig)
# st.markdown("---")

# HIDE STREAMLIT STYLE
hide_streamlit_style = """
                        <style>
                        #MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}
                        header {visibility: hidden;}
                        </style>
                        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
