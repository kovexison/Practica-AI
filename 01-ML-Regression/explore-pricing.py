import pandas as pd
import folium
import streamlit as st
import plotly.express as px
from streamlit_folium import st_folium
import branca.colormap as cm

# -------------------------------
# Load dataset
df = pd.read_csv('./dataset/housing.csv')
df.dropna(inplace=True)

# -------------------------------
# Streamlit config
st.set_page_config(page_title="🏡 California Housing Explorer", layout="wide")
st.title("🏡 California Housing Explorer")
st.markdown("Exploră vizual și interactiv prețurile locuințelor din California.")

# -------------------------------
# Sidebar: filters
st.sidebar.header("🔎 Filtrare")
income_min, income_max = float(df["median_income"].min()), float(df["median_income"].max())
income_range = st.sidebar.slider("Venit median ($)", income_min, income_max, (income_min, income_max))

value_min, value_max = int(df["median_house_value"].min()), int(df["median_house_value"].max())
value_range = st.sidebar.slider("Valoare mediană locuință ($)", value_min, value_max, (value_min, value_max), step=10000)

ocean_options = df["ocean_proximity"].unique().tolist()
ocean_filter = st.sidebar.multiselect("Proximitate ocean", options=ocean_options, default=ocean_options)

# -------------------------------
# Filtered data
filtered_df = df[
    (df["median_income"] >= income_range[0]) & (df["median_income"] <= income_range[1]) &
    (df["median_house_value"] >= value_range[0]) & (df["median_house_value"] <= value_range[1]) &
    (df["ocean_proximity"].isin(ocean_filter))
]

if filtered_df.empty:
    st.warning("⚠️ Nicio locuință nu se potrivește cu filtrele actuale.")
    st.stop()

st.write(f"📊 {len(filtered_df)} locuințe găsite după filtre.")

# -------------------------------
import plotly.express as px

fig_map = px.scatter_mapbox(
    filtered_df,
    lat="latitude",
    lon="longitude",
    color="median_house_value",
    size="median_income",
    color_continuous_scale="Viridis",
    mapbox_style="carto-positron",
    zoom=5,
    height=600
)
st.plotly_chart(fig_map, use_container_width=True)


# -------------------------------
# Table (optional)
with st.expander("📋 Vezi datele filtrate"):
    st.dataframe(filtered_df.reset_index(drop=True))

# -------------------------------
# Download button
st.download_button(
    label="💾 Descarcă datele filtrate (.csv)",
    data=filtered_df.to_csv(index=False),
    file_name="california_filtrat.csv",
    mime="text/csv"
)

# -------------------------------
# Plotly visualization
st.header("📈 Vizualizare personalizată")
plot_type = st.selectbox("Alege tipul de plot", ["Scatter", "Histogramă", "Boxplot"])
col_options = ["median_income", "median_house_value", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households"]

if plot_type == "Scatter":
    x_col = st.selectbox("Axa X", options=col_options, index=0)
    y_col = st.selectbox("Axa Y", options=col_options, index=1)
    fig = px.scatter(
        filtered_df, x=x_col, y=y_col, color="ocean_proximity",
        hover_data=["median_house_value", "median_income"]
    )
elif plot_type == "Histogramă":
    hist_col = st.selectbox("Coloană", options=col_options, index=0)
    fig = px.histogram(filtered_df, x=hist_col, color="ocean_proximity")
elif plot_type == "Boxplot":
    y_box = st.selectbox("Axa Y", options=col_options, index=0)
    fig = px.box(filtered_df, x="ocean_proximity", y=y_box, points="all")

st.plotly_chart(fig, use_container_width=True)
