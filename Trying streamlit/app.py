import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="3D Scatter Selection Demo", layout="wide")

st.title("Plotly 3D Scatter in Streamlit")
st.write("Select points in the chart and see their data below.")

# -----------------------------
# Generate random data
# -----------------------------
np.random.seed(42)
n = 150

df = pd.DataFrame({
    "x": np.random.normal(0, 1, n),
    "y": np.random.normal(0, 1, n),
    "z": np.random.normal(0, 1, n),
    "color_value": np.random.uniform(0, 100, n),
    "size": np.random.uniform(6, 20, n),
})

df["label"] = [f"Point {i}" for i in range(n)]
df["id"] = df.index  # useful for mapping selections back to rows

# -----------------------------
# Controls
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    point_size_scale = st.slider("Point size scale", 0.5, 3.0, 1.2, 0.1)

with col2:
    opacity = st.slider("Opacity", 0.1, 1.0, 0.85, 0.05)

# -----------------------------
# Build figure
# -----------------------------
fig = px.scatter_3d(
    df,
    x="x",
    y="y",
    z="z",
    color="color_value",          # 4th dimension = color
    size="size",
    hover_name="label",
    hover_data={"id": True, "x": ":.3f", "y": ":.3f", "z": ":.3f", "color_value": ":.2f", "size": ":.2f"},
    title="Random 3D Scatterplot",
)

fig.update_traces(
    marker=dict(opacity=opacity, sizeref=8 / point_size_scale),
    selector=dict(mode="markers")
)

fig.update_layout(
    height=700,
    margin=dict(l=0, r=0, t=50, b=0),
)

# -----------------------------
# Show chart with selection support
# -----------------------------
event = st.plotly_chart(
    fig,
    key="random_3d_scatter",
    on_select="rerun",
    selection_mode=["points"],
    use_container_width=True,
)

# -----------------------------
# Parse selected points
# -----------------------------
st.subheader("Selected data")

selected_ids = []

if event and "selection" in event and event["selection"]:
    points = event["selection"].get("points", [])
    for p in points:
        # Plotly selection event often includes point_index
        point_index = p.get("point_index")
        if point_index is not None:
            selected_ids.append(point_index)

if selected_ids:
    selected_df = df.iloc[selected_ids].copy()
    st.success(f"You selected {len(selected_df)} point(s).")
    st.dataframe(selected_df, use_container_width=True)

    st.subheader("Summary of selected points")
    st.write({
        "count": len(selected_df),
        "avg_x": round(selected_df["x"].mean(), 3),
        "avg_y": round(selected_df["y"].mean(), 3),
        "avg_z": round(selected_df["z"].mean(), 3),
        "avg_color_value": round(selected_df["color_value"].mean(), 2),
    })
else:
    st.info("No points selected yet. Click or select points in the chart.")
    st.dataframe(df.head(10), use_container_width=True)