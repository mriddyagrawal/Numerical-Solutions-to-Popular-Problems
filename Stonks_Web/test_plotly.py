import streamlit as st
import plotly.graph_objects as go

st.title("Test Plotly Select")
fig = go.Figure(data=[go.Scatter3d(x=[1, 2], y=[3, 4], z=[5, 6], mode='markers')])
event = st.plotly_chart(fig, on_select="rerun")
st.write(event)
