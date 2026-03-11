import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backend import fetch_hourly_data, backtest_basic_dip_strategy, buy_basic_dip_strategy_timeseries

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Stonks Explorer", page_icon="📈", layout="wide")
st.title("📈 Stonks Strategy Explorer")
st.markdown("Interact with the Plotly heatmaps to find the most profitable trading parameters.")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("Data Parameters")
ticker_input = st.sidebar.text_input("Ticker Symbol(s)", "NVDA, AAPL")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
if len(tickers) > 2:
    st.sidebar.warning("Maximum of 2 tickers allowed for side-by-side comparison. Only showing first two.")
    tickers = tickers[:2]

st.sidebar.header("Grid Search Parameters")
sell_min = st.sidebar.slider("Min Sell Multiplier", 1.01, 1.05, 1.01, 0.01)
sell_max = st.sidebar.slider("Max Sell Multiplier", 1.06, 1.20, 1.10, 0.01)

buy_min = st.sidebar.slider("Min Buy Multiplier", 0.80, 0.95, 0.90, 0.01)
buy_max = st.sidebar.slider("Max Buy Multiplier", 0.96, 0.99, 0.99, 0.01)

grid_resolution = st.sidebar.slider("Grid Resolution (N x N)", 10, 100, 50, 10)

st.sidebar.markdown("---")
st.sidebar.header("Visualization Filters")
minimum_profit = st.sidebar.slider("Minimum Profit Threshold (%)", -50.0, 100.0, 10.0, 5.0, help="Filters out points in the 3D Scatterplot below this profit margin.")

sell_multipliers = np.round(np.linspace(sell_min, sell_max, grid_resolution), decimals=3)
buy_multipliers = np.round(np.linspace(buy_min, buy_max, grid_resolution), decimals=3)

# String labels for Plotly explicitly formatted
x_labels = [f"{x:.3f}" for x in sell_multipliers]
y_labels = [f"{y:.3f}" for y in buy_multipliers]

# ==========================================
# CACHED BACKTEST RUNNER (1 to 20 Hours)
# ==========================================
@st.cache_data(show_spinner="Running 160k Backtests...")
def run_all_backtests(closes, buy_arr, sell_arr, baseline_profit):
    # Tests wait periods from 1 to 20
    wait_periods = list(range(1, 21))
    results = {}
    
    for wait in wait_periods:
        profits = backtest_basic_dip_strategy(closes, buy_arr, sell_arr, wait)
        # Convert raw profit to % difference from baseline
        results[wait] = (profits - baseline_profit) * 100 / baseline_profit

    global_min = min(arr.min() for arr in results.values())
    global_max = max(arr.max() for arr in results.values())
    return results, global_min, global_max, wait_periods

# ==========================================
# MAIN WINDOW: DUAL COLUMNS
# ==========================================
if not tickers:
    st.warning("Please enter at least one ticker.")
    st.stop()

cols = st.columns(len(tickers))

for col_idx, ticker_symbol in enumerate(tickers):
    with cols[col_idx]:
        st.header(f"Results for **{ticker_symbol}**")
        
        with st.spinner(f"Fetching data for {ticker_symbol}..."):
            df = fetch_hourly_data(ticker_symbol)

        if df.empty:
            continue

        closes_array = df["Close"].values
        first_close = closes_array[0]
        last_close = closes_array[-1]
        baseline_profit = (100 / first_close) * last_close

        st.markdown(f"**Data Rows:** {len(closes_array)} | **Baseline Buy & Hold:** ${baseline_profit:.2f} (from $100)")

        results_dict, global_min, global_max, wait_periods = run_all_backtests(closes_array, buy_multipliers, sell_multipliers, baseline_profit)

        # ==========================================
        # TABS
        # ==========================================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Single Wait", 
            "Grid Matrix", 
            "Stacked 3D", 
            "3D Scatter",
            "Strategy Drilldown"
        ])

        # --- TAB 1: Single Wait Heatmap ---
        with tab1:
            st.subheader(f"2D Heatmap")
            selected_wait = st.slider(f"Wait Period (Hours) for {ticker_symbol}", 1, 20, 5, key=f"wait_{ticker_symbol}")
            
            heatmap_data = results_dict[selected_wait]
            
            fig1 = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=x_labels,
                y=y_labels,
                colorscale=[[0, "red"], [0.5, "white"], [1, "green"]],
                zmid=0,
                colorbar=dict(title="Profit (%)")
            ))
            
            fig1.update_layout(
                title=f"Wait={selected_wait}h",
                xaxis_title="Sell Mult",
                yaxis_title="Buy Mult",
                height=600,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig1, use_container_width=True)

        # --- TAB 2: Grid Search Matrix ---
        with tab2:
            st.subheader(f"Grid Search (1-20h)")
            
            fig2 = make_subplots(
                rows=5, cols=4, 
                subplot_titles=[f"{w}h" for w in wait_periods],
                shared_xaxes=True,
                shared_yaxes=True,
                vertical_spacing=0.03,
                horizontal_spacing=0.03
            )
            
            for w_idx, wait in enumerate(wait_periods):
                row = (w_idx // 4) + 1
                col = (w_idx % 4) + 1
                
                fig2.add_trace(go.Heatmap(
                    z=results_dict[wait],
                    x=x_labels,
                    y=y_labels,
                    colorscale=[[0, "red"], [0.5, "white"], [1, "green"]],
                    zmin=global_min if global_min < 0 else -1,
                    zmax=global_max if global_max > 0 else 1,
                    zmid=0,
                    coloraxis="coloraxis"
                ), row=row, col=col)

            fig2.update_layout(
                coloraxis=dict(
                    colorscale=[[0, "red"], [0.5, "white"], [1, "green"]],
                    cmin=global_min,
                    cmax=global_max,
                    cmid=0,
                    colorbar=dict(title="Profit (%)", len=0.8) # scale down colorbar slightly for columns
                ),
                height=1200,
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig2, use_container_width=True)

        # --- TAB 3: Stacked 3D Surface ---
        with tab3:
            st.subheader("3D Stacked Surfaces")
            
            fig3 = go.Figure()
            
            for w_idx, wait in enumerate(wait_periods):
                profits_matrix = results_dict[wait]
                Z_surface = np.full_like(profits_matrix, wait)
                
                fig3.add_trace(go.Surface(
                    z=Z_surface,
                    x=x_labels,
                    y=y_labels,
                    surfacecolor=profits_matrix,
                    colorscale=[[0, "red"], [0.5, "white"], [1, "green"]],
                    cmin=global_min,
                    cmax=global_max,
                    cmid=0,
                    opacity=0.8,
                    name=f"Wait {wait}",
                    showscale=(w_idx == 0)
                ))

            fig3.update_layout(
                scene=dict(
                    xaxis_title='Sell',
                    yaxis_title='Buy',
                    zaxis_title='Wait',
                    zaxis=dict(dtick=2)
                ),
                height=700,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig3, use_container_width=True)

        # --- TAB 4: 3D Scatterplot ---
        with tab4:
            st.subheader(f"3D Scatterplot (Min Profit: {minimum_profit}%)")
            
            X, Y, Z, C = [], [], [], []
            
            Sell_mesh, Buy_mesh = np.meshgrid(sell_multipliers, buy_multipliers)
            
            for wait in wait_periods:
                profits_matrix = results_dict[wait]
                
                X.append(Sell_mesh.flatten())
                Y.append(Buy_mesh.flatten())
                Z.append(np.full(profits_matrix.size, wait))
                C.append(profits_matrix.flatten())

            X = np.concatenate(X)
            Y = np.concatenate(Y)
            Z = np.concatenate(Z)
            C = np.concatenate(C)
            
            # Apply the Profit Threshold Filter
            mask = C >= minimum_profit
            X_plot, Y_plot, Z_plot, C_plot = X[mask], Y[mask], Z[mask], C[mask]

            fig4 = go.Figure()
            if len(C_plot) > 0:
                fig4.add_trace(go.Scatter3d(
                    x=X_plot,
                    y=Y_plot,
                    z=Z_plot,
                    mode='markers',
                    marker=dict(
                        size=6, # INCREASED SIZE HERE
                        color=C_plot,
                        colorscale=[[0, "red"], [0.5, "white"], [1, "green"]],
                        cmin=global_min,
                        cmax=global_max,
                        cmid=0,
                        opacity=0.8,
                        colorbar=dict(title="Profit (%)")
                    ),
                    text=[f"Profit: {c:.2f}%<br>Buy: {y:.3f}<br>Sell: {x:.3f}<br>Wait: {z}" \
                          for x, y, z, c in zip(X_plot, Y_plot, Z_plot, C_plot)],
                    hoverinfo='text'
                ))
            else:
                st.warning(f"No configurations produced a profit >= {minimum_profit}% for {ticker_symbol}.")
            
            fig4.update_layout(
                scene=dict(
                    xaxis_title='Sell',
                    yaxis_title='Buy',
                    zaxis_title='Wait'
                ),
                height=800,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            
            st.plotly_chart(fig4, use_container_width=True, key=f"scatter_{ticker_symbol}")

        # --- TAB 5: Strategy Drilldown ---
        with tab5:
            st.subheader("Strategy Drilldown")
            st.markdown("**(Plotly 3D objects lack click interactivity). Use the inputs below to drill down into a specific strategy's timeline!**")
            
            col_buy, col_sell, col_wait = st.columns(3)
            with col_buy:
                buy_target = st.number_input("Buy Multiplier", min_value=0.01, max_value=2.00, value=0.90, step=0.005, format="%f", key=f"buy_in_{ticker_symbol}")
            with col_sell:
                sell_target = st.number_input("Sell Multiplier", min_value=0.01, max_value=5.00, value=1.05, step=0.01, format="%f", key=f"sell_in_{ticker_symbol}")
            with col_wait:
                wait_target = st.number_input("Wait Period (Hours)", min_value=1, max_value=100, value=5, step=1, key=f"wait_in_{ticker_symbol}")
            
            history = buy_basic_dip_strategy_timeseries(closes_array, buy_target, sell_target, wait_target)
            baseline_history = (100 / closes_array[0]) * closes_array
            
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(y=baseline_history, mode='lines', name="Baseline (Buy & Hold)", line=dict(color='gray', dash='dash', width=2)))
            fig_ts.add_trace(go.Scatter(y=history, mode='lines', name="Strategy Performance", line=dict(color='#00CC96', width=3)))
            
            fig_ts.update_layout(
                title=f"Portfolio Value over Time - {ticker_symbol} (Starting Capital: $100)",
                xaxis_title="Hours Passed",
                yaxis_title="Account Value ($)",
                height=500,
                margin=dict(l=0, r=0, t=40, b=0),
                hovermode="x unified"
            )
            st.plotly_chart(fig_ts, use_container_width=True, key=f"ts_{ticker_symbol}")
