import os
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import lru_cache

# Import from local backend
from backend import fetch_hourly_data, run_all_backtests, buy_basic_dip_strategy_timeseries

# -----------------------------------------------------
# CACHING BACKTEST RESULTS (Server-side)
# -----------------------------------------------------
@lru_cache(maxsize=32)
def get_backtest_results(ticker_symbol, buy_min, buy_max, sell_min, sell_max, grid_resolution):
    df = fetch_hourly_data(ticker_symbol)
    if df.empty:
        return None
    
    closes_array = df["Close"].values
    baseline_profit = (100 / closes_array[0]) * closes_array[-1]

    sell_multipliers = np.round(np.linspace(sell_min, sell_max, grid_resolution), decimals=3)
    buy_multipliers = np.round(np.linspace(buy_min, buy_max, grid_resolution), decimals=3)
    
    results_dict, global_min, global_max, wait_periods = run_all_backtests(
        closes_array, buy_multipliers, sell_multipliers, baseline_profit
    )
    return {
        "results_dict": results_dict,
        "global_min": global_min,
        "global_max": global_max,
        "wait_periods": wait_periods,
        "sell_multipliers": sell_multipliers,
        "buy_multipliers": buy_multipliers,
        "closes_array": closes_array,
        "baseline_profit": baseline_profit
    }

# -----------------------------------------------------
# APP CONFIG
# -----------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Stonks Explorer"

# -----------------------------------------------------
# LAYOUT
# -----------------------------------------------------
sidebar = dbc.Card(
    [
        html.H4("Stonks Explorer", className="display-6"),
        html.Hr(),
        html.P("Interact with the Plotly heatmaps to find profitable strategies.", className="lead"),
        
        dbc.Label("Ticker Symbol"),
        dbc.Input(id="ticker-input", value="NVDA", type="text", debounce=True),
        html.Br(),
        
        html.H5("Grid Search Parameters"),
        dbc.Label("Min Sell Multiplier"),
        dcc.Slider(1.01, 1.05, 0.01, value=1.01, id="sell-min", marks={1.01: "1.01", 1.05: "1.05"}),
        
        dbc.Label("Max Sell Multiplier"),
        dcc.Slider(1.06, 1.20, 0.01, value=1.10, id="sell-max", marks={1.06: "1.06", 1.20: "1.20"}),
        
        dbc.Label("Min Buy Multiplier"),
        dcc.Slider(0.80, 0.95, 0.01, value=0.90, id="buy-min", marks={0.80: "0.80", 0.95: "0.95"}),
        
        dbc.Label("Max Buy Multiplier"),
        dcc.Slider(0.96, 0.99, 0.01, value=0.99, id="buy-max", marks={0.96: "0.96", 0.99: "0.99"}),
        
        dbc.Label("Grid Resolution (N x N)"),
        dcc.Slider(10, 100, 10, value=50, id="grid-res", marks={10: "10", 100: "100"}),
        
        html.Hr(),
        html.H5("Visualization Filters"),
        dbc.Label("Minimum Profit Threshold (%) (For 3D Scatter)"),
        dcc.Slider(-50.0, 100.0, 5.0, value=10.0, id="min-profit", marks={-50: "-50", 0: "0", 100: "100"}),
        
        html.Br(),
        dbc.Button("Run Backtests", id="run-btn", color="primary", className="w-100"),
        html.Br(),
        html.Div(id="status-msg", className="mt-3 text-warning")
    ],
    body=True,
    className="h-100",
)

content = dbc.Tabs(
    [
        dbc.Tab(
            [
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Wait Period (Hours)"),
                        dcc.Slider(1, 20, 1, value=5, id="wait-slider-2d", marks={i: str(i) for i in range(1, 21)})
                    ])
                ]),
                dcc.Graph(id="fig-2d", style={"height": "75vh"})
            ], 
            label="Single Wait", tab_id="tab-1"
        ),
        dbc.Tab(
            [
                html.Br(),
                dcc.Graph(id="fig-grid", style={"height": "85vh"})
            ],
            label="Grid Matrix", tab_id="tab-2"
        ),
        dbc.Tab(
            [
                html.Br(),
                dcc.Graph(id="fig-stacked", style={"height": "85vh"})
            ],
            label="Stacked 3D", tab_id="tab-3"
        ),
        dbc.Tab(
            [
                html.Br(),
                html.P("Click on a point to view its Strategy Drilldown!"),
                dcc.Graph(id="fig-scatter", style={"height": "80vh"})
            ],
            label="3D Scatter", tab_id="tab-4"
        ),
        dbc.Tab(
            [
                html.Br(),
                dbc.Row([
                    dbc.Col([dbc.Label("Buy Multiplier"), dbc.Input(id="drill-buy", type="number", value=0.90, step=0.005)]),
                    dbc.Col([dbc.Label("Sell Multiplier"), dbc.Input(id="drill-sell", type="number", value=1.05, step=0.01)]),
                    dbc.Col([dbc.Label("Wait Period"), dbc.Input(id="drill-wait", type="number", value=5, step=1)]),
                    dbc.Col([
                        html.Br(),
                        dcc.Checklist(
                            id="toggle-shade",
                            options=[{"label": " Show Trade Overlays", "value": "show"}],
                            value=["show"],
                            inline=True,
                            className="mt-2"
                        )
                    ])
                ]),
                html.Br(),
                dcc.Graph(id="fig-drill", style={"height": "65vh"})
            ],
            label="Strategy Drilldown", tab_id="tab-5"
        )
    ],
    active_tab="tab-1",
    id="tabs"
)

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, width=3),
                dbc.Col(
                    [
                        dbc.Row([
                            dbc.Col(html.H2(id="header-title", children="Results for NVDA"), width=8),
                            dbc.Col(html.Div(id="baseline-metrics", className="text-end mt-2"), width=4)
                        ]),
                        content
                    ], 
                    width=9,
                    className="p-3"
                ),
            ],
            className="h-100"
        )
    ],
    fluid=True,
    style={"height": "100vh"}
)

# -----------------------------------------------------
# CALLBACKS
# -----------------------------------------------------

@app.callback(
    [
        Output("status-msg", "children"),
        Output("header-title", "children"),
        Output("baseline-metrics", "children"),
        Output("fig-2d", "figure"),
        Output("fig-grid", "figure"),
        Output("fig-stacked", "figure"),
    ],
    [Input("run-btn", "n_clicks"), Input("wait-slider-2d", "value")],
    [
        State("ticker-input", "value"),
        State("sell-min", "value"),
        State("sell-max", "value"),
        State("buy-min", "value"),
        State("buy-max", "value"),
        State("grid-res", "value")
    ]
)
def update_main_figures(n_clicks, wait_2d, ticker, sell_min, sell_max, buy_min, buy_max, grid_res):
    ticker = ticker.strip().upper()
    if not ticker:
        return "Please enter a ticker", "", "", go.Figure(), go.Figure(), go.Figure()

    res = get_backtest_results(ticker, buy_min, buy_max, sell_min, sell_max, grid_res)
    if not res:
        return f"No data for {ticker}", f"No data", "", go.Figure(), go.Figure(), go.Figure()

    results_dict = res["results_dict"]
    sell_multipliers = res["sell_multipliers"]
    buy_multipliers = res["buy_multipliers"]
    global_min = res["global_min"]
    global_max = res["global_max"]
    wait_periods = res["wait_periods"]
    baseline = res["baseline_profit"]
    rows_len = len(res["closes_array"])

    x_labels = [f"{x:.3f}" for x in sell_multipliers]
    y_labels = [f"{y:.3f}" for y in buy_multipliers]

    # --- 2D Heatmap ---
    heatmap_data = results_dict.get(wait_2d, np.zeros((grid_res, grid_res)))
    fig1 = go.Figure(data=go.Heatmap(
        z=heatmap_data, x=x_labels, y=y_labels,
        colorscale=[[0, "red"], [0.5, "white"], [1, "green"]],
        zmid=0, colorbar=dict(title="Profit (%)")
    ))
    fig1.update_layout(title=f"Wait={wait_2d}h", xaxis_title="Sell Mult", yaxis_title="Buy Mult", margin=dict(l=0, r=0, t=40, b=0), template="plotly_dark")

    # --- Grid Matrix ---
    fig2 = make_subplots(
        rows=5, cols=4, subplot_titles=[f"{w}h" for w in wait_periods],
        shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.03, horizontal_spacing=0.03
    )
    for w_idx, wait in enumerate(wait_periods):
        row = (w_idx // 4) + 1
        col = (w_idx % 4) + 1
        fig2.add_trace(go.Heatmap(
            z=results_dict[wait], x=x_labels, y=y_labels,
            colorscale=[[0, "red"], [0.5, "white"], [1, "green"]],
            zmin=global_min if global_min < 0 else -1,
            zmax=global_max if global_max > 0 else 1,
            zmid=0, coloraxis="coloraxis"
        ), row=row, col=col)
    fig2.update_layout(
        coloraxis=dict(colorscale=[[0, "red"], [0.5, "white"], [1, "green"]], cmin=global_min, cmax=global_max, cmid=0, colorbar=dict(title="Profit (%)", len=0.8)),
        margin=dict(l=0, r=0, t=30, b=0), template="plotly_dark"
    )

    # --- Stacked 3D ---
    fig3 = go.Figure()
    for w_idx, wait in enumerate(wait_periods):
        prof = results_dict[wait]
        Z_surface = np.full_like(prof, wait)
        fig3.add_trace(go.Surface(
            z=Z_surface, x=x_labels, y=y_labels, surfacecolor=prof,
            colorscale=[[0, "red"], [0.5, "white"], [1, "green"]],
            cmin=global_min, cmax=global_max, cmid=0, opacity=0.8,
            name=f"Wait {wait}", showscale=(w_idx == 0)
        ))
    fig3.update_layout(scene=dict(xaxis_title='Sell', yaxis_title='Buy', zaxis_title='Wait', zaxis=dict(dtick=2)), margin=dict(l=0, r=0, t=0, b=0), template="plotly_dark")

    metrics_text = f"Rows: {rows_len} | Baseline: ${baseline:.2f}"
    return "Loaded Successfully!", f"Results for {ticker}", metrics_text, fig1, fig2, fig3


@app.callback(
    Output("fig-scatter", "figure"),
    [Input("run-btn", "n_clicks"), Input("min-profit", "value")],
    [
        State("ticker-input", "value"),
        State("sell-min", "value"),
        State("sell-max", "value"),
        State("buy-min", "value"),
        State("buy-max", "value"),
        State("grid-res", "value")
    ]
)
def update_scatter(n_clicks, min_profit, ticker, sell_min, sell_max, buy_min, buy_max, grid_res):
    ticker = ticker.strip().upper()
    if not ticker:
        return go.Figure()

    res = get_backtest_results(ticker, buy_min, buy_max, sell_min, sell_max, grid_res)
    if not res:
        return go.Figure()

    results_dict = res["results_dict"]
    sell_multipliers = res["sell_multipliers"]
    buy_multipliers = res["buy_multipliers"]
    wait_periods = res["wait_periods"]
    global_min = res["global_min"]
    global_max = res["global_max"]

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
    
    mask = C >= min_profit
    X_plot, Y_plot, Z_plot, C_plot = X[mask], Y[mask], Z[mask], C[mask]

    fig4 = go.Figure()
    if len(C_plot) > 0:
        fig4.add_trace(go.Scatter3d(
            x=X_plot, y=Y_plot, z=Z_plot,
            mode='markers',
            marker=dict(
                size=6, color=C_plot,
                colorscale=[[0, "red"], [0.5, "white"], [1, "green"]],
                cmin=global_min, cmax=global_max, cmid=0,
                opacity=0.8, colorbar=dict(title="Profit (%)")
            ),
            text=[f"Profit: {c:.2f}%<br>Buy: {y:.3f}<br>Sell: {x:.3f}<br>Wait: {z}" for x, y, z, c in zip(X_plot, Y_plot, Z_plot, C_plot)],
            hoverinfo='text'
        ))
    
    fig4.update_layout(scene=dict(xaxis_title='Sell', yaxis_title='Buy', zaxis_title='Wait'), margin=dict(l=0, r=0, t=0, b=0), template="plotly_dark")
    return fig4


@app.callback(
    [Output("drill-buy", "value"),
     Output("drill-sell", "value"),
     Output("drill-wait", "value")],
    [Input("fig-scatter", "clickData")],
    prevent_initial_call=True
)
def update_drilldown_from_scatter(clickData):
    if clickData and "points" in clickData:
        point = clickData["points"][0]
        # x is sell, y is buy, z is wait
        return point["y"], point["x"], point["z"]
    return dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output("fig-drill", "figure"),
    [Input("drill-buy", "value"),
     Input("drill-sell", "value"),
     Input("drill-wait", "value"),
     Input("ticker-input", "value"),
     Input("toggle-shade", "value")],
    [
        State("sell-min", "value"),
        State("sell-max", "value"),
        State("buy-min", "value"),
        State("buy-max", "value"),
        State("grid-res", "value")    
    ]
)
def update_drilldown_fig(buy_val, sell_val, wait_val, ticker, show_shade, s_min, s_max, b_min, b_max, res_val):
    ticker = ticker.strip().upper()
    if not ticker or buy_val is None or sell_val is None or wait_val is None:
        return go.Figure()

    res = get_backtest_results(ticker, b_min, b_max, s_min, s_max, res_val)
    if not res:
        return go.Figure()
    
    closes_array = res["closes_array"]
    history, buy_indices, sell_indices = buy_basic_dip_strategy_timeseries(closes_array, buy_val, sell_val, int(wait_val))
    baseline_history = (100 / closes_array[0]) * closes_array
    
    fig_ts = go.Figure()
    
    # 1. Baseline - Solid Blue
    fig_ts.add_trace(go.Scatter(
        y=baseline_history, 
        mode='lines', 
        name="Baseline (Buy & Hold)", 
        line=dict(color='#007BFF', dash='solid', width=2)
    ))
    
    # 2. Strategy Performance Line
    fig_ts.add_trace(go.Scatter(
        y=history, 
        mode='lines', 
        name=f"Strategy B:{buy_val} S:{sell_val} W:{wait_val}", 
        line=dict(color='#00CC96', width=2)
    ))
    
    # 3. Buy Markers
    if len(buy_indices) > 0:
        fig_ts.add_trace(go.Scatter(
            x=buy_indices,
            y=history[buy_indices],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='lime', size=8, symbol='triangle-up'),
            hoverinfo='skip'
        ))

    # 4. Sell Markers
    if len(sell_indices) > 0:
        fig_ts.add_trace(go.Scatter(
            x=sell_indices,
            y=history[sell_indices],
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red', size=8, symbol='triangle-down'),
            hoverinfo='skip'
        ))

    shapes = []
    # 5. Trade Shading Overlay
    if show_shade and len(show_shade) > 0:
        # Shade Invested Periods (Buy to Sell)
        for i in range(len(buy_indices)):
            buy_idx = buy_indices[i]
            
            if i < len(sell_indices):
                sell_idx = sell_indices[i]
            else:
                sell_idx = len(history) - 1
            
            # Avoid division by zero
            if history[buy_idx] > 0 and baseline_history[buy_idx] > 0:
                strat_ret = (history[sell_idx] - history[buy_idx]) / history[buy_idx]
                base_ret = (baseline_history[sell_idx] - baseline_history[buy_idx]) / baseline_history[buy_idx]
                
                if strat_ret >= base_ret:
                    shade_color = "rgba(0, 255, 0, 0.15)" # Green
                else:
                    shade_color = "rgba(255, 0, 0, 0.15)" # Red

                shapes.append(dict(
                    type="rect",
                    xref="x", yref="paper",
                    x0=buy_idx, y0=0,
                    x1=sell_idx, y1=1,
                    fillcolor=shade_color,
                    layer="below",
                    line_width=0
                ))
        
        # Shade Uninvested Periods (Sell to Next Buy)
        # The portfolio value stays flat (in cash) during these periods, but baseline keeps moving
        for i in range(len(sell_indices)):
            sell_idx = sell_indices[i]
            
            if i + 1 < len(buy_indices):
                next_buy_idx = buy_indices[i + 1]
            else:
                next_buy_idx = len(history) - 1

            if history[sell_idx] > 0 and baseline_history[sell_idx] > 0:
                strat_ret = (history[next_buy_idx] - history[sell_idx]) / history[sell_idx]
                base_ret = (baseline_history[next_buy_idx] - baseline_history[sell_idx]) / baseline_history[sell_idx]

                if strat_ret >= base_ret:
                    # Even uninvested, if strategy beats baseline (baseline dropped), it's green
                    shade_color = "rgba(0, 255, 0, 0.15)" 
                else:
                    shade_color = "rgba(255, 0, 0, 0.15)"

                shapes.append(dict(
                    type="rect",
                    xref="x", yref="paper",
                    x0=sell_idx, y0=0,
                    x1=next_buy_idx, y1=1,
                    fillcolor=shade_color,
                    layer="below",
                    line_width=0
                ))

    fig_ts.update_layout(
        title=f"Portfolio Value over Time - {ticker} (Starting Capital: $100)",
        xaxis_title="Hours Passed",
        yaxis_title="Account Value ($)",
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
        template="plotly_dark",
        shapes=shapes
    )
    return fig_ts

# -----------------------------------------------------
# RUN SERVER
# -----------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)
