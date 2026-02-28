import plotly.graph_objects as go

def plot_strategy(df):

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Market"
    ))

    # EMA lines
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["EMA_20"],
        mode='lines',
        name='EMA 20'
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["EMA_50"],
        mode='lines',
        name='EMA 50'
    ))

    # Buy markers
    buys = df[df["prediction"] == 1]
    fig.add_trace(go.Scatter(
        x=buys.index,
        y=buys["Close"],
        mode="markers",
        marker=dict(symbol="triangle-up", size=10),
        name="BUY"
    ))

    # Sell markers
    sells = df[df["prediction"] == -1]
    fig.add_trace(go.Scatter(
        x=sells.index,
        y=sells["Close"],
        mode="markers",
        marker=dict(symbol="triangle-down", size=10),
        name="SELL"
    ))

    fig.update_layout(title="ML Trading Strategy - Buy/Sell Signals")

    fig.show()
