import pandas as pd
import numpy as np
import moexalgo as ma
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_all_history(security, start_date, end_date, period):
    df = ma.Stock(security).candles(start=start_date, end=end_date, period=period)
    df.set_index('begin', inplace = True)
    return df

moex_tickers = ['GAZP', 'SBER', 'AFLT', 'T', 'VTBR', 'LKOH', 'PIKK', 'NVTK', 'GMKN', 'SNGSP', 'YDEX', 'PLZL', 'MAGN', 'ROSN', 'SNGS', 'AFKS', 'ALRS', 'NLMK',
                'CHMF', 'TRNFP', 'SVCB', 'RTKM', 'TATN', 'RUAL', 'BSPB', 'POSI', 'MOEX', 'VKCO', 'MTSS', 'IRAO', 'HEAD', 'UPRO', 'PHOR', 'ENPG', 'FLOT', 'CBOM',
                 'UGLD'
]

def fetch_prices(tickers, start_date, end_date, period = '1d'):

    prices = ma.Ticker('IMOEX2').candles(
        start = start_date,
        end = end_date,
        period = period
    )

    prices.set_index('begin', inplace = True)
    prices = prices[['close']]
    prices.columns = ['IMOEX']

    for ticker in tickers:
        try:
            df = get_all_history(ticker, start_date, end_date, period)
            df = df[['close']]
            df.columns = [ticker]
            prices = pd.merge(prices, df, how = 'outer', left_index=True, right_index=True)
        except Exception as e:
            print(f"Ошибка при загрузке {ticker}: {e}")
            continue
    
    prices.ffill(inplace=True)
    
    return prices

def select_top_momentum_tickers(
    prices,
    ema_short=20,
    ema_long=60,
    sma_trend=100,
    top_k=5,
    short_k=0  
):
    if len(prices) < max(ema_long, sma_trend):
        raise ValueError(f"Недостаточно данных: требуется минимум {max(ema_long, sma_trend)} дней.")

    momentum_scores = {}

    eligible_for_long = set()
    eligible_for_short = set()

    for ticker in prices.columns:
        price_series = prices[ticker].dropna()
        if len(price_series) < max(ema_long, sma_trend):
            continue
        
        ema_short_series = price_series.ewm(span=ema_short, adjust=False).mean()
        ema_long_series = price_series.ewm(span=ema_long, adjust=False).mean()
        sma_trend_series = price_series.rolling(window=sma_trend, min_periods=sma_trend).mean()
        
        ema_short_last = ema_short_series.iloc[-1]
        ema_long_last = ema_long_series.iloc[-1]
        sma_trend_last = sma_trend_series.iloc[-1]
        price_last = price_series.iloc[-1]
        
        if price_last >= sma_trend_last:
            eligible_for_long.add(ticker)
        elif price_last < sma_trend_last:
            eligible_for_short.add(ticker)
        
        smooth_momentum = (ema_short_last - ema_long_last) / price_last
        momentum_scores[ticker] = smooth_momentum

    sorted_items = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
    
    long_candidates = [ticker for ticker, _ in sorted_items if ticker in eligible_for_long]
    long_tickers = long_candidates[:top_k]
    
    short_candidates = [ticker for ticker, _ in sorted_items if ticker in eligible_for_short]
    short_tickers = short_candidates[-short_k:] if short_k > 0 else []

    return long_tickers, short_tickers

def plot_backtest_results(portfolio_value, benchmark_value, title="Моментум-ротация vs IMOEX"):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Кумулятивная доходность", "Относительное превышение (Альфа)"),
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, name="Портфель", line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=benchmark_value.index, y=benchmark_value, name="IMOEX", line=dict(color='orange')), row=1, col=1)

    alpha = portfolio_value / benchmark_value
    fig.add_trace(go.Scatter(x=alpha.index, y=alpha, name="Альфа", line=dict(color='green')), row=2, col=1)
    fig.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(title=title, height=700, showlegend=True)
    fig.update_yaxes(title_text="Множитель капитала", row=1, col=1)
    fig.update_yaxes(title_text="Альфа", row=2, col=1)
    fig.update_xaxes(title_text="Дата", row=2, col=1)

    fig.show()

def backtest_momentum_rotation(
    tickers,
    start_date, 
    end_date,
    ema_long = 60,
    ema_short = 15,
    sma_trend = 150,
    top_k=5,
    short_k=5,           
    rebalance_freq_days=10,
    benchmark_ticker='IMOEX', 
    stop_pct = 0.08
):
    if short_k is None:
        short_k = 0

    data = fetch_prices(tickers, start_date, end_date)

    position_tracker = {
        ticker: {'active': False, 'high': np.nan, 'low': np.nan, 'type': None}
        for ticker in tickers
    }

    portfolio_value = pd.Series(index=data.index, dtype=float)
    portfolio_value.iloc[0] = 1.0 
    current_weights = pd.Series(0.0, index=tickers)
    trades_log = []
    last_rebalance_idx = 0

    for i in range(1, len(data)):
        current_date = data.index[i]
        price_today = data[tickers].iloc[i]
        price_yesterday = data[tickers].iloc[i - 1]

        for ticker in tickers:
            if not position_tracker[ticker]['active']:
                continue

            pos_type = position_tracker[ticker]['type']
            if pos_type == 'long':

                current_high = max(position_tracker[ticker]['high'], price_today[ticker])
                position_tracker[ticker]['high'] = current_high
                drawdown = (current_high - price_today[ticker]) / current_high
                if drawdown >= stop_pct:
                    current_weights[ticker] = 0.0
                    position_tracker[ticker]['active'] = False

            elif pos_type == 'short':
                current_low = min(position_tracker[ticker]['low'], price_today[ticker])
                position_tracker[ticker]['low'] = current_low

                drawup = (price_today[ticker] - current_low) / current_low
                if drawup >= stop_pct:
                    current_weights[ticker] = 0.0
                    position_tracker[ticker]['active'] = False

        if (i - last_rebalance_idx) >= rebalance_freq_days:
            hist = data[tickers].iloc[:i]#?
            try:
                long_tickers, short_tickers = select_top_momentum_tickers(
                    hist, 
                        ema_short=ema_short,
                        ema_long=ema_long,
                        sma_trend=sma_trend,
                        top_k=top_k,
                        short_k=short_k, 
                )
            except ValueError:
                pass
            else:
                current_weights = pd.Series(0.0, index=tickers)

                total_long = len(long_tickers)
                total_short = len(short_tickers)
                
                if long_tickers:
                    weight_long = 0.5 / (total_long)
                    for ticker in long_tickers:
                        current_weights[ticker] = weight_long
                        position_tracker[ticker] = {
                            'active': True,
                            'high': price_today[ticker],
                            'low': np.nan,
                            'type': 'long'
                        }

                if short_tickers:
                    weight_short = -0.5 / (total_short) 
                    for ticker in short_tickers:
                        current_weights[ticker] = weight_short
                        position_tracker[ticker] = {
                            'active': True,
                            'high': np.nan,
                            'low': price_today[ticker],
                            'type': 'short'
                        }

                last_rebalance_idx = i
                trades_log.append((current_date, {'long': long_tickers.copy(), 'short': short_tickers.copy()}))

        daily_rets = price_today / price_yesterday - 1
        portfolio_return = (current_weights * daily_rets).sum()
        portfolio_value.iloc[i] = portfolio_value.iloc[i - 1] * (1 + portfolio_return)

    benchmark_price = data[benchmark_ticker]
    benchmark_value = benchmark_price / benchmark_price.iloc[0]

    if not trades_log:
        raise ValueError("Ни одной перебалансировки не произошло.")

    first_trade_date = trades_log[0][0]
    if first_trade_date not in portfolio_value.index:
        first_trade_date = portfolio_value.index[portfolio_value.index >= first_trade_date][0]

    portfolio_trimmed = portfolio_value.loc[first_trade_date:]
    benchmark_trimmed = benchmark_value.loc[first_trade_date:]

    portfolio_final = portfolio_trimmed / portfolio_trimmed.iloc[0]
    benchmark_final = benchmark_trimmed / benchmark_trimmed.iloc[0]

    return data, portfolio_final, benchmark_final, trades_log

def generate_trades_with_stops(
    prices,
    ema_short=15,
    ema_long=60,
    sma_trend=150,
    top_k=5,
    short_k=5,
    stop_pct=0.08
):
  
    long_tickers, short_tickers = select_top_momentum_tickers(
        prices,
        ema_short=ema_short,
        ema_long=ema_long,
        sma_trend=sma_trend,
        top_k=top_k,
        short_k=short_k
    )
    
    current_prices = prices.iloc[-1]
    trades = []
    
    for ticker in long_tickers:
        if pd.isna(current_prices[ticker]):
            continue
        stop_price = current_prices[ticker] * (1 - stop_pct)
        trades.append({
            'ticker': ticker,
            'direction': 'long',
            'entry_price': current_prices[ticker],
            'stop_price': stop_price
        })

    for ticker in short_tickers:
        if pd.isna(current_prices[ticker]):
            continue
        stop_price = current_prices[ticker] * (1 + stop_pct)
        trades.append({
            'ticker': ticker,
            'direction': 'short',
            'entry_price': current_prices[ticker],
            'stop_price': stop_price
        })
    
    return trades