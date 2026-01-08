import streamlit as st
import pandas as pd
import numpy as np
import moexalgo as ma
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from func import get_all_history, moex_tickers, fetch_prices, select_top_momentum_tickers, backtest_momentum_rotation, plot_backtest_results, generate_trades_with_stops


st.title("Приложение для отбора акций для Лонг-Шорт стратегии")
st.write("С помощью данного приложения вы можете настроить отбор акицй для одновременного открытия Лонг и Шорт позиций, чтобы зарабатывать на разнице в динамике изменения цен на основе соотношения экспоненциальных скользящих средних")

with st.form("Параметры отбора"):
    period = st.text_input("Период анализа ('1h', '1d', '1w')", value="1d")
    if period not in ['1h', '1d', '1w']:
        st.error("Неверный период анализа. Пожалуйста, выберите из '1h', '1d', '1w'.")
        
    col1 , col2 = st.columns(2)
    start_date = col1.date_input("Дата начала анализа", value=datetime.today()-timedelta(365))
    end_date = col2.date_input("Дата окончания анализа", value=datetime.today())
    st.warning("Для корректной работы приложения рекомендуется выбирать период не менее 12 месяцев")

    rebalance_freq_days = st.number_input("Частота ребалансировки (в днях)", min_value=1, value=10)
    ema_short = st.number_input("Период короткой EMA", min_value=1, value=20)
    ema_long = st.number_input("Период длинной EMA", min_value=1, value=60)
    sma_trend = st.number_input("Период SMA для тренда", min_value=1, value=150)
    top_k = st.number_input("Количество топ акций для Лонга", min_value=1, value=5)
    short_k = st.number_input("Количество топ акций для Шорта", min_value=1, value=5)
    stop_pct = st.number_input("Cтоп-лосс на позицию, %", min_value=0, value=8)/100
    run_backtest = st.checkbox("Запустить бэктест стратегии?")
    submit_button = st.form_submit_button(label="Запустить отбор акций")
    
@st.cache_data(ttl=3600)
def load_moex_data(moex_tickers, start_date, end_date, period):
    return fetch_prices(moex_tickers, start_date, end_date, period)

if submit_button:
    with st.spinner("Загружаем данные с MOEX..."):
        df_prices = load_moex_data(moex_tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), period)
    
    st.success("Данные загружены!")

    with st.spinner("Отбор акций..."):
        selected_tickers = generate_trades_with_stops(
            df_prices,
            ema_short=ema_short,
            ema_long=ema_long,
            sma_trend=sma_trend,
            top_k=top_k,
            short_k=short_k,
            stop_pct=stop_pct
        )

    if not selected_tickers:
        st.warning("Не удалось сформировать список акций для торговли. Попробуйте изменить параметры.")
    else:
        st.subheader("Рекомендуемые позиции")
        
        trades_df = pd.DataFrame(selected_tickers)
        trades_df['entry_price'] = trades_df['entry_price'].round(2)
        trades_df['stop_price'] = trades_df['stop_price'].round(2)

        trades_df['sort_key'] = trades_df['direction'].map({'long': 0, 'short': 1})
        trades_df = trades_df.sort_values(['sort_key', 'ticker']).drop(columns='sort_key').reset_index(drop=True)
        
        st.dataframe(trades_df.style.format({
            'entry_price': "{:.2f}",
            'stop_price': "{:.2f}"
        }), use_container_width=True)

        st.subheader("Графики сигналов для выбранных акций")
        
        all_selected = trades_df['ticker'].unique()
        for ticker in all_selected:
            if ticker not in df_prices.columns:
                continue
                
            price_series = df_prices[ticker].dropna()
            if len(price_series) < max(ema_long, sma_trend):
                continue

            ema_short_series = price_series.ewm(span=ema_short, adjust=False).mean()
            ema_long_series = price_series.ewm(span=ema_long, adjust=False).mean()
            sma_trend_series = price_series.rolling(window=sma_trend, min_periods=sma_trend).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=price_series.index, y=price_series, name='Цена', line=dict(color='black')))
            fig.add_trace(go.Scatter(x=ema_short_series.index, y=ema_short_series, name=f'EMA {ema_short}', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=ema_long_series.index, y=ema_long_series, name=f'EMA {ema_long}', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=sma_trend_series.index, y=sma_trend_series, name=f'SMA {sma_trend}', line=dict(color='green', dash='dash')))
            
            direction = trades_df[trades_df['ticker'] == ticker]['direction'].iloc[0]
            entry_price = trades_df[trades_df['ticker'] == ticker]['entry_price'].iloc[0]
            stop_price = trades_df[trades_df['ticker'] == ticker]['stop_price'].iloc[0]
            
            fig.add_hline(y=entry_price, line_dash="dot", line_color="gray", annotation_text="Вход")
            fig.add_hline(y=stop_price, line_dash="dash", line_color="orange", annotation_text="Стоп")
            
            fig.update_layout(
                title=f"{ticker} — {direction.upper()}",
                xaxis_title="Дата",
                yaxis_title="Цена",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        if run_backtest:
            with st.spinner("Выполняется бэктест..."):
                try:
                    _, portfolio_val, benchmark_val, _ = backtest_momentum_rotation(
                        tickers=moex_tickers,
                        start_date=start_date - timedelta(365),
                        end_date=end_date,
                        ema_short=ema_short,
                        ema_long=ema_long,
                        sma_trend=sma_trend,
                        top_k=top_k,
                        short_k=short_k,
                        rebalance_freq_days=rebalance_freq_days,
                        benchmark_ticker='IMOEX',
                        stop_pct=stop_pct
                    )

                    fig_bt = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=("Кумулятивная доходность", "Альфа (Портфель / IMOEX)"),
                        row_heights=[0.7, 0.3]
                    )

                    fig_bt.add_trace(go.Scatter(x=portfolio_val.index, y=portfolio_val, name="Портфель", line=dict(color='blue')), row=1, col=1)
                    fig_bt.add_trace(go.Scatter(x=benchmark_val.index, y=benchmark_val, name="IMOEX", line=dict(color='orange')), row=1, col=1)

                    alpha = portfolio_val / benchmark_val
                    fig_bt.add_trace(go.Scatter(x=alpha.index, y=alpha, name="Альфа", line=dict(color='green')), row=2, col=1)
                    fig_bt.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1)

                    fig_bt.update_layout(title="Результаты бэктеста", height=600, showlegend=True)
                    fig_bt.update_yaxes(title_text="Множитель капитала", row=1, col=1)
                    fig_bt.update_yaxes(title_text="Альфа", row=2, col=1)
                    fig_bt.update_xaxes(title_text="Дата", row=2, col=1)

                    st.plotly_chart(fig_bt, use_container_width=True)

                    total_return_port = portfolio_val.iloc[-1] - 1
                    total_return_bench = benchmark_val.iloc[-1] - 1
                    alpha_final = alpha.iloc[-1] - 1

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Доходность портфеля", f"{total_return_port:.2%}")
                    col2.metric("Доходность IMOEX", f"{total_return_bench:.2%}")
                    col3.metric("Альфа", f"{alpha_final:.2%}")

                except Exception as e:
                    st.error(f"Ошибка при бэктесте: {e}")
        
