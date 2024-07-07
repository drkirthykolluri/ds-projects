import pandas as pd
import yfinance as yf
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "plotly_white"

#collect stock price data

#define tickers for apple and google
apple_ticker = 'AAPL'
google_ticker = 'GOOGL'

#define date range for the last quarter
start_date = '2023-07-01'
end_date = '2023-09-30'

apple_data = yf.download(apple_ticker, start = start_date, end = end_date)
google_data = yf.download(google_ticker, start = start_date, end = end_date)

#calculate daily returns
apple_data['Daily_Return'] = apple_data['Adj Close'].pct_change()
google_data['Daily_Return'] = google_data['Adj Close'].pct_change()

#create a figure to visualize daily returns
fig = go.Figure()
fig.add_trace(go.Scatter(x=apple_data.index, y=apple_data['Daily_Return'], mode='lines', name='Apple', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=google_data.index, y=google_data['Daily_Return'], mode='lines', name='Google', line=dict(color='green')))

fig.update_layout(title='Daily Returns for Apple and Google (Last Quarter)',
                  xaxis_title='Date', yaxis_title='Daily Return',
                  legend=dict(x=0.02, y=0.95))

fig.show()

#calculate cumulative product
apple_cumulative_return = (1 + apple_data['Daily_Return']).cumprod()- 1
google_cumulative_return = (1 + google_data['Daily_Return']).cumprod()- 1

#figure to visualizee cumulative returns

fig = go.Figure()
fig.add_trace(go.Scatter(x=apple_cumulative_return.index, y=apple_cumulative_return, mode='lines', name='Apple', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=apple_cumulative_return.index, y=google_cumulative_return, mode='lines', name='Google', line=dict(color='green')))
fig.update_layout(title='Cumulative Returns for Apple and Google (Last Quarter)',
                  xaxis_title='Date', yaxis_title='Cumulative Return',
                  legend=dict(x=0.02, y=0.95))
fig.show()

#calculate standard deviation of daily returns
apple_volatility = apple_data['Daily_Return'].std()
google_volatility = google_data['Daily_Return'].std()

#create a figure to compare volatility
fig1 = go.Figure()
fig1.add_bar(x=['Apple', 'Google'], y=[apple_volatility, google_volatility],
             text=[f'{apple_volatility:.4f}', f'{google_volatility:.4f}'],
             textposition='auto', marker=dict(color=['blue', 'green']))
fig1.update_layout(title='Volatility Comparison (Last Quarter)',
                   xaxis_title='Stock', yaxis_title='Volatility (Standard Deviation)',
                   bargap=0.5)
fig1.show()