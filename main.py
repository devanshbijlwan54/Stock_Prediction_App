
import streamlit as st 
from datetime import date
import pandas as pd
import yfinance as yf

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.subheader("Data")
start_date = st.date_input(
    "Select start date",
    value=date(2020, 1, 1),
    min_value=date(1990, 1, 1),
    max_value=date.today()
)

end_date = st.date_input(
    "Select end date",
    value=date.today(),
    min_value=start_date,
    max_value=date.today()
)
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL","GOOG","MSFT","GME")
selected_stock = st.text_input(
    "Enter stock ticker symbol (e.g. AAPL, TSLA, INFY, RELIANCE.NS)",
    value="AAPL"
).upper().strip()
n_months = st.slider("Years of Prediction", 1 , 4)
period = n_months * 12

@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start, end)

    # Flatten columns if MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])

    return data

data_load_state =st.text("Load data...")
data = load_data(selected_stock, start_date, end_date)
data_load_state.text("Loading data...done!")


st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

#days to analyze
summary_days = st.number_input(
    "How many future days for Buy/Sell summary?",
    min_value=1,
    max_value=60,
    value=7
)

# Forecasting 
# ---------- Forecasting ----------
df_train = data.loc[:, ['Date', 'Close']].copy()

df_train.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# FORCE 1-D Series
df_train['y'] = df_train['y'].astype(float)
df_train['ds'] = pd.to_datetime(df_train['ds'])

df_train = df_train.dropna()

# 🔍 FINAL SAFETY CHECK
assert isinstance(df_train['y'], pd.Series)
assert df_train['y'].ndim == 1

#temporary debug

st.write(df_train['y'].shape)
st.write(df_train.head())

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

#buy/sell signals
# ---------- Buy / Sell Summary ----------
summary_df = forecast[['ds', 'yhat']].copy()

# Keep only future dates
summary_df = summary_df[summary_df['ds'] > df_train['ds'].max()]
summary_df = summary_df.head(summary_days)

# Calculate daily change
summary_df['change'] = summary_df['yhat'].diff()

# Define thresholds (noise filter)
threshold = summary_df['yhat'].mean() * 0.002  # 0.2%

def signal(row):
    if row['change'] > threshold:
        return "BUY 📈"
    elif row['change'] < -threshold:
        return "SELL 📉"
    else:
        return "HOLD ➖"

summary_df['Signal'] = summary_df.apply(signal, axis=1)

summary_df.dropna(inplace=True)

st.subheader('Forecast data')
st.write(forecast.tail())
st.subheader("Forecast chart")

fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Forecast components")

fig2 = m.plot_components(forecast)
st.pyplot(fig2)

#disply summary table
st.subheader("Buy / Sell Prediction Summary")
st.dataframe(
    summary_df[['ds', 'yhat', 'Signal']]
    .rename(columns={
        'ds': 'Date',
        'yhat': 'Predicted Price'
    }),
    use_container_width=True
)

#recommendation
buy_days = summary_df[summary_df['Signal'].str.contains("BUY")]
sell_days = summary_df[summary_df['Signal'].str.contains("SELL")]

st.subheader("Model Insight")

if len(buy_days) > len(sell_days):
    st.success(
        f"📈 Model indicates a bullish trend for the next {summary_days} days.\n"
        "More BUY signals than SELL signals."
    )
elif len(sell_days) > len(buy_days):
    st.warning(
        f"📉 Model indicates a bearish trend for the next {summary_days} days.\n"
        "More SELL signals than BUY signals."
    )
else:
    st.info(
        f"➖ Model predicts a sideways market for the next {summary_days} days."
    )