import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from math import sqrt

st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")

# ---------- COLORFUL CSS ----------
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#ff9a9e,#fad0c4,#fbc2eb,#a18cd1);
background-size: 400% 400%;
font-family: 'Segoe UI', sans-serif;
}

/* Header */
.header{
background: linear-gradient(90deg,#667eea,#764ba2);
padding:30px;
border-radius:10px;
text-align:center;
color:white;
margin-bottom:20px;
}

.header h1{ 
font-size:45px;
}

.header p{
font-size:18px;
}

/* KPI Cards */

.card-container{
display:flex;
gap:20px;
justify-content:center;
margin-top:20px;
}

.card{
padding:25px;
border-radius:12px;
width:260px;
text-align:center;
color:white;
font-weight:bold;
box-shadow:0px 5px 15px rgba(0,0,0,0.2);
transition:transform 0.3s;
}

.card:hover{
transform:scale(1.07);
}

.card1{background:linear-gradient(120deg,#ff758c,#ff7eb3);}
.card2{background:linear-gradient(120deg,#43cea2,#185a9d);}
.card3{background:linear-gradient(120deg,#f7971e,#ffd200);}

/* Section Boxes */

.section{
background:white;
padding:25px;
border-radius:12px;
margin-top:20px;
box-shadow:0px 5px 15px rgba(0,0,0,0.15);
}

/* Colored Section Variants */

.section-blue{background:#e3f2fd;}
.section-green{background:#e8f5e9;}
.section-purple{background:#f3e5f5;}
.section-orange{background:#fff3e0;}

/* Table */

table{
width:100%;
border-collapse:collapse;
margin-top:15px;
}

th{
background:#6a11cb;
color:white;
padding:12px;
}

td{
padding:10px;
border-bottom:1px solid #ddd;
text-align:center;
}

/* Best model box */

.bestmodel{
background:linear-gradient(120deg,#00c6ff,#0072ff);
color:white;
padding:15px;
border-radius:10px;
font-size:18px;
text-align:center;
margin-top:20px;
}

</style>

<div class="header">
<h1>📈 Demand Forecasting Dashboard</h1>
<p>Forecast Demand using Holt-Winters, ARIMA and Prophet Models</p>
</div>

""", unsafe_allow_html=True)


# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload demand.csv", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.markdown('<div class="section section-blue"><h2>📂 Dataset Preview</h2></div>', unsafe_allow_html=True)
    st.dataframe(data.head())

    # ---------- PREPROCESSING ----------
    data['week'] = pd.to_datetime(data['week'], format='%d/%m/%y')
    data.set_index('week', inplace=True)

    weekly_data = data['units_sold'].resample('W').sum()

    # ---------- KPI METRICS ----------
    total_sales = int(weekly_data.sum())
    avg_sales = int(weekly_data.mean())
    max_sales = int(weekly_data.max())

    st.markdown(f"""
    <div class="card-container">

    <div class="card card1">
    <h3>Total Units Sold</h3>
    <h1>{total_sales}</h1>
    </div>

    <div class="card card2">
    <h3>Average Weekly Sales</h3>
    <h1>{avg_sales}</h1>
    </div>

    <div class="card card3">
    <h3>Peak Weekly Sales</h3>
    <h1>{max_sales}</h1>
    </div>

    </div>
    """, unsafe_allow_html=True)

    # ---------- SALES TREND ----------
    st.markdown('<div class="section section-green"><h2>📊 Weekly Sales Trend</h2></div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(weekly_data, color="#6a11cb", linewidth=2)
    ax.set_xlabel("Week")
    ax.set_ylabel("Units Sold")
    ax.set_title("Weekly Sales")
    st.pyplot(fig)

    # ---------- TRAIN TEST SPLIT ----------
    train_data = weekly_data[:int(0.8*len(weekly_data))]
    test_data = weekly_data[int(0.8*len(weekly_data)):]

    # ---------- HOLT WINTERS ----------
    hw_model = ExponentialSmoothing(train_data, seasonal='add', seasonal_periods=52).fit()
    hw_predictions = hw_model.predict(start=test_data.index[0], end=test_data.index[-1])
    hw_rmse = sqrt(mean_squared_error(test_data, hw_predictions))

    # ---------- ARIMA ----------
    arima_model = ARIMA(train_data, order=(1,0,0)).fit()
    arima_predictions = arima_model.predict(start=test_data.index[0], end=test_data.index[-1])
    arima_rmse = sqrt(mean_squared_error(test_data, arima_predictions))

    # ---------- PROPHET ----------
    prophet_df = weekly_data.reset_index()
    prophet_df.columns = ['ds','y']

    train_prophet = prophet_df[:int(0.8*len(prophet_df))]
    test_prophet = prophet_df[int(0.8*len(prophet_df)):]

    prophet_model = Prophet(yearly_seasonality=True)
    prophet_model.fit(train_prophet)

    future = prophet_model.make_future_dataframe(periods=len(test_prophet))
    forecast = prophet_model.predict(future)

    prophet_pred = forecast['yhat'][-len(test_prophet):]
    prophet_rmse = sqrt(mean_squared_error(test_prophet['y'], prophet_pred))

    # ---------- MODEL COMPARISON ----------
    st.markdown('<div class="section section-purple"><h2>🤖 Model Performance</h2></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <table>
    <tr>
    <th>Model</th>
    <th>RMSE</th>
    </tr>

    <tr>
    <td>Holt-Winters</td>
    <td>{hw_rmse:.2f}</td>
    </tr>

    <tr>
    <td>ARIMA</td>
    <td>{arima_rmse:.2f}</td>
    </tr>

    <tr>
    <td>Prophet</td>
    <td>{prophet_rmse:.2f}</td>
    </tr>

    </table>
    """, unsafe_allow_html=True)

    # ---------- BEST MODEL ----------
    rmse_values = [hw_rmse, arima_rmse, prophet_rmse]
    model_names = ['Holt-Winters','ARIMA','Prophet']
    best_model = model_names[rmse_values.index(min(rmse_values))]

    st.markdown(f"""
    <div class="bestmodel">
    🏆 Best Forecasting Model : <b>{best_model}</b>
    </div>
    """, unsafe_allow_html=True)

    # ---------- FORECAST GRAPH ----------
    st.markdown('<div class="section section-orange"><h2>📈 Forecast Comparison</h2></div>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(10,5))

    ax2.plot(weekly_data, label="Original", color="black")
    ax2.plot(hw_predictions, label="Holt-Winters", color="blue")
    ax2.plot(arima_predictions, label="ARIMA", color="green")
    ax2.plot(test_prophet['ds'], prophet_pred.values, label="Prophet", color="red")

    ax2.legend()
    ax2.set_xlabel("Week")
    ax2.set_ylabel("Units Sold")

    st.pyplot(fig2)

else:
    st.info("📂 Upload demand.csv to start forecasting")