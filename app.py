# ✅ Enhanced Car Price Prediction App with all "Extra Edge" features

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error

# Set page config
st.set_page_config(page_title="🚘 Enhanced Car Price Predictor", layout="wide")

# Dark mode styling
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #0d1117 !important;
        color: #c9d1d9 !important;
    }
    .stSidebar, .css-6qob1r, .st-bf, .st-c7 {
        background-color: #161b22 !important;
    }
    .stButton > button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5em 1em;
    }
    .stButton > button:hover {
        background-color: #2ea043;
    }
    .stNumberInput input, .stSelectbox div[role="button"] {
        background-color: #21262d;
        color: #c9d1d9;
        border: 1px solid #30363d;
    }
    .metric-box {
        background-color: #21262d;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(255,255,255,0.03);
    }
    .prediction-form {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(255,255,255,0.05);
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚘 Enhanced Car Price Prediction Dashboard")
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

if uploaded_file:
    df = load_data(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("🎯 Select Target Column")
    target_column = st.selectbox("Select your target (Price) column:", df.columns)

    if target_column:
        st.markdown("---")
        st.subheader("📊 Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Target Distribution**")
            plt.style.use('dark_background')
            sns.histplot(df[target_column], kde=True, color="#58a6ff")
            st.pyplot(plt.gcf())
            plt.clf()
        with col2:
            st.markdown("**Correlation Heatmap**")
            sns.heatmap(df.corr(), annot=False, cmap="mako")
            st.pyplot(plt.gcf())
            plt.clf()

        st.markdown("---")
        st.subheader("⚙️ Model Training & Comparison")

        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Random Forest": RandomForestRegressor(),
            "Linear Regression": LinearRegression(),
            "XGBoost": XGBRegressor(verbosity=0)
        }

        selected_model_name = st.selectbox("Choose a model to train:", list(models.keys()))
        model = models[selected_model_name]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-box"><h4>R² Score</h4><h2 style="color:#2ea043">{r2:.2f}</h2></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-box"><h4>RMSE</h4><h2 style="color:#f85149">{rmse:.2f}</h2></div>', unsafe_allow_html=True)

        st.subheader("📈 Feature Importances")
        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=X.columns)
            importances.sort_values().plot(kind='barh', color="#58a6ff")
            st.pyplot(plt.gcf())
            plt.clf()

        st.markdown("---")
        st.subheader("🧮 Predict Car Price")
        st.markdown('<div class="prediction-form">', unsafe_allow_html=True)
        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(f"{col}", value=float(df[col].median()))

        if st.button("🚀 Predict Price"):
            input_df = pd.DataFrame([input_data])
            predicted_price = model.predict(input_df)[0]
            st.success(f"💰 Estimated Price: ₹{predicted_price:,.2f}")

            st.subheader("📊 Visual Price Gauge")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted_price,
                title={'text': "Estimated Price"},
                gauge={'axis': {'range': [None, y.max()*1.1]}}
            ))
            st.plotly_chart(fig)

            pred_df = pd.DataFrame([input_data])
            pred_df['Predicted_Price'] = predicted_price
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Prediction as CSV", csv, "prediction.csv", "text/csv")

        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("📂 Please upload a CSV file to start.")