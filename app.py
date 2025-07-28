import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense


import streamlit as st
import hashlib
import os
import json

# ---------- USER AUTH SETUP ----------
USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_login(username, password):
    users = load_users()
    return username in users and users[username] == hash_password(password)

def signup_user(username, password):
    users = load_users()
    users[username] = hash_password(password)
    save_users(users)

def change_password(username, new_password):
    users = load_users()
    users[username] = hash_password(new_password)
    save_users(users)

# ---------- APP STATE ----------
if "page" not in st.session_state:
    st.session_state.page = "signup"
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ---------- NAVIGATION MENU ----------
if st.session_state.page == "dashboard":
    with st.sidebar:
        st.markdown(f"👤 **{st.session_state.username}**")
        if st.button("🔑 Change Password"):
            st.session_state.page = "change_pass"
        if st.button("🚪 Logout"):
            st.session_state.page = "login"
            st.session_state.logged_in = False
            st.rerun()

# ---------- SIGNUP PAGE ----------
if st.session_state.page == "signup":
    st.title("📝 Sign Up")
    new_user = st.text_input("Choose a username")
    new_pass = st.text_input("Choose a 4-digit numeric password", type="password")
    confirm_pass = st.text_input("Confirm password", type="password")

    if st.button("Sign Up"):
        if not new_user or not new_pass or not confirm_pass:
            st.error("All fields are required")
        elif not new_pass.isdigit() or len(new_pass) != 4:
            st.error("Password must be 4 digits")
        elif new_pass != confirm_pass:
            st.error("Passwords do not match")
        else:
            signup_user(new_user, new_pass)
            st.success("Account created successfully!")
            st.session_state.page = "login"

# ---------- LOGIN PAGE ----------
if st.session_state.page == "login":
    st.title("🔒 Login to Stock Predictor")
    login_user = st.text_input("Username")
    login_pass = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_login(login_user, login_pass):
            st.success(f"Welcome {login_user}!")
            st.session_state.logged_in = True
            st.session_state.username = login_user
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            st.error("Invalid username or password")

# ---------- PASSWORD CHANGE PAGE ----------
if st.session_state.page == "change_pass":
    st.title("🔄 Change Password")
    old_pass = st.text_input("Current Password", type="password")
    new_pass = st.text_input("New 4-digit Password", type="password")
    confirm_pass = st.text_input("Confirm New Password", type="password")

    if st.button("Update Password"):
        if not old_pass or not new_pass or not confirm_pass:
            st.error("All fields required")
        elif not new_pass.isdigit() or len(new_pass) != 4:
            st.error("Password must be 4 digits")
        elif new_pass != confirm_pass:
            st.error("New passwords do not match")
        elif not check_login(st.session_state.username, old_pass):
            st.error("Current password is incorrect")
        else:
            change_password(st.session_state.username, new_pass)
            st.success("Password updated successfully")
            st.session_state.page = "dashboard"
            st.rerun()

# ---------- DASHBOARD ----------
if st.session_state.get("page") == "dashboard" and st.session_state.logged_in:
    st.title("📈 Stock Price Trend Prediction using LSTM")

    company_map = {
        "Apple Inc": "AAPL",
        "Microsoft": "MSFT",
        "Google (Alphabet)": "GOOGL",
        "Tesla": "TSLA"
    }

    company_name = st.selectbox("Select Company", list(company_map.keys()))
    ticker = company_map[company_name]

    if ticker:
        with st.spinner(f"Downloading data for {ticker}..."):
            df = yf.download(ticker, start="2015-01-01", end="2023-12-31")

        if df.empty:
            st.error("No data found. Try another stock symbol.")
        else:
            df = df[['Close']].dropna()
            df.reset_index(inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            st.success(f"Data for {ticker} loaded!")
            st.line_chart(df['Close'])

            # Preprocessing
            data = df[['Close']].copy()

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            time_step = 60
            X, y = [], []
            for i in range(time_step, len(scaled_data)):
                X.append(scaled_data[i - time_step:i, 0])
                y.append(scaled_data[i, 0])

            X, y = np.array(X), np.array(y)
            X = X.reshape(X.shape[0], X.shape[1], 1)

            model = Sequential()
            model.add(Input(shape=(X.shape[1], 1)))
            model.add(LSTM(50, return_sequences=True))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            with st.spinner("Training model..."):
                model.fit(X, y, epochs=3, batch_size=32, verbose=0)

            predicted = model.predict(X)
            predicted_price = scaler.inverse_transform(predicted)
            actual_price = scaler.inverse_transform(y.reshape(-1, 1))

            st.subheader("Actual vs Predicted Price")
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(actual_price, label='Actual Price', color='blue')
            ax1.plot(predicted_price, label='Predicted Price', color='orange')
            ax1.legend()
            st.pyplot(fig1)

            # Technical Indicators
            st.subheader("Technical Indicators")
            data['MA_14'] = data['Close'].rolling(window=14).mean()

            def calculate_rsi(series, period=14):
                delta = series.diff()
                gain = delta.clip(lower=0)
                loss = -1 * delta.clip(upper=0)
                avg_gain = gain.rolling(window=period).mean()
                avg_loss = loss.rolling(window=period).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            data['RSI_14'] = calculate_rsi(data['Close'])

            fig2, (ax2_1, ax2_2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax2_1.plot(data['Close'], label='Close Price', color='blue')
            ax2_1.plot(data['MA_14'], label='14-Day MA', color='orange')
            ax2_1.legend()
            ax2_2.plot(data['RSI_14'], label='14-Day RSI', color='purple')
            ax2_2.axhline(70, color='red', linestyle='--')
            ax2_2.axhline(30, color='green', linestyle='--')
            ax2_2.legend()
    st.pyplot(fig2)
