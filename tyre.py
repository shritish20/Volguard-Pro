import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
from scipy.stats import linregress
import xgboost as xgb
import pickle
from io import BytesIO
import os
from time import time
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import base64
from streamlit_autorefresh import st_autorefresh
from arch import arch_model

# --- DATABASE SETUP (In Memory) ---
Base = declarative_base()
class TradeLog(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    strategy = Column(String)
    instrument_token = Column(String)
    entry_price = Column(Float)
    exit_price = Column(Float)
    quantity = Column(Float)
    realized_pnl = Column(Float)
    unrealized_pnl = Column(Float)
    timestamp_entry = Column(DateTime, default=datetime.utcnow)
    timestamp_exit = Column(DateTime)
    status = Column(String, default="open")
    regime_score = Column(Float)
    notes = Column(Text)

class JournalEntry(Base):
    __tablename__ = 'journals'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(Text)
    mood = Column(String)
    tags = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

engine = create_engine('sqlite:///:memory:', connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_session = next(get_db())

# --- Helper Functions for Database ---
def log_trade(trade_data):
    try:
        trade = TradeLog(**trade_data)
        db_session.add(trade)
        db_session.commit()
        db_session.refresh(trade)
        return True
    except Exception as e:
        db_session.rollback()
        st.warning(f"⚠️ Error logging trade: {e}")
        return False

def add_journal_entry(entry_data):
    try:
        entry = JournalEntry(**entry_data)
        db_session.add(entry)
        db_session.commit()
        db_session.refresh(entry)
        return True
    except Exception as e:
        db_session.rollback()
        st.warning(f"⚠️ Error adding journal entry: {e}")
        return False

def get_all_trades():
    return db_session.query(TradeLog).order_by(TradeLog.timestamp_entry.desc()).limit(50).all()

def get_all_journals():
    return db_session.query(JournalEntry).order_by(JournalEntry.timestamp.desc()).limit(50).all()

def trades_to_dataframe():
    trades = get_all_trades()
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([{
        "ID": t.id,
        "Strategy": t.strategy,
        "Instrument": t.instrument_token,
        "Entry Price": t.entry_price,
        "Exit Price": t.exit_price,
        "Quantity": t.quantity,
        "Realized P&L": t.realized_pnl,
        "Status": t.status,
        "Regime Score": t.regime_score,
        "Timestamp": t.timestamp_entry.strftime("%Y-%m-%d %H:%M"),
    } for t in get_all_trades()])

def journals_to_dataframe():
    journals = get_all_journals()
    if not journals:
        return pd.DataFrame()
    return pd.DataFrame([{
        "ID": j.id,
        "Title": j.title,
        "Content": j.content,
        "Mood": j.mood,
        "Tags": j.tags,
        "Timestamp": j.timestamp.strftime("%Y-%m-%d %H:%M"),
    } for j in get_all_journals()])

# --- Export Functionality ---
def download_df(df, filename, button_label="Download CSV"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{button_label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def download_excel(df, filename, sheet_name="Sheet1", button_label="Download Excel"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    b64 = base64.b64encode(output.getvalue()).decode()
    file_size = round(len(output.getvalue()) / 1024, 2)
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{button_label} ({file_size} KB)</a>'
    st.markdown(href, unsafe_allow_html=True)

# --- SESSION STATE SETUP ---
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Dashboard"
if "order_clicked" not in st.session_state:
    st.session_state.order_clicked = False
if "selected_strategy" not in st.session_state:
    st.session_state.selected_strategy = ""
if "selected_lots" not in st.session_state:
    st.session_state.selected_lots = 1
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "access_token" not in st.session_state:
    st.session_state.access_token = ""
if "expiry_type_selector" not in st.session_state:
    st.session_state.expiry_type_selector = "Weekly"

# --- ALL STRATEGIES ---
all_strategies = [
    "Iron Fly", "Iron Condor", "Jade Lizard", "Straddle",
    "Calendar Spread", "Bull Put Spread", "Wide Strangle", "ATM Strangle"
]

# --- Configuration ---
def get_config(access_token, expiry_type):
    config = {
        "access_token": access_token,
        "base_url": "https://api.upstox.com/v2",
        "headers": {
            "accept": "application/json",
            "Api-Version": "2.0",
            "Authorization": f"Bearer {access_token}"
        },
        "instrument_key": "NSE_INDEX|Nifty 50",
        "event_url": "https://raw.githubusercontent.com/shritish20/VolGuard/main/upcoming_events.csv",
        "ivp_url": "https://raw.githubusercontent.com/shritish20/VolGuard/main/ivp.csv",
        "nifty_url": "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv",
        "total_funds": 2000000,
        "risk_config": {
            "Iron Fly": {"capital_pct": 0.30, "risk_per_trade_pct": 0.01},
            "Iron Condor": {"capital_pct": 0.25, "risk_per_trade_pct": 0.015},
            "Jade Lizard": {"capital_pct": 0.20, "risk_per_trade_pct": 0.01},
            "Straddle": {"capital_pct": 0.15, "risk_per_trade_pct": 0.02},
            "Calendar Spread": {"capital_pct": 0.10, "risk_per_trade_pct": 0.01},
            "Bull Put Spread": {"capital_pct": 0.15, "risk_per_trade_pct": 0.01},
            "Wide Strangle": {"capital_pct": 0.10, "risk_per_trade_pct": 0.015},
            "ATM Strangle": {"capital_pct": 0.10, "risk_per_trade_pct": 0.015}
        },
        "daily_risk_limit_pct": 0.02,
        "weekly_risk_limit_pct": 0.03,
        "lot_size": 50
    }

    @st.cache_data(ttl=3600)
    def get_expiries(_config):
        try:
            url = f"{_config['base_url']}/option/contract"
            params = {"instrument_key": _config["instrument_key"]}
            res = requests.get(url, headers=_config["headers"], params=params)
            if res.status_code == 200:
                df = pd.DataFrame(res.json()["data"])
                df["expiry"] = pd.to_datetime(df["expiry"])
                weekly = df[df["weekly"] == True]["expiry"].drop_duplicates().sort_values()
                monthly = df[df["weekly"] == False]["expiry"].drop_duplicates().sort_values()
                return weekly, monthly
            return pd.Series(), pd.Series()
        except Exception as e:
            st.warning(f"Error fetching expiries: {e}")
            return pd.Series(), pd.Series()

    weekly_exp, monthly_exp = get_expiries(config)

    expiry_date = None
    if expiry_type == "Weekly" and not weekly_exp.empty:
        expiry_date = weekly_exp.iloc[0].strftime("%Y-%m-%d")
    elif expiry_type == "Monthly" and not monthly_exp.empty:
        expiry_date = monthly_exp.iloc[0].strftime("%Y-%m-%d")
    else:
        expiry_date = datetime.now().strftime("%Y-%m-%d")
        if weekly_exp.empty and monthly_exp.empty:
            st.warning("Could not fetch valid expiry. Defaulting to today.")

    config["expiry_date"] = expiry_date
    return config

# --- Data Fetching and Calculation Functions ---
@st.cache_data(ttl=300)
def fetch_option_chain(config):
    try:
        url = f"{config['base_url']}/option/chain"
        params = {"instrument_key": config['instrument_key'], "expiry_date": config['expiry_date']}
        res = requests.get(url, headers=config['headers'], params=params)
        if res.status_code == 200:
            return res.json()["data"]
        st.error(f":warning: Error fetching option chain: {res.status_code} - {res.text}")
        return []
    except Exception as e:
        st.error(f":warning: Exception in fetch_option_chain: {e}")
        return []

@st.cache_data(ttl=60)
def get_indices_quotes(config):
    try:
        url = f"{config['base_url']}/market-quote/quotes?instrument_key=NSE_INDEX|India VIX,NSE_INDEX|Nifty 50"
        res = requests.get(url, headers=config['headers'])
        if res.status_code == 200:
            data = res.json().get("data", {})
            vix = data["NSE_INDEX:India VIX"]["last_price"] if "NSE_INDEX:India VIX" in data else None
            nifty = data["NSE_INDEX:Nifty 50"]["last_price"] if "NSE_INDEX:Nifty 50" in data else None
            return vix, nifty
        st.error(f":warning: Error fetching indices quotes: {res.status_code} - {res.text}")
        return None, None
    except Exception as e:
        st.error(f":warning: Exception in get_indices_quotes: {e}")
        return None, None

@st.cache_data(ttl=3600)
def load_upcoming_events(config):
    try:
        df = pd.read_csv(config['event_url'])
        df["Datetime"] = pd.to_datetime(df["Date"].str.strip() + " " + df["Time"].str.strip(), format="%d-%b %H:%M", errors="coerce")
        current_year = datetime.now().year
        df["Datetime"] = df["Datetime"].apply(
            lambda dt: dt.replace(year=current_year) if pd.notnull(dt) and dt.year == 1900 else dt
        )
        now = datetime.now()
        expiry_dt = datetime.strptime(config['expiry_date'], "%Y-%m-%d")
        mask = (df["Datetime"] >= now) & (df["Datetime"] <= expiry_dt)
        filtered = df.loc[mask, ["Datetime", "Event", "Classification", "Forecast", "Prior"]]
        return filtered.sort_values("Datetime").reset_index(drop=True)
    except Exception as e:
        st.warning(f":warning: Failed to load upcoming events: {e}")
        return pd.DataFrame(columns=["Datetime", "Event", "Classification", "Forecast", "Prior"])

@st.cache_data(ttl=3600)
def load_ivp(config, avg_iv):
    try:
        iv_df = pd.read_csv(config['ivp_url'])
        iv_df.dropna(subset=["ATM_IV"], inplace=True)
        iv_df = iv_df.tail(30)
        ivp = round((iv_df["ATM_IV"] < avg_iv).sum() / len(iv_df) * 100, 2)
        return ivp
    except Exception as e:
        st.warning(f":warning: Exception in load_ivp: {e}")
        return 0

@st.cache_data(ttl=3600)
def load_xgboost_model():
    try:
        model_url = "https://raw.githubusercontent.com/shritish20/VolGuard-Pro/main/xgb_vol_model_v2.pkl"
        response = requests.get(model_url)
        if response.status_code == 200:
            model = pickle.load(BytesIO(response.content))
            return model
        st.error(f":warning: Error fetching XGBoost model: {response.status_code} - {response.text}")
        return None
    except Exception as e:
        st.error(f":warning: Exception in load_xgboost_model: {e}")
        return None

def predict_xgboost_volatility(model, atm_iv, realized_vol, ivp, pcr, vix, days_to_expiry, garch_vol):
    try:
        features = pd.DataFrame({
            'ATM_IV': [atm_iv],
            'Realized_Vol': [realized_vol],
            'IVP': [ivp],
            'PCR': [pcr],
            'VIX': [vix],
            'Days_to_Expiry': [days_to_expiry],
            'GARCH_Predicted_Vol': [garch_vol]
        })
        if model is not None:
            prediction = model.predict(features)[0]
            return round(float(prediction), 2)
        return 0
    except Exception as e:
        st.warning(f":warning: Exception in predict_xgboost_volatility: {e}")
        return 0

def extract_seller_metrics(option_chain, spot_price):
    try:
        atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
        call = atm["call_options"]
        put = atm["put_options"]
        return {
            "strike": atm["strike_price"],
            "straddle_price": call["market_data"]["ltp"] + put["market_data"]["ltp"],
            "avg_iv": (call["option_greeks"]["iv"] + put["option_greeks"]["iv"]) / 2,
            "theta": call["option_greeks"]["theta"] + put["option_greeks"]["theta"],
            "vega": call["option_greeks"]["vega"] + put["option_greeks"]["vega"],
            "delta": call["option_greeks"]["delta"] + put["option_greeks"]["delta"],
            "gamma": call["option_greeks"]["gamma"] + put["option_greeks"]["gamma"],
            "pop": ((call["option_greeks"]["pop"] + put["option_greeks"]["pop"]) / 2),
        }
    except Exception as e:
        st.warning(f":warning: Exception in extract_seller_metrics: {e}")
        return {}

def full_chain_table(option_chain, spot_price):
    try:
        chain_data = []
        for opt in option_chain:
            strike = opt["strike_price"]
            # Widen the range for a more comprehensive view
            if abs(strike - spot_price) <= 500:
                call = opt["call_options"]
                put = opt["put_options"]
                chain_data.append({
                    "Strike": strike,
                    "Call IV": call["option_greeks"]["iv"],
                    "Put IV": put["option_greeks"]["iv"],
                    "IV Skew": call["option_greeks"]["iv"] - put["option_greeks"]["iv"],
                    "Total Theta": call["option_greeks"]["theta"] + put["option_greeks"]["theta"],
                    "Total Vega": call["option_greeks"]["vega"] + put["option_greeks"]["vega"],
                    "Straddle Price": call["market_data"]["ltp"] + put["market_data"]["ltp"],
                    "Total OI": call["market_data"]["oi"] + put["market_data"]["oi"]
                })
        return pd.DataFrame(chain_data)
    except Exception as e:
        st.warning(f":warning: Exception in full_chain_table: {e}")
        return pd.DataFrame()

def market_metrics(option_chain, expiry_date):
    try:
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        days_to_expiry = (expiry_dt - datetime.now()).days
        call_oi = sum(opt["call_options"]["market_data"]["oi"] for opt in option_chain if "call_options" in opt and "market_data" in opt["call_options"])
        put_oi = sum(opt["put_options"]["market_data"]["oi"] for opt in option_chain if "put_options" in opt and "market_data" in opt["put_options"])
        pcr = put_oi / call_oi if call_oi != 0 else 0
        strikes = sorted(set(opt["strike_price"] for opt in option_chain))
        max_pain_strike = 0
        min_pain = float('inf')
        for strike in strikes:
            pain_at_strike = 0
            for opt in option_chain:
                if "call_options" in opt and "market_data" in opt["call_options"]:
                     pain_at_strike += max(0, strike - opt["strike_price"]) * opt["call_options"]["market_data"]["oi"]
                if "put_options" in opt and "market_data" in opt["put_options"]:
                     pain_at_strike += max(0, opt["strike_price"] - strike) * opt["put_options"]["market_data"]["oi"]
            if pain_at_strike < min_pain:
                min_pain = pain_at_strike
                max_pain_strike = strike
        return {"days_to_expiry": days_to_expiry, "pcr": round(pcr, 2), "max_pain": max_pain_strike}
    except Exception as e:
        st.warning(f":warning: Exception in market_metrics: {e}")
        return {"days_to_expiry": 0, "pcr": 0, "max_pain": 0}

@st.cache_data(ttl=3600)
def calculate_volatility(config, seller_avg_iv):
    try:
        df = pd.read_csv(config['nifty_url'])
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%Y")
        df = df.sort_values('Date')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)
        hv_7 = np.std(df["Log_Returns"][-7:]) * np.sqrt(252) * 100
        model = arch_model(df["Log_Returns"], vol="Garch", p=1, q=1)
        res = model.fit(disp="off")
        forecast = res.forecast(horizon=7)
        garch_7d = np.mean(np.sqrt(forecast.variance.iloc[-1]) * np.sqrt(252) * 100)
        iv_rv_spread = round(seller_avg_iv - hv_7, 2)
        return hv_7, garch_7d, iv_rv_spread
    except Exception as e:
        st.warning(f":warning: Exception in calculate_volatility: {e}")
        return 0, 0, 0

def calculate_iv_skew_slope(full_chain_df):
    try:
        if full_chain_df.empty or len(full_chain_df) < 2:
            return 0
        slope, _, _, _, _ = linregress(full_chain_df["Strike"], full_chain_df["IV Skew"])
        return round(slope, 4)
    except Exception as e:
        st.warning(f":warning: Exception in calculate_iv_skew_slope: {e}")
        return 0

def calculate_regime(atm_iv, ivp, realized_vol, garch_vol, straddle_price, spot_price, pcr, vix, iv_skew_slope):
    expected_move = (straddle_price / spot_price) * 100
    vol_spread = atm_iv - realized_vol
    regime_score = 0
    regime_score += 10 if ivp > 80 else -10 if ivp < 20 else 0
    regime_score += 10 if vol_spread > 10 else -10 if vol_spread < -10 else 0
    regime_score += 10 if vix > 20 else -10 if vix < 10 else 0
    regime_score += 5 if pcr > 1.2 else -5 if pcr < 0.8 else 0
    regime_score += 5 if abs(iv_skew_slope) > 0.001 else 0
    regime_score += 10 if expected_move > 1.5 else -10 if expected_move < 0.5 else 0 # Adjusted thresholds
    regime_score += 5 if garch_vol > realized_vol * 1.2 else -5 if garch_vol < realized_vol * 0.8 else 0

    if regime_score > 20:
        return regime_score, "🔥 High Vol Trend", "Market in high volatility — ideal for premium selling.", "High IVP, elevated VIX, and wide straddle suggest strong premium opportunities."
    elif regime_score > 10:
        return regime_score, "⚡ Elevated Volatility", "Above-average volatility — favor range-bound strategies.", "Moderate IVP and IV-RV spread indicate potential for mean-reverting moves."
    elif regime_score > -10:
        return regime_score, "😊 Neutral Volatility", "Balanced market — flexible strategy selection.", "IV and RV aligned, with moderate PCR and skew."
    else:
        return regime_score, "📉 Low Volatility", "Low volatility — cautious selling or long vega plays.", "Low IVP, tight straddle, and low VIX suggest limited movement."

def suggest_strategy(regime_label, ivp, iv_minus_rv, days_to_expiry, event_df, expiry_date, straddle_price, spot_price):
    strategies = []
    rationale = []
    event_warning = None
    event_window = 3 if ivp > 80 else 2
    high_impact_event_near = False
    event_impact_score = 0
    for _, row in event_df.iterrows():
        try:
            dt = pd.to_datetime(row["Datetime"])
            level = row["Classification"]
            if level == "High" and (0 <= (datetime.strptime(expiry_date, "%Y-%m-%d") - dt).days <= event_window):
                high_impact_event_near = True
            if level == "High" and pd.notnull(row["Forecast"]) and pd.notnull(row["Prior"]):
                forecast = float(str(row["Forecast"]).strip("%")) if "%" in str(row["Forecast"]) else float(row["Forecast"])
                prior = float(str(row["Prior"]).strip("%")) if "%" in str(row["Prior"]) else float(row["Prior"])
                if abs(forecast - prior) > 0.5:
                    event_impact_score += 1
        except Exception as e:
            st.warning(f":warning: Error processing event row: {row}. Error: {e}")
            continue
    if high_impact_event_near:
        event_warning = f"⚠️ High-impact event within {event_window} days of expiry. Prefer defined-risk strategies."
    if event_impact_score > 0:
        rationale.append(f"High-impact events with significant forecast deviations ({event_impact_score} events).")

    expected_move_pct = (straddle_price / spot_price) * 100

    if "High Vol" in regime_label:
        strategies = ["Iron Condor", "Wide Strangle"]
        rationale.append("Strong IV premium — neutral strategies for premium capture.")
    elif "Elevated" in regime_label:
        strategies = ["Iron Fly", "Jade Lizard"]
        rationale.append("Volatility above average — range-bound strategies offer favorable reward-risk.")
    elif "Neutral" in regime_label:
        if days_to_expiry >= 3:
            strategies = ["Jade Lizard", "Bull Put Spread"]
            rationale.append("Market balanced — slight directional bias strategies offer edge.")
        else:
            strategies = ["Iron Fly", "ATM Strangle"]
            rationale.append("Tight expiry — quick theta-based capture.")
    elif "Low Vol" in regime_label:
        if days_to_expiry > 7:
            strategies = ["Straddle", "Calendar Spread"]
            rationale.append("Low IV with longer expiry — benefit from potential IV increase.")
        else:
            strategies = ["Straddle", "ATM Strangle"]
            rationale.append("Low IV — premium collection favorable but monitor for breakout risk.")

    if event_impact_score > 0 or high_impact_event_near:
        strategies = [s for s in strategies if "Iron" in s or "Lizard" in s or "Spread" in s or "Condor" in s]
        rationale.append("Defined-risk strategies are favored due to upcoming events.")

    if ivp > 85 and iv_minus_rv > 5:
        rationale.append(f"Volatility overpriced (IVP: {ivp}%, IV-RV: {iv_minus_rv}%) — ideal for selling premium.")
    elif ivp < 30:
        rationale.append(f"Volatility underpriced (IVP: {ivp}%) — avoid unhedged selling.")
    rationale.append(f"Expected move: ±{expected_move_pct:.2f}% based on straddle price.")
    return strategies, " | ".join(rationale), event_warning

@st.cache_data(ttl=60)
def get_funds_and_margin(config):
    try:
        url = f"{config['base_url']}/user/get-funds-and-margin?segment=SEC"
        res = requests.get(url, headers=config['headers'])
        if res.status_code == 200:
            data = res.json().get("data", {})
            equity_data = data.get("equity", {})
            return {
                "available_margin": float(equity_data.get("available_margin", 0)),
                "used_margin": float(equity_data.get("used_margin", 0)),
                "total_funds": float(equity_data.get("notional_cash", 0))
            }
        st.warning(f":warning: Error fetching funds and margin: {res.status_code} - {res.text}")
        return {"available_margin": 0, "used_margin": 0, "total_funds": 0}
    except Exception as e:
        st.error(f":warning: Exception in get_funds_and_margin: {e}")
        return {"available_margin": 0, "used_margin": 0, "total_funds": 0}

def calculate_sharpe_ratio():
    try:
        # Placeholder for actual portfolio returns
        daily_returns = np.random.normal(0.001, 0.01, 252)
        annual_return = np.mean(daily_returns) * 252
        annual_volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.06) / annual_volatility # Assuming 6% risk-free rate
        return round(sharpe_ratio, 2)
    except Exception as e:
        st.warning(f":warning: Exception in calculate_sharpe_ratio: {e}")
        return 0

def calculate_strategy_premium(orders, lot_size):
    total_premium = 0
    for order in orders:
        price = order.get("current_price", 0)
        qty = order.get("quantity", 0)
        if order["transaction_type"] == "SELL":
            total_premium += price * qty
        else:
            total_premium -= price * qty
    premium_per_lot = total_premium / (lot_size * len(orders) / 4) # Rough estimate
    return premium_per_lot, total_premium

def calculate_strategy_margin(config, strategy_details):
    try:
        instruments = []
        for order in strategy_details["orders"]:
            instruments.append({
                "instrument_token": order["instrument_key"],
                "quantity": abs(order["quantity"]),
                "transaction_type": order["transaction_type"],
                "product": "D"
            })
        if not instruments:
            return 0
            
        url = f"{config['base_url']}/charges/brokerage" # Using brokerage endpoint which also returns margin
        payload = {
            "instrument": instruments,
            "order_type": "MARKET",
            "product": "D",
            "transaction_type": "SELL", # This is a placeholder; API calculates for the basket
            "quantity": config['lot_size']
        }
        res = requests.post(url, headers=config['headers'], json=payload)
        
        if res.status_code == 200:
            data = res.json().get("data", {})
            return round(data.get("required_margin", {}).get("total", 0), 2)
        
        # Fallback to older margin endpoint if brokerage fails
        url_margin = f"{config['base_url']}/charges/margin"
        res_margin = requests.post(url_margin, headers=config['headers'], json={"instruments": instruments})
        if res_margin.status_code == 200:
            data_margin = res_margin.json().get("data", {})
            return round(data_margin.get("required_margin", 0), 2)
            
        st.warning(f":warning: Failed to calculate margin: {res.status_code} - {res.text}")
        return 0
    except Exception as e:
        st.warning(f":warning: Error calculating strategy margin: {e}")
        return 0

def place_multi_leg_orders(config, orders):
    try:
        # Buy legs first to get margin benefit
        sorted_orders = sorted(orders, key=lambda x: 0 if x["transaction_type"] == "BUY" else 1)
        payload = []
        for idx, order in enumerate(sorted_orders):
            correlation_id = f"volg{idx}{int(time() % 100000)}"
            payload.append({
                "quantity": abs(order["quantity"]),
                "product": "D",
                "validity": "DAY",
                "price": 0, # CRITICAL FIX: Set to 0 for MARKET orders
                "tag": f"volguard_{order['instrument_key'].split('|')[1][:10]}",
                "instrument_token": order["instrument_key"],
                "order_type": "MARKET",
                "transaction_type": order["transaction_type"],
                "disclosed_quantity": 0,
                "trigger_price": 0,
                "is_amo": False,
                "correlation_id": correlation_id
            })
        url = f"{config['base_url']}/order/multi/place"
        
        # CRITICAL FIX: The API expects the list directly, not a dict containing the list.
        res = requests.post(url, headers=config['headers'], json=payload) 
        
        if res.status_code == 200:
            st.success("✅ Multi-leg order placed successfully!")
            return True
        else:
            st.error(f"❌ Failed to place multi-leg order: {res.status_code} - {res.text}")
            return False
    except Exception as e:
        st.error(f"⚠️ Error placing multi-leg order: {e}")
        return False

def create_gtt_order(config, instrument_token, trigger_price, transaction_type="SELL", tag="SL"):
    try:
        url = f"{config['base_url']}/order/gtt/place"
        payload = {
            "instrument_token": instrument_token,
            "transaction_type": transaction_type,
            "quantity": config["lot_size"],
            "order_type": "LIMIT",
            "product": "D",
            "price": trigger_price * 1.01, # Set a limit price slightly worse than trigger
            "trigger_price": trigger_price,
            "disclosed_quantity": 0
        }
        res = requests.post(url, headers=config['headers'], json=payload)
        if res.status_code == 200:
            st.success(f"✅ GTT order placed for {instrument_token}")
            return True
        else:
            st.warning(f"⚠️ GTT failed: {res.status_code} - {res.text}")
            return False
    except Exception as e:
        st.error(f"⚠️ Error creating GTT: {e}")
        return False

# --- Strategy Definitions ---
def get_strategy_details(strategy_name, option_chain, spot_price, config, ivp, lots=1):
    func_map = {
        "Iron Fly": _iron_fly_calc,
        "Iron Condor": _iron_condor_calc,
        "Jade Lizard": _jade_lizard_calc,
        "Straddle": _straddle_calc,
        "Calendar Spread": _calendar_spread_calc,
        "Bull Put Spread": _bull_put_spread_calc,
        "Wide Strangle": _wide_strangle_calc,
        "ATM Strangle": _atm_strangle_calc
    }
    if strategy_name not in func_map:
        st.warning(f"⚠️ Strategy {strategy_name} not supported.")
        return None
    try:
        if strategy_name in ["Iron Fly", "Iron Condor"]:
            detail = func_map[strategy_name](option_chain, spot_price, config, ivp, lots=lots)
        else:
            detail = func_map[strategy_name](option_chain, spot_price, config, lots=lots)
    except Exception as e:
        st.warning(f"⚠️ Error calculating {strategy_name} details: {e}")
        return None

    if detail and "orders" in detail and detail["orders"]:
        ltp_map = {}
        for opt in option_chain:
            if "call_options" in opt and "market_data" in opt["call_options"]:
                ltp_map[opt["call_options"]["instrument_key"]] = opt["call_options"]["market_data"].get("ltp", 0)
            if "put_options" in opt and "market_data" in opt["put_options"]:
                ltp_map[opt["put_options"]["instrument_key"]] = opt["put_options"]["market_data"].get("ltp", 0)

        updated_orders = []
        premium = 0
        for order in detail["orders"]:
            key = order["instrument_key"]
            ltp = ltp_map.get(key, 0)
            qty = order["quantity"]
            updated_orders.append({**order, "current_price": ltp})
            if order["transaction_type"] == "SELL":
                premium += ltp * qty
            else:
                premium -= ltp * qty
        
        detail["orders"] = updated_orders
        detail["premium"] = premium / (lots * config["lot_size"])
        detail["premium_total"] = premium

        # Simplified Max Loss/Profit
        if strategy_name in ["Iron Fly", "Iron Condor", "Bull Put Spread"]:
            wing_width = abs(detail["strikes"][0] - detail["strikes"][1]) if len(detail["strikes"]) >= 2 else 50
            detail["max_loss"] = (wing_width * lots * config["lot_size"]) - detail["premium_total"]
            detail["max_profit"] = detail["premium_total"]
        elif strategy_name == "Jade Lizard":
            wing_width = abs(detail["strikes"][1] - detail["strikes"][2])
            detail["max_loss"] = (wing_width * lots * config["lot_size"]) - detail["premium_total"]
            detail["max_profit"] = detail["premium_total"]
        elif strategy_name in ["Straddle", "Wide Strangle", "ATM Strangle"]:
            detail["max_loss"] = float("inf")
            detail["max_profit"] = detail["premium_total"]
        else: # Calendar, etc.
            detail["max_loss"] = detail["premium_total"] if detail["premium_total"] < 0 else 0
            detail["max_profit"] = float("inf")

    return detail

def find_option_by_strike(option_chain, strike, option_type):
    try:
        for opt in option_chain:
            if abs(opt["strike_price"] - strike) < 0.01:
                key = "call_options" if option_type == "CE" else "put_options"
                if key in opt:
                    return opt[key]
        st.warning(f"⚠️ No option found for strike {strike} {option_type}")
        return None
    except Exception as e:
        st.warning(f"⚠️ Exception in find_option_by_strike: {e}")
        return None

def get_dynamic_wing_distance(ivp, straddle_price):
    if ivp >= 80:
        multiplier = 0.40
    elif ivp <= 20:
        multiplier = 0.20
    else:
        multiplier = 0.30
    raw_distance = straddle_price * multiplier
    return max(50, int(round(raw_distance / 50.0)) * 50)  # Round to nearest 50, min 50

def _iron_fly_calc(option_chain, spot_price, config, ivp, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    straddle_price = atm["call_options"]["market_data"]["ltp"] + atm["put_options"]["market_data"]["ltp"]
    wing_distance = get_dynamic_wing_distance(ivp, straddle_price)
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike, "PE")
    ce_long_opt = find_option_by_strike(option_chain, strike + wing_distance, "CE")
    pe_long_opt = find_option_by_strike(option_chain, strike - wing_distance, "PE")
    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        st.error("⚠️ Invalid options for Iron Fly.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Iron Fly", "strikes": [strike - wing_distance, strike, strike + wing_distance], "orders": orders}

def _iron_condor_calc(option_chain, spot_price, config, ivp, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    straddle_price = atm["call_options"]["market_data"]["ltp"] + atm["put_options"]["market_data"]["ltp"]
    short_wing_distance = get_dynamic_wing_distance(ivp, straddle_price)
    long_wing_distance = short_wing_distance + 50 # Fixed 50 point wide condor
    
    ce_short_opt = find_option_by_strike(option_chain, strike + short_wing_distance, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike - short_wing_distance, "PE")
    ce_long_opt = find_option_by_strike(option_chain, strike + long_wing_distance, "CE")
    pe_long_opt = find_option_by_strike(option_chain, strike - long_wing_distance, "PE")
    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        st.error("⚠️ Invalid options for Iron Condor.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Iron Condor", "strikes": [strike - long_wing_distance, strike - short_wing_distance, strike + short_wing_distance, strike + long_wing_distance], "orders": orders}

def _jade_lizard_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 50
    put_strike = atm["strike_price"] - 50
    put_long_strike = put_strike - 50
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, put_long_strike, "PE")
    if not all([ce_short_opt, pe_short_opt, pe_long_opt]):
        st.error("⚠️ Invalid options for Jade Lizard.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Jade Lizard", "strikes": [call_strike, put_strike, put_long_strike], "orders": orders}

def _straddle_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike, "PE")
    if not all([ce_short_opt, pe_short_opt]):
        st.error("⚠️ Invalid options for Straddle.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    return {"strategy": "Straddle", "strikes": [strike, strike], "orders": orders}

def _calendar_spread_calc(option_chain, spot_price, config, lots):
    # This requires multiple expiries, which is complex for this script. Placeholder.
    st.warning("Calendar Spread requires multiple expiries and is not fully implemented.")
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    near_leg = find_option_by_strike(option_chain, strike, "CE")
    # A true calendar would need to fetch a different expiry chain
    far_leg = near_leg 
    if not all([near_leg, far_leg]):
        return None
    return {"strategy": "Calendar Spread", "strikes": [strike, strike], "orders": []}


def _bull_put_spread_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    short_strike = atm["strike_price"] - 50
    long_strike = short_strike - 50
    pe_short_opt = find_option_by_strike(option_chain, short_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, long_strike, "PE")
    if not all([pe_short_opt, pe_long_opt]):
        st.error("⚠️ Invalid options for Bull Put Spread.")
        return None
    orders = [
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Bull Put Spread", "strikes": [long_strike, short_strike], "orders": orders}

def _wide_strangle_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 100
    put_strike = atm["strike_price"] - 100
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    if not all([ce_short_opt, pe_short_opt]):
        st.error("⚠️ Invalid options for Wide Strangle.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    return {"strategy": "Wide Strangle", "strikes": [put_strike, call_strike], "orders": orders}

def _atm_strangle_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 50
    put_strike = atm["strike_price"] - 50
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    if not all([ce_short_opt, pe_short_opt]):
        st.error("⚠️ Invalid options for ATM Strangle.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    return {"strategy": "ATM Strangle", "strikes": [put_strike, call_strike], "orders": orders}

def evaluate_full_risk(trades_df, config, regime_label, vix):
    try:
        total_funds = config.get('total_funds', 2000000)
        daily_risk_limit = config['daily_risk_limit_pct'] * total_funds
        weekly_risk_limit = config['weekly_risk_limit_pct'] * total_funds
        max_drawdown_pct = 0.05 if vix > 20 else 0.03 if vix > 12 else 0.02
        max_drawdown = max_drawdown_pct * total_funds
        strategy_summary = []
        total_cap_used = total_risk_used = total_realized_pnl = total_vega = 0
        flags = []
        if trades_df.empty:
            strategy_summary.append({
                "Strategy": "None", "Capital Used": 0, "Cap Limit": total_funds, "% Used": 0,
                "Potential Risk": 0, "Risk Limit": total_funds * 0.01,
                "P&L": 0, "Vega": 0, "Risk OK?": "✅"
            })
        else:
            for _, row in trades_df.iterrows():
                strat = row["strategy"]
                capital_used = row["capital_used"]
                potential_risk = row["potential_loss"]
                pnl = row["realized_pnl"]
                sl_hit = row["sl_hit"]
                vega = row["vega"]
                cfg = config['risk_config'].get(strat, {"capital_pct": 0.1, "risk_per_trade_pct": 0.01})
                risk_factor = 1.2 if "High Vol" in regime_label else 0.8 if "Low Vol" in regime_label else 1.0
                max_cap = cfg["capital_pct"] * total_funds
                max_risk = cfg["risk_per_trade_pct"] * max_cap * risk_factor
                risk_ok = potential_risk <= max_risk
                strategy_summary.append({
                    "Strategy": strat,
                    "Capital Used": capital_used,
                    "Cap Limit": round(max_cap),
                    "% Used": round(capital_used / max_cap * 100, 2) if max_cap else 0,
                    "Potential Risk": potential_risk,
                    "Risk Limit": round(max_risk),
                    "P&L": pnl,
                    "Vega": vega,
                    "Risk OK?": "✅" if risk_ok else "❌"
                })
                total_cap_used += capital_used
                total_risk_used += potential_risk
                total_realized_pnl += pnl
                total_vega += vega
                if not risk_ok:
                    flags.append(f"❌ {strat} exceeded risk limit")
                if sl_hit:
                    flags.append(f"⚠️ {strat} shows possible revenge trading")
        net_dd = -total_realized_pnl if total_realized_pnl < 0 else 0
        exposure_pct = round(total_cap_used / total_funds * 100, 2) if total_funds else 0
        risk_pct = round(total_risk_used / total_funds * 100, 2) if total_funds else 0
        dd_pct = round(net_dd / total_funds * 100, 2) if total_funds else 0
        portfolio_summary = {
            "Total Funds": total_funds,
            "Capital Deployed": total_cap_used,
            "Exposure Percent": exposure_pct,
            "Risk on Table": total_risk_used,
            "Risk Percent": risk_pct,
            "Daily Risk Limit": daily_risk_limit,
            "Weekly Risk Limit": weekly_risk_limit,
            "Realized P&L": total_realized_pnl,
            "Drawdown ₹": net_dd,
            "Drawdown Percent": dd_pct,
            "Max Drawdown Allowed": max_drawdown,
            "Flags": flags
        }
        return pd.DataFrame(strategy_summary), portfolio_summary
    except Exception as e:
        st.error(f"⚠️ Exception in evaluate_full_risk: {e}")
        return pd.DataFrame([{ "Strategy": "None", "Capital Used": 0, "Cap Limit": 2000000, "% Used": 0, "Potential Risk": 0, "Risk Limit": 20000, "P&L": 0, "Vega": 0, "Risk OK?": "✅" }]), {}

def fetch_trade_data(config, full_chain_df):
    try:
        url_positions = f"{config['base_url']}/portfolio/short-term-positions"
        res_positions = requests.get(url_positions, headers=config['headers'])
        positions = res_positions.json().get("data", []) if res_positions.status_code == 200 else []
        
        # Simple logic to group legs into strategies (placeholder)
        trades_df_list = []
        for pos in positions:
            instrument = pos.get("instrument_token", "")
            # Basic strategy identification
            strategy = "Unknown"
            if pos.get("product") == "I": # Assuming intraday for options strategies
                 if "CE" in instrument or "PE" in instrument:
                      strategy = "Options Leg" # Needs more logic to group into straddles, etc.

            capital = abs(pos["quantity"] * pos["average_price"])
            trades_df_list.append({
                "strategy": strategy,
                "capital_used": capital,
                "potential_loss": capital * 0.1, # Placeholder risk
                "realized_pnl": pos.get("pnl", 0),
                "trades_today": 1, # Placeholder
                "sl_hit": pos.get("pnl", 0) < -abs(capital * 0.05),
                "vega": full_chain_df["Total Vega"].mean() if not full_chain_df.empty else 0,
                "instrument_token": instrument
            })
        return pd.DataFrame(trades_df_list) if trades_df_list else pd.DataFrame(columns=["strategy", "capital_used", "potential_loss", "realized_pnl", "trades_today", "sl_hit", "vega", "instrument_token"])
    except Exception as e:
        st.error(f"⚠️ Exception in fetch_trade_data: {e}")
        return pd.DataFrame()

# --- Plotting Functions ---
def plot_allocation_pie(strategy_df, config):
    if strategy_df.empty or strategy_df["Capital Used"].sum() == 0:
        st.info("No capital deployed to plot allocation.")
        return

    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0E1117')
    ax.pie(strategy_df["Capital Used"], labels=strategy_df["Strategy"], autopct='%1.1f%%', startangle=90, textprops={'color': 'white'})
    ax.set_title("Capital Allocation by Strategy", color="white")
    st.pyplot(fig)

def plot_drawdown_trend(portfolio_summary):
    # Placeholder data for drawdown trend
    np.random.seed(42)
    daily_pnl = np.random.normal(-500, 3000, 30)
    cumulative_pnl = np.cumsum(daily_pnl)
    drawdowns = cumulative_pnl - np.maximum.accumulate(cumulative_pnl)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0E1117')
    ax.plot(range(30), drawdowns, color="#00BFFF", fillstyle='full')
    ax.fill_between(range(30), drawdowns, 0, color="#00BFFF", alpha=0.3)
    ax.axhline(-portfolio_summary.get("Max Drawdown Allowed", 0), linestyle="--", color="red", label="Max Drawdown Limit")
    ax.set_title("Drawdown Trend (₹)", color="white")
    ax.set_facecolor('#1A1C24')
    ax.tick_params(colors='white')
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(facecolor='#1A1C24', labelcolor='white')
    st.pyplot(fig)

def plot_margin_gauge(funds_data):
    if not funds_data or funds_data.get("total_funds", 0) == 0:
        st.info("No funds data to display margin.")
        return
        
    used_pct = (funds_data["used_margin"] / funds_data["total_funds"] * 100) if funds_data["total_funds"] else 0
    fig, ax = plt.subplots(figsize=(8, 2), facecolor='#0E1117')
    ax.barh(["Margin"], [100], color='#2E2F38')
    ax.barh(["Margin"], [used_pct], color="#00BFFF")
    ax.set_xlim(0, 100)
    ax.set_title("Margin Utilization (%)", color="white")
    ax.text(used_pct + 2, 0, f"{used_pct:.1f}%", va='center', color='white', weight='bold')
    ax.axis('off')
    st.pyplot(fig)

def plot_vol_comparison(seller, hv_7, garch_7d, xgb_vol):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0E1117')
    vols = [seller['avg_iv'], hv_7, garch_7d, xgb_vol]
    labels = ['ATM IV', 'Realized Vol (7D)', 'GARCH Forecast', 'XGBoost Forecast']
    colors = ['#00BFFF', '#FF6347', '#32CD32', '#FFD700']
    bars = ax.bar(labels, vols, color=colors)
    ax.set_title("Volatility Comparison (%)", color="white")
    ax.set_ylabel("Annualized Volatility (%)", color="white")
    ax.tick_params(axis='x', colors='white', rotation=25)
    ax.tick_params(axis='y', colors='white')
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_facecolor('#1A1C24')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom', color='white')
    st.pyplot(fig)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="VolGuard - Trading Copilot", layout="wide", initial_sidebar_state="expanded")
st_autorefresh(interval=5 * 60 * 1000, key="refresh") # 5 min refresh
st.markdown("""
<style>
.main { background-color: #0E1117; color: white; }
.metric-box { 
    background-color: #1A1C24; 
    padding: 1rem; 
    border-radius: 0.75rem; 
    margin-bottom: 1rem; 
    border: 1px solid #2E2F38;
}
.metric-box h3 { color: #6495ED; font-size: 1em; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-box .value { font-size: 1.8em; font-weight: bold; color: #00BFFF; }
.stDataFrame { border: 1px solid #2E2F38; }
.stButton>button { border-radius: 0.5rem; }
section[data-testid="stSidebar"] {
    background-color: #1A1C24;
}
section[data-testid="stSidebar"] .stTextInput > div > div > input,
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #2E2F38;
    color: white;
    border: 1px solid #00BFFF;
}
</style>""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🚀 VolGuard Pro")
st.sidebar.markdown("Your Options Trading Copilot")

access_token = st.sidebar.text_input("Enter Upstox Access Token", type="password", value=st.session_state.access_token)

if st.sidebar.button("Login"):
    if access_token:
        # Simple validation by trying to get profile
        headers = {"accept": "application/json", "Api-Version": "2.0", "Authorization": f"Bearer {access_token}"}
        test_url = "https://api.upstox.com/v2/user/profile"
        try:
            res = requests.get(test_url, headers=headers)
            if res.status_code == 200:
                st.session_state.access_token = access_token
                st.session_state.logged_in = True
                st.sidebar.success("✅ Logged in successfully!")
                st.rerun()
            else:
                st.sidebar.error(f"❌ Invalid token or API error.")
        except Exception as e:
            st.sidebar.error(f"⚠️ Error validating token: {e}")
    else:
        st.sidebar.error("❌ Please enter an access token.")

if st.session_state.logged_in and st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.access_token = ""
    st.cache_data.clear()
    st.rerun()

# Main App Logic
if st.session_state.logged_in:
    # Moved expiry selector here to fix duplicate key error
    expiry_type = st.sidebar.radio("📅 Expiry Type", ["Weekly", "Monthly"], key="expiry_type_selector", horizontal=True)
    
    config = get_config(st.session_state.access_token, expiry_type)
    
    @st.cache_data(show_spinner="Analyzing market data...", ttl=120)
    def load_all_data(_config):
        option_chain = fetch_option_chain(_config)
        if not option_chain:
            return (None,) * 22 # Return tuple of Nones
        
        spot_price = option_chain[0].get("underlying_spot_price")
        vix, nifty = get_indices_quotes(_config)
        if not all([spot_price, vix, nifty]):
            st.error("❌ Failed to fetch critical market data (Spot, VIX).")
            return (None,) * 22

        seller = extract_seller_metrics(option_chain, spot_price)
        full_chain_df = full_chain_table(option_chain, spot_price)
        market = market_metrics(option_chain, _config['expiry_date'])
        ivp = load_ivp(_config, seller.get("avg_iv", 0))
        hv_7, garch_7d, iv_rv_spread = calculate_volatility(_config, seller.get("avg_iv", 0))
        
        xgb_model = load_xgboost_model()
        xgb_vol = predict_xgboost_volatility(xgb_model, seller.get("avg_iv", 0), hv_7, ivp, market.get("pcr", 0), vix, market.get("days_to_expiry", 0), garch_7d)
        
        iv_skew_slope = calculate_iv_skew_slope(full_chain_df)
        regime_score, regime, regime_note, regime_explanation = calculate_regime(
            seller.get("avg_iv", 0), ivp, hv_7, garch_7d, seller.get("straddle_price", 0), spot_price, market.get("pcr", 0), vix, iv_skew_slope)
        
        event_df = load_upcoming_events(_config)
        strategies, strategy_rationale, event_warning = suggest_strategy(
            regime, ivp, iv_rv_spread, market.get("days_to_expiry", 0), event_df, _config['expiry_date'], seller.get("straddle_price", 0), spot_price)
        
        # Pass ivp to get_strategy_details
        strategy_details = [detail for strat in strategies if (detail := get_strategy_details(strat, option_chain, spot_price, _config, ivp, lots=1)) is not None]
        
        funds_data = get_funds_and_margin(_config)
        _config['total_funds'] = funds_data.get('total_funds', _config['total_funds'])
        
        trades_df = fetch_trade_data(_config, full_chain_df)
        strategy_df, portfolio_summary = evaluate_full_risk(trades_df, _config, regime, vix)
        sharpe_ratio = calculate_sharpe_ratio()
        
        return (option_chain, spot_price, vix, nifty, seller, full_chain_df, market, ivp, hv_7, garch_7d, xgb_vol, iv_rv_spread, iv_skew_slope, regime_score, regime, regime_note, regime_explanation, event_df, strategies, strategy_rationale, event_warning, strategy_details, strategy_df, portfolio_summary, funds_data, sharpe_ratio)

    # Unpack all data
    data_tuple = load_all_data(config)
    if data_tuple[0] is None:
        st.error("Dashboard loading failed. Please check token or API status.")
        st.stop()
        
    (option_chain, spot_price, vix, nifty, seller, full_chain_df, market, ivp, hv_7, garch_7d, xgb_vol, iv_rv_spread, iv_skew_slope, regime_score, regime, regime_note, regime_explanation, event_df, strategies, strategy_rationale, event_warning, strategy_details, strategy_df, portfolio_summary, funds_data, sharpe_ratio) = data_tuple

    # Main Dashboard UI
    st.markdown("<h1 style='text-align: center;'>Market Insights Dashboard</h1>", unsafe_allow_html=True)
    st.info(f"⏳ Auto-refreshing every 5 minutes. Last refresh: {datetime.now().strftime('%H:%M:%S')}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-box'><h3>Nifty 50 Spot</h3><div class='value'>₹{nifty:.2f}</div></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-box'><h3>India VIX</h3><div class='value'>{vix:.2f}</div></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-box'><h3>ATM IV</h3><div class='value'>{seller['avg_iv']:.2f}%</div></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-box'><h3>IVP</h3><div class='value'>{ivp}%</div></div>", unsafe_allow_html=True)

    col5, col6, col7, col8 = st.columns(4)
    col5.markdown(f"<div class='metric-box'><h3>Straddle Price</h3><div class='value'>₹{seller['straddle_price']:.2f}</div></div>", unsafe_allow_html=True)
    col6.markdown(f"<div class='metric-box'><h3>PCR</h3><div class='value'>{market['pcr']:.2f}</div></div>", unsafe_allow_html=True)
    col7.markdown(f"<div class='metric-box'><h3>Days to Expiry</h3><div class='value'>{market['days_to_expiry']}</div></div>", unsafe_allow_html=True)
    col8.markdown(f"<div class='metric-box'><h3>Max Pain</h3><div class='value'>{market['max_pain']:.0f}</div></div>", unsafe_allow_html=True)

    tabs = ["Dashboard", "Option Chain", "Strategy Hub", "Risk & Portfolio", "Manual Order", "Journal"]
    st.session_state.active_tab = st.sidebar.radio("Navigate", tabs, index=tabs.index(st.session_state.active_tab))
    st.sidebar.markdown("---")
    st.sidebar.info(f"Selected Expiry: **{config['expiry_date']}**")

    # --- TAB: Dashboard ---
    if st.session_state.active_tab == "Dashboard":
        st.subheader("Volatility Landscape")
        plot_vol_comparison(seller, hv_7, garch_7d, xgb_vol)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='metric-box'><h3>Breakeven Range</h3><div class='value' style='font-size: 1.4em;'>{seller['strike'] - seller['straddle_price']:.0f} – {seller['strike'] + seller['straddle_price']:.0f}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-box'><h3>IV - RV Spread</h3><div class='value'>{iv_rv_spread:+.2f}%</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-box'><h3>IV Skew Slope</h3><div class='value'>{iv_skew_slope:.4f}</div></div>", unsafe_allow_html=True)

        st.subheader("ATM Greeks Snapshot")
        gc1, gc2, gc3, gc4, gc5 = st.columns(5)
        gc1.metric("Delta", f"{seller['delta']:.3f}")
        gc2.metric("Theta", f"₹{seller['theta']:.2f}")
        gc3.metric("Vega", f"₹{seller['vega']:.2f}")
        gc4.metric("Gamma", f"{seller['gamma']:.4f}")
        gc5.metric("POP", f"{seller['pop']:.2f}%")
        
        st.subheader("Upcoming Events Before Expiry")
        if not event_df.empty:
            st.dataframe(event_df.style.set_properties(**{'background-color': '#1A1C24', 'color': 'white'}), use_container_width=True)
        else:
            st.info("No major upcoming events before expiry.")

    # --- TAB: Option Chain ---
    elif st.session_state.active_tab == "Option Chain":
        st.subheader("📊 Interactive Option Chain")
        st.dataframe(full_chain_df.style.set_properties(**{'background-color': '#1A1C24', 'color': 'white'}).format({
            "Call IV": "{:.2f}", "Put IV": "{:.2f}", "IV Skew": "{:.2f}", 
            "Total Theta": "{:.2f}", "Total Vega": "{:.2f}", "Straddle Price": "{:.2f}"
        }), use_container_width=True)

        st.subheader("📈 Visual Greeks Analysis")
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), facecolor='#0E1117')
        
        # Call vs Put IV
        axs[0].bar(full_chain_df["Strike"], full_chain_df["Call IV"], color="#1f77b4", width=20, label="Call IV")
        axs[0].bar(full_chain_df["Strike"], full_chain_df["Put IV"], color="#ff7f0e", width=20, alpha=0.7, label="Put IV")
        axs[0].set_title("Call vs Put IV", color='white')
        axs[0].legend()

        # Total Theta
        axs[1].bar(full_chain_df["Strike"], full_chain_df["Total Theta"], color="#2ca02c", width=20)
        axs[1].set_title("Total Theta (Decay)", color='white')

        # Total Vega
        axs[2].bar(full_chain_df["Strike"], full_chain_df["Total Vega"], color="#d62728", width=20)
        axs[2].set_title("Total Vega (Volatility Risk)", color='white')

        for ax in axs:
            ax.set_facecolor('#1A1C24')
            ax.tick_params(colors='white')
            ax.grid(True, linestyle=":", alpha=0.4)
            ax.axvline(spot_price, color='cyan', linestyle='--', label='Spot Price')
        
        st.pyplot(fig)
        
    # --- TAB: Strategy Hub ---
    elif st.session_state.active_tab == "Strategy Hub":
        st.markdown(f"<div class='metric-box'><h3>Regime: {regime}</h3><p>Score: {regime_score:.2f} | {regime_note}<br><i>{regime_explanation}</i></p></div>", unsafe_allow_html=True)
        st.subheader("💡 Recommended Strategies")
        if strategies:
            st.success(f"**Top Suggestions:** {', '.join(strategies)}")
            st.info(f"**Rationale:** {strategy_rationale}")
            if event_warning:
                st.warning(event_warning)
            
            for detail in strategy_details:
                with st.expander(f"**{detail['strategy']}** - View Details & Trade"):
                    order_df = pd.DataFrame(detail["orders"])
                    st.dataframe(order_df[['instrument_key', 'transaction_type', 'quantity', 'current_price']].style.set_properties(**{'background-color': '#2E2F38', 'color': 'white'}), use_container_width=True)
                    
                    margin = calculate_strategy_margin(config, detail)
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Est. Margin", f"₹{margin:,.2f}")
                    sc2.metric("Max Profit", f"₹{detail.get('max_profit', 0):,.2f}")
                    sc3.metric("Max Loss", f"₹{detail.get('max_loss', 0):,.2f}")

                    lots = st.number_input(f"Lots for {detail['strategy']}", min_value=1, value=1, step=1, key=f"lots_{detail['strategy']}")
                    if st.button(f"Place {detail['strategy']} Order", key=f"place_{detail['strategy']}"):
                        final_detail = get_strategy_details(detail['strategy'], option_chain, spot_price, config, ivp, lots=lots)
                        if final_detail:
                            place_multi_leg_orders(config, final_detail["orders"])
        else:
            st.info("No specific strategies recommended for the current market conditions.")
            
    # --- TAB: Risk & Portfolio ---
    elif st.session_state.active_tab == "Risk & Portfolio":
        st.subheader("🛡️ Risk & Portfolio Overview")
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Total Funds", f"₹{funds_data.get('total_funds', 0):,.2f}")
        pc2.metric("Available Margin", f"₹{funds_data.get('available_margin', 0):,.2f}")
        pc3.metric("Used Margin", f"₹{funds_data.get('used_margin', 0):,.2f}")
        pc4.metric("Sharpe Ratio (mock)", f"{sharpe_ratio:.2f}")
        
        plot_margin_gauge(funds_data)
        
        st.subheader("Strategy Risk Summary")
        if not strategy_df.empty:
            st.dataframe(strategy_df.style.set_properties(**{'background-color': '#1A1C24', 'color': 'white'}), use_container_width=True)
            if portfolio_summary.get("Flags"):
                st.warning(f"⚠️ Risk Alerts: {' | '.join(portfolio_summary['Flags'])}")
        else:
            st.info("No active strategies or positions found.")
            
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Capital Allocation")
            plot_allocation_pie(strategy_df, config)
        with c2:
            st.subheader("Drawdown Trend")
            plot_drawdown_trend(portfolio_summary)

    # --- TAB: Manual Order ---
    elif st.session_state.active_tab == "Manual Order":
        st.subheader("📥 Manual Multi-Leg Order")
        selected_strategy = st.selectbox("Select Strategy Template", all_strategies, key="manual_strategy")
        lots = st.number_input("Number of Lots", min_value=1, value=1, step=1, key="manual_lots")
        
        detail = get_strategy_details(selected_strategy, option_chain, spot_price, config, ivp, lots=lots)

        if detail:
            st.write("📋 Order Legs")
            order_df = pd.DataFrame(detail["orders"])
            st.dataframe(order_df[['instrument_key', 'transaction_type', 'quantity', 'current_price']].style.set_properties(**{'background-color': '#1A1C24', 'color': 'white'}), use_container_width=True)

            margin = calculate_strategy_margin(config, detail)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Est. Margin", f"₹{margin * lots:,.2f}")
            mc2.metric("Total Premium", f"₹{detail['premium_total']:,.2f}")
            mc3.metric("Max Loss", f"₹{detail.get('max_loss', 0):,.2f}")

            if st.button("🚀 Place Manual Order"):
                place_multi_leg_orders(config, detail["orders"])
        else:
            st.error("❌ Could not generate details for the selected strategy.")
            
    # --- TAB: Journal ---
    elif st.session_state.active_tab == "Journal":
        st.header("📁 Logs & Journal")
        tab_logs, tab_journal = st.tabs(["Trade Logs", "Trading Journal"])
        with tab_logs:
            st.subheader("📊 Recent Trades")
            trades_df = trades_to_dataframe()
            if not trades_df.empty:
                st.dataframe(trades_df.style.set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
                download_df(trades_df, "trades.csv")
            else:
                st.info("No trade logs found.")
        with tab_journal:
            st.subheader("📝 Add Journal Entry")
            with st.form("journal_form"):
                entry_title = st.text_input("Title")
                entry_text = st.text_area("Observations & Decisions")
                mood = st.selectbox("Mood", ["😄 Happy", "😌 Calm", "😰 Stressed", "😤 Angry", "😴 Tired"])
                tags = st.text_input("Tags (comma-separated)", "e.g., Nifty, HighIV, FED_event")
                if st.form_submit_button("Save Entry"):
                    if add_journal_entry({"title": entry_title, "content": entry_text, "mood": mood, "tags": tags}):
                        st.success("✅ Journal entry saved!")
            
            st.subheader("📓 Recent Entries")
            journals_df = journals_to_dataframe()
            if not journals_df.empty:
                 st.dataframe(journals_df.style.set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
            else:
                 st.info("No journal entries yet.")

else:
    st.markdown("<h1 style='text-align: center;'>Welcome to VolGuard Pro</h1>", unsafe_allow_html=True)
    st.info("Please enter your Upstox Access Token in the sidebar to begin.")
