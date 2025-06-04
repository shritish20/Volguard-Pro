import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
from scipy.stats import linregress
import xgboost as xgb
import pickle
from io import BytesIO
import base64
from streamlit_autorefresh import st_autorefresh
from arch import arch_model
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import time

# --- DATABASE SETUP ---
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

# --- DATABASE HELPER FUNCTIONS ---
def log_trade(trade_data):
    try:
        trade = TradeLog(**trade_data)
        db_session.add(trade)
        db_session.commit()
        db_session.refresh(trade)
        return True
    except Exception as e:
        db_session.rollback()
        st.error(f"⚠️ Error logging trade: {e}")
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
        st.error(f"⚠️ Error adding journal entry: {e}")
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
        "Timestamp": t.timestamp_entry.strftime("%Y-%m-%d %H:%M")
    } for t in trades])

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
        "Timestamp": j.timestamp.strftime("%Y-%m-%d %H:%M")
    } for j in journals])

# --- EXPORT FUNCTIONS ---
def download_df(df, filename, button_label="Download CSV"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode('utf-8')
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{button_label}</a>'
    return href

def download_excel(df, filename, sheet_name="Sheet1", button_label="Download Excel"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    b64 = base64.b64encode(output.getvalue()).decode('utf-8')
    file_size = round(len(output.getvalue()) / 1024, 2)
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{button_label} ({file_size} KB)</a>'
    return href

# --- SESSION STATE INITIALIZATION ---
def init_session_state():
    defaults = {
        "active_tab": "Dashboard",
        "order_clicked": False,
        "selected_strategy": "",
        "selected_lots": 1,
        "logged_in": False,
        "access_token": "",
        "expiry_type": "Weekly"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- STRATEGIES ---
all_strategies = [
    "Iron Strike", "Iron Condor", "Jade Lizard", "Straddle",
    "Calendar Spread", "Bull Put Spread", "Wide Strangle", "ATM Strangle"
]

# --- CONFIGURATION ---
def get_config(access_token):
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
            "Iron Strike": {"capital_pct": 0.30, "risk_per_trade_pct": 0.01},
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
        "lot_size": 75
    }

    def get_expiries():
        try:
            url = f"{config['base_url']}/option/contract"
            params = {"instrument_key": config["instrument_key"]}
            res = requests.get(url, headers=config["headers"], params=params)
            if res.status_code == 200:
                df = pd.DataFrame(res.json()["data"])
                df["expiry"] = pd.to_datetime(df["expiry"])
                weekly = df[df["weekly"] == True]["expiry"].drop_duplicates().sort_values()
                monthly = df[df["weekly"] == False]["expiry"].drop_duplicates().sort_values()
                return weekly, monthly
            return [], []
        except Exception as e:
            st.error(f"⚠️ Error fetching expiries: {e}")
            return [], []

    weekly_exp, monthly_exp = get_expiries()
    expiry_type = st.session_state.get("expiry_type", "Weekly")
    expiry_date = None
    if expiry_type == "Weekly" and len(weekly_exp) > 0:
        expiry_date = weekly_exp.iloc[0].strftime("%Y-%m-%d")
    elif expiry_type == "Monthly" and len(monthly_exp) > 0:
        expiry_date = monthly_exp.iloc[0].strftime("%Y-%m-%d")
    else:
        expiry_date = datetime.now().strftime("%Y-%m-%d")
        st.warning("Could not fetch valid expiry. Defaulting to today.")
    config["expiry_date"] = expiry_date
    return config

# --- DATA FETCHING ---
@st.cache_data(ttl=300)
def fetch_option_chain(config):
    try:
        url = f"{config['base_url']}/option/chain"
        params = {"instrument_key": config['instrument_key'], "expiry_date": config['expiry_date']}
        res = requests.get(url, headers=config['headers'], params=params)
        if res.status_code == 200:
            return res.json()["data"]
        st.error(f"⚠️ Error fetching option chain: {res.status_code} - {res.text}")
        return []
    except Exception as e:
        st.error(f"⚠️ Exception in fetch_option_chain: {e}")
        return None

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
        st.error(f"⚠️ Error fetching indices quotes: {res.status_code} - {res.text}")
        return None, None
    except Exception as e:
        st.error(f"⚠️ Exception in get_indices_quotes: {e}")
        return None, None

@st.cache_data
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
        st.warning(f"⚠️ Failed to load upcoming events: {e}")
        return pd.DataFrame(columns=["Datetime", "Event", "Classification", "Forecast", "Prior"])

@st.cache_data
def load_ivp(config, avg_iv):
    try:
        iv_df = pd.read_csv(config['ivp_url'])
        iv_df.dropna(subset=["ATM_IV"], inplace=True)
        iv_df = iv_df.tail(30)
        ivp = round((iv_df["ATM_IV"] < avg_iv).sum() / len(iv_df) * 100, 2)
        return ivp
    except Exception as e:
        st.warning(f"⚠️ Exception in load_ivp: {e}")
        return 0

@st.cache_data
def load_xgboost_model():
    try:
        model_url = "https://raw.githubusercontent.com/shritish20/VolGuard-Pro/main/xgb_vol_model_v2.pkl"
        response = requests.get(model_url)
        if response.status_code == 200:
            model = pickle.load(BytesIO(response.content))
            return model
        st.error(f"⚠️ Error fetching XGBoost model: {response.status_code} - {response.text}")
        return None
    except Exception as e:
        st.error(f"⚠️ Exception in load_xgboost_model: {e}")
        return None

# --- CALCULATION FUNCTIONS ---
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
        st.warning(f"⚠️ Exception in predict_xgboost_volatility: {e}")
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
            "pop": ((call["option_greeks"]["pop"] + put["option_greeks"]["pop"]) / 2)
        }
    except Exception as e:
        st.warning(f"⚠️ Exception in extract_seller_metrics: {e}")
        return {}

def full_chain_table(option_chain, spot_price):
    try:
        chain_data = []
        for opt in option_chain:
            strike = opt["strike_price"]
            if abs(strike - spot_price) <= 300:
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
        st.warning(f"⚠️ Exception in full_chain_table: {e}")
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
                if "call_options" in opt:
                    pain_at_strike += max(0, strike - opt["strike_price"]) * opt["call_options"]["market_data"]["oi"]
                if "put_options" in opt:
                    pain_at_strike += max(0, opt["strike_price"] - strike) * opt["put_options"]["market_data"]["oi"]
            if pain_at_strike < min_pain:
                min_pain = pain_at_strike
                max_pain_strike = strike
        return {"days_to_expiry": days_to_expiry, "pcr": round(pcr, 2), "max_pain": max_pain_strike}
    except Exception as e:
        st.warning(f"⚠️ Exception in market_metrics: {e}")
        return {"days_to_expiry": 0, "pcr": 0, "max_pain": 0}

@st.cache_data
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
        st.warning(f"⚠️ Exception in calculate_volatility: {e}")
        return 0, 0, 0

def calculate_iv_skew_slope(full_chain_df):
    try:
        if full_chain_df.empty:
            return 0
        slope, _, _, _, _ = linregress(full_chain_df["Strike"], full_chain_df["IV Skew"])
        return round(slope, 4)
    except Exception as e:
        st.warning(f"⚠️ Exception in calculate_iv_skew_slope: {e}")
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
    regime_score += 10 if expected_move > 0.05 else -10 if expected_move < 0.02 else 0
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
            st.warning(f"⚠️ Error processing event row: {row}. Error: {e}")
            continue
    if high_impact_event_near:
        event_warning = f"⚠️ High-impact event within {event_window} days of expiry. Prefer defined-risk strategies."
    if event_impact_score > 0:
        rationale.append(f"High-impact events with significant forecast deviations ({event_impact_score} events).")
    expected_move_pct = (straddle_price / spot_price) * 100
    if regime_label == "🔥 High Vol Trend":
        strategies = ["Iron Strike", "Wide Strangle"]
        rationale.append("Strong IV premium — neutral strategies for premium capture.")
    elif regime_label == "⚡ Elevated Volatility":
        strategies = ["Iron Condor", "Jade Lizard"]
        rationale.append("Volatility above average — range-bound strategies offer favorable reward-risk.")
    elif regime_label == "😊 Neutral Volatility":
        if days_to_expiry >= 3:
            strategies = ["Jade Lizard", "Bull Put Spread"]
            rationale.append("Market balanced — slight directional bias strategies offer edge.")
        else:
            strategies = ["Iron Strike"]
            rationale.append("Tight expiry — quick theta-based capture via short Iron Strike.")
    elif regime_label == "📉 Low Volatility":
        if days_to_expiry > 7:
            strategies = ["Straddle", "Calendar Spread"]
            rationale.append("Low IV with longer expiry — benefit from potential IV increase.")
        else:
            strategies = ["Straddle", "ATM Strangle"]
            rationale.append("Low IV — premium collection favorable but monitor for breakout risk.")
    if event_impact_score > 0 and not high_impact_event_near:
        strategies = [s for s in strategies if "Iron" in s or "Lizard" in s or "Spread" in s]
    if ivp > 85 and iv_minus_rv > 5:
        rationale.append(f"Volatility overpriced (IVP: {ivp}%, IV-RV: {iv_minus_rv}%) — ideal for selling premium.")
    elif ivp < 30:
        rationale.append(f"Volatility underpriced (IVP: {ivp}%) — avoid unhedged selling.")
    rationale.append(f"Expected move: ±{expected_move_pct:.2f}% based on straddle price.")
    return strategies, " | ".join(rationale), event_warning

@st.cache_data
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
        st.warning(f"⚠️ Error fetching funds and margin: {res.status_code} - {res.text}")
        return {"available_margin": 0, "used_margin": 0, "total_funds": 0}
    except Exception as e:
        st.error(f"⚠️ Exception in get_funds_and_margin: {e}")
        return {"available_margin": 0, "used_margin": 0, "total_funds": 0}

def calculate_sharpe_ratio():
    try:
        daily_returns = np.random.normal(0.001, 0.01, 252)
        annual_return = np.mean(daily_returns) * 252
        annual_volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.06 / 252) / annual_volatility
        return round(sharpe_ratio, 2)
    except Exception as e:
        st.warning(f"⚠️ Exception in calculate_sharpe_ratio: {e}")
        return 0

def calculate_strategy_margin(config, strategy_details):
    try:
        instruments = [
            {
                "instrument_key": order["instrument_key"],
                "quantity": abs(order["quantity"]),
                "transaction_type": order["transaction_type"],
                "product": "D"
            }
            for order in strategy_details["orders"]
        ]
        url = f"{config['base_url']}/charges/margin"
        res = requests.post(url, headers=config['headers'], json={"instruments": instruments})
        if res.status_code == 200:
            data = res.json().get("data", {})
            total_margin = 0
            if isinstance(data, list):
                total_margin = sum(item.get("total_margin", 0) for item in data)
            elif isinstance(data, dict):
                margins = data.get("margins", [])
                if isinstance(margins, list):
                    total_margin = sum(item.get("total_margin", 0) for item in margins)
                else:
                    total_margin = data.get("required_margin", 0) or data.get("final_margin", 0)
            return round(total_margin, 2)
        st.warning(f"⚠️ Failed to calculate margin: {res.status_code} - {res.text}")
        return 0
    except Exception as e:
        st.warning(f"⚠️ Error calculating strategy margin: {e}")
        return 0

def place_multi_leg_orders(config, orders):
    try:
        sorted_orders = sorted(orders, key=lambda x: 0 if x["transaction_type"] == "BUY" else 1)
        payload = []
        for idx, order in enumerate(sorted_orders):
            correlation_id = f"s{idx}_{int(time.time()) % 1000000}"
            payload.append({
                "quantity": abs(order["quantity"]),
                "product": "D",
                "validity": "DAY",
                "price": order.get("current_price", 0),
                "tag": f"{order['instrument_key']}_leg_{idx}",
                "slice": False,
                "instrument_token": order["instrument_key"],
                "order_type": "MARKET",
                "transaction_type": order["transaction_type"],
                "disclosed_quantity": 0,
                "trigger_price": 0,
                "is_amo": False,
                "correlation_id": correlation_id
            })
        url = f"{config['base_url']}/order/multi/place"
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
        url = f"{config['base_url'].replace('v2', 'v3')}/order/gtt/place"
        payload = {
            "type": "SINGLE",
            "quantity": config["lot_size"],
            "product": "D",
            "rules": [{
                "strategy": "ENTRY",
                "trigger_type": "ABOVE" if transaction_type == "SELL" else "BELOW",
                "trigger_price": trigger_price
            }],
            "instrument_token": instrument_token,
            "transaction_type": transaction_type,
            "tag": tag
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

# --- STRATEGY CALCULATIONS ---
def get_strategy_details(strategy_name, option_chain, spot_price, config, lots=1):
    func_map = {
        "Iron Strike": _iron_fly_calc,
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
        detail = func_map[strategy_name](option_chain, spot_price, config, lots=lots)
    except Exception as e:
        st.warning(f"⚠️ Error calculating {strategy_name} details: {e}")
        return None
    if detail:
        ltp_map = {}
        for opt in option_chain:
            if "call_options" in opt and "market_data" in opt["call_options"]:
                ltp_map[opt["call_options"]["instrument_key"]] = opt["call_options"]["market_data"].get("ltp", 0)
            if "put_options" in opt and "market_data" in opt["put_options"]:
                ltp_map[opt["put_options"]["instrument_key"]] = opt["put_options"]["market_data"].get("ltp", 0)
        updated_orders = []
        prices = {}
        for order in detail["orders"]:
            key = order["instrument_key"]
            ltp = ltp_map.get(key, 0)
            prices[key] = ltp
            updated_orders.append({**order, "current_price": ltp})
        detail["orders"] = updated_orders
        detail["pricing"] = prices
        premium = 0
        for order in detail["orders"]:
            price = order["current_price"]
            qty = order["quantity"]
            if order["transaction_type"] == "SELL":
                premium += price * qty
            else:
                premium -= price * qty
        detail["premium"] = premium / config["lot_size"]
        detail["premium_total"] = premium
        wing_width = 0
        if strategy_name == "Iron Strike":
            wing_width = abs(detail["strikes"][0] - detail["strikes"][2])
            detail["max_loss"] = (wing_width - detail["premium"]) * config["lot_size"] * lots if premium > 0 else float('inf')
        elif strategy_name == "Iron Condor":
            wing_width = abs(detail["strikes"][2] - detail["strikes"][0])
            detail["max_loss"] = (wing_width - detail["premium"]) * config["lot_size"] * lots if premium > 0 else float('inf')
        elif strategy_name == "Jade Lizard":
            wing_width = abs(detail["strikes"][1] - detail["strikes"][2])
            detail["max_loss"] = (wing_width - detail["premium"]) * config["lot_size"] * lots if premium > 0 else float('inf')
        elif strategy_name == "Bull Put Spread":
            wing_width = abs(detail["strikes"][0] - detail["strikes"][1])
            detail["max_loss"] = (wing_width - detail["premium"]) * config["lot_size"] * lots if premium > 0 else float('inf')
        elif strategy_name in ["Straddle", "Wide Strangle", "ATM Strangle"]:
            detail["max_loss"] = float("inf")
        elif strategy_name == "Calendar Spread":
            detail["max_loss"] = detail["premium"]
            detail["max_profit"] = float("inf")
        detail["max_profit"] = detail["premium_total"] if strategy_name != "Calendar Spread" else float("inf")
    return detail

def find_option_by_strike(option_chain, strike, option_type):
    try:
        for opt in option_chain:
            if abs(opt["strike_price"] - strike) < 0.01:
                if option_type == "CE" and "call_options" in opt:
                    return opt["call_options"]
                elif option_type == "PE" and "put_options" in opt:
                    return opt["put_options"]
        st.warning(f"⚠️ No option found for strike {strike} {option_type}")
        return None
    except Exception as e:
        st.warning(f"⚠️ Exception in find_option_by_strike: {e}")
        return None

def get_dynamic_wing_distance(ivp, straddle_price):
    if ivp >= 80:
        multiplier = 0.35
    elif ivp <= 20:
        multiplier = 0.2
    else:
        multiplier = 0.25
    raw_distance = straddle_price * multiplier
    return int(round(raw_distance / 50.0)) * 50

def _iron_fly_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    straddle_price = atm["call_options"]["market_data"]["ltp"] + atm["put_options"]["market_data"]["ltp"]
    ivp = load_ivp(config, atm["call_options"]["option_greeks"]["iv"])
    wing_distance = get_dynamic_wing_distance(ivp, straddle_price)
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike, "PE")
    ce_long_opt = find_option_by_strike(option_chain, strike + wing_distance, "CE")
    pe_long_opt = find_option_by_strike(option_chain, strike - wing_distance, "PE")
    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        st.error("⚠️ Invalid options for Iron Strike.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Iron Strike", "strikes": [strike + wing_distance, strike, strike - wing_distance], "orders": orders}

def _iron_condor_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    straddle_price = atm["call_options"]["market_data"].get("ltp", 0) + atm["put_options"]["market_data"].get("ltp", 0)
    ivp = load_ivp(config, atm["call_options"]["option_greeks"].get("iv", 0))
    short_wing_distance = get_dynamic_wing_distance(ivp, straddle_price)
    long_wing_distance = int(round(short_wing_distance * 1.5 / 50)) * 50
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
    return {"strategy": "Iron Condor", "strikes": [strike + long_wing_distance, strike + short_wing_distance, strike - short_wing_distance, strike - long_wing_distance], "orders": orders}

def _jade_lizard_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 50
    put_strike = atm["strike_price"] - 50
    put_long_strike = atm["strike_price"] - 100
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
    atm = min(option_chain, key=lambda x: abs(x["strike_price"])
    return {"strategy": "Iron Fly"}
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
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    near_leg = find_option_by_strike(option_chain, strike, "CE")
    far_leg = find_option_by_strike(option_chain, strike, "CE")
    if not all([near_leg, far_leg]):
        st.error("⚠️ Invalid options for Calendar Spread.")
        return None
    orders = [
        {"instrument_key": near_leg["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": far_leg["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Calendar Spread", "strikes": [strike], "orders": orders}

def _bull_put_spread_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    short_strike = atm["strike_price"] - 50
    long_strike = atm["strike_price"] - 100
    pe_short_opt = find_option_by_strike(option_chain, short_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, long_strike, "PE")
    if not all([pe_short_opt, pe_long_opt]):
        st.error("⚠️ Invalid options for Bull Put Spread.")
        return None
    orders = [
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Bull Put Spread", "strikes": [short_strike, long_strike], "orders": orders}

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
    return {"strategy": "Wide Strangle", "strikes": [call_strike, put_strike], "orders": orders}

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
    return {"strategy": "ATM Strangle", "strikes": [call_strike, put_strike], "orders": orders}

# --- Evaluate Risk ---
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
                "P&L": 0, "Vega": 0, "Risk OK": "✅"
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
                risk_factor = 1.2 if regime_label == "🔥 High Volatility Trend" else 0.8 if regime_label == "📉 Low Volatility" else 1.0
                max_cap = cfg["capital_pct"] * total_funds
                max_risk = cfg["risk_per_trade_pct"] * max_cap * risk_factor
                risk_ok = "✅" if potential_risk <= max_risk else "❌"
                strategy_summary.append({
                    "Strategy": strat,
                    "Capital Used": capital_used,
                    "Cap Limit": round(max_cap),
                    "% Used": round(capital_used / max_cap * 100, 2) if max_cap else 0,
                    "Potential Risk": potential_risk,
                    "Risk Limit": round(max_risk),
                    "P&L": pnl,
                    "Vega": vega,
                    "Risk OK": risk_ok
                })
                total_cap_used += capital_used
                total_risk_used += potential_risk
                total_realized_pnl += pnl
                total_vega += vega
                if not risk_ok:
                    flags.append(f"❌ {strat} exceeded risk limit")
                if sl_hit:
                    flags.append(f"⚠️ {strat} indicates possible revenge trading")

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
    ""

except Exception as e:
        st.error(f"⚠️ Exception in evaluate_full_risk: {e}")
        return pd.DataFrame([{
            "Strategy": "None", "Capital Used": 0, "Cap Limit": 2000000, "% Used": 0,
            "Potential Risk": 0, "Risk Limit": 20000, "P&L": 0, "Vega": 0, "Risk OK": "✅"
        }]), {
            "Total Funds": 2000000,
            "Capital Deployed": 0,
            "Exposure Percent": 0,
            "Risk on Table": 0,
            "Risk Percent": 0,
            "Daily Risk Limit": 40000,
            "Weekly Risk Limit": 60000,
            "Realized P&L": 0,
            "Drawdown ₹": 0,
            "Drawdown Percent": 0,
            "Max Drawdown Allowed": 40000,
            "Flags": []
        }

def fetch_trade_data(config, full_chain_df):
    try:
        url_positions = f"https{config['base_url']/portfolio/short-term/positions"}
        res_positions = requests.get(url_positions, headers=config['headers'])
        positions = res_positions.json().get("data", []) if res_positions.status_code == 200 else []
        trade_counts = {}
        for pos in positions:
            instrument = pos.get("instrument_token", "")
            strategy = "Unknown"
            if pos.get("product") == "D" and pos.get("quantity") < 0 and pos.get("average_price") > 0:
                if "CE" in instrument or "PE" in instrument:
                    strategy = "Straddle"
                else:
                    strategy = "Iron Condor"
            trade_counts[strategy] = trade_counts.get(strategy, 0) + 1
        trades_df_list = []
        for pos in positions:
            instrument = pos.get("instrument_token", "")
            strategy = "Unknown"
            if pos.get("product") == "D":
                if pos.get("quantity") < 0 and pos.get("average_price") > 0:
                    if "CE" in instrument or "PE" in instrument:
                        strategy = "Straddle"
                    else:
                        strategy = "Iron Condor"
                capital_used = pos["quantity"] * pos["average_price"]
                trades_df_list.append({
                    "strategy": strategy,
                    "capital_used": abs(capital_used),
                    "potential_loss": abs(capital_used * 0.1),
                    "realized_pnl": pos["pnl"],
                    "trades_to_sl_hit": trade_counts.get(strategy, 0),
                    "sl_hit": pos["pnl"] < -abs(capital_used * 0.05),
                    "vega": full_chain_df["Total Vega"].mean() if not full_chain_df.empty else 0,
                    "instrument_token": instrument
                })
        return pd.DataFrame(trades_df_list) if trades_df_list else pd.DataFrame()
    except Exception as e:
        st.error(f"⚠️ Exception in fetch_trade_data: {e}")
        return pd.DataFrame()

# --- PLOTTING ---
def plot_allocation_pie(strategy_df, config):
    if strategy_df.empty or strategy_df["Capital Used"].sum() == 0:
        st.info("No strategy data to plot allocation.")
        return
    strategy_df = strategy_df[strategy_df["Capital Used"] > 0]
    fig = px.pie(
        strategy_df,
        names="Strategy",
        values="Capital Used",
        title="Capital Allocation by Strategy",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig.update_layout(
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white",
        title_font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_drawdown_trend(portfolio_summary):
    np.random.seed(42)
    drawdowns = np.cumsum(np.random.normal(-1000, 5000, 30))
    drawdowns = np.maximum(drawdowns, -portfolio_summary["Max Drawdown Allowed"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(30)),
        y=drawdowns,
        mode='lines',
        name='Drawdown',
        line=dict(color='#00BFFF')
    ))
    fig.add_hline(
        y=-hline(y= -portfolio_summary["Max Drawdown Allowed"],
        line_dash="dash",
        line_color="red",
        name="Max Drawdown"
    )
    fig.update_layout(
        title="Drawdown Trend (₹)"),
        xaxis_title="Days",
        yaxis_title="Drawdown (₹)"),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white",
        title_font_color="white",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_margin_gauge(funds_data):
    used_pct = (funds_data["used_margin"] / funds_data["total_funds"] * 100) if funds_data["total_funds"] else 0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=used_pct,
            title={'text': "Margin Utilization (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00BFFF"},
                'bgcolor': "#1A1C24",
                'bordercolor': "white"
            }
        ))
    fig.update_layout(
        paper_bgcolor="#0E1117",
        font_color="white",
        title_font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_vol_comparison(seller, hv_7, garch_7d, xgb_vol):
    fig = go.Figure(data=[
        go.Bar(
            x=['ATM IV', 'Realized Vol', 'GARCH Vol', 'XGBoost Vol'],
            y=[seller.get('avg_iv', 0), hv_7, garch_7d, xgb_vol],
            marker_color=['#00BFFF', '#FF6347', '#32CD32', '#FFD700']
        )
    ])
    fig.update_layout(
        title="Volatility Comparison (%)",
        xaxis_title="Volatility Type",
        yaxis_title="Annualized Volatility (%)",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white",
        title_font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_chain_analysis(full_chain_df):
    if full_chain_df.empty:
        st.info("No option chain data to plot.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=full_chain_df["Strike"],
        y=full_chain_df["IV Skew"],
        mode='lines',
        name='IV Skew',
        line=dict(color='#00BFFF')
    ))
    fig.add_trace(go.Scatter(
        x=full_chain_df["Strike"],
        y=full_chain_df["Total OI"],
        mode='lines',
        name='Total OI',
        yaxis="y2",
        line=dict(color='#FF6347')
    ))
    fig.update_layout(
        title="Option Chain: IV Skew vs Total OI",
        xaxis_title="Strike",
        yaxis=dict(title="IV Skew", titlefont=dict(color="#00BFFF"), tickfont=dict(color="#00BFFF")),
        yaxis2=dict(title="Total OI", titlefont=dict(color="#FF6347"), tickfont=dict(color="#FF6347"), overlaying="y", side="right"),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white",
        title_font_color="white",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_payoff(orders, spot_price, wing_width):
    prices = np.arange(spot_price - wing_width * 2, spot_price + wing_width * 2, 10)
    payoff = np.zeros_like(prices, dtype=float)
    for order in orders:
        try:
            strike = float(order["instrument_key"].split("_")[-1])
            price = order.get("current_price", 0)
            qty = order["quantity"]
            mult = 1 if order["transaction_type"] == "SELL" else -1
            if "CE" in order["instrument_key"]:
                payoff += mult * qty * np.maximum(prices - strike, 0) - mult * qty * price
            elif "PE" in order["instrument_key"]:
                payoff += mult * qty * np.maximum(strike - prices, 0) - mult * qty * price
        except Exception as e:
            st.warning(f"⚠️ Error calculating payoff for {order['instrument_key']}: {e}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices,
        y=payoff,
        mode='lines',
        name='Payoff',
        line=dict(color='#00BFFF')
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Strategy Payoff",
        xaxis_title="Underlying Price",
        yaxis_title="P&L",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="white",
        title_font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- UI COMPONENTS ---
def render_login():
    st.markdown("""
    <style>
    .login-container { background-color: #1A1C24; padding: 20px; border-radius: 10px; }
    .login-title { font-size: 1.5em; color: #00BFFF; margin-bottom: 20px; }
    </style>
    <div class='login-container'>
        <div class='login-title'>🔐 Login to VolGuard</div>
    </div>
    """, unsafe_allow_html=True)
    with st.form("login_form"):
        access_token = st.text_input("Upstox Access Token", type="password", value=st.session_state.access_token)
        submit = st.form_submit_button("Login")
        if submit and access_token:
            config = get_config(access_token)
            test_url = f"{config['base_url']}/user/profile"
            try:
                res = requests.get(test_url, headers=config['headers'])
                if res.status_code == 200:
                    st.session_state.access_token = access_token
                    st.session_state.logged_in = True
                    st.success("✅ Logged in successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Invalid token: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"⚠️ Error validating token: {e}")
        elif submit:
            st.error("❌ Please enter an access token.")
    if st.session_state.logged_in:
        if st.button("Logout", key="logout_button"):
            st.session_state.logged_in = False
            st.session_state.access_token = ""
            st.cache_data.clear()
            st.rerun()

def render_sidebar():
    st.sidebar.markdown("""
    <style>
    .sidebar-header { font-size: 1.8em; color: #00BFFF; margin-bottom: 20px; }
    </style>
    <div class='sidebar-header'>🚀 VolGuard</div>
    <div style='color: white; margin-bottom: 20px;'>Options Trading Dashboard</div>
    """, unsafe_allow_html=True)
    
    with st.sidebar.expander("⚙️ Settings", expanded=True):
        st.session_state.expiry_type = st.radio(
            "📅 Expiry Type",
            ["Weekly", "Monthly"],
            index=0 if st.session_state.get("expiry_type", "Weekly") == "Weekly" else 1,
            key="expiry_type_selector"
        )
    
    st.sidebar.markdown("---")
    tabs = ["Dashboard", "Option Chain", "Strategies", "Portfolio", "Orders", "Journal"]
    st.session_state.active_tab = st.sidebar.radio("📋 Navigate", tabs, index=tabs.index(st.session_state.active_tab))

def render_dashboard(option_chain, spot_price, vix, nifty, seller, full_chain_df, market, ivp, hv_7, garch_7d, xgb_vol, iv_rv_spread, iv_skew_slope, regime_score, regime, regime_note, regime_explanation, event_df):
    st.markdown("<h1 style='text-align: center; color: white;'>📊 Market Dashboard</h1>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-box'><h3>Nifty 50</h3><div class='value'>₹{nifty:.2f}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-box'><h3>India VIX</h3><div class='value'>{vix:.2f}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-box'><h3>ATM Strike</h3><div class='value'>{seller.get('strike', 0):.0f}</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-box'><h3>Straddle Price</h3><div class='value'>₹{seller.get('straddle_price', 0):.2f}</div></div>", unsafe_allow_html=True)
    
    st.subheader("📈 Volatility Overview")
    plot_vol_comparison(seller, hv_7, garch_7d, xgb_vol)
    
    col_metrics = st.columns(3)
    with col_metrics[0]:
        st.markdown(f"<div class='metric-box'><h4>IV-RV Spread</h4>{iv_rv_spread:+.2f}%</div>", unsafe_allow_html=True)
    with col_metrics[1]:
        st.markdown(f"<div class='metric-box'><h4>IV Skew Slope</h4>{iv_skew_slope:.4f}</div>", unsafe_allow_html=True)
    with col_metrics[2]:
        st.markdown(f"<div class='metric-box'><h4>Expiry Date</h4>{config['expiry_date']}</div>", unsafe_allow_html=True)
    
    st.subheader("📍 Market Regime")
    st.markdown(f"<div class='metric-box'><h3>{regime}</h3><p>Score: {regime_score:.2f}<br>{regime_note}<br><i>{regime_explanation}</i></p></div>", unsafe_allow_html=True)
    
    st.subheader("🔔 Upcoming Events")
    if not event_df.empty:
        st.dataframe(event_df.style.set_properties(**{'background-color': '#1A1C24', 'color': 'white'}), use_container_width=True)
    else:
        st.info("No events before expiry.")

def render_option_chain(option_chain, spot_price, seller):
    st.subheader("📊 Option Chain")
    all_expiries = sorted(set(c["expiry"][:10] for c in option_chain))
    expiry_override = st.selectbox("Choose Expiry", all_expiries, index=0)
    filtered_chain = [c for c in option_chain if c["expiry"][:10] == expiry_override]
    chain_df = full_chain_table(filtered_chain, spot_price)
    
    if not chain_df.empty:
        plot_chain_analysis(chain_df)
        st.dataframe(chain_df.style.set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
        st.markdown(download_df(chain_df, f"option_chain_{expiry_override}.csv", "📥 Download Option Chain CSV"), unsafe_allow_html=True)
    else:
        st.warning("⚠️ No data for selected expiry.")

def render_strategies(strategies, strategy_rationale, event_warning, option_chain, spot_price, config, regime_score):
    st.subheader("📈 Strategy Recommendations")
    if strategies:
        st.success(f"**Suggested Strategies:** {', '.join(strategies)}")
        st.info(f"**Rationale:** {strategy_rationale}")
        if event_warning:
            st.warning(event_warning)
        for strat in strategies:
            with st.expander(f"🔍 {strat} Details"):
                detail = get_strategy_details(strat, option_chain, spot_price, config, lots=st.session_state.selected_lots)
                if detail:
                    order_df = pd.DataFrame({
                        "Instrument": [o["instrument_key"] for o in detail["orders"]],
                        "Type": [o["transaction_type"] for o in detail["orders"]],
                        "Quantity": [abs(o["quantity"]) for o in detail["orders"]],
                        "Price": [o.get("current_price", 0) for o in detail["orders"]]
                    })
                    st.dataframe(order_df.style.set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
                    st.markdown(f"**Premium per Lot:** ₹{detail['premium']:.2f}")
                    st.markdown(f"**Total Premium:** ₹{detail['premium_total']:.2f}")
                    st.markdown(f"**Max Profit:** ₹{detail['max_profit']:.2f}")
                    st.markdown(f"**Max Loss:** {'Unlimited' if detail['max_loss'] == float('inf') else f'₹{detail['max_loss']:.2f}'}")
                    margin = calculate_strategy_margin(config, detail)
                    st.markdown(f"**Required Margin:** ₹{margin:.2f}")
                    plot_payoff(detail["orders"], spot_price, max(abs(detail["strikes"][0] - spot_price), abs(detail["strikes"][-1] - spot_price)))
                    if st.button(f"🚀 Execute {strat}", key=f"execute_{strat}"):
                        if place_multi_leg_orders(config, detail["orders"]):
                            trade_data = {
                                "strategy": strat,
                                "instrument_token": ", ".join([o["instrument_key"] for o in detail["orders"]]),
                                "entry_price": detail["premium"],
                                "quantity": config["lot_size"] * st.session_state.selected_lots,
                                "realized_pnl": 0,
                                "unrealized_pnl": 0,
                                "regime_score": regime_score
                            }
                            log_trade(trade_data)
                            st.success(f"✅ Trade logged for {strat}")
                else:
                    st.warning(f"⚠️ Unable to calculate details for {strat}")
    else:
        st.warning("⚠️ No strategies recommended.")

def render_portfolio(trades_df, config, regime_label, vix, funds_data):
    st.subheader("📈 Portfolio")
    strategy_df, portfolio_summary = evaluate_full_risk(trades_df, config, regime_label, vix)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-box'><h3>Total Funds</h3>₹{portfolio_summary['Total Funds']:.2f}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-box'><h3>Capital Deployed</h3>₹{portfolio_summary['Capital Deployed']:.2f}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-box'><h3>Exposure</h3>{portfolio_summary['Exposure Percent']:.2f}%</div>", unsafe_allow_html=True)
    
    st.subheader("📊 Allocation")
    plot_allocation_pie(strategy_df, config)
    
    st.subheader("📉 Drawdown")
    plot_drawdown_trend(portfolio_summary)
    
    st.subheader("📊 Margin")
    plot_margin_gauge(funds_data)
    
    st.subheader("🔍 Risk Details")
    st.dataframe(strategy_df.style.set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
    
    if portfolio_summary["Flags"]:
        st.warning("⚠️ Risk Flags: " + " | ".join(portfolio_summary["Flags"]))
    
    st.markdown(download_excel(strategy_df, "portfolio_summary.xlsx", "Portfolio Summary", "📥 Download Portfolio Excel"), unsafe_allow_html=True)

def render_orders(config, option_chain, spot_price):
    st.subheader("📝 Orders")
    strategy = st.selectbox("Select Strategy", all_strategies, index=0, key="order_strategy")
    lots = st.number_input("Number of Lots", min_value=1, value=1, step=1, key="order_lots")
    detail = get_strategy_details(strategy, option_chain, spot_price, config, lots)
    
    if detail:
        st.markdown("**Order Summary**")
        order_df = pd.DataFrame({
            "Instrument": [o["instrument_key"] for o in detail["orders"]],
            "Type": [o["transaction_type"] for o in detail["orders"]],
            "Quantity": [abs(o["quantity"]) for o in detail["orders"]],
            "Price": [o.get("current_price", 0) for o in detail["orders"]]
        })
        st.dataframe(order_df.style.set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
        st.markdown(f"**Total Premium:** ₹{detail['premium_total']:.2f}")
        margin = calculate_strategy_margin(config, detail)
        st.markdown(f"**Required Margin:** ₹{margin:.2f}")
        plot_payoff(detail["orders"], spot_price, max(abs(detail["strikes"][0] - spot_price), abs(detail["strikes"][-1] - spot_price)))
        
        if st.button("🚀 Place Order", key="place_order"):
            if place_multi_leg_orders(config, detail["orders"]):
                trade_data = {
                    "strategy": strategy,
                    "instrument_token": ", ".join([o["instrument_key"] for o in detail["orders"]]),
                    "entry_price": detail["premium"],
                    "quantity": config["lot_size"] * lots,
                    "realized_pnl": 0,
                    "unrealized_pnl": 0,
                    "regime_score": 0
                }
                log_trade(trade_data)
                st.success("✅ Order placed and logged!")

def render_journal():
    st.subheader("📖 Journal")
    with st.form("journal_form"):
        title = st.text_input("Entry Title")
        content = st.text_area("Entry Content")
        mood = st.selectbox("Mood", ["Positive 😊", "Neutral 😐", "Negative 😟"])
        tags = st.text_input("Tags (comma-separated)")
        if st.form_submit_button("Add Entry"):
            entry_data = {
                "title": title,
                "content": content,
                "mood": mood,
                "tags": tags
            }
            if add_journal_entry(entry_data):
                st.success("✅ Journal entry added!")
    
    st.subheader("📜 Entries")
    journal_df = journals_to_dataframe()
    if not journal_df.empty:
        st.dataframe(journal_df.style.set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
        st.markdown(download_df(journal_df, "journal_entries.csv", "📥 Download Journal CSV"), unsafe_allow_html=True)
    else:
        st.info("No journal entries yet.")

# --- MAIN APP ---
def main():
    st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    .metric-box { background-color: #1A1C24; padding: 15px; border-radius: 10px; text-align: center; color: white; margin-bottom: 10px; }
    .metric-box h3, .metric-box h4 { color: #00BFFF; margin: 0; }
    .metric-box .value { font-size: 1.5em; color: white; }
    .stTabs [data-baseweb="tab-list"] { background-color: #1A1C24; }
    .stTabs [data-baseweb="tab"] { color: white; }
    .stTabs [data-baseweb="tab"]:hover { background-color: #2E2F38; }
    .stTabs [aria-selected="true"] { background-color: #00BFFF; color: white; }
    .stDataFrame { background-color: #1A1C24; color: white; }
    .stSelectbox > div, .stTextInput > div > div > input, .stNumberInput > div > div > input {
        background-color: #2E2F38; color: white; border: 1px solid #00BFFF; border-radius: 5px; }
    .stButton > button { background-color: #00BFFF; color: white; border-radius: 5px; padding: 10px 20px; }
    .stButton > button:hover { background-color: #0099CC; }
    .stExpander { background-color: #1A1C24; }
    .stExpander > div > div > label { color: white; }
    </style>
    """, unsafe_allow_html=True)
    
    st_autorefresh(interval=60 * 1000, key="data_refresh")
    
    if not st.session_state.logged_in:
        render_login()
        return
    
    global config
    config = get_config(st.session_state.access_token)
    render_sidebar()
    
    with st.spinner("Fetching market data..."):
        option_chain = fetch_option_chain(config)
        vix, nifty = get_indices_quotes(config)
        event_df = load_upcoming_events(config)
        funds_data = get_funds_and_margin(config)
        model = load_xgboost_model()
    
    if not option_chain or nifty is None:
        st.error("⚠️ Failed to fetch market data.")
        return
    
    spot_price = nifty
    seller = extract_seller_metrics(option_chain, spot_price)
    full_chain_df = full_chain_table(option_chain, spot_price)
    market = market_metrics(option_chain, config['expiry_date'])
    hv_7, garch_7d, iv_rv_spread = calculate_volatility(config, seller.get('avg_iv', 0))
    ivp = load_ivp(config, seller.get('avg_iv', 0))
    iv_skew_slope = calculate_iv_skew_slope(full_chain_df)
    xgb_vol = predict_xgboost_volatility(model, seller.get('avg_iv', 0), hv_7, ivp, market['pcr'], vix, market['days_to_expiry'], garch_7d)
    regime_score, regime, regime_note, regime_explanation = calculate_regime(
        seller.get('avg_iv', 0), ivp, hv_7, garch_7d, seller.get('straddle_price', 0), spot_price, market['pcr'], vix, iv_skew_slope
    )
    strategies, strategy_rationale, event_warning = suggest_strategy(
        regime, ivp, iv_rv_spread, market['days_to_expiry'], event_df, config['expiry_date'], seller.get('straddle_price', 0), spot_price
    )
    trades_df = fetch_trade_data(config, full_chain_df)
    
    tab_functions = {
        "Dashboard": render_dashboard,
        "Option Chain": render_option_chain,
        "Strategies": render_strategies,
        "Portfolio": render_portfolio,
        "Orders": render_orders,
        "Journal": render_journal
    }
    
    active_tab = st.session_state.active_tab
    if active_tab == "Dashboard":
        render_dashboard(option_chain, spot_price, vix, nifty, seller, full_chain_df, market, ivp, hv_7, garch_7d, xgb_vol, iv_rv_spread, iv_skew_slope, regime_score, regime, regime_note, regime_explanation, event_df)
    elif active_tab == "Option Chain":
        render_option_chain(option_chain, spot_price, seller)
    elif active_tab == "Strategies":
        render_strategies(strategies, strategy_rationale, event_warning, option_chain, spot_price, config, regime_score)
    elif active_tab == "Portfolio":
        render_portfolio(trades_df, config, regime, vix, funds_data)
    elif active_tab == "Orders":
        render_orders(config, option_chain, spot_price)
    elif active_tab == "Journal":
        render_journal()

if __name__ == "__main__":
    main()
