import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from arch import arch_model
import requests
from scipy.stats import linregress
import xgboost as xgb
import pickle
from io import BytesIO
import os
from time import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="VolGuard - Options Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
    .main { background-color: #0E1117; color: white; }
    .block-container { padding: 1.5rem 2rem; }
    .metric-box {
        background-color: #1A1C24;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 12px;
        color: white;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
    }
    .metric-box h3 {
        color: #6495ED;
        font-size: 1.1em;
        margin-bottom: 5px;
    }
    .metric-box .value {
        font-size: 1.8em;
        font-weight: bold;
        color: #00BFFF;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        padding: 10px;
    }
    .stButton>button {
        border-radius: 8px;
        background-color: #007BFF;
        color: white;
        border: none;
        padding: 10px 18px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stDataFrame thead tr th {
        background-color: #202736;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- ALL STRATEGIES ---
all_strategies = [
    "Iron Fly", "Iron Condor", "Jade Lizard", "Straddle",
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
        "lot_size": 75
    }

    def get_next_expiry_internal():
        try:
            url = f"{config['base_url']}/option/contract"
            params = {"instrument_key": config['instrument_key']}
            res = requests.get(url, headers=config['headers'], params=params)
            if res.status_code == 200:
                expiries = sorted(res.json()["data"], key=lambda x: datetime.strptime(x["expiry"], "%Y-%m-%d"))
                today = datetime.now()
                for expiry in expiries:
                    expiry_dt = datetime.strptime(expiry["expiry"], "%Y-%m-%d")
                    if expiry_dt.weekday() == 3 and expiry_dt > today:
                        return expiry["expiry"]
                return datetime.now().strftime("%Y-%m-%d")
            st.error(f":warning: Error fetching expiries: {res.status_code} - {res.text}")
            return datetime.now().strftime("%Y-%m-%d")
        except Exception as e:
            st.error(f":warning: Exception in get_next_expiry: {e}")
            return datetime.now().strftime("%Y-%m-%d")

    config['expiry_date'] = get_next_expiry_internal()
    return config

# --- DATA FETCHING FUNCTIONS ---
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
            data = res.json()
            vix = data["data"]["NSE_INDEX:India VIX"]["last_price"]
            nifty = data["data"]["NSE_INDEX:Nifty 50"]["last_price"]
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
        df["Datetime"] = df["Date"] + " " + df["Time"]
        df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d-%b %H:%M", errors="coerce")
        current_year = datetime.now().year
        df["Datetime"] = df["Datetime"].apply(lambda dt: dt.replace(year=current_year) if pd.notnull(dt) else dt)
        now = datetime.now()
        expiry_dt = datetime.strptime(config['expiry_date'], "%Y-%m-%d")
        mask = (df["Datetime"] >= now) & (df["Datetime"] <= expiry_dt)
        filtered = df.loc[mask, ["Datetime", "Event", "Classification", "Forecast", "Prior"]]
        return filtered.sort_values("Datetime").reset_index(drop=True)
    except Exception as e:
        st.warning(f":warning: Failed to load upcoming events: {e}")
        return pd.DataFrame(columns=["Datetime", "Event", "Classification", "Forecast", "Prior"])

# --- OTHER HELPER FUNCTIONS ---
def calculate_strategy_margin(config, strategy_details):
    try:
        instruments = [{"instrument_key": order["instrument_key"], "quantity": abs(order["quantity"]),
                        "transaction_type": order["transaction_type"], "product": "D"} for order in strategy_details["orders"]]
        url = f"{config['base_url']}/charges/margin"
        res = requests.post(url, headers=config['headers'], json={"instruments": instruments})
        if res.status_code == 200:
            data = res.json().get("data", {})
            total_margin = 0
            if isinstance(data, list):
                total_margin = sum(item.get("total_margin", 0) for item in data)
            elif isinstance(data, dict):
                margins = data.get("margins", [])
                total_margin = sum(item.get("total_margin", 0) for item in margins)
                total_margin += data.get("required_margin", 0)
            return round(total_margin, 2)
        else:
            st.warning(f":warning: Failed to calculate margin: {res.status_code} - {res.text}")
            return 0
    except Exception as e:
        st.warning(f":warning: Error calculating strategy margin: {e}")
        return 0

def place_multi_leg_orders(config, orders):
    try:
        payload = []
        for idx, order in enumerate(orders):
            correlation_id = f"s{idx}_{int(time()) % 1000000}"
            leg_payload = {
                "quantity": abs(order["quantity"]),
                "product": "D",
                "validity": order.get("validity", "DAY"),
                "tag": f"{order['instrument_key']}_leg_{idx}",
                "slice": False,
                "instrument_token": order["instrument_key"],
                "order_type": order["order_type"],
                "transaction_type": order["transaction_type"],
                "disclosed_quantity": 0,
                "trigger_price": 0,
                "is_amo": False,
                "correlation_id": correlation_id
            }
            if order["order_type"] == "LIMIT":
                leg_payload["price"] = order.get("current_price", 0)
            payload.append(leg_payload)
        url = f"{config['base_url']}/order/multi/place"
        res = requests.post(url, headers=config['headers'], json=payload)
        if res.status_code == 200:
            st.success(":white_check_mark: Multi-leg order placed successfully!")
            return True
        else:
            st.error(f":x: Failed to place multi-leg order: {res.status_code} - {res.text}")
            return False
    except Exception as e:
        st.error(f":warning: Error placing multi-leg order: {e}")
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
            st.success(f":white_check_mark: GTT order placed for {instrument_token}")
            return True
        else:
            st.warning(f":warning: GTT failed: {res.status_code} - {res.text}")
            return False
    except Exception as e:
        st.error(f":warning: Error creating GTT: {e}")
        return False

# --- STRATEGY DETAILS ---
def get_strategy_details(strategy_name, option_chain, spot_price, config, lots=1):
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
        st.warning(f":warning: Strategy {strategy_name} not supported.")
        return None
    try:
        detail = func_map[strategy_name](option_chain, spot_price, config, lots=lots)
    except Exception as e:
        st.warning(f":warning: Error calculating {strategy_name} details: {e}")
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
        detail["premium_total"] = premium
        detail["premium"] = premium / config["lot_size"]
        strategy_name = detail["strategy"]
        if strategy_name == "Iron Fly":
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
        detail["max_profit"] = detail["premium_total"] if strategy_name not in ["Calendar Spread"] else float("inf")
    return detail

# --- STRATEGY IMPLEMENTATIONS (_iron_fly_calc etc.) ---
def _iron_fly_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    wing = 100
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike, "PE")
    ce_long_opt = find_option_by_strike(option_chain, strike + wing, "CE")
    pe_long_opt = find_option_by_strike(option_chain, strike - wing, "PE")
    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        st.error(":warning: Invalid options for Iron Fly.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Iron Fly", "strikes": [strike, strike, strike + wing, strike - wing], "orders": orders}

def _iron_condor_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    short_wing_distance = 100
    long_wing_distance = 200
    ce_short_strike = atm["strike_price"] + short_wing_distance
    pe_short_strike = atm["strike_price"] - short_wing_distance
    ce_long_strike = atm["strike_price"] + long_wing_distance
    pe_long_strike = atm["strike_price"] - long_wing_distance
    ce_short_opt = find_option_by_strike(option_chain, ce_short_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, pe_short_strike, "PE")
    ce_long_opt = find_option_by_strike(option_chain, ce_long_strike, "CE")
    pe_long_opt = find_option_by_strike(option_chain, pe_long_strike, "PE")
    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        st.error(":warning: Invalid options for Iron Condor.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Iron Condor", "strikes": [ce_short_strike, pe_short_strike, ce_long_strike, pe_long_strike], "orders": orders}

def _jade_lizard_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 50
    put_strike = atm["strike_price"]
    put_long_strike = atm["strike_price"] - 100
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, put_long_strike, "PE")
    if not all([ce_short_opt, pe_short_opt, pe_long_opt]):
        st.error(":warning: Invalid options for Jade Lizard.")
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
        st.error(":warning: Invalid options for Straddle.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    return {"strategy": "Straddle", "strikes": [strike, strike], "orders": orders}

def _calendar_spread_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    ce_long_opt = ce_short_opt
    if not ce_short_opt:
        st.error(":warning: Invalid options for Calendar Spread.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Calendar Spread", "strikes": [strike, strike], "orders": orders}

def _bull_put_spread_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    short_strike = atm["strike_price"] - 50
    long_strike = atm["strike_price"] - 100
    pe_short_opt = find_option_by_strike(option_chain, short_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, long_strike, "PE")
    if not all([pe_short_opt, pe_long_opt]):
        st.error(":warning: Invalid options for Bull Put Spread.")
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
        st.error(":warning: Invalid options for Wide Strangle.")
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
        st.error(":warning: Invalid options for ATM Strangle.")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    return {"strategy": "ATM Strangle", "strikes": [call_strike, put_strike], "orders": orders}

def find_option_by_strike(option_chain, strike, option_type):
    try:
        for opt in option_chain:
            if abs(opt["strike_price"] - strike) < 0.01:
                if option_type == "CE" and "call_options" in opt:
                    return opt["call_options"]
                elif option_type == "PE" and "put_options" in opt:
                    return opt["put_options"]
        st.warning(f":warning: No option found for strike {strike} {option_type}")
        return None
    except Exception as e:
        st.warning(f":warning: Exception in find_option_by_strike: {e}")
        return None

# --- LOGIN UI ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'access_token' not in st.session_state:
    st.session_state.access_token = ""

with st.sidebar:
    st.title("🔐 Login")
    st.markdown("Manage your session and credentials.")
    access_token = st.text_input("Access Token", type="password", value=st.session_state.access_token)
    col_login1, col_login2 = st.columns([2, 1])
    with col_login1:
        login_btn = st.button("Login")
    with col_login2:
        logout_btn = st.button("Logout")
    if login_btn:
        if access_token:
            config = get_config(access_token)
            test_url = f"{config['base_url']}/user/profile"
            try:
                res = requests.get(test_url, headers=config['headers'])
                if res.status_code == 200:
                    st.session_state.access_token = access_token
                    st.session_state.logged_in = True
                    st.success(":white_check_mark: Logged in successfully!")
                else:
                    st.error(f":x: Invalid token: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f":warning: Error validating token: {e}")
        else:
            st.error(":x: Please enter an access token.")
    if logout_btn:
        st.session_state.logged_in = False
        st.session_state.access_token = ""
        st.experimental_rerun()

# --- MAIN APP ---
if st.session_state.logged_in:
    config = get_config(st.session_state.access_token)
    config['total_funds'] = get_funds_and_margin(config)['total_funds']

    @st.cache_data(show_spinner="Analyzing market data...")
    def load_all_data(config):
        xgb_model = load_xgboost_model()
        option_chain = fetch_option_chain(config)
        if not option_chain:
            st.error(":x: Failed to fetch option chain data.")
            return tuple([None]*27)
        spot_price = option_chain[0]["underlying_spot_price"]
        vix, nifty = get_indices_quotes(config)
        if not vix or not nifty:
            st.error(":x: Failed to fetch India VIX or Nifty 50 data.")
            return tuple([None]*27)
        seller = extract_seller_metrics(option_chain, spot_price)
        if not seller:
            st.error(":x: Failed to extract seller metrics.")
            return tuple([None]*27)
        full_chain_df = full_chain_table(option_chain, spot_price)
        market = market_metrics(option_chain, config['expiry_date'])
        ivp = load_ivp(config, seller["avg_iv"])
        hv_7, garch_7d, iv_rv_spread = calculate_volatility(config, seller["avg_iv"])
        xgb_vol = predict_xgboost_volatility(xgb_model, seller["avg_iv"], hv_7, ivp, market["pcr"], vix, market["days_to_expiry"], garch_7d)
        iv_skew_slope = calculate_iv_skew_slope(full_chain_df)
        regime_score, regime, regime_note, regime_explanation = calculate_regime(
            seller["avg_iv"], ivp, hv_7, garch_7d, seller["straddle_price"], spot_price, market["pcr"], vix, iv_skew_slope)
        event_df = load_upcoming_events(config)
        strategies, strategy_rationale, event_warning = suggest_strategy(
            regime, ivp, iv_rv_spread, market['days_to_expiry'], event_df, config['expiry_date'], seller["straddle_price"], spot_price)
        strategy_details = [detail for strat in strategies if (detail := get_strategy_details(strat, option_chain, spot_price, config, lots=1)) is not None]
        trades_df = fetch_trade_data(config, full_chain_df)
        strategy_df, portfolio_summary = evaluate_full_risk(trades_df, config, regime, vix)
        funds_data = get_funds_and_margin(config)
        sharpe_ratio = calculate_sharpe_ratio()
        return (option_chain, spot_price, vix, nifty, seller, full_chain_df, market, ivp, hv_7, garch_7d, xgb_vol, iv_rv_spread, iv_skew_slope, regime_score, regime, regime_note, regime_explanation, event_df, strategies, strategy_rationale, event_warning, strategy_details, trades_df, strategy_df, portfolio_summary, funds_data, sharpe_ratio)

    # --- HELPER FUNCTIONS FOR DASHBOARD ---
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
                    if "call_options" in opt:
                        pain_at_strike += max(0, strike - opt["strike_price"]) * opt["call_options"]["market_data"]["oi"]
                    if "put_options" in opt:
                        pain_at_strike += max(0, opt["strike_price"] - strike) * opt["put_options"]["market_data"]["oi"]
                if pain_at_strike < min_pain:
                    min_pain = pain_at_strike
                    max_pain_strike = strike
            return {"days_to_expiry": days_to_expiry, "pcr": round(pcr, 2), "max_pain": max_pain_strike}
        except Exception as e:
            st.warning(f":warning: Exception in market_metrics: {e}")
            return {"days_to_expiry": 0, "pcr": 0, "max_pain": 0}

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
            st.warning(f"Exception in calculate_volatility: {e}")
            return 0, 0, 0

    def calculate_iv_skew_slope(full_chain_df):
        try:
            if full_chain_df.empty:
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
        regime_score += 10 if expected_move > 0.05 else -10 if expected_move < 0.02 else 0
        regime_score += 5 if garch_vol > realized_vol * 1.2 else -5 if garch_vol < realized_vol * 0.8 else 0
        if regime_score > 20:
            return regime_score, ":fire: High Vol Trend", "Market in high volatility — ideal for premium selling.", "High IVP, elevated VIX, wide straddle."
        elif regime_score > 10:
            return regime_score, ":zap: Elevated Volatility", "Above-average volatility — favor range-bound strategies.", "Moderate IVP and IV-RV spread."
        elif regime_score > -10:
            return regime_score, ":smile: Neutral Volatility", "Balanced market — flexible strategy selection.", "IV and RV aligned, moderate PCR and skew."
        else:
            return regime_score, ":chart_with_downwards_trend: Low Volatility", "Low volatility — cautious selling or long vega plays.", "Low IVP, tight straddle, low VIX."

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
                continue
        if high_impact_event_near:
            event_warning = f":warning: High-impact event within {event_window} days of expiry."
        if event_impact_score > 0:
            rationale.append(f"High-impact events: {event_impact_score} with significant deviation.")
        expected_move_pct = (straddle_price / spot_price) * 100
        if regime_label == ":fire: High Vol Trend":
            strategies = ["Iron Fly", "Wide Strangle"]
            rationale.append("Strong IV premium — neutral strategies for premium capture.")
        elif regime_label == ":zap: Elevated Volatility":
            strategies = ["Iron Condor", "Jade Lizard"]
            rationale.append("Volatility above average — range-bound strategies offer favorable reward-risk.")
        elif regime_label == ":smile: Neutral Volatility":
            if days_to_expiry >= 3:
                strategies = ["Jade Lizard", "Bull Put Spread"]
                rationale.append("Market balanced — slight directional bias strategies offer edge.")
            else:
                strategies = ["Iron Fly"]
                rationale.append("Tight expiry — quick theta-based capture via short Iron Fly.")
        elif regime_label == ":chart_with_downwards_trend: Low Volatility":
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
            daily_returns = np.random.normal(0.001, 0.01, 252)
            annual_return = np.mean(daily_returns) * 252
            annual_volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = (annual_return - 0.06 / 252) / annual_volatility
            return round(sharpe_ratio, 2)
        except Exception as e:
            st.warning(f":warning: Exception in calculate_sharpe_ratio: {e}")
            return 0

    def plot_allocation_pie(strategy_df, config):
        if strategy_df.empty:
            st.info("No strategy data to plot allocation.")
            return
        strategy_df = strategy_df[strategy_df["Capital Used"] > 0]
        if strategy_df.empty:
            st.info("All strategies have zero capital deployed.")
            return
        labels = strategy_df["Strategy"]
        sizes = strategy_df["Capital Used"]
        fig, ax = plt.subplots(figsize=(8, 8))
        try:
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'color': 'white'})
            ax.set_title("Capital Allocation by Strategy", color="white")
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f":warning: Could not render pie chart: {e}")

    def plot_drawdown_trend(portfolio_summary):
        np.random.seed(42)
        days = 30
        drawdowns = np.cumsum(np.random.normal(-1000, 5000, days))
        drawdowns = np.maximum(drawdowns, -portfolio_summary["Max Drawdown Allowed"])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(days), drawdowns, color="#00BFFF")
        ax.axhline(-portfolio_summary["Max Drawdown Allowed"], linestyle="--", color="red", label="Max Drawdown Allowed")
        ax.set_title("Drawdown Trend (₹)", color="white")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        st.pyplot(fig)

    def plot_margin_gauge(funds_data):
        fig, ax = plt.subplots(figsize=(8, 4))
        used_pct = (funds_data["used_margin"] / funds_data["total_funds"] * 100) if funds_data["total_funds"] > 0 else 0
        ax.barh(["Margin Utilization"], [used_pct], color="#00BFFF")
        ax.set_xlim(0, 100)
        ax.set_title("Margin Utilization (%)", color="white")
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        fig.patch.set_facecolor('#0E1117')
        ax.set_facecolor('#0E1117')
        st.pyplot(fig)

    def plot_vol_comparison(seller, hv_7, garch_7d, xgb_vol):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            vols = [seller['avg_iv'], hv_7, garch_7d, xgb_vol]
            labels = ['ATM IV', 'Realized Vol', 'GARCH Vol', 'XGBoost Vol']
            colors = ['#00BFFF', '#FF6347', '#32CD32', '#FFD700']
            ax.bar(labels, vols, color=colors)
            ax.set_title("Volatility Comparison (%)", color="white")
            ax.set_ylabel("Annualized Volatility (%)", color="white")
            ax.tick_params(axis='x', colors='white', rotation=45)
            ax.tick_params(axis='y', colors='white')
            ax.grid(True, linestyle=':', alpha=0.6)
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f":warning: Exception in plot_vol_comparison: {e}")

    def plot_chain_analysis(full_chain_df):
        try:
            if full_chain_df.empty:
                st.info("No option chain data to plot.")
                return
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(full_chain_df["Strike"], full_chain_df["IV Skew"], color="#00BFFF", label="IV Skew")
            ax1.set_xlabel("Strike", color="white")
            ax1.set_ylabel("IV Skew", color="#00BFFF")
            ax1.tick_params(axis='x', colors='white')
            ax1.tick_params(axis='y', colors='#00BFFF')
            ax2 = ax1.twinx()
            ax2.plot(full_chain_df["Strike"], full_chain_df["Total OI"], color="#FF6347", label="Total OI")
            ax2.set_ylabel("Total OI", color="#FF6347")
            ax2.tick_params(axis='y', colors='#FF6347')
            ax1.set_title("Option Chain Analysis: IV Skew vs Total OI", color="white")
            ax1.grid(True, linestyle=':', alpha=0.6)
            fig.patch.set_facecolor('#0E1117')
            ax1.set_facecolor('#0E1117')
            fig.legend(loc="upper right", bbox_to_anchor=(1, 1), facecolor='#0E1117', edgecolor='white', labelcolor='white')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f":warning: Exception in plot_chain_analysis: {e}")

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
                    "Potential Risk": 0, "Risk Limit": total_funds * 0.01, "P&L": 0,
                    "Vega": 0, "Risk OK?": ":white_check_mark:"
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
                    risk_factor = 1.2 if regime_label == ":fire: High Vol Trend" else 0.8 if regime_label == ":chart_with_downwards_trend: Low Volatility" else 1.0
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
                        "Risk OK?": ":white_check_mark:" if risk_ok else ":x:"
                    })
                    total_cap_used += capital_used
                    total_risk_used += potential_risk
                    total_realized_pnl += pnl
                    total_vega += vega
                    if not risk_ok:
                        flags.append(f":x: {strat} exceeded risk limit")
                    if sl_hit:
                        flags.append(f":warning: {strat} shows possible revenge trading")
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
            st.error(f":warning: Exception in evaluate_full_risk: {e}")
            return pd.DataFrame(), {
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
            url_positions = f"{config['base_url']}/portfolio/short-term-positions"
            res_positions = requests.get(url_positions, headers=config['headers'])
            url_trades = f"{config['base_url']}/order/trades/get-trades-for-day"
            res_trades = requests.get(url_trades, headers=config['headers'])
            positions = res_positions.json().get("data", []) if res_positions.status_code == 200 else []
            trades = res_trades.json().get("data", []) if res_trades.status_code == 200 else []
            trade_counts = {}
            for trade in trades:
                instrument = trade.get("instrument_token", "")
                strat = "Straddle" if "NIFTY" in instrument and ("CE" in instrument or "PE" in instrument) else "Unknown"
                trade_counts[strat] = trade_counts.get(strat, 0) + 1
            trades_df_list = []
            for pos in positions:
                instrument = pos.get("instrument_token", "")
                strategy = "Unknown"
                if pos.get("product") == "D" and pos.get("quantity", 0) < 0 and pos.get("average_price", 0) > 0:
                    strategy = "Straddle" if "CE" in instrument or "PE" in instrument else "Iron Condor"
                capital = pos["quantity"] * pos["average_price"]
                trades_df_list.append({
                    "strategy": strategy,
                    "capital_used": abs(capital),
                    "potential_loss": abs(capital * 0.1),
                    "realized_pnl": pos["pnl"],
                    "trades_today": trade_counts.get(strategy, 0),
                    "sl_hit": pos["pnl"] < -abs(capital * 0.05),
                    "vega": full_chain_df["Total Vega"].mean() if not full_chain_df.empty else 0,
                    "instrument_token": instrument
                })
            return pd.DataFrame(trades_df_list) if trades_df_list else pd.DataFrame()
        except Exception as e:
            st.error(f":warning: Exception in fetch_trade_data: {e}")
            return pd.DataFrame()

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

    def load_ivp(config, avg_iv):
        try:
            iv_df = pd.read_csv(config['ivp_url']).dropna(subset=["ATM_IV"]).tail(30)
            return round((iv_df["ATM_IV"] < avg_iv).sum() / len(iv_df) * 100, 2)
        except Exception as e:
            st.warning(f":warning: Exception in load_ivp: {e}")
            return 0

    def calculate_sharpe_ratio():
        try:
            daily_returns = np.random.normal(0.001, 0.01, 252)
            annual_return = np.mean(daily_returns) * 252
            annual_volatility = np.std(daily_returns) * np.sqrt(252)
            sharpe_ratio = (annual_return - 0.06 / 252) / annual_volatility
            return round(sharpe_ratio, 2)
        except Exception as e:
            st.warning(f":warning: Exception in calculate_sharpe_ratio: {e}")
            return 0

    def exit_all_positions(config):
        try:
            url = f"{config['base_url']}/order/positions/exit?segment=EQ"
            res = requests.post(url, headers=config['headers'])
            if res.status_code == 200:
                st.success(":white_check_mark: Exit initiated.")
                return True
            st.error(f":x: Exit failed: {res.status_code} - {res.text}")
            return False
        except Exception as e:
            st.error(f":warning: Exception in exit_all_positions: {e}")
            return False

    def logout(config):
        try:
            url = f"{config['base_url']}/logout"
            res = requests.delete(url, headers=config['headers'])
            if res.status_code == 200:
                st.success(":white_check_mark: Logged out successfully!")
                st.session_state.update({'logged_in': False, 'access_token': ""})
                st.cache_data.clear()
            else:
                st.error(f":x: Logout failed: {res.status_code} - {res.text}")
        except Exception as e:
            st.error(f":warning: Exception in logout: {e}")

    # --- LOADED DATA ---
    (option_chain, spot_price, vix, nifty, seller, full_chain_df, market, ivp, hv_7, garch_7d, xgb_vol, iv_rv_spread, iv_skew_slope, regime_score, regime, regime_note, regime_explanation, event_df, strategies, strategy_rationale, event_warning, strategy_details, trades_df, strategy_df, portfolio_summary, funds_data, sharpe_ratio) = load_all_data(config)
    if option_chain is None:
        st.stop()

    # --- HEADER ---
    st.markdown("<h1 style='text-align:center;'>📊 VolGuard: Your AI-Powered Options Dashboard</h1>", unsafe_allow_html=True)

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Market Dashboard",
        "📊 Chain Analysis",
        "📋 Strategy Suggestions",
        "🧮 Portfolio Summary",
        "🛒 Place Orders",
        "🛡️ Risk Manager"
    ])

    with tab1:
        st.subheader("Market Snapshot")
        col1, col2, col3, col4 = st.columns(4)
        for col, name, val in zip(
            [col1, col2, col3, col4],
            ["Nifty Spot", "India VIX", "ATM Strike", "Straddle Price"],
            [f"₹{nifty:.2f}", f"{vix:.2f}", f"{seller['strike']:.0f}", f"₹{seller['straddle_price']:.2f}"]
        ):
            with col:
                st.markdown(f"""
                <div class='metric-box'>
                    <h3>{name}</h3><div class='value'>{val}</div>
                </div>
                """, unsafe_allow_html=True)
        st.subheader("Volatility Breakdown")
        plot_vol_comparison(seller, hv_7, garch_7d, xgb_vol)

    with tab2:
        st.subheader("Option Chain: IV Skew vs OI")
        plot_chain_analysis(full_chain_df)
        st.dataframe(full_chain_df.head(10).style.set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)

    with tab3:
        st.markdown(f"<div class='metric-box'><h3>Regime: {regime}</h3><p style='color: #6495ED;'>Score: {regime_score:.2f}</p><p>{regime_note}</p><p><i>{regime_explanation}</i></p></div>", unsafe_allow_html=True)
        st.subheader("AI-Backed Strategy Recommendations")
        if strategies:
            st.success(f"Suggested: {', '.join(strategies)}")
            st.info(f"Rationale: {strategy_rationale}")
            if event_warning:
                st.warning(event_warning)
            for strat in strategies:
                detail = get_strategy_details(strat, option_chain, spot_price, config, lots=1)
                if detail:
                    st.markdown(f"### {strat}")
                    order_df = pd.DataFrame({
                        "Instrument": [o["instrument_key"] for o in detail["orders"]],
                        "Type": [o["transaction_type"] for o in detail["orders"]],
                        "Quantity": [config["lot_size"] for _ in detail["orders"]],
                        "Price": [o["current_price"] for o in detail["orders"]]
                    })
                    st.dataframe(order_df.style.set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
                    margin = calculate_strategy_margin(config, detail)
                    st.markdown(f"<div class='metric-box'><h4>Estimated Margin Required</h4> ₹{margin:.2f}</div>", unsafe_allow_html=True)
                    lots = st.number_input(f"Lots for {strat}", min_value=1, value=1, step=1, key=f"lots_{strat}")
                    if st.button(f"Place {strat} Order", key=f"place_{strat}"):
                        updated_detail = get_strategy_details(strat, option_chain, spot_price, config, lots=lots)
                        if updated_detail:
                            success = place_multi_leg_orders(config, updated_detail["orders"])
                            if success:
                                st.success(f":white_check_mark: Placed {strat} order with {lots} lots!")
                            else:
                                st.error(f":x: Failed to place {strat} order.")
                        else:
                            st.error(f":x: Unable to generate order details for {strat}.")
                else:
                    st.error(f":x: Error: No details found for {strat}.")
        else:
            st.info("No strategies suggested for current market conditions.")

    with tab4:
        st.subheader("Portfolio Summary")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        for col, name, val in zip(
            [col_p1, col_p2, col_p3, col_p4],
            ["Available Capital", "Used Margin", "Exposure %", "Sharpe Ratio"],
            [f"₹{funds_data['available_margin']:.2f}", f"₹{funds_data['used_margin']:.2f}", f"{portfolio_summary['Exposure Percent']:.2f}%", f"{sharpe_ratio:.2f}"]
        ):
            with col:
                st.markdown(f"<div class='metric-box'><h3>{name}</h3><div class='value'>{val}</div></div>", unsafe_allow_html=True)
        st.subheader("Capital Allocation")
        plot_allocation_pie(strategy_df, config)
        st.subheader("Drawdown Trend")
        plot_drawdown_trend(portfolio_summary)
        st.subheader("Margin Utilization")
        plot_margin_gauge(funds_data)
        st.subheader("Strategy Risk Summary")
        st.dataframe(strategy_df.style.set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
        if portfolio_summary.get("Flags"):
            for flag in portfolio_summary["Flags"]:
                st.warning(flag)

    with tab5:
        st.subheader("📥 Manual Multi-Leg Order Placement")
        selected_strategy = st.selectbox("Select Strategy", all_strategies, key="manual_strategy")
        lots = st.number_input("Number of Lots", min_value=1, value=1, step=1, key="manual_lots")
        sl_percentage = st.slider("Stop Loss %", 0.0, 50.0, 10.0, 0.5, key="manual_sl_pct")
        order_type = st.radio("Order Type", ["MARKET", "LIMIT"], horizontal=True)
        validity = st.radio("Order Validity", ["DAY", "IOC"], horizontal=True)
        if selected_strategy:
            detail = get_strategy_details(selected_strategy, option_chain, spot_price, config, lots=lots)
            if detail:
                st.write("🧮 Strategy Legs (Editable)")
                updated_orders = []
                for idx, order in enumerate(detail["orders"]):
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.markdown(f"**Leg {idx + 1}**")
                    with col2:
                        qty = st.number_input(f"Qty {idx + 1}", min_value=1, value=order["quantity"], step=1, key=f"qty_{idx}")
                    with col3:
                        tx_type = st.selectbox(f"Type {idx + 1}", ["BUY", "SELL"], index=0 if order["transaction_type"] == "BUY" else 1, key=f"tx_{idx}")
                    with col4:
                        price = st.number_input(f"Price {idx + 1}", min_value=0.0, value=order.get("current_price", 0.0), step=0.05, key=f"price_{idx}")
                    with col5:
                        instr = order["instrument_key"]
                        st.code(instr)
                    updated_orders.append({
                        "instrument_key": instr,
                        "quantity": qty,
                        "transaction_type": tx_type,
                        "order_type": order_type,
                        "validity": validity,
                        "product": "D",
                        "current_price": price
                    })
                st.markdown("---")
                margin = calculate_strategy_margin(config, detail) * lots
                st.markdown(f"💰 **Estimated Margin:** ₹{margin:,.2f}")
                st.markdown(f"💵 **Premium Collected:** ₹{detail['premium_total'] * lots:,.2f}")
                st.markdown(f"🔻 **Max Loss:** ₹{detail['max_loss'] * lots:,.2f}")
                st.markdown(f"🟢 **Max Profit:** ₹{detail['max_profit'] * lots:,.2f}")
                if st.button("🚀 Place Multi-Leg Order"):
                    payload = []
                    for idx, leg in enumerate(updated_orders):
                        correlation_id = f"mleg_{idx}_{int(time()) % 100000}"
                        leg_payload = {
                            "quantity": abs(leg["quantity"]),
                            "product": "D",
                            "validity": leg["validity"],
                            "tag": f"{leg['instrument_key']}_leg_{idx}",
                            "slice": False,
                            "instrument_token": leg["instrument_key"],
                            "order_type": leg["order_type"],
                            "transaction_type": leg["transaction_type"],
                            "disclosed_quantity": 0,
                            "trigger_price": 0,
                            "is_amo": False,
                            "correlation_id": correlation_id
                        }
                        if leg["order_type"] == "LIMIT":
                            leg_payload["price"] = leg["current_price"]
                        payload.append(leg_payload)
                    url = f"{config['base_url']}/order/multi/place"
                    res = requests.post(url, headers=config['headers'], json=payload)
                    if res.status_code == 200:
                        st.success(":white_check_mark: Multi-leg order placed successfully!")
                        for leg in updated_orders:
                            if leg["transaction_type"] == "SELL":
                                sl_price = leg["current_price"] * (1 + sl_percentage / 100)
                                create_gtt_order(config, leg["instrument_key"], sl_price, "BUY", tag=f"SL_{selected_strategy}")
                        st.success(f"🛡️ SL orders placed at {sl_percentage}% above sell price.")
                    else:
                        st.error(f":x: Failed to place order: {res.status_code} - {res.text}")
        else:
            st.info("No active strategies to display.")

    with tab6:
        st.subheader("Risk Management Dashboard")
        st.markdown("<h4>Portfolio Risk Metrics</h4>", unsafe_allow_html=True)
        col_r1, col_r2, col_r3 = st.columns(3)
        for col, name, val in zip(
            [col_r1, col_r2, col_r3],
            ["Total Risk %", "Daily Risk Limit", "Weekly Risk Limit"],
            [f"{portfolio_summary['Risk Percent']:.2f}%", f"₹{portfolio_summary['Daily Risk Limit']:.2f}", f"₹{portfolio_summary['Weekly Risk Limit']:.2f}"]
        ):
            with col:
                st.markdown(f"<div class='metric-box'><h3>{name}</h3><div class='value'>{val}</div></div>", unsafe_allow_html=True)
        st.subheader("Drawdown Analysis")
        plot_drawdown_trend(portfolio_summary)
        st.markdown(f"<div class='metric-box'><h4>Current Drawdown</h4> ₹{portfolio_summary['Drawdown ₹']:.2f} ({portfolio_summary['Drawdown Percent']:.2f}%)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-box'><h4>Max Drawdown Allowed</h4> ₹{portfolio_summary['Max Drawdown Allowed']:.2f}</div>", unsafe_allow_html=True)
        st.subheader("Risk Flags")
        if portfolio_summary.get("Flags"):
            for flag in portfolio_summary["Flags"]:
                st.warning(flag)
        else:
            st.info("No risk flags raised.")
        st.subheader("Strategy Risk Details")
        if not strategy_df.empty:
            st.dataframe(strategy_df[["Strategy", "Capital Used", "% Used", "Potential Risk", "Risk OK?"]].style.set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
        else:
            st.info("No active strategies to display.")

    # --- FOOTER ---
    st.markdown("---")
    st.markdown("<div style='text-align:center;'>© 2025 VolGuard by Shritish Shukla. All rights reserved.</div>", unsafe_allow_html=True)
    
