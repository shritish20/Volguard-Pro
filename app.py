import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
from arch import arch_model
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go

# Set page config for wide layout and custom title
st.set_page_config(
    page_title="Volguard Pro - Your Trading Copilot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional, trading-focused, dark-themed look
st.markdown("""
    <style>
    .main {background-color: #1a1f2b; color: #e6e9ef;}
    .sidebar .sidebar-content {background-color: #252b3a; color: #ffffff;}
    .stButton>button {
        background-color: #28a745;
        color: white;
        border-radius: 6px;
        font-weight: bold;
        padding: 10px 20px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #218838;
        color: #ffffff;
    }
    .stTextInput>div>input {
        background-color: #2c3344;
        color: #ffffff;
        border-radius: 6px;
        border: 1px solid #28a745;
        padding: 8px;
    }
    h1, h2, h3, h4 {color: #ffffff; font-family: 'Helvetica Neue', sans-serif;}
    .metric-card {
        background-color: #252b3a;
        border-radius: 8px;
        padding: 15px;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card .metric-label {font-size: 14px; color: #a3bffa;}
    .metric-card .metric-value {font-size: 18px; font-weight: bold; color: #ffffff;}
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
        background-color: #2c3344;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        margin-right: 5px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #28a745;
        color: #ffffff;
    }
    .stDataFrame {
        background-color: #252b3a;
        border-radius: 8px;
        color: #ffffff;
    }
    .stDataFrame table {
        background-color: #2c3344;
        color: #ffffff;
        border-collapse: collapse;
        width: 100%;
    }
    .stDataFrame th, .stDataFrame td {
        color: #ffffff;
        border: 1px solid #28a745;
        padding: 8px;
        text-align: center;
    }
    .stDataFrame tr:hover {background-color: #3a4257;}
    .st-expander {background-color: #2c3344; color: #ffffff; border-radius: 8px;}
    .footer {
        text-align: center;
        color: #a3bffa;
        font-size: 14px;
        margin-top: 20px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'access_token' not in st.session_state:
    st.session_state.access_token = ""
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'config' not in st.session_state:
    st.session_state.config = None
if 'option_chain' not in st.session_state:
    st.session_state.option_chain = []

# Sidebar
with st.sidebar:
    # Replace with your actual logo URL when hosted
    st.image("https://via.placeholder.com/150x50.png?text=Volguard+Pro", use_column_width=True)
    st.header("Volguard Pro - Your Trading Copilot")
    st.session_state.access_token = st.text_input("Enter Upstox API Access Token", type="password")
    if st.button("Fetch Data"):
        if st.session_state.access_token:
            st.session_state.data_loaded = True
            st.success("Fetching data...")
        else:
            st.error("Please enter a valid access token.")

# Auto-refresh every 4 minutes (240 seconds)
st_autorefresh(interval=240000, key="data_refresh")

# Configuration function
def get_config():
    config = {
        "base_url": "https://api.upstox.com/v2",
        "headers": {
            "accept": "application/json",
            "Api-Version": "2.0",
            "Authorization": f"Bearer {st.session_state.access_token}"
        },
        "instrument_key": "NSE_INDEX|Nifty 50",
        "event_url": "https://raw.githubusercontent.com/shritish20/VolGuard/main/upcoming_events.csv",
        "ivp_url": "https://raw.githubusercontent.com/shritish20/VolGuard/main/ivp.csv",
        "nifty_url": "https://raw.githubusercontent.com/shritish20/VolGuard/main/nifty_50.csv",
        "total_capital": 2000000,
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

    def get_next_expiry():
        try:
            url = f"{config['base_url']}/option/contract"
            params = {"instrument_key": config['instrument_key']}
            res = requests.get(url, headers=config['headers'], params=params, timeout=5)
            res.raise_for_status()
            expiries = sorted(res.json()["data"], key=lambda x: datetime.strptime(x["expiry"], "%Y-%m-%d"))
            today = datetime.now()
            for expiry in expiries:
                expiry_dt = datetime.strptime(expiry["expiry"], "%Y-%m-%d")
                if expiry_dt.weekday() == 3 and expiry_dt > today:
                    return expiry["expiry"]
            return datetime.now().strftime("%Y-%m-%d")
        except Exception as e:
            st.error(f"Failed to fetch expiries: {e}")
            return datetime.now().strftime("%Y-%m-%d")

    config['expiry_date'] = get_next_expiry()
    return config

# Data fetching and processing functions
@st.cache_data(show_spinner=False)
def fetch_option_chain(config):
    try:
        url = f"{config['base_url']}/option/chain"
        params = {"instrument_key": config['instrument_key'], "expiry_date": config['expiry_date']}
        res = requests.get(url, headers=config['headers'], params=params, timeout=5)
        res.raise_for_status()
        return res.json()["data"]
    except Exception as e:
        st.error(f"Failed to fetch option chain: {e}")
        return []

@st.cache_data(show_spinner=False)
def get_indices_quotes(config):
    try:
        url = f"{config['base_url']}/market-quote/quotes?instrument_key=NSE_INDEX|India VIX,NSE_INDEX|Nifty 50"
        res = requests.get(url, headers=config['headers'], timeout=5)
        res.raise_for_status()
        data = res.json()
        vix = data["data"]["NSE_INDEX:India VIX"]["last_price"]
        nifty = data["data"]["NSE_INDEX:Nifty 50"]["last_price"]
        return vix, nifty
    except Exception as e:
        st.error(f"Failed to fetch indices quotes: {e}")
        return None, None

@st.cache_data(show_spinner=False)
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
        st.error(f"Failed to load upcoming events: {e}")
        return pd.DataFrame(columns=["Datetime", "Event", "Classification", "Forecast", "Prior"])

@st.cache_data(show_spinner=False)
def load_ivp(config, avg_iv):
    try:
        iv_df = pd.read_csv(config['ivp_url'])
        iv_df.dropna(subset=["ATM_IV"], inplace=True)
        iv_df = iv_df.tail(30)
        ivp = round((iv_df["ATM_IV"] < avg_iv).sum() / len(iv_df) * 100, 2)
        return ivp
    except Exception as e:
        st.error(f"Failed to load IVP: {e}")
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
            "pop": (call["option_greeks"]["pop"] + put["option_greeks"]["pop"]) / 2 * 100
        }
    except Exception as e:
        st.error(f"Failed to extract seller metrics: {e}")
        return {}

def full_chain_table(option_chain, spot_price):
    try:
        chain_data = []
        for opt in option_chain:
            strike = opt["strike_price"]
            call = opt["call_options"]
            put = opt["put_options"]
            if abs(strike - spot_price) <= 300:
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
        st.error(f"Failed to create chain table: {e}")
        return pd.DataFrame()

def market_metrics(option_chain, expiry_date):
    try:
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        days_to_expiry = (expiry_dt - datetime.now()).days
        call_oi = sum(opt["call_options"]["market_data"]["oi"] for opt in option_chain)
        put_oi = sum(opt["put_options"]["market_data"]["oi"] for opt in option_chain)
        pcr = put_oi / call_oi if call_oi != 0 else 0
        max_pain = max(range(int(min(opt["strike_price"] for opt in option_chain)),
                            int(max(opt["strike_price"] for opt in option_chain)) + 1, 50),
                       key=lambda strike: sum(max(0, strike - opt["strike_price"]) * opt["call_options"]["market_data"]["oi"] +
                                             max(0, opt["strike_price"] - strike) * opt["put_options"]["market_data"]["oi"]
                                             for opt in option_chain))
        return {"days_to_expiry": days_to_expiry, "pcr": round(pcr, 2), "max_pain": max_pain}
    except Exception as e:
        st.error(f"Failed to calculate market metrics: {e}")
        return {"days_to_expiry": 0, "pcr": 0, "max_pain": 0}

@st.cache_data(show_spinner=False)
def calculate_volatility(config, seller):
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
        iv_rv_spread = round(seller['avg_iv'] - hv_7, 2)
        return hv_7, garch_7d, iv_rv_spread
    except Exception as e:
        st.error(f"Failed to calculate volatility: {e}")
        return 0, 0, 0

def calculate_iv_skew_slope(full_chain_df):
    try:
        if full_chain_df.empty:
            return 0
        from scipy.stats import linregress
        slope, _, _, _, _ = linregress(full_chain_df["Strike"], full_chain_df["IV Skew"])
        return slope
    except Exception as e:
        st.error(f"Failed to calculate IV skew slope: {e}")
        return 0

def calculate_regime(atm_iv, ivp, realized_vol, garch_vol, straddle_price, spot_price, pcr, vix, iv_skew_slope):
    try:
        expected_move = straddle_price / spot_price
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
            regime = "🔥 High Vol Trend"
            note = "Market in high volatility — ideal for premium selling with defined risk."
            explanation = "High IVP, elevated VIX, and wide straddle suggest strong premium opportunities."
        elif regime_score > 10:
            regime = "🚧 Elevated Volatility"
            note = "Above-average volatility — favor range-bound strategies."
            explanation = "Moderate IVP and IV-RV spread indicate potential for mean-reverting moves."
        elif regime_score > -10:
            regime = "😴 Neutral Volatility"
            note = "Balanced market — flexible strategy selection."
            explanation = "IV and RV aligned, with moderate PCR and skew."
        else:
            regime = "💤 Low Volatility"
            note = "Low volatility — cautious selling or long vega plays."
            explanation = "Low IVP, tight straddle, and low VIX suggest limited movement."
        return regime_score, regime, note, explanation
    except Exception as e:
        st.error(f"Failed to calculate regime: {e}")
        return 0, "Unknown", "Error calculating regime.", ""

def suggest_strategy(regime_label, ivp, iv_minus_rv, days_to_expiry, event_df, expiry_date, straddle_price, spot_price):
    strategies = []
    rationale = []
    event_warning = None
    event_window = 3 if ivp > 80 else 2
    high_impact_event_near = False
    event_impact_score = 0
    for _, row in event_df.iterrows():
        dt, level = row["Datetime"], row["Classification"]
        if (level == "High") and (0 <= (datetime.strptime(expiry_date, "%Y-%m-%d") - dt).days <= event_window):
            high_impact_event_near = True
        if level == "High" and pd.notnull(row["Forecast"]) and pd.notnull(row["Prior"]):
            try:
                forecast = float(str(row["Forecast"]).strip("%")) if "%" in str(row["Forecast"]) else float(row["Forecast"])
                prior = float(str(row["Prior"]).strip("%")) if "%" in str(row["Prior"]) else float(row["Prior"])
                if abs(forecast - prior) > 0.5:
                    event_impact_score += 1
            except:
                continue
    if high_impact_event_near:
        event_warning = f"⚠️ High-impact event within {event_window} days of expiry. Prefer defined-risk strategies."
    if event_impact_score > 0:
        rationale.append(f"High-impact events with significant forecast deviations ({event_impact_score} events).")
    expected_move_pct = (straddle_price / spot_price) * 100
    if regime_label == "🔥 High Vol Trend":
        if high_impact_event_near or event_impact_score > 0:
            strategies = ["Iron Fly", "Wide Strangle"]
            rationale.append("High volatility with major event — use defined-risk structures.")
        else:
            strategies = ["Iron Fly", "Wide Strangle"]
            rationale.append("Strong IV premium — neutral strategies for premium capture.")
    elif regime_label == "🚧 Elevated Volatility":
        strategies = ["Iron Condor", "Jade Lizard"]
        rationale.append("Volatility above average — range-bound strategies offer favorable reward-risk.")
    elif regime_label == "😴 Neutral Volatility":
        if days_to_expiry >= 3:
            strategies = ["Jade Lizard", "Bull Put Spread"]
            rationale.append("Market balanced — slight directional bias strategies offer edge.")
        else:
            strategies = ["Iron Fly"]
            rationale.append("Tight expiry — quick theta-based capture via short Iron Fly.")
    elif regime_label == "💤 Low Volatility":
        if days_to_expiry > 7:
            strategies = ["Straddle", "Calendar Spread"]
            rationale.append("Low IV with longer expiry — benefit from potential IV increase.")
        else:
            strategies = ["Straddle", "ATM Strangle"]
            rationale.append("Low IV — premium collection favorable but monitor for breakout risk.")
    if event_impact_score > 0 and not high_impact_event_near:
        strategies = [s for s in strategies if "Iron" in s or "Lizard" in s]
    if ivp > 85 and iv_minus_rv > 5:
        rationale.append(f"Volatility overpriced (IVP: {ivp}%, IV-RV: {iv_minus_rv}%) — ideal for selling premium.")
    elif ivp < 30:
        rationale.append(f"Volatility underpriced (IVP: {ivp}%) — avoid unhedged selling.")
    rationale.append(f"Expected move: ±{expected_move_pct:.2f}% based on straddle price.")
    return strategies, " | ".join(rationale), event_warning

def fetch_trade_data(config, full_chain_df):
    try:
        url_positions = f"{config['base_url']}/portfolio/short-term-positions"
        res_positions = requests.get(url_positions, headers=config['headers'], timeout=5)
        url_trades = f"{config['base_url']}/order/trades/get-trades-for-day"
        res_trades = requests.get(url_trades, headers=config['headers'], timeout=5)
        if res_positions.status_code == 200 and res_trades.status_code == 200:
            positions = res_positions.json()["data"]
            trades = res_trades.json()["data"]
            trade_counts = {}
            for trade in trades:
                instrument = trade["instrument_key"]
                strat = "Straddle" if "CE" in instrument or "PE" in instrument else "Iron Condor"
                trade_counts[strat] = trade_counts.get(strat, 0) + 1
            trades_df = []
            for pos in positions:
                instrument = pos["instrument_key"]
                strat = "Straddle" if pos.get("option_type") in ["CE", "PE"] else "Iron Condor"
                capital = pos["quantity"] * pos["average_price"] * pos["multiplier"]
                url_brokerage = f"{config['base_url']}/charges/brokerage"
                params_brokerage = {
                    "instrument_token": instrument,
                    "quantity": pos["quantity"],
                    "product": "D",
                    "transaction_type": "SELL" if pos["quantity"] < 0 else "BUY",
                    "price": pos["average_price"]
                }
                res_brokerage = requests.get(url_brokerage, headers=config['headers'], params=params_brokerage, timeout=5)
                brokerage = res_brokerage.json()["data"]["charges"]["total"] if res_brokerage.status_code == 200 else 0
                trades_df.append({
                    "strategy": strat,
                    "capital_used": capital,
                    "potential_loss": capital * 0.1,
                    "realized_pnl": pos["pnl"] - brokerage,
                    "trades_today": trade_counts.get(strat, 1),
                    "sl_hit": pos["pnl"] < -capital * 0.05,
                    "vega": full_chain_df["Total Vega"].mean() if not full_chain_df.empty else 0
                })
            return pd.DataFrame(trades_df) if trades_df else pd.DataFrame()
        st.error(f"Failed to fetch positions or trades: {res_positions.status_code}, {res_trades.status_code}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to fetch trade data: {e}")
        return pd.DataFrame()

def evaluate_full_risk(trades_df, config, regime_label):
    try:
        daily_risk_limit = config['daily_risk_limit_pct'] * config['total_capital']
        weekly_risk_limit = config['weekly_risk_limit_pct'] * config['total_capital']
        strategy_summary = []
        total_cap_used = 0
        total_risk_used = 0
        total_realized_pnl = 0
        total_vega = 0
        flags = []
        for _, row in trades_df.iterrows():
            strat = row["strategy"]
            capital_used = row["capital_used"]
            risk = row["potential_loss"]
            pnl = row["realized_pnl"]
            sl_hit = row["sl_hit"]
            trades_today = row["trades_today"]
            vega = row["vega"]
            cfg = config['risk_config'].get(strat, {"capital_pct": 0.1, "risk_per_trade_pct": 0.01})
            risk_factor = 0.8 if regime_label == "🔥 High Vol Trend" else 1.1 if regime_label == "💤 Low Volatility" else 1.0
            max_cap = cfg["capital_pct"] * config['total_capital']
            max_risk = cfg["risk_per_trade_pct"] * max_cap * risk_factor
            risk_ok = risk <= max_risk
            strategy_summary.append({
                "Strategy": strat,
                "Capital Used": capital_used,
                "Cap Limit": round(max_cap),
                "% Used": round(capital_used / max_cap * 100, 2),
                "Potential Risk": risk,
                "Risk Limit": round(max_risk),
                "P&L": pnl,
                "Vega": vega,
                "Risk OK?": "✅" if risk_ok else "❌"
            })
            total_cap_used += capital_used
            total_risk_used += risk
            total_realized_pnl += pnl
            total_vega += vega * (capital_used / config['total_capital'])
            if not risk_ok:
                flags.append(f"❌ {strat} exceeded risk limit")
            if sl_hit and trades_today > 3:
                flags.append(f"⚠️ {strat} shows possible revenge trading (SL hit + {trades_today} trades)")
        net_dd = -total_realized_pnl if total_realized_pnl < 0 else 0
        exposure_pct = round(total_cap_used / config['total_capital'] * 100, 2)
        risk_pct = round(total_risk_used / config['total_capital'] * 100, 2)
        dd_pct = round(net_dd / config['total_capital'] * 100, 2)
        portfolio_summary = {
            "Total Capital": config['total_capital'],
            "Capital Deployed": total_cap_used,
            "Exposure %": exposure_pct,
            "Risk on Table": total_risk_used,
            "Risk %": risk_pct,
            "Daily Risk Limit": daily_risk_limit,
            "Weekly Risk Limit": weekly_risk_limit,
            "Realized P&L": total_realized_pnl,
            "Drawdown ₹": net_dd,
            "Drawdown %": dd_pct,
            "Portfolio Vega": round(total_vega, 2),
            "Flags": flags
        }
        return pd.DataFrame(strategy_summary), portfolio_summary
    except Exception as e:
        st.error(f"Failed to evaluate risk: {e}")
        return pd.DataFrame(), {}

def find_option_by_strike(option_chain, strike, option_type):
    try:
        for opt in option_chain:
            if abs(opt["strike_price"] - strike) < 0.01:
                return opt["call_options"] if option_type == "CE" else opt["put_options"]
        st.error(f"No option found for strike {strike} {option_type}")
        return None
    except Exception as e:
        st.error(f"Failed to find option by strike: {e}")
        return None

def place_order(config, instrument_key, quantity, transaction_type, order_type="MARKET", price=0):
    try:
        url = f"{config['base_url'].replace('v2', 'v3')}/order/place"
        payload = {
            "quantity": quantity,
            "product": "D",
            "validity": "DAY",
            "price": price,
            "instrument_token": instrument_key,
            "order_type": order_type,
            "transaction_type": transaction_type,
            "disclosed_quantity": 0,
            "trigger_price": 0,
            "is_amo": False
        }
        res = requests.post(url, headers=config['headers'], json=payload, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                order_data = data.get("data", {})
                if "order_ids" in order_data and order_data["order_ids"]:
                    return order_data["order_ids"][0]
                elif "order_id" in order_data:
                    return order_data["order_id"]
                st.error(f"Unexpected response format: {data}")
                return None
            st.error(f"Unexpected response status: {data}")
            return None
        elif res.status_code == 400:
            data = res.json()
            errors = data.get("errors", [])
            for error in errors:
                if error.get("errorCode") == "UDAPI100060":
                    st.error(f"Order failed: Insufficient funds for {instrument_key}")
                    return None
                else:
                    st.error(f"Order failed: {error.get('message', 'Unknown error')} for {instrument_key}")
                    return None
        st.error(f"Error placing order: {res.status_code} - {res.text}")
        return None
    except Exception as e:
        st.error(f"Failed to place order: {e}")
        return None

def get_option_greeks(config, instrument_keys):
    try:
        url = f"{config['base_url'].replace('v2', 'v3')}/market-quote/option-greek"
        params = {"instrument_key": ",".join(instrument_keys)}
        res = requests.get(url, headers=config['headers'], params=params, timeout=5)
        if res.status_code == 200:
            return res.json()["data"]
        st.error(f"Failed to fetch Greeks: {res.status_code} - {res.text}")
        return {}
    except Exception as e:
        st.error(f"Failed to fetch option Greeks: {e}")
        return {}

# Strategy functions
def iron_fly(option_chain, spot_price, config, lots=1):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    wing = 100
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike, "PE")
    ce_long_opt = find_option_by_strike(option_chain, strike + wing, "CE")
    pe_long_opt = find_option_by_strike(option_chain, strike - wing, "PE")
    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        st.error("Missing options for Iron Fly")
        return None
    instrument_keys = [ce_short_opt["instrument_key"], pe_short_opt["instrument_key"],
                      ce_long_opt["instrument_key"], pe_long_opt["instrument_key"]]
    greeks = get_option_greeks(config, instrument_keys)
    ce_short_price = greeks.get(ce_short_opt["instrument_key"], {}).get("last_price", ce_short_opt["market_data"]["ltp"])
    pe_short_price = greeks.get(pe_short_opt["instrument_key"], {}).get("last_price", pe_short_opt["market_data"]["ltp"])
    ce_long_price = greeks.get(ce_long_opt["instrument_key"], {}).get("last_price", ce_long_opt["market_data"]["ltp"])
    pe_long_price = greeks.get(pe_long_opt["instrument_key"], {}).get("last_price", pe_long_opt["market_data"]["ltp"])
    premium = (ce_short_price + pe_short_price - ce_long_price - pe_long_price) * config["lot_size"] * lots
    max_loss = (wing - premium/config["lot_size"]) * config["lot_size"] * lots
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Iron Fly", "strikes": [strike, strike, strike + wing, strike - wing],
            "premium": premium, "max_loss": max_loss, "max_profit": premium, "orders": orders}

def iron_condor(option_chain, spot_price, config, lots=1):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    short_wing = 100
    long_wing = 200
    ce_short_opt = find_option_by_strike(option_chain, atm["strike_price"] + short_wing, "CE")
    pe_short_opt = find_option_by_strike(option_chain, atm["strike_price"] - short_wing, "PE")
    ce_long_opt = find_option_by_strike(option_chain, atm["strike_price"] + long_wing, "CE")
    pe_long_opt = find_option_by_strike(option_chain, atm["strike_price"] - long_wing, "PE")
    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        st.error("Missing options for Iron Condor")
        return None
    instrument_keys = [ce_short_opt["instrument_key"], pe_short_opt["instrument_key"],
                      ce_long_opt["instrument_key"], pe_long_opt["instrument_key"]]
    greeks = get_option_greeks(config, instrument_keys)
    ce_short_price = greeks.get(ce_short_opt["instrument_key"], {}).get("last_price", ce_short_opt["market_data"]["ltp"])
    pe_short_price = greeks.get(pe_short_opt["instrument_key"], {}).get("last_price", pe_short_opt["market_data"]["ltp"])
    ce_long_price = greeks.get(ce_long_opt["instrument_key"], {}).get("last_price", ce_long_opt["market_data"]["ltp"])
    pe_long_price = greeks.get(pe_long_opt["instrument_key"], {}).get("last_price", pe_long_opt["market_data"]["ltp"])
    premium = (ce_short_price + pe_short_price - ce_long_price - pe_long_price) * config["lot_size"] * lots
    max_loss = ((long_wing - short_wing) - premium/config["lot_size"]) * config["lot_size"] * lots
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Iron Condor", "strikes": [atm["strike_price"] + short_wing, atm["strike_price"] - short_wing,
            atm["strike_price"] + long_wing, atm["strike_price"] - long_wing],
            "premium": premium, "max_loss": max_loss, "max_profit": premium, "orders": orders}

def jade_lizard(option_chain, spot_price, config, lots=1):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 50
    put_strike = atm["strike_price"]
    put_long_strike = atm["strike_price"] - 100
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, put_long_strike, "PE")
    if not all([ce_short_opt, pe_short_opt, pe_long_opt]):
        st.error("Missing options for Jade Lizard")
        return None
    instrument_keys = [ce_short_opt["instrument_key"], pe_short_opt["instrument_key"], pe_long_opt["instrument_key"]]
    greeks = get_option_greeks(config, instrument_keys)
    ce_short_price = greeks.get(ce_short_opt["instrument_key"], {}).get("last_price", ce_short_opt["market_data"]["ltp"])
    pe_short_price = greeks.get(pe_short_opt["instrument_key"], {}).get("last_price", pe_short_opt["market_data"]["ltp"])
    pe_long_price = greeks.get(pe_long_opt["instrument_key"], {}).get("last_price", pe_long_opt["market_data"]["ltp"])
    premium = (ce_short_price + pe_short_price - pe_long_price) * config["lot_size"] * lots
    max_loss = (put_strike - put_long_strike - premium/config["lot_size"]) * config["lot_size"] * lots
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Jade Lizard", "strikes": [call_strike, put_strike, put_long_strike],
            "premium": premium, "max_loss": max_loss, "max_profit": premium, "orders": orders}

def straddle(option_chain, spot_price, config, lots=1):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike, "PE")
    if not all([ce_short_opt, pe_short_opt]):
        st.error("Missing options for Straddle")
        return None
    instrument_keys = [ce_short_opt["instrument_key"], pe_short_opt["instrument_key"]]
    greeks = get_option_greeks(config, instrument_keys)
    ce_short_price = greeks.get(ce_short_opt["instrument_key"], {}).get("last_price", ce_short_opt["market_data"]["ltp"])
    pe_short_price = greeks.get(pe_short_opt["instrument_key"], {}).get("last_price", pe_short_opt["market_data"]["ltp"])
    premium = (ce_short_price + pe_short_price) * config["lot_size"] * lots
    max_loss = float("inf")
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    return {"strategy": "Straddle", "strikes": [strike, strike],
            "premium": premium, "max_loss": max_loss, "max_profit": premium, "orders": orders}

def calendar_spread(option_chain, spot_price, config, lots=1):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    if not ce_short_opt:
        st.error("Missing options for Calendar Spread")
        return None
    ce_long_opt = ce_short_opt
    instrument_keys = [ce_short_opt["instrument_key"]]
    greeks = get_option_greeks(config, instrument_keys)
    ce_short_price = greeks.get(ce_short_opt["instrument_key"], {}).get("last_price", ce_short_opt["market_data"]["ltp"])
    ce_long_price = ce_short_price
    premium = 0
    max_loss = premium
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Calendar Spread", "strikes": [strike, strike],
            "premium": premium, "max_loss": max_loss, "max_profit": float("inf"), "orders": orders}

def bull_put_spread(option_chain, spot_price, config, lots=1):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    short_strike = atm["strike_price"] - 50
    long_strike = atm["strike_price"] - 100
    pe_short_opt = find_option_by_strike(option_chain, short_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, long_strike, "PE")
    if not all([pe_short_opt, pe_long_opt]):
        st.error("Missing options for Bull Put Spread")
        return None
    instrument_keys = [pe_short_opt["instrument_key"], pe_long_opt["instrument_key"]]
    greeks = get_option_greeks(config, instrument_keys)
    pe_short_price = greeks.get(pe_short_opt["instrument_key"], {}).get("last_price", pe_short_opt["market_data"]["ltp"])
    pe_long_price = greeks.get(pe_long_opt["instrument_key"], {}).get("last_price", pe_long_opt["market_data"]["ltp"])
    premium = (pe_short_price - pe_long_price) * config["lot_size"] * lots
    max_loss = (short_strike - long_strike - premium/config["lot_size"]) * config["lot_size"] * lots
    orders = [
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Bull Put Spread", "strikes": [short_strike, long_strike],
            "premium": premium, "max_loss": max_loss, "max_profit": premium, "orders": orders}

def wide_strangle(option_chain, spot_price, config, lots=1):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 100
    put_strike = atm["strike_price"] - 100
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    if not all([ce_short_opt, pe_short_opt]):
        st.error("Missing options for Wide Strangle")
        return None
    instrument_keys = [ce_short_opt["instrument_key"], pe_short_opt["instrument_key"]]
    greeks = get_option_greeks(config, instrument_keys)
    ce_short_price = greeks.get(ce_short_opt["instrument_key"], {}).get("last_price", ce_short_opt["market_data"]["ltp"])
    pe_short_price = greeks.get(pe_short_opt["instrument_key"], {}).get("last_price", pe_short_opt["market_data"]["ltp"])
    premium = (ce_short_price + pe_short_price) * config["lot_size"] * lots
    max_loss = float("inf")
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    return {"strategy": "Wide Strangle", "strikes": [call_strike, put_strike],
            "premium": premium, "max_loss": max_loss, "max_profit": premium, "orders": orders}

def atm_strangle(option_chain, spot_price, config, lots=1):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 50
    put_strike = atm["strike_price"] - 50
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    if not all([ce_short_opt, pe_short_opt]):
        st.error("Missing options for ATM Strangle")
        return None
    instrument_keys = [ce_short_opt["instrument_key"], pe_short_opt["instrument_key"]]
    greeks = get_option_greeks(config, instrument_keys)
    ce_short_price = greeks.get(ce_short_opt["instrument_key"], {}).get("last_price", ce_short_opt["market_data"]["ltp"])
    pe_short_price = greeks.get(pe_short_opt["instrument_key"], {}).get("last_price", pe_short_opt["market_data"]["ltp"])
    premium = (ce_short_price + pe_short_price) * config["lot_size"] * lots
    max_loss = float("inf")
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    return {"strategy": "ATM Strangle", "strikes": [call_strike, put_strike],
            "premium": premium, "max_loss": max_loss, "max_profit": premium, "orders": orders}

def plot_vol_comparison(seller, hv_7, garch_7d):
    fig = go.Figure(data=[
        go.Bar(name='ATM IV', x=['ATM IV'], y=[seller["avg_iv"]], marker_color='#28a745'),
        go.Bar(name='Realized Vol (7D)', x=['Realized Vol (7D)'], y=[hv_7], marker_color='#ffc107'),
        go.Bar(name='GARCH Vol (7D)', x=['GARCH Vol (7D)'], y=[garch_7d], marker_color='#dc3545')
    ])
    fig.update_layout(
        title="Volatility Comparison",
        yaxis_title="Annualized Volatility (%)",
        template="plotly_dark",
        showlegend=True,
        height=400,
        paper_bgcolor="#1a1f2b",
        plot_bgcolor="#1a1f2b",
        font=dict(color="#e6e9ef"),
        annotations=[dict(x=x, y=y + 0.4, text=f"{y:.2f}%", showarrow=False, font=dict(color="#ffffff"))
                     for x, y in zip(['ATM IV', 'Realized Vol (7D)', 'GARCH Vol (7D)'], [seller["avg_iv"], hv_7, garch_7d])]
    )
    return fig

def plot_chain_analysis(full_chain_df):
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=full_chain_df["Strike"], y=full_chain_df["IV Skew"], mode='lines+markers', name='IV Skew', line=dict(color='#28a745')))
    fig1.add_hline(y=0, line_dash="dash", line_color="#a3bffa")
    fig1.update_layout(title="IV Skew", xaxis_title="Strike", yaxis_title="IV Skew (%)", template="plotly_dark", height=300, paper_bgcolor="#1a1f2b", plot_bgcolor="#1a1f2b", font=dict(color="#e6e9ef"))
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=full_chain_df["Strike"], y=full_chain_df["Total Theta"], mode='lines+markers', name='Total Theta', line=dict(color='#ffc107')))
    fig2.update_layout(title="Total Theta", xaxis_title="Strike", yaxis_title="Theta", template="plotly_dark", height=300, paper_bgcolor="#1a1f2b", plot_bgcolor="#1a1f2b", font=dict(color="#e6e9ef"))
    
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=full_chain_df["Strike"], y=full_chain_df["Straddle Price"], mode='lines+markers', name='Straddle Price', line=dict(color='#dc3545')))
    fig3.update_layout(title="Straddle Price", xaxis_title="Strike", yaxis_title="Price (₹)", template="plotly_dark", height=300, paper_bgcolor="#1a1f2b", plot_bgcolor="#1a1f2b", font=dict(color="#e6e9ef"))
    
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=full_chain_df["Strike"], y=full_chain_df["Total OI"], name='Total OI', marker_color='#28a745'))
    fig4.update_layout(title="Total OI", xaxis_title="Strike", yaxis_title="OI", template="plotly_dark", height=300, paper_bgcolor="#1a1f2b", plot_bgcolor="#1a1f2b", font=dict(color="#e6e9ef"))
    
    return [fig1, fig3, fig2]
    
def plot_payoff_diagram(strategy_details, spot_price, config):
    fig = go.Figure()
    strikes = np.linspace(spot_price - 50, spot_price + 50, 100)
    for detail in strategy_details:
        try:
            payoffs = np.zeros_like(strikes)
            for order in range(len(detail["orders"])):
                instrument = detail["orders"][order]["instrument_key"]
                qty = detail["orders"][order]["quantity"]
                is_buy = detail["orders"][order]["transaction_type"] == "BUY"
                multiplier = 1 if is_buy == 1 else -1
                strike = detail["strikes"][order]
                is_call = "CE" in instrument
                price = detail["pricing"].get(instrument, {}).get("last_price", 0)
                if is_call:
                    payoff = multiplier * (np.maximum(0, strikes - strike) - price)
                else:
                    payoff = multiplier * (np.maximum(0, strike - strikes) - price)
                payoffs += payoff * qty / config["lot_size"]
            fig.add_trace(go.Scatter(x=strikes, y=payoffs, mode='lines', name=detail["strategy"], line=dict(color='#28a745')))
        except Exception as e:
            st.error(f"Error plotting payoff for {detail['strategy']}: {e}")
            continue
    fig.add_vline(x=spot_price, line_dash="dash", line_color="#a3bffa", annotation_text="Spot Price")
    fig.update_layout(
        title="Payoff Diagram",
        xaxis_title="Underlying Price",
        yaxis_title="Payoff (₹)",
        template="plotly_dark",
        showlegend=True,
        height=400,
        paper_bgcolor="#1a1f2b",
        plot_bgcolor="#1a1f2b",
        font=dict(color="#e6e9ef")
    )
    return fig

# Main app logic
if st.session_state.data_loaded and st.session_state.access_token:
    config = get_config()
    st.session_state.config = config
    option_chain = fetch_option_chain(config)
    if not option_chain:
        st.error("Failed to fetch option chain data.")
        st.stop()
    spot_price = option_chain[0]["underlying_spot_price"]
    vix, nifty = get_indices_quotes(config)
    if vix is None or nifty is None:
        st.error("Failed to fetch VIX or Nifty data.")
        st.stop()
    seller = extract_seller_metrics(option_chain, spot_price)
    if not seller:
        st.error("Failed to extract seller metrics.")
        st.stop()
    full_chain_df = full_chain_table(option_chain, spot_price)
    if full_chain_df.empty:
        st.error("Failed to create chain table.")
        st.stop()
    market = market_metrics(option_chain, config['expiry_date'])
    if not market["pcr"]:
        st.error("Failed to calculate market metrics.")
        st.stop()
    ivp = load_ivp(config, seller["avg_iv"])
    hv_7, garch_7d, iv_rv_spread = calculate_volatility(config, seller)
    iv_skew_slope = calculate_iv_skew_slope(full_chain_df)
    regime_score, regime, regime_note, regime_explanation = calculate_regime(
        atm_iv=seller["avg_iv"],
        ivp=ivp,
        realized_vol=hv_7,
        garch_vol=garch_7d,
        straddle_price=seller["straddle_price"],
        spot_price=spot_price,
        pcr=market["pcr"],
        vix=vix,
        iv_skew_slope=iv_skew_slope
    )
    event_df = load_upcoming_events(config)
    strategies, strategy_rationale, event_warning = suggest_strategy(
        regime_label=regime,
        ivp=ivp,
        iv_minus_rv=iv_rv_spread,
        days_to_expiry=market['days_to_expiry'],
        event_df=event_df,
        expiry_date=config['expiry_date'],
        straddle_price=seller["straddle_price"],
        spot_price=spot_price
    )
    strategy_details = []
    func_map = {
        "Iron Fly": iron_fly,
        "Iron Condor": iron_condor,
        "Jade Lizard": jade_lizard,
        "Straddle": straddle,
        "Calendar Spread": calendar_spread,
        "Bull Put Spread": bull_put_spread,
        "Wide Strangle": wide_strangle,
        "ATM Strangle": atm_strangle
    }
    for strat in strategies:
        strat_clean = strat.replace("(hedged)", "").replace("with strict stop", "").replace("short ", "").strip()
        if strat_clean in func_map:
            detail = func_map[strat_clean](option_chain, spot_price, config)
            if detail and detail["premium"] >= 0:
                instrument_keys = [order["instrument_key"] for order in detail["orders"]]
                detail["pricing"] = get_option_greeks(config, instrument_keys)
                strategy_details.append(detail)
    trades_df = fetch_trade_data(config, full_chain_df)
    strategy_df, portfolio_summary = evaluate_full_risk(trades_df, config, regime)

    # Tabs for dashboard sections
    tab1, tab2, tab3, tab4 = st.tabs(["Market Overview", "Option Chain", "Events & Regime", "Strategies & Orders"])

    with tab1:
        st.header("Market Overview")
        st.subheader("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            {"label": "Spot Price", "value": f"₹{spot_price:.0f}", "color": "#28a745"},
            {"label": "ATM Strike", "value": f"{seller['strike']:.0f}", "color": "#28a745"},
            {"label": "Straddle Price", "value": f"₹{seller['straddle_price']:.2f}", "color": "#28a745"},
            {"label": "Breakeven Range", "value": f"{seller['strike']} - {seller['straddle_price']:.0f} - {seller['strike'] + seller['straddle_price']:.0f}", "color": "#28a745"},
            {"label": "ATM IV", "value": f"{seller['avg_iv']:.2f}%", "color": "#28a745"},
            {"label": "Realized Vol (7D)", "value": f"{hv_7:.2f}%", "color": "#ffc107"},
            {"label": "GARCH Vol (7D)", "value": f"{garch_7d:.2f}%", "color": "#ffc107"},
            {"label": "IV-RV Spread", "value": f"{iv_rv_spread:.2f}%", "color": "#dc3545" if iv_rv_spread > 0 else "#28a745"},
            {"label": "IV Percentile (IVP)", "value": f"{ivp:.2f}%", "color": "#28a745"},
            {"label": "Theta (Total)", "value": f"₹{seller['theta']:.2f}", "color": "#28a745"},
            {"label": "Vega (IV Risk)", "value": f"₹{seller['vega']:.2f}", "color": "#28a745"},
            {"label": "Delta (Total)", "value": f"{seller['delta']:.4f}", "color": "#28a745"},
            {"label": "Gamma (Total)", "value": f"{seller['gamma']:.6f}", "color": "#28a745"},
            {"label": "POP (Avg.)", "value": f"{seller['pop']:.2f}%", "color": "#28a745"},
            {"label": "Days to Expiry", "value": f"{market['days_to_expiry']} days", "color": "#28a745"},
            {"label": "PCR (OI Ratio)", "value": f"{market['pcr']:.2f}", "color": "#28a745"},
            {"label": "Max Pain", "value": f"{market['max_pain']:.0f}", "color": "#28a745"},
            {"label": "IV Skew Slope", "value": f"{iv_skew_slope:.4f}", "color": "#28a745"}
        ]
        for i, metric in enumerate(metrics):
            col = cols[i % len(cols)]
            col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{metric['label']}</div>
                    <div class="metric-value" style="color: {metric['color']}">{metric['value']}</div>
                </div>
            """, unsafe_allow_html=True)
        st.subheader("Volatility Analysis")
        st.plotly_chart(plot_vol_comparison(seller, hv_7, garch_7d), use_container_width=True)
        st.subheader("Portfolio Summary")
        portfolio_metrics = [
            {"label": "Total Capital", "value": f"₹{portfolio_summary['Total Capital']:,.2f}", "color": "#28a745"},
            {"label": "Capital Deployed", "value": f"₹{portfolio_summary['Capital Deployed']:,.2f}", "color": "#28a745"},
            {"label": "Exposure %", "value": f"{portfolio_summary['Exposure %']:.2f}%", "color": "#28a745"},
            {"label": "Risk on Table", "value": f"₹{portfolio_summary['Risk on Table']:,.2f}", "color": "#dc3545" if portfolio_summary['Risk %'] > 2 else "#28a745"},
            {"label": "Risk %", "value": f"{portfolio_summary['Risk %']:.2f}%", "color": "#dc3545" if portfolio_summary['Risk %'] > 2 else "#28a745"},
            {"label": "Daily Risk Limit", "value": f"₹{portfolio_summary['Daily Risk Limit']:,.2f}", "color": "#28a745"},
            {"label": "Weekly Risk Limit", "value": f"₹{portfolio_summary['Weekly Risk Limit']:,.2f}", "color": "#28a745"},
            {"label": "Realized P&L", "value": f"₹{portfolio_summary['Realized P&L']:,.2f}", "color": "#28a745" if portfolio_summary['Realized P&L'] >= 0 else "#dc3545"},
            {"label": "Drawdown %", "value": f"{portfolio_summary['Drawdown %']:.2f}%", "color": "#dc3545" if portfolio_summary['Drawdown %'] > 0 else "#28a745"},
            {"label": "Portfolio Vega", "value": f"{portfolio_summary['Portfolio Vega']:.2f}", "color": "#28a745"}
        ]
        cols = st.columns(3)
        for i, metric in enumerate(portfolio_metrics):
            col = cols[i % 3]
            col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{metric['label']}</div>
                    <div class="metric-value" style="color: {metric['color']}">{metric['value']}</div>
                </div>
            """, unsafe_allow_html=True)
        if portfolio_summary.get("Flags"):
            st.warning("🚨 Risk Alerts:")
            for flag in portfolio_summary["Flags"]:
                st.markdown(f"- {flag}")
        else:
            st.success("✅ All risks within limits.")
        st.subheader("Risk Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=portfolio_summary['Risk %'],
            title={'text': "Portfolio Risk %"},
            gauge={
                'axis': {'range': [0, 5], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#28a745"},
                'steps': [
                    {'range': [0, 2], 'color': "#28a745"},
                    {'range': [2, 4], 'color': "#ffc107"},
                    {'range': [4, 5], 'color': "#dc3545"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'value': portfolio_summary['Daily Risk Limit'] / portfolio_summary['Total Capital'] * 100
                }
            }
        ))
        fig.update_layout(template="plotly_dark", height=300, paper_bgcolor="#1a1f2b", font=dict(color="#e6e9ef"))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Option Chain Analysis")
        st.subheader("ATM ±300 Chain")
        st.dataframe(full_chain_df, style=lambda x: x.style.set_properties(**{
            'background-color': '#2c3344',
            'color': '#ffffff',
            'border-color': '#28a745',
            'text-align': 'center'
        }).format({
            'Strike': '{:.0f}',
            'Call IV': '{:.2f}',
            'Put IV': '{:.2f}',
            'IV Skew': '{:.2f}',
            'Total Theta': '{:.2f}',
            'Total Vega': '{:.2f}',
            'Straddle Price': '{:.2f}',
            'Total OI': '{:.0f}'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#28a745'), ('color', '#ffffff')]},
            {'selector': 'tr:hover', 'props': [('background-color', '#3a4257')]}
        ]))
        eff_df = full_chain_df.copy()
        eff_df["Theta/Vega"] = eff_df["Total Theta"] / eff_df["Total Vega"]
        eff_df = eff_df[["Strike", "Total Theta", "Total Vega", "Theta/Vega"]].sort_values("Theta/Vega", ascending=False)
        st.subheader("Theta/Vega Ranking")
        st.dataframe(eff_df, style=lambda x: x.style.set_properties(**{
            'background-color': '#2c3344',
            'color': '#ffffff',
            'border-color': '#28a745',
            'text-align': 'center'
        }).format({
            'Strike': '{:.0f}',
            'Total Theta': '{:.2f}',
            'Total Vega': '{:.2f}',
            'Theta/Vega': '{:.2f}'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#28a745'), ('color', '#ffffff')]},
            {'selector': 'tr:hover', 'props': [('background-color', '#3a4257')]}
        ]))
        st.subheader("Chain Analysis")
        figs = plot_chain_analysis(full_chain_df)
        col1, col2 = st.columns(2)
        col1.plotly_chart(figs[0], use_container_width=True)
        col1.plotly_chart(figs[1], use_container_width=True)
        col2.plotly_chart(figs[2], use_container_width=True)

    with tab3:
        st.header("Events & Volatility Regime")
        st.subheader("Upcoming Events")
        if not event_df.empty:
            # Format event_df to handle non-numeric values
            event_df_display = event_df.copy()
            event_df_display['Forecast'] = event_df_display['Forecast'].apply(lambda x: f"{float(x.strip('%')):.2f}" if isinstance(x, str) and '%' in x else x if pd.notnull(x) else '')
            event_df_display['Prior'] = event_df_display['Prior'].apply(lambda x: f"{float(x.strip('%')):.2f}" if isinstance(x, str) and '%' in x else x if pd.notnull(x) else '')
            st.dataframe(event_df_display, style=lambda x: x.style.set_properties(**{
                'background-color': '#2c3344',
                'color': '#ffffff',
                'border-color': '#28a745',
                'text-align': 'center'
            }).format({
                'Datetime': lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else '',
                'Forecast': '{}',
                'Prior': '{}'
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#28a745'), ('color', '#ffffff')]},
                {'selector': 'tr:hover', 'props': [('background-color', '#3a4257')]}
            ]))
            if any((datetime.strptime(config['expiry_date'], "%Y-%m-%d") - dt).days <= 7 and impact == "High" 
                   for dt, impact in zip(event_df["Datetime"], event_df["Classification"])):
                st.warning("⚠️ High-impact event within 7 days of expiry.")
        else:
            st.info("No upcoming events found.")
        st.subheader("Volatility Regime")
        st.markdown(f"""
            <div style="background-color: #2c3344; padding: 15px; border-radius: 8px;">
                <strong>Regime:</strong> {regime} (Score: {regime_score:.2f})<br>
                <strong>Note:</strong> {regime_note}<br>
                <strong>Details:</strong> {regime_explanation}
            </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.header("Strategies & Order Placement")
        st.subheader("Suggested Strategies")
        st.markdown(f"<strong>Strategies:</strong> {', '.join(strategies)}", unsafe_allow_html=True)
        st.markdown(f"<em>Rationale:</em> {strategy_rationale}", unsafe_allow_html=True)
        if event_warning:
            st.warning(event_warning)
        st.subheader("Strategy Details")
        for detail in strategy_details:
            with st.expander(f"{detail['strategy']}"):
                st.markdown(f"""
                    <strong>Strikes:</strong> {detail['strikes']}<br>
                    <strong>Premium:</strong> ₹{detail['premium']:.2f}<br>
                    <strong>Max Profit:</strong> ₹{detail['max_profit']:.2f}<br>
                    <strong>Max Loss:</strong> {'₹' if detail['max_loss'] != float('inf') else 'Unlimited'}
                """, unsafe_allow_html=True)
        st.subheader("Payoff Diagram")
        st.plotly_chart(plot_payoff_diagram(strategy_details, spot_price, config), use_container_width=True)
        st.subheader("Execute Strategy")
        strategy_options = [detail["strategy"] for detail in strategy_details]
        strategy_choice = st.selectbox("Choose Strategy", strategy_options)
        lots = st.number_input("Number of Lots", min_value=1, max_value=50, value=1)
        if st.button("Execute Order"):
            for detail in strategy_details:
                if detail['strategy'] == strategy_choice:
                    order_ids = []
                    failed_orders = []
                    for order in detail["orders"]:
                        order_id = place_order(config, order["instrument_key"], order["quantity"] * lots, order["transaction_type"])
                        if order_id:
                            order_ids.append(order_id)
                            st.success(f"Order placed: {order_id} for {order['instrument_key']} (Qty: {order['quantity'] * lots})")
                        else:
                            failed_orders.append(order["instrument_key"])
                            st.error(f"Order failed for {order['instrument_key']}")
                    if order_ids:
                        st.success(f"Successfully placed {len(order_ids)}/{len(detail['orders'])} orders for {strategy_choice}.")
                        if failed_orders:
                            st.warning(f"Warning: {len(failed_orders)} orders failed: {failed_orders}")
                    else:
                        st.error(f"All orders failed for {strategy_choice}.")
                    break
        st.subheader("Risk Summary")
        st.dataframe(strategy_df, style=lambda x: x.style.set_properties(**{
            'background-color': '#2c3f44',
            'color': '#ffffff',
            'border-color': '#28a745',
            'text-align': 'center'
        }).format({
            'Capital Used': '{:.2f}',
            'Cap Limit': '{:.0f}',
            '% Used': '{:.2f}',
            'Potential Risk': '{:.1f}',
            'Risk Limit': '{:.2f}',
            'P&L': '{:.2f}',
            'Vega': '{:.2f}',
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#28a745'), ('color', '#ffffff')]},
            {'selector': 'tr:hover', 'props': [('background-color', '#3a4257')]}
        ]))
    
    # Footer
    st.markdown("<div class='footer'>Crafted with ❤️ by Shritish Shukla</div>", unsafe_allow_html=True)

else:
    st.info("Enter your Upstox API access token in the sidebar and click 'Fetch Data' to begin.")
