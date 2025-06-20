import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from arch import arch_model
import requests
from scipy.stats import linregress

# --- Configuration ---
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
            st.error(f"Error fetching expiries: {res.status_code} - {res.text}")
            return datetime.now().strftime("%Y-%m-%d")
        except Exception as e:
            st.error(f"Exception in get_next_expiry: {e}")
            return datetime.now().strftime("%Y-%m-%d")

    config['expiry_date'] = get_next_expiry_internal()
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
        st.error(f"Error fetching option chain: {res.status_code} - {res.text}")
        return []
    except Exception as e:
        st.error(f"Exception in fetch_option_chain: {e}")
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
        st.error(f"Error fetching indices quotes: {res.status_code} - {res.text}")
        return None, None
    except Exception as e:
        st.error(f"Exception in get_indices_quotes: {e}")
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
        st.warning(f"âš ï¸ Failed to load upcoming events: {e}")
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
        st.warning(f"Exception in load_ivp: {e}")
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
        st.warning(f"Exception in extract_seller_metrics: {e}")
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
        st.warning(f"Exception in full_chain_table: {e}")
        return pd.DataFrame()

def market_metrics(option_chain, expiry_date):
    try:
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        days_to_expiry = (expiry_dt - datetime.now()).days
        call_oi = sum(opt["call_options"]["market_data"]["oi"] for opt in option_chain if "call_options" in opt and "market_data" in opt["call_options"])
        put_oi = sum(opt["put_options"]["market_data"]["oi"] for opt in option_chain if "put_options" in opt and "market_data" in opt["put_options"])
        pcr = put_oi / call_oi if call_oi != 0 else 0
        strikes = sorted(list(set([opt["strike_price"] for opt in option_chain])))
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
        st.warning(f"Exception in market_metrics: {e}")
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
        st.warning(f"Exception in calculate_volatility: {e}")
        return 0, 0, 0

def calculate_iv_skew_slope(full_chain_df):
    try:
        if full_chain_df.empty:
            return 0
        slope, _, _, _, _ = linregress(full_chain_df["Strike"], full_chain_df["IV Skew"])
        return slope
    except Exception as e:
        st.warning(f"Exception in calculate_iv_skew_slope: {e}")
        return 0

def calculate_regime(atm_iv, ivp, realized_vol, garch_vol, straddle_price, spot_price, pcr, vix, iv_skew_slope):
    try:
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
            regime = "ðŸ”¥ High Vol Trend"
            note = "Market in high volatility â€” ideal for premium selling with defined risk."
            explanation = "High IVP, elevated VIX, and wide straddle suggest strong premium opportunities."
        elif regime_score > 10:
            regime = "ðŸš§ Elevated Volatility"
            note = "Above-average volatility â€” favor range-bound strategies."
            explanation = "Moderate IVP and IV-RV spread indicate potential for mean-reverting moves."
        elif regime_score > -10:
            regime = "ðŸ˜´ Neutral Volatility"
            note = "Balanced market â€” flexible strategy selection."
            explanation = "IV and RV aligned, with moderate PCR and skew."
        else:
            regime = "ðŸ’¤ Low Volatility"
            note = "Low volatility â€” cautious selling or long vega plays."
            explanation = "Low IVP, tight straddle, and low VIX suggest limited movement."
        return regime_score, regime, note, explanation
    except Exception as e:
        st.warning(f"Exception in calculate_regime: {e}")
        return 0, "Unknown", "Error calculating regime.", ""

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
            if (level == "High") and (0 <= (datetime.strptime(expiry_date, "%Y-%m-%d") - dt).days <= event_window):
                high_impact_event_near = True
            if level == "High" and pd.notnull(row["Forecast"]) and pd.notnull(row["Prior"]):
                forecast = float(str(row["Forecast"]).strip("%")) if "%" in str(row["Forecast"]) else float(row["Forecast"])
                prior = float(str(row["Prior"]).strip("%")) if "%" in str(row["Prior"]) else float(row["Prior"])
                if abs(forecast - prior) > 0.5:
                    event_impact_score += 1
        except Exception as e:
            st.warning(f"Error processing event row: {row}. Error: {e}")
            continue
    if high_impact_event_near:
        event_warning = f"âš ï¸ High-impact event within {event_window} days of expiry. Prefer defined-risk strategies."
    if event_impact_score > 0:
        rationale.append(f"High-impact events with significant forecast deviations ({event_impact_score} events).")
    expected_move_pct = (straddle_price / spot_price) * 100
    if regime_label == "ðŸ”¥ High Vol Trend":
        if high_impact_event_near or event_impact_score > 0:
            strategies = ["Iron Fly", "Wide Strangle"]
            rationale.append("High volatility with major event â€” use defined-risk structures.")
        else:
            strategies = ["Iron Fly", "Wide Strangle"]
            rationale.append("Strong IV premium â€” neutral strategies for premium capture.")
    elif regime_label == "ðŸš§ Elevated Volatility":
        strategies = ["Iron Condor", "Jade Lizard"]
        rationale.append("Volatility above average â€” range-bound strategies offer favorable reward-risk.")
    elif regime_label == "ðŸ˜´ Neutral Volatility":
        if days_to_expiry >= 3:
            strategies = ["Jade Lizard", "Bull Put Spread"]
            rationale.append("Market balanced â€” slight directional bias strategies offer edge.")
        else:
            strategies = ["Iron Fly"]
            rationale.append("Tight expiry â€” quick theta-based capture via short Iron Fly.")
    elif regime_label == "ðŸ’¤ Low Volatility":
        if days_to_expiry > 7:
            strategies = ["Straddle", "Calendar Spread"]
            rationale.append("Low IV with longer expiry â€” benefit from potential IV increase.")
        else:
            strategies = ["Straddle", "ATM Strangle"]
            rationale.append("Low IV â€” premium collection favorable but monitor for breakout risk.")
    if event_impact_score > 0 and not high_impact_event_near:
        strategies = [s for s in strategies if "Iron" in s or "Lizard" in s or "Spread" in s]
    if ivp > 85 and iv_minus_rv > 5:
        rationale.append(f"Volatility overpriced (IVP: {ivp}%, IV-RV: {iv_minus_rv}%) â€” ideal for selling premium.")
    elif ivp < 30:
        rationale.append(f"Volatility underpriced (IVP: {ivp}%) â€” avoid unhedged selling.")
    rationale.append(f"Expected move: Â±{expected_move_pct:.2f}% based on straddle price.")
    return strategies, " | ".join(rationale), event_warning

@st.cache_data(ttl=60)
def fetch_trade_data(config, full_chain_df):
    try:
        url_positions = f"{config['base_url']}/portfolio/short-term-positions"
        res_positions = requests.get(url_positions, headers=config['headers'])
        url_trades = f"{config['base_url']}/order/trades/get-trades-for-day"
        res_trades = requests.get(url_trades, headers=config['headers'])
        positions = []
        trades = []
        if res_positions.status_code == 200:
            positions = res_positions.json().get("data", [])
        else:
            st.warning(f"Error fetching positions: {res_positions.status_code} - {res_positions.text}")
        if res_trades.status_code == 200:
            trades = res_trades.json().get("data", [])
        else:
            st.warning(f"Error fetching trades: {res_trades.status_code} - {res_trades.text}")
        trade_counts = {}
        for trade in trades:
            instrument = trade.get("instrument_token", "")
            strat = "Straddle" if "NIFTY" in instrument and ("CE" in instrument or "PE" in instrument) else "Unknown"
            trade_counts[strat] = trade_counts.get(strat, 0) + 1
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
            capital = pos["quantity"] * pos["average_price"]
            trades_df_list.append({
                "strategy": strategy,
                "capital_used": abs(capital),
                "potential_loss": abs(capital * 0.1),
                "realized_pnl": pos["pnl"],
                "trades_today": trade_counts.get(strategy, 0),
                "sl_hit": pos["pnl"] < -abs(capital * 0.05),
                "vega": full_chain_df["Total Vega"].mean() if not full_chain_df.empty else 0
            })
        return pd.DataFrame(trades_df_list) if trades_df_list else pd.DataFrame()
    except Exception as e:
        st.error(f"Exception in fetch_trade_data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_funds_and_margin(config):
    try:
        url = f"{config['base_url']}/user/get-funds-and-margin?segment=EQ"
        res = requests.get(url, headers=config['headers'])
        if res.status_code == 200:
            data = res.json().get("data", {})
            return {
                "available_margin": data.get("available_margin", 0),
                "used_margin": data.get("used_margin", 0),
                "total_funds": data.get("equity", {}).get("total_funds", 0)
            }
        st.warning(f"Error fetching funds and margin: {res.status_code} - {res.text}")
        return {"available_margin": 0, "used_margin": 0, "total_funds": 0}
    except Exception as e:
        st.warning(f"Exception in get_funds_and_margin: {e}")
        return {"available_margin": 0, "used_margin": 0, "total_funds": 0}

@st.cache_data(ttl=60)
def get_option_greeks(config, instrument_keys):
    try:
        if not instrument_keys:
            return {}
        url = f"{config['base_url'].replace('v2', 'v3')}/market-quote/option-greek"
        params = {"instrument_key": ",".join(instrument_keys)}
        res = requests.get(url, headers=config['headers'], params=params)
        if res.status_code == 200:
            return res.json()["data"]
        st.warning(f"Error fetching Greeks: {res.status_code} - {res.text}")
        return {}
    except Exception as e:
        st.warning(f"Exception in get_option_greeks: {e}")
        return {}

def calculate_sharpe_ratio():
    try:
        # Simulate 252 daily returns (1 year) for an option selling portfolio
        np.random.seed(42)  # For reproducibility
        daily_returns = np.random.normal(0.001, 0.01, 252)  # Mean 0.1%, Std 1%
        annual_return = np.mean(daily_returns) * 252
        annual_volatility = np.std(daily_returns) * np.sqrt(252)
        risk_free_rate = 0.06 / 252  # 6% annual risk-free rate, daily
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        return round(sharpe_ratio, 2)
    except Exception as e:
        st.warning(f"Exception in calculate_sharpe_ratio: {e}")
        return 0

def evaluate_full_risk(trades_df, config, regime_label, vix):
    try:
        daily_risk_limit = config['daily_risk_limit_pct'] * config['total_capital']
        weekly_risk_limit = config['weekly_risk_limit_pct'] * config['total_capital']
        max_drawdown_pct = 0.05 if vix > 20 else 0.03 if vix > 12 else 0.02
        max_drawdown = max_drawdown_pct * config['total_capital']
        strategy_summary = []
        total_cap_used = 0
        total_risk_used = 0
        total_realized_pnl = 0
        total_vega = 0
        total_theta = 0
        flags = []
        for _, row in trades_df.iterrows():
            strat = row["strategy"]
            capital_used = row["capital_used"]
            potential_risk = row["potential_loss"]
            pnl = row["realized_pnl"]
            sl_hit = row["sl_hit"]
            trades_today = row["trades_today"]
            vega = row["vega"]
            cfg = config['risk_config'].get(strat, {"capital_pct": 0.1, "risk_per_trade_pct": 0.01})
            risk_factor = 1.2 if regime_label == "ðŸ”¥ High Vol Trend" else 0.8 if regime_label == "ðŸ’¤ Low Volatility" else 1.0
            max_cap = cfg["capital_pct"] * config['total_capital']
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
                "Risk OK?": "âœ…" if risk_ok else "âŒ"
            })
            total_cap_used += capital_used
            total_risk_used += potential_risk
            total_realized_pnl += pnl
            total_vega += vega
            if not risk_ok:
                flags.append(f"âŒ {strat} exceeded risk limit")
            if sl_hit and trades_today > 3:
                flags.append(f"âš ï¸ {strat} shows possible revenge trading (SL hit + {trades_today} trades)")
        net_dd = -total_realized_pnl if total_realized_pnl < 0 else 0
        exposure_pct = round(total_cap_used / config['total_capital'] * 100, 2) if config['total_capital'] else 0
        risk_pct = round(total_risk_used / config['total_capital'] * 100, 2) if config['total_capital'] else 0
        dd_pct = round(net_dd / config['total_capital'] * 100, 2) if config['total_capital'] else 0
        portfolio_summary = {
            "Total Capital": config['total_capital'],
            "Capital Deployed": total_cap_used,
            "Exposure %": exposure_pct,
            "Risk on Table": total_risk_used,
            "Risk %": risk_pct,
            "Daily Risk Limit": daily_risk_limit,
            "Weekly Risk Limit": weekly_risk_limit,
            "Realized P&L": total_realized_pnl,
            "Drawdown â‚¹": net_dd,
            "Drawdown %": dd_pct,
            "Portfolio Vega": round(total_vega, 2),
            "Max Drawdown Allowed": max_drawdown,
            "Flags": flags
        }
        return pd.DataFrame(strategy_summary), portfolio_summary
    except Exception as e:
        st.error(f"Exception in evaluate_full_risk: {e}")
        return pd.DataFrame(), {}

def find_option_by_strike(option_chain, strike, option_type):
    try:
        for opt in option_chain:
            if abs(opt["strike_price"] - strike) < 0.01:
                if option_type == "CE" and "call_options" in opt:
                    return opt["call_options"]
                elif option_type == "PE" and "put_options" in opt:
                    return opt["put_options"]
        st.warning(f"No option found for strike {strike} {option_type}")
        return None
    except Exception as e:
        st.warning(f"Exception in find_option_by_strike: {e}")
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
        res = requests.post(url, headers=config['headers'], json=payload)
        st.info(f"Order response for {instrument_key}: {res.status_code} - {res.text}")
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
        st.error(f"Exception in place_order: {e}")
        return None

def exit_all_positions(config):
    try:
        url = f"{config['base_url']}/order/positions/exit?segment=EQ"
        res = requests.post(url, headers=config['headers'])
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                order_ids = data.get("data", {}).get("order_ids", [])
                st.success(f"Successfully initiated exit for {len(order_ids)} positions.")
                return order_ids
            st.error(f"Unexpected response status: {data}")
            return []
        elif res.status_code == 400:
            errors = res.json().get("errors", [])
            for error in errors:
                if error.get("errorCode") in ["UDAPI1108", "UDAPI1109"]:
                    st.warning(f"No open positions to exit or limit exceeded: {error.get('message')}")
                else:
                    st.error(f"Exit failed: {error.get('message')}")
            return []
        st.error(f"Error exiting positions: {res.status_code} - {res.text}")
        return []
    except Exception as e:
        st.error(f"Exception in exit_all_positions: {e}")
        return []

def logout(config):
    try:
        url = f"{config['base_url']}/logout"
        res = requests.delete(url, headers=config['headers'])
        if res.status_code == 200:
            st.success("Successfully logged out.")
            st.session_state.access_token = ""
            st.session_state.logged_in = False
            st.cache_data.clear()
        else:
            st.error(f"Logout failed: {res.status_code} - {res.text}")
    except Exception as e:
        st.error(f"Exception in logout: {e}")

# --- Strategy Definitions ---
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
        return None
    detail = func_map[strategy_name](option_chain, spot_price, config, lots)
    if detail:
        instrument_keys = [order["instrument_key"] for order in detail["orders"]]
        current_greeks = get_option_greeks(config, instrument_keys)
        updated_orders = []
        prices = {}
        for order in detail["orders"]:
            key = order["instrument_key"]
            ltp = current_greeks.get(key, {}).get("last_price", 0)
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

def _iron_fly_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    wing = 100
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike, "PE")
    ce_long_opt = find_option_by_strike(option_chain, strike + wing, "CE")
    pe_long_opt = find_option_by_strike(option_chain, strike - wing, "PE")
    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        st.error("Error: Missing options for Iron Fly")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Iron Fly", "strikes": [strike, strike, strike + wing, strike - wing],
            "premium": 0, "max_loss": 0, "max_profit": 0, "orders": orders}

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
        st.error("Error: Missing options for Iron Condor")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Iron Condor", "strikes": [ce_short_strike, pe_short_strike, ce_long_strike, pe_long_strike],
            "premium": 0, "max_loss": 0, "max_profit": 0, "orders": orders}

def _jade_lizard_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 50
    put_strike = atm["strike_price"]
    put_long_strike = atm["strike_price"] - 100
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, put_long_strike, "PE")
    if not all([ce_short_opt, pe_short_opt, pe_long_opt]):
        st.error("Error: Missing options for Jade Lizard")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Jade Lizard", "strikes": [call_strike, put_strike, put_long_strike],
            "premium": 0, "max_loss": 0, "max_profit": 0, "orders": orders}

def _straddle_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike, "PE")
    if not all([ce_short_opt, pe_short_opt]):
        st.error("Error: Missing options for Straddle")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    return {"strategy": "Straddle", "strikes": [strike, strike],
            "premium": 0, "max_loss": float("inf"), "max_profit": 0, "orders": orders}

def _calendar_spread_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    ce_long_opt = ce_short_opt
    if not ce_short_opt:
        st.error("Error: Missing options for Calendar Spread")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Calendar Spread", "strikes": [strike, strike],
            "premium": 0, "max_loss": 0, "max_profit": float("inf"), "orders": orders}

def _bull_put_spread_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    short_strike = atm["strike_price"] - 50
    long_strike = atm["strike_price"] - 100
    pe_short_opt = find_option_by_strike(option_chain, short_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, long_strike, "PE")
    if not all([pe_short_opt, pe_long_opt]):
        st.error("Error: Missing options for Bull Put Spread")
        return None
    orders = [
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY"}
    ]
    return {"strategy": "Bull Put Spread", "strikes": [short_strike, long_strike],
            "premium": 0, "max_loss": 0, "max_profit": 0, "orders": orders}

def _wide_strangle_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 100
    put_strike = atm["strike_price"] - 100
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    if not all([ce_short_opt, pe_short_opt]):
        st.error("Error: Missing options for Wide Strangle")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    return {"strategy": "Wide Strangle", "strikes": [call_strike, put_strike],
            "premium": 0, "max_loss": float("inf"), "max_profit": 0, "orders": orders}

def _atm_strangle_calc(option_chain, spot_price, config, lots):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    call_strike = atm["strike_price"] + 50
    put_strike = atm["strike_price"] - 50
    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")
    if not all([ce_short_opt, pe_short_opt]):
        st.error("Error: Missing options for ATM Strangle")
        return None
    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
    ]
    return {"strategy": "ATM Strangle", "strikes": [call_strike, put_strike],
            "premium": 0, "max_loss": float("inf"), "max_profit": 0, "orders": orders}

# --- Plotting Functions ---
def plot_payoff_diagram(strategy_details, spot_price, config):
    if not strategy_details:
        st.warning("No strategy details to plot for payoff diagram.")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("darkgrid")
    min_strike = min(min(d["strikes"]) for d in strategy_details if d["strikes"]) - 200
    max_strike = max(max(d["strikes"]) for d in strategy_details if d["strikes"]) + 200
    strikes = np.linspace(min_strike, max_strike, 200)
    for detail in strategy_details:
        payoffs = np.zeros_like(strikes)
        for order in detail["orders"]:
            instrument_key = order["instrument_key"]
            qty = order["quantity"]
            is_buy = order["transaction_type"] == "BUY"
            multiplier = 1 if is_buy else -1
            try:
                parts = instrument_key.split('|')[1].replace('Nifty 50', '')
                strike_str = ''.join(filter(str.isdigit, parts))
                order_strike = float(strike_str)
            except (ValueError, IndexError):
                order_strike = detail["strikes"][detail["orders"].index(order)]
            is_call = "CE" in instrument_key
            price = order.get("current_price", 0)
            if is_call:
                payoff = multiplier * (np.maximum(0, strikes - order_strike) - price)
            else:
                payoff = multiplier * (np.maximum(0, order_strike - strikes) - price)
            payoffs += payoff * (qty / config["lot_size"])
        ax.plot(strikes, payoffs, label=detail["strategy"], linewidth=2)
    ax.axvline(spot_price, linestyle="--", color="yellow", label=f"Spot Price: {spot_price:.0f}")
    ax.axhline(0, linestyle="--", color="red", label="Breakeven")
    ax.set_title("ðŸ“Š Payoff Diagram", color="white")
    ax.set_xlabel("Underlying Price", color="white")
    ax.set_ylabel("Payoff (â‚¹)", color="white")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['top'].set_color('white')
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    st.pyplot(fig)

def plot_vol_comparison(seller, hv_7, garch_7d):
    labels = ['ATM IV', 'Realized Vol (7D)', 'GARCH Vol (7D)']
    values = [seller["avg_iv"], hv_7, garch_7d]
    colors = ['#00BFFF', '#32CD32', '#FF4500']
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{bar.get_height():.2f}%", ha='center', va='bottom', color='white')
    ax.set_title("ðŸ“Š Volatility Comparison: IV vs RV vs GARCH", color="white")
    ax.set_ylabel("Annualized Volatility (%)", color="white")
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['top'].set_color('white')
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    st.pyplot(fig)

def plot_chain_analysis(full_chain_df):
    if full_chain_df.empty:
        st.warning("No option chain data to plot.")
        return
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    line_color_1 = "#8A2BE2"
    line_color_2 = "#3CB371"
    line_color_3 = "#FFD700"
    bar_palette = "viridis"
    sns.lineplot(data=full_chain_df, x="Strike", y="IV Skew", ax=axes[0, 0], marker="o", color=line_color_1)
    axes[0, 0].set_title("IV Skew", color="white")
    axes[0, 0].axhline(0, linestyle='--', color="gray")
    axes[0, 0].tick_params(axis='x', colors='white', rotation=45)
    axes[0, 0].tick_params(axis='y', colors='white')
    axes[0, 0].set_xlabel("Strike", color="white")
    axes[0, 0].set_ylabel("IV Skew", color="white")
    sns.lineplot(data=full_chain_df, x="Strike", y="Total Theta", ax=axes[0, 1], marker="o", color=line_color_2)
    axes[0, 1].set_title("Total Theta", color="white")
    axes[0, 1].tick_params(axis='x', colors='white', rotation=45)
    axes[0, 1].tick_params(axis='y', colors='white')
    axes[0, 1].set_xlabel("Strike", color="white")
    axes[0, 1].set_ylabel("Total Theta", color="white")
    sns.lineplot(data=full_chain_df, x="Strike", y="Straddle Price", ax=axes[1, 0], marker="o", color=line_color_3)
    axes[1, 0].set_title("Straddle Price", color="white")
    axes[1, 0].tick_params(axis='x', colors='white', rotation=45)
    axes[1, 0].tick_params(axis='y', colors='white')
    axes[1, 0].set_xlabel("Strike", color="white")
    axes[1, 0].set_ylabel("Straddle Price", color="white")
    sns.barplot(data=full_chain_df, x="Strike", y="Total OI", ax=axes[1, 1], palette=bar_palette)
    axes[1, 1].set_title("Total OI", color="white")
    axes[1, 1].tick_params(axis='x', colors='white', rotation=45)
    axes[1, 1].tick_params(axis='y', colors='white')
    axes[1, 1].set_xlabel("Strike", color="white")
    axes[1, 1].set_ylabel("Total OI", color="white")
    fig.patch.set_facecolor('#0E1117')
    for ax in axes.flatten():
        ax.set_facecolor('#0E1117')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.spines['top'].set_color('white')
    plt.tight_layout()
    st.pyplot(fig)

def plot_allocation_pie(strategy_df, config):
    if strategy_df.empty:
        st.warning("No strategy data to plot allocation.")
        return
    labels = strategy_df["Strategy"]
    sizes = strategy_df["Capital Used"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'color': 'white'})
    ax.set_title("Capital Allocation by Strategy", color="white")
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    st.pyplot(fig)

def plot_drawdown_trend(portfolio_summary):
    # Simulate drawdown trend since no trade history
    np.random.seed(42)
    days = 30
    drawdowns = np.cumsum(np.random.normal(-1000, 5000, days))
    drawdowns = np.maximum(drawdowns, -portfolio_summary["Max Drawdown Allowed"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(days), drawdowns, color="#00BFFF", label="Drawdown Trend")
    ax.axhline(-portfolio_summary["Max Drawdown Allowed"], linestyle="--", color="red", label="Max Drawdown Allowed")
    ax.set_title("Drawdown Trend (â‚¹)", color="white")
    ax.set_xlabel("Days", color="white")
    ax.set_ylabel("Drawdown (â‚¹)", color="white")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['top'].set_color('white')
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
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['top'].set_color('white')
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    st.pyplot(fig)

# --- Streamlit UI ---
st.set_page_config(
    page_title="Volguard - Your Trading Copilot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stApp {
        background-color: #0E1117;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border-radius: 5px;
    }
    .stAlert {
        background-color: #333333;
        color: white;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1E90FF;
    }
    .metric-box {
        background-color: #1A1C24;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
        color: white;
    }
    .metric-box h3 {
        color: #6495ED;
        margin-bottom: 5px;
    }
    .metric-box .value {
        font-size: 1.8em;
        font-weight: bold;
        color: #00BFFF;
    }
    .small-metric-box {
        background-color: #1A1C24;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 5px;
        box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.1);
        color: white;
        font-size: 0.9em;
    }
    .small-metric-box h4 {
        color: #6495ED;
        margin-bottom: 3px;
        font-size: 1em;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #262730;
        color: white;
        border-radius: 5px 5px 0 0;
       2160    padding: 10px 15px;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        background-color: #3A3B40;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #1E90FF;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Volguard - Your Trading Copilot")

# Initialize session state
if 'access_token' not in st.session_state:
    st.session_state.access_token = ""
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

access_token = st.sidebar.text_input("Enter Upstox Access Token", type="password", value=st.session_state.access_token)
if st.sidebar.button("Login"):
    if access_token:
        # Validate token by making a test API call
        config = get_config(access_token)
        test_url = f"{config['base_url']}/user/profile"
        try:
            res = requests.get(test_url, headers=config['headers'])
            if res.status_code == 200:
                st.session_state.access_token = access_token
                st.session_state.logged_in = True
                st.sidebar.success("Logged in successfully!")
            else:
                st.sidebar.error(f"Invalid token: {res.status_code} - {res.text}")
        except Exception as e:
            st.sidebar.error(f"Error validating token: {e}")
    else:
        st.sidebar.error("Please enter an access token.")

if st.session_state.logged_in and st.sidebar.button("Logout"):
    config = get_config(st.session_state.access_token)
    logout(config)
    st.experimental_rerun()

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared. Reloading data...")

if st.session_state.logged_in and access_token:
    config = get_config(access_token)
    st.sidebar.success("Access Token provided. Fetching data...")

    @st.cache_data(show_spinner="Analyzing market data...")
    def load_all_data(config):
        option_chain = fetch_option_chain(config)
        if not option_chain:
            st.error("Failed to fetch option chain data.")
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        spot_price = option_chain[0]["underlying_spot_price"]
        vix, nifty = get_indices_quotes(config)
        if vix is None or nifty is None:
            st.error("Failed to fetch India VIX or Nifty 50 data.")
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        seller = extract_seller_metrics(option_chain, spot_price)
        if not seller:
            st.error("Failed to extract seller metrics.")
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        full_chain_df = full_chain_table(option_chain, spot_price)
        if full_chain_df.empty:
            st.error("Failed to create full option chain table.")
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        market = market_metrics(option_chain, config['expiry_date'])
        ivp = load_ivp(config, seller["avg_iv"])
        hv_7, garch_7d, iv_rv_spread = calculate_volatility(config, seller["avg_iv"])
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
        strategy_details = [get_strategy_details(strat, option_chain, spot_price, config, lots=1) for strat in strategies if get_strategy_details(strat, option_chain, spot_price, config, lots=1)]
        trades_df = fetch_trade_data(config, full_chain_df)
        strategy_df, portfolio_summary = evaluate_full_risk(trades_df, config, regime, vix)
        funds_data = get_funds_and_margin(config)
        sharpe_ratio = calculate_sharpe_ratio()
        return (option_chain, spot_price, vix, nifty, seller, full_chain_df, market,
                ivp, hv_7, garch_7d, iv_rv_spread, iv_skew_slope, regime_score, regime,
                regime_note, regime_explanation, event_df, strategies, strategy_rationale,
                event_warning, strategy_details, trades_df, strategy_df, portfolio_summary, funds_data, sharpe_ratio)

    (option_chain, spot_price, vix, nifty, seller, full_chain_df, market,
     ivp, hv_7, garch_7d, iv_rv_spread, iv_skew_slope, regime_score, regime,
     regime_note, regime_explanation, event_df, strategies, strategy_rationale,
     event_warning, strategy_details, trades_df, strategy_df, portfolio_summary, funds_data, sharpe_ratio) = load_all_data(config)

    if option_chain is None:
        st.stop()

    st.markdown("---")
    st.markdown("<h1 style='text-align: center; color: #1E90FF;'>Market Insights Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("Current Market Snapshot")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-box'><h3>ðŸ“ˆ Nifty 50 Spot</h3><div class='value'>{nifty:.2f}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-box'><h3>ðŸŒ¡ï¸ India VIX</h3><div class='value'>{vix:.2f}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-box'><h3>ðŸŽ¯ ATM Strike</h3><div class='value'>{seller['strike']:.0f}</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-box'><h3>ðŸ’° Straddle Price</h3><div class='value'>â‚¹{seller['straddle_price']:.2f}</div></div>", unsafe_allow_html=True)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown(f"<div class='metric-box'><h3>ðŸ“‰ ATM IV</h3><div class='value'>{seller['avg_iv']:.2f}%</div></div>", unsafe_allow_html=True)
    with col6:
        st.markdown(f"<div class='metric-box'><h3>ðŸ“Š IVP</h3><div class='value'>{ivp}%</div></div>", unsafe_allow_html=True)
    with col7:
        st.markdown(f"<div class='metric-box'><h3>â³ Days to Expiry</h3><div class='value'>{market['days_to_expiry']}</div></div>", unsafe_allow_html=True)
    with col8:
        st.markdown(f"<div class='metric-box'><h3>ðŸ” PCR</h3><div class='value'>{market['pcr']:.2f}</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["ðŸ“Š Dashboard", "â›“ï¸ Option Chain Analysis", "ðŸ’¡ Strategy Suggestions", "ðŸ“ˆ Risk & Portfolio", "ðŸš€ Place Orders", "ðŸ›¡ï¸ Risk Management Dashboard"])

    with tab1:
        st.markdown("<h2 style='color: #1E90FF;'>Market Overview</h2>", unsafe_allow_html=True)
        col_t1_1, col_t1_2 = st.columns([0.6, 0.4])
        with col_t1_1:
            st.subheader("Volatility Landscape")
            plot_vol_comparison(seller, hv_7, garch_7d)
            st.markdown(f"<div class='small-metric-box'><h4>ðŸ§® IV - RV Spread:</h4> {iv_rv_spread:+.2f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-metric-box'><h4>ðŸ“‰ IV Skew Slope:</h4> {iv_skew_slope:.4f}</div>", unsafe_allow_html=True)
            st.subheader("Breakeven & Max Pain")
            st.markdown(f"<div class='small-metric-box'><h4>ðŸ“Š Breakeven Range:</h4> {seller['strike'] - seller['straddle_price']:.0f} â€“ {seller['strike'] + seller['straddle_price']:.0f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-metric-box'><h4>ðŸŽ¯ Max Pain:</h4> {market['max_pain']:.0f}</div>", unsafe_allow_html=True)
        with col_t1_2:
            st.subheader("Greeks at ATM")
            st.markdown(f"<div class='small-metric-box'><h4>â³ Theta (Total):</h4> â‚¹{seller['theta']:.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-metric-box'><h4>ðŸŒªï¸ Vega (IV Risk):</h4> â‚¹{seller['vega']:.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-metric-box'><h4>ðŸ“ Delta:</h4> {seller['delta']:.4f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-metric-box'><h4>âš¡ Gamma:</h4> {seller['gamma']:.6f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-metric-box'><h4>ðŸŽ¯ POP (Avg):</h4> {seller['pop']:.2f}%</div>", unsafe_allow_html=True)
            st.subheader("Upcoming Events")
            if not event_df.empty:
                st.dataframe(event_df.style.set_properties(**{'background-color': '#1A1C24', 'color': 'white'}), use_container_width=True)
                if event_warning:
                    st.warning(event_warning)
            else:
                st.info("No upcoming events before expiry.")

    with tab2:
        st.markdown("<h2 style='color: #1E90FF;'>Option Chain Analysis</h2>", unsafe_allow_html=True)
        plot_chain_analysis(full_chain_df)
        st.subheader("ATM Â±300 Chain Table")
        st.dataframe(full_chain_df.style.set_properties(**{'background-color': '#1A1C24', 'color': 'white'}), use_container_width=True)
        st.subheader("Theta/Vega Ranking")
        eff_df = full_chain_df.copy()
        if not eff_df.empty:
            eff_df["Theta/Vega"] = eff_df["Total Theta"] / eff_df["Total Vega"]
            eff_df = eff_df[["Strike", "Total Theta", "Total Vega", "Theta/Vega"]].sort_values("Theta/Vega", ascending=False)
            st.dataframe(eff_df.style.format(precision=2).set_properties(**{'background-color': '#1A1C24', 'color': 'white'}), use_container_width=True)
        else:
            st.info("No data for Theta/Vega ranking.")

    with tab3:
        st.markdown("<h2 style='color: #1E90FF;'>Strategy Suggestions</h2>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='metric-box'><h3>ðŸ§  Volatility Regime: {regime}</h3><p style='color: #6495ED;'>Score: {regime_score:.2f}</p><p>{regime_note}</p><p><i>{regime_explanation}</i></p>",
            unsafe_allow_html=True,
        )
        st.subheader("Recommended Strategies")
        if strategies:
            st.success(f"**Suggested Strategies:** {', '.join(strategies)}")
            st.info(f"**Rationale:** {strategy_rationale}")
            if event_warning:
                st.warning(event_warning)
            for strat in strategies:
                st.markdown(f"### {strat}")
                detail = get_strategy_details(strat, option_chain, spot_price, config, lots=1)
                if detail:
                    # Display strategy details
                    st.markdown(f"**Strategy:** {detail['strategy']}")
                    st.markdown(f"**Strikes:** {', '.join(map(str, detail['strikes']))}")
                    st.markdown(f"**Premium Collected:** â‚¹{detail['premium_total']:.2f}")
                    st.markdown(
                        f"**Max Profit:** â‚¹{detail['max_profit']:.2f}"
                        if detail['max_profit'] != float('inf')
                        else "**Max Profit:** Unlimited"
                    )
                    st.markdown(
                        f"**Max Loss:** â‚¹{detail['max_loss']:.2f}"
                        if detail['max_loss'] != float('inf')
                        else "**Max Loss:** Unlimited"
                    )
                    # Display individual legs
                    st.markdown("#### Individual Legs (for 1 lot):")
                    order_df = pd.DataFrame(
                        {
                            "Instrument": [order["instrument_key"] for order in detail["orders"]],
                            "Type": [order["transaction_type"] for order in detail["orders"]],
                            "Option Type": ["CE" if "CE" in order["instrument_key"] else "PE" for order in detail["orders"]],
                            "Strike": detail["strikes"],
                            "Quantity (per lot)": [config["lot_size"] for _ in detail["orders"]],
                            "LTP": [order.get("current_price", 0) for order in detail["orders"]],
                        }
                    )
                    st.dataframe(
                        order_df.style.format(precision=2).set_properties(
                            **{"background-color": "#1A1C24", "color": "white"}
                        ),
                        use_container_width=True,
                    )
                    # Payoff diagram
                    st.subheader("Payoff Diagram")
                    plot_payoff_diagram([detail], spot_price, config)
                    # Risk metrics
                    st.subheader("Risk Metrics")
                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.markdown(
                            f"<div class='small-metric-box'><h4>ðŸ’° Premium:</h4> â‚¹{detail['premium_total']:.2f}</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='small-metric-box'><h4>ðŸ“‰ Max Loss:</h4> â‚¹{detail['max_loss']:.2f}</div>"
                            if detail['max_loss'] != float('inf')
                            else "<div class='small-metric-box'><h4>ðŸ“‰ Max Loss:</h4> Unlimited</div>",
                            unsafe_allow_html=True,
                        )
                    with col_r2:
                        st.markdown(
                            f"<div class='small-metric-box'><h4>ðŸ“ˆ Max Profit:</h4> â‚¹{detail['max_profit']:.2f}</div>"
                            if detail['max_profit'] != float('inf')
                            else "<div class='small-metric-box'><h4>ðŸ“ˆ Max Profit:</h4> Unlimited</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='small-metric-box'><h4>ðŸŽ¯ Risk/Reward:</h4> {detail['max_loss'] / detail['max_profit']:.2f}</div>"
                            if detail['max_profit'] > 0 and detail['max_loss'] != float('inf')
                            else "<div class='small-metric-box'><h4>ðŸŽ¯ Risk/Reward:</h4> Undefined</div>",
                            unsafe_allow_html=True,
                        )
                    with col_r3:
                        capital_required = detail['premium_total'] * 10  # Approx margin
                        st.markdown(
                            f"<div class='small-metric-box'><h4>ðŸ’¸ Capital Required:</h4> â‚¹{capital_required:.2f}</div>",
                            unsafe_allow_html=True,
                        )
                    # Order placement
                    lots = st.number_input(
                        f"Lots for {strat}", min_value=1, value=1, step=1, key=f"lots_{strat}"
                    )
                    if st.button(f"Place {strat} Order", key=f"place_{strat}"):
                        updated_detail = get_strategy_details(strat, option_chain, spot_price, config, lots)
                        if updated_detail:
                            for order in updated_detail["orders"]:
                                order_id = place_order(
                                    config,
                                    order["instrument_key"],
                                    order["quantity"],
                                    order["transaction_type"],
                                )
                                if not order_id:
                                    st.error(f"Failed to place order for {order['instrument_key']}")
                            st.success(f"Placed {strat} order with {lots} lots!")
                        else:
                            st.error(f"Failed to generate {strat} order details.")
                else:
                    st.error(f"Unable to fetch details for {strat}.")
        else:
            st.info("No strategies suggested for the current market conditions.")

    with tab4:
        st.markdown("<h2 style='color: #1E90FF;'>Risk & Portfolio Analysis</h2>", unsafe_allow_html=True)
        st.subheader("Portfolio Summary")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            st.markdown(
                f"<div class='metric-box'><h3>ðŸ’° Total Capital</h3><div class='value'>â‚¹{portfolio_summary['Total Capital']:.2f}</div></div>",
                unsafe_allow_html=True,
            )
        with col_p2:
            st.markdown(
                f"<div class='metric-box'><h3>ðŸ“ˆ Capital Deployed</h3><div class='value'>â‚¹{portfolio_summary['Capital Deployed']:.2f}</div></div>",
                unsafe_allow_html=True,
            )
        with col_p3:
            st.markdown(
                f"<div class='metric-box'><h3>âš–ï¸ Exposure %</h3><div class='value'>{portfolio_summary['Exposure %']:.2f}%</div></div>",
                unsafe_allow_html=True,
            )
        with col_p4:
            st.markdown(
                f"<div class='metric-box'><h3>ðŸ“‰ Drawdown %</h3><div class='value'>{portfolio_summary['Drawdown %']:.2f}%</div></div>",
                unsafe_allow_html=True,
            )
        st.subheader("Strategy Risk Table")
        if not strategy_df.empty:
            st.dataframe(
                strategy_df.style.format(precision=2).set_properties(
                    **{"background-color": "#1A1C24", "color": "white"}
                ),
                use_container_width=True,
            )
            st.subheader("Capital Allocation")
            plot_allocation_pie(strategy_df, config)
            st.subheader("Risk Heatmap")
            plot_risk_heatmap(strategy_df)
        else:
            st.info("No active strategies to display.")
        if portfolio_summary.get("Flags"):
            st.subheader("Risk Alerts")
            for flag in portfolio_summary["Flags"]:
                st.error(flag)

    with tab5:
        st.markdown("<h2 style='color: #1E90FF;'>Place Orders</h2>", unsafe_allow_html=True)
        st.subheader("Select Strategy to Place Order")
        selected_strategy = st.selectbox("Choose Strategy", strategies, key="order_strategy")
        if selected_strategy:
            lots = st.number_input("Number of Lots", min_value=1, value=1, step=1, key="order_lots")
            detail = get_strategy_details(selected_strategy, option_chain, spot_price, config, lots)
            if detail:
                st.subheader(f"{selected_strategy} Order Details")
                order_df = pd.DataFrame(
                    {
                        "Instrument": [order["instrument_key"] for order in detail["orders"]],
                        "Type": [order["transaction_type"] for order in detail["orders"]],
                        "Option Type": ["CE" if "CE" in order["instrument_key"] else "PE" for order in detail["orders"]],
                        "Strike": detail["strikes"],
                        "Quantity": [order["quantity"] for order in detail["orders"]],
                        "LTP": [order.get("current_price", 0) for order in detail["orders"]],
                    }
                )
                st.dataframe(
                    order_df.style.format(precision=2).set_properties(
                        **{"background-color": "#1A1C24", "color": "white"}
                    ),
                    use_container_width=True,
                )
                st.markdown(
                    f"<div class='metric-box'><h3>ðŸ’° Total Premium:</h3><div class='value'>â‚¹{detail['premium_total']:.2f}</div></div>",
                    unsafe_allow_html=True,
                )
                if st.button("Confirm and Place Order", key="confirm_order"):
                    for order in detail["orders"]:
                        order_id = place_order(
                            config,
                            order["instrument_key"],
                            order["quantity"],
                            order["transaction_type"],
                        )
                        if not order_id:
                            st.error(f"Failed to place order for {order['instrument_key']}")
                    st.success(f"Placed {selected_strategy} order with {lots} lots!")
            else:
                st.error(f"Unable to fetch details for {selected_strategy}.")

    with tab6:
        st.markdown("<h2 style='color: #1E90FF;'>Risk Management Dashboard</h2>", unsafe_allow_html=True)
        st.subheader("Portfolio Risk Overview")
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.markdown(
                f"<div class='metric-box'><h3>ðŸ“‰ Total Risk</h3><div class='value'>â‚¹{portfolio_summary['Risk on Table']:.2f}</div></div>",
                unsafe_allow_html=True,
            )
        with col_r2:
            st.markdown(
                f"<div class='metric-box'><h3>ðŸ“Š Sharpe Ratio</h3><div class='value'>{sharpe_ratio:.2f}</div></div>",
                unsafe_allow_html=True,
            )
        with col_r3:
            portfolio_beta = calculate_portfolio_beta(positions, nifty)
            st.markdown(
                f"<div class='metric-box'><h3>Î² Portfolio Beta</h3><div class='value'>{portfolio_beta:.2f}</div></div>",
                unsafe_allow_html=True,
            )
        with col_r4:
            margin_pct = (funds_data["used_margin"] / funds_data["total_funds"] * 100) if funds_data["total_funds"] > 0 else 0
            st.markdown(
                f"<div class='metric-box'><h3>ðŸ’¸ Margin Utilization</h3><div class='value'>{margin_pct:.2f}%</div></div>",
                unsafe_allow_html=True,
            )
        st.subheader("Greeks Exposure")
        total_delta, total_theta, total_vega, total_gamma = 0.0, 0.0, 0.0, 0.0
        for pos in positions:
            try:
                instrument_key = pos.get("instrument_token", "")
                greeks = get_option_greeks(config, [instrument_key]).get(instrument_key, {})
                total_delta += greeks.get("delta", 0) * pos.get("quantity", 0)
                total_theta += greeks.get("theta", 0) * pos.get("quantity", 0)
                total_vega += greeks.get("vega", 0) * pos.get("quantity", 0)
                total_gamma += greeks.get("gamma", 0) * pos.get("quantity", 0)
            except Exception as e:
                st.warning(f"Error calculating Greeks for {instrument_key}: {e}")
                continue
        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        with col_g1:
            st.markdown(
                f"<div class='small-metric-box'><h4>ðŸ“ Delta</h4>{total_delta:.2f}</div>",
                unsafe_allow_html=True,
            )
        with col_g2:
            st.markdown(
                f"<div class='small-metric-box'><h4>â³ Theta</h4>â‚¹{total_theta:.2f}</div>",
                unsafe_allow_html=True,
            )
        with col_g3:
            st.markdown(
                f"<div class='small-metric-box'><h4>ðŸŒªï¸ Vega</h4>â‚¹{total_vega:.2f}</div>",
                unsafe_allow_html=True,
            )
            if total_vega > 1000:
                st.error("âš ï¸ High Vega exposure! Risk of volatility spike.")
        with col_g4:
            st.markdown(
                f"<div class='small-metric-box'><h4>âš¡ Gamma</h4>{total_gamma:.6f}</div>",
                unsafe_allow_html=True,
            )
        st.subheader("Capital Allocation")
        plot_allocation_pie(strategy_df, config)
        st.subheader("Drawdown Control")
        plot_drawdown_trend(portfolio_summary)
        max_drawdown_limit = 0.05 if vix > 20 else 0.03 if vix > 12 else 0.02
        if portfolio_summary["Drawdown %"] > max_drawdown_limit * 100:
            st.error(f"âš ï¸ Drawdown ({portfolio_summary['Drawdown %']:.2f}%) exceeds limit ({max_drawdown_limit*100:.2f}%)!")
        st.subheader("All Positions")
        if positions:
            pos_df = pd.DataFrame(positions)
            pos_df = pos_df[["instrument_token", "quantity", "average_price", "pnl"]]
            pos_df.columns = ["Instrument", "Quantity", "Avg Price", "P&L"]
            st.dataframe(
                pos_df.style.format({
                    "Avg Price": "{:.2f}",
                    "P&L": "{:.2f}",
                    "Quantity": "{:.0f}"
                }).set_properties(
                    **{"background-color": "#1A1C24", "color": "white"}
                ),
                use_container_width=True,
            )
        else:
            st.info("No open positions.")
        st.subheader("Risk Actions")
        if st.button("Exit All Positions", key="exit_all"):
            if st.checkbox("Confirm Exit All Positions", key="confirm_exit_all"):
                order_ids = exit_all_positions(config)
                if order_ids:
                    st.success(f"Initiated exit for {len(order_ids)} positions.")
                else:
                    st.warning("No positions to exit or error occurred.")
        st.subheader("Margin Utilization")
        plot_margin_gauge(funds_data)
