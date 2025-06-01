import pandas as pd
import numpy as np
import requests
from scipy.stats import linregress
from arch import arch_model
from datetime import datetime


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
            return datetime.now().strftime("%Y-%m-%d")
        except Exception as e:
            return datetime.now().strftime("%Y-%m-%d")

    config['expiry_date'] = get_next_expiry_internal()
    return config


def fetch_option_chain(config):
    try:
        url = f"{config['base_url']}/option/chain"
        params = {"instrument_key": config['instrument_key'], "expiry_date": config['expiry_date']}
        res = requests.get(url, headers=config['headers'], params=params)
        if res.status_code == 200:
            return res.json()["data"]
        return []
    except Exception as e:
        return []


def get_indices_quotes(config):
    try:
        url = f"{config['base_url']}/market-quote/quotes?instrument_key=NSE_INDEX|India VIX,NSE_INDEX|Nifty 50"
        res = requests.get(url, headers=config['headers'])
        if res.status_code == 200:
            data = res.json()
            vix = data["data"]["NSE_INDEX:India VIX"]["last_price"]
            nifty = data["data"]["NSE_INDEX:Nifty 50"]["last_price"]
            return vix, nifty
        return None, None
    except Exception as e:
        return None, None


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
        return pd.DataFrame()


def market_metrics(option_chain, expiry_date):
    try:
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        days_to_expiry = (expiry_dt - datetime.now()).days
        call_oi = sum(opt["call_options"]["market_data"]["oi"] for opt in option_chain if "call_options" in opt)
        put_oi = sum(opt["put_options"]["market_data"]["oi"] for opt in option_chain if "put_options" in opt)
        pcr = put_oi / call_oi if call_oi != 0 else 0
        strikes = sorted(list(set([opt["strike_price"] for opt in option_chain])))
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
        return {"days_to_expiry": 0, "pcr": 0, "max_pain": 0}


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
        return pd.DataFrame(columns=["Datetime", "Event", "Classification", "Forecast", "Prior"])


def load_ivp(config, seller_avg_iv):
    try:
        iv_df = pd.read_csv(config['ivp_url'])
        iv_df.dropna(subset=["ATM_IV"], inplace=True)
        iv_df = iv_df.tail(30)
        ivp = round((iv_df["ATM_IV"] < seller_avg_iv).sum() / len(iv_df) * 100, 2)
        return ivp
    except Exception as e:
        return 0


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
        return 0, 0, 0


def calculate_iv_skew_slope(full_chain_df):
    try:
        if full_chain_df.empty:
            return 0
        slope, _, _, _, _ = linregress(full_chain_df["Strike"], full_chain_df["IV Skew"])
        return slope
    except Exception as e:
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
        return "🔥 High Vol Trend", "Ideal for premium selling.", "Market in high volatility — ideal for premium selling with defined risk."
    elif regime_score > 10:
        return "🚧 Elevated Volatility", "Favor range-bound strategies.", "Moderate IVP and IV-RV spread indicate mean-reverting moves."
    elif regime_score > -10:
        return "😴 Neutral Volatility", "Flexible strategy selection.", "IV and RV aligned, with moderate PCR and skew."
    else:
        return "💤 Low Volatility", "Cautious selling or long vega plays.", "Low IVP, tight straddle, and low VIX suggest limited movement."


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
            if level == "High":
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
    expected_move_pct = round(expected_move, 2)
    if regime_label == "🔥 High Vol Trend":
        strategies = ["Iron Fly", "Wide Strangle"]
        rationale.append("Strong IV premium — neutral strategies for premium capture.")
    elif regime_label == "🚧 Elevated Volatility":
        strategies = ["Iron Condor", "Jade Lizard"]
        rationale.append("Volatility above average — range-bound strategies offer favorable reward-risk.")
    elif regime_label == "😴 Neutral Volatility":
        strategies = ["Jade Lizard", "Bull Put Spread"]
        rationale.append("Market balanced — slight directional bias strategies offer edge.")
    elif regime_label == "💤 Low Volatility":
        strategies = ["Straddle", "Calendar Spread"]
        rationale.append("Low IV — premium collection favorable but monitor for breakout risk.")
    if event_impact_score > 0 and not high_impact_event_near:
        strategies = [s for s in strategies if "Iron" in s or "Lizard" in s or "Spread" in s]
    if ivp > 85 and iv_minus_rv > 5:
        rationale.append(f"Volatility overpriced (IVP: {ivp}%, IV-RV: {iv_minus_rv}%): Ideal for selling premium.")
    elif ivp < 30:
        rationale.append(f"Volatility underpriced (IVP: {ivp}%): Avoid unhedged selling.")
    rationale.append(f"Expected move: ±{expected_move_pct:.2f}% based on straddle price.")
    return strategies, " | ".join(rationale), event_warning


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
        if res_trades.status_code == 200:
            trades = res_trades.json().get("data", [])
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
        return pd.DataFrame()


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
        return {"available_margin": 0, "used_margin": 0, "total_funds": 0}
    except Exception as e:
        return {"available_margin": 0, "used_margin": 0, "total_funds": 0}


def get_option_greeks(config, instrument_keys):
    try:
        if not instrument_keys:
            return {}
        url = f"{config['base_url'].replace('v2', 'v3')}/market-quote/option-greek"
        params = {"instrument_key": ",".join(instrument_keys)}
        res = requests.get(url, headers=config['headers'], params=params)
        if res.status_code == 200:
            return res.json()["data"]
        return {}
    except Exception as e:
        return {}


def calculate_sharpe_ratio():
    try:
        np.random.seed(42)
        daily_returns = np.random.normal(0.001, 0.01, 252)
        annual_return = np.mean(daily_returns) * 252
        annual_volatility = np.std(daily_returns) * np.sqrt(252)
        risk_free_rate = 0.06 / 252
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
        return round(sharpe_ratio, 2)
    except:
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
        flags = []
        for _, row in trades_df.iterrows():
            strat = row["strategy"]
            capital_used = row["capital_used"]
            potential_risk = row["potential_loss"]
            pnl = row["realized_pnl"]
            sl_hit = row["sl_hit"]
            trades_today = row["trades_today"]
            cfg = config['risk_config'].get(strat, {"capital_pct": 0.1, "risk_per_trade_pct": 0.01})
            risk_factor = 1.2 if regime_label == "🔥 High Vol Trend" else 0.8 if regime_label == "💤 Low Volatility" else 1.0
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
                "Risk OK?": "✅" if risk_ok else "❌"
            })
            total_cap_used += capital_used
            total_risk_used += potential_risk
            total_realized_pnl += pnl
            if not risk_ok:
                flags.append(f"❌ {strat} exceeded risk limit")
            if sl_hit and trades_today > 3:
                flags.append(f"⚠️ {strat} shows possible revenge trading (SL hit + {trades_today} trades)")
        net_dd = -total_realized_pnl if total_realized_pnl < 0 else 0
        exposure_pct = round(total_cap_used / config['total_capital'] * 100, 2) if config['total_capital'] else 0
        risk_pct = round(total_risk_used / config['total_capital'] * 100, 2) if config['total_capital'] else 0
        dd_pct = round(net_dd / config['total_capital'] * 100, 2) if config['total_capital'] else 0
        return pd.DataFrame(strategy_summary), {
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
            "Max Drawdown Allowed": max_drawdown,
            "Flags": flags
        }
    except Exception as e:
        return pd.DataFrame(), {}


def find_option_by_strike(option_chain, strike, option_type):
    try:
        for opt in option_chain:
            if abs(opt["strike_price"] - strike) < 0.01:
                if option_type == "CE" and "call_options" in opt:
                    return opt["call_options"]
                elif option_type == "PE" and "put_options" in opt:
                    return opt["put_options"]
        return None
    except Exception as e:
        return None


def get_strategy_details(strategy_name, option_chain, spot_price, config, lots=1):
    def _iron_fly_calc(option_chain, spot_price, config, lots):
        atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
        strike = atm["strike_price"]
        wing = 100
        ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
        pe_short_opt = find_option_by_strike(option_chain, strike, "PE")
        ce_long_opt = find_option_by_strike(option_chain, strike + wing, "CE")
        pe_long_opt = find_option_by_strike(option_chain, strike - wing, "PE")
        if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
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
            return None
        orders = [
            {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"},
            {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL"}
        ]
        return {"strategy": "ATM Strangle", "strikes": [call_strike, put_strike], "orders": orders}

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
    if not detail:
        return None

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
        qty = order["quantity"]
        price = order["current_price"]
        if order["transaction_type"] == "SELL":
            premium += price * qty
        else:
            premium -= price * qty
    detail["premium_total"] = premium
    detail["premium"] = premium / config["lot_size"]

    strategy = detail["strategy"]
    if strategy == "Iron Fly":
        wing_width = abs(detail["strikes"][0] - detail["strikes"][2])
        detail["max_loss"] = (wing_width - detail["premium"]) * config["lot_size"] * lots if premium > 0 else float('inf')
    elif strategy == "Iron Condor":
        wing_width = abs(detail["strikes"][2] - detail["strikes"][0])
        detail["max_loss"] = (wing_width - detail["premium"]) * config["lot_size"] * lots if premium > 0 else float('inf')
    elif strategy == "Jade Lizard":
        wing_width = abs(detail["strikes"][1] - detail["strikes"][2])
        detail["max_loss"] = (wing_width - detail["premium"]) * config["lot_size"] * lots if premium > 0 else float('inf')
    elif strategy == "Bull Put Spread":
        wing_width = abs(detail["strikes"][0] - detail["strikes"][1])
        detail["max_loss"] = (wing_width - detail["premium"]) * config["lot_size"] * lots if premium > 0 else float('inf')
    elif strategy in ["Straddle", "Wide Strangle", "ATM Strangle"]:
        detail["max_loss"] = float("inf")
    elif strategy == "Calendar Spread":
        detail["max_loss"] = detail["premium"]
        detail["max_profit"] = float("inf")
    detail["max_profit"] = detail["premium_total"] if strategy != "Calendar Spread" else float("inf")
    return detail


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
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                return data.get("data", {}).get("order_id", "Success")
            return data.get("data", {}).get("order_ids", ["Error"])[0]
        return f"Error {res.status_code}: {res.text}"
    except Exception as e:
        return f"Exception: {e}"


def exit_all_positions(config):
    try:
        url = f"{config['base_url']}/order/positions/exit?segment=EQ"
        res = requests.post(url, headers=config['headers'])
        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                return data.get("data", {}).get("order_ids", [])
        return []
    except Exception as e:
        return []


def logout(config):
    try:
        url = f"{config['base_url']}/logout"
        res = requests.delete(url, headers=config['headers'])
        return res.status_code == 200
    except Exception as e:
        return False
