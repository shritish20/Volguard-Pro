
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from arch import arch_model
import time # Added for polling delay

# --- Configuration ---
def get_config():
    config = {
        "access_token": "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiIzUUNUTVMiLCJqdGkiOiI2ODNiOGJhZjVhMGZiMDZlNTdiODcwYTQiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6dHJ1ZSwiaWF0IjoxNzQ4NzMyODQ3LCJpc3MiOiJ1ZGFwaS1nYXRld2F5LXNlcnZpY2UiLCJleHAiOjE3NDg4MTUyMDB9.s6qb4uAyaDjQ6nkncK8KtBSojRXGZC8x1whJlO5mHOI",  # Replace with your actual Upstox API token
        "base_url": "https://api.upstox.com/v2",
        "headers": {
            "accept": "application/json",
            "Api-Version": "2.0",
            "Authorization": "Bearer your_valid_access_token_here" # This will be updated below
        },
        "instrument_key": "NSE_INDEX|Nifty 50",
        "event_url": "https://raw.githubusercontent.com/shritish20/VolGuard/refs/heads/main/upcoming_events.csv",
        "ivp_url": "https://raw.githubusercontent.com/shritish20/VolGuard/refs/heads/main/ivp.csv",
        "nifty_url": "https://raw.githubusercontent.com/shritish20/VolGuard/refs/heads/main/nifty_50.csv",
        "total_capital": 2000000, # This will be updated by API call
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
        "lot_size": 75  # Correct Nifty lot size
    }
    # Update authorization header with the actual token
    config['headers']['Authorization'] = f"Bearer {config['access_token']}"

    def get_next_expiry_date():
        try:
            url = f"{config['base_url']}/option/contract"
            params = {"instrument_key": config['instrument_key']}
            res = requests.get(url, headers=config['headers'], params=params)
            if res.status_code == 200:
                # Filter for Nifty 50 only, if the instrument_key is for index
                # Assuming the response for Nifty 50 instrument_key directly gives Nifty option expiries
                expiries = sorted(res.json()["data"], key=lambda x: datetime.strptime(x["expiry"], "%Y-%m-%d"))
                today = datetime.now()
                for expiry in expiries:
                    expiry_dt = datetime.strptime(expiry["expiry"], "%Y-%m-%d")
                    # Check for Thursday expiry and expiry date is in the future
                    if expiry_dt.weekday() == 3 and expiry_dt > today:  # Thursday
                        return expiry["expiry"]
                # Fallback if no future Thursday expiry found (e.g., end of month)
                print("No future Thursday expiry found, returning current date.")
                return datetime.now().strftime("%Y-%m-%d")
            print(f"Error fetching expiries: {res.status_code} - {res.text}")
            return datetime.now().strftime("%Y-%m-%d")
        except Exception as e:
            print(f"Exception in get_next_expiry: {e}")
            return datetime.now().strftime("%Y-%m-%d")

    config['expiry_date'] = get_next_expiry_date()
    return config

config = get_config()

# --- Upstox API Functions ---

def fetch_option_chain(config):
    try:
        url = f"{config['base_url']}/option/chain"
        params = {"instrument_key": config['instrument_key'], "expiry_date": config['expiry_date']}
        res = requests.get(url, headers=config['headers'], params=params)
        if res.status_code == 200:
            return res.json()["data"]
        print(f"Error fetching option chain: {res.status_code} - {res.text}")
        return []
    except Exception as e:
        print(f"Exception in fetch_option_chain: {e}")
        return []

def get_indices_quotes(config):
    try:
        url = f"{config['base_url']}/market-quote/quotes?instrument_key=NSE_INDEX|India VIX,NSE_INDEX|Nifty 50"
        res = requests.get(url, headers=config['headers'])
        if res.status_code == 200:
            data = res.json()["data"]
            # Ensure keys match the actual response structure
            vix_data = data.get("NSE_INDEX:India VIX", {})
            nifty_data = data.get("NSE_INDEX:Nifty 50", {})

            vix = vix_data.get("last_price")
            nifty = nifty_data.get("last_price")

            if vix is not None and nifty is not None:
                print(f"🌡️ India VIX: {vix}")
                print(f"📈 Nifty 50 Spot: {nifty}")
                return vix, nifty
            else:
                print("Could not find India VIX or Nifty 50 last_price in response.")
                return None, None
        print(f"Error fetching indices quotes: {res.status_code} - {res.text}")
        return None, None
    except Exception as e:
        print(f"Exception in get_indices_quotes: {e}")
        return None, None

def get_option_greeks(config, instrument_keys):
    try:
        if not instrument_keys:
            return {}
        # Use v3 endpoint for option Greeks
        url = f"{config['base_url'].replace('v2', 'v3')}/market-quote/option-greek"
        params = {"instrument_key": ",".join(instrument_keys)}
        res = requests.get(url, headers=config['headers'], params=params)
        if res.status_code == 200:
            return res.json()["data"]
        print(f"Error fetching Greeks: {res.status_code} - {res.text}")
        return {}
    except Exception as e:
        print(f"Exception in get_option_greeks: {e}")
        return {}

def place_order(config, instrument_key, quantity, transaction_type, order_type="MARKET", price=0):
    try:
        url = f"{config['base_url'].replace('v2', 'v3')}/order/place"
        payload = {
            "quantity": quantity,
            "product": "D", # Assuming NRML for F&O. Verify with Upstox documentation.
            "validity": "DAY",
            "price": price,
            "instrument_token": instrument_key, # Use instrument_token for placing order
            "order_type": order_type,
            "transaction_type": transaction_type,
            "disclosed_quantity": 0,
            "trigger_price": 0,
            "is_amo": False,
            "tag": "VolGuard" # Optional: Add a tag to identify your orders
        }
        res = requests.post(url, headers=config['headers'], json=payload)
        print(f"Order response for {instrument_key}: {res.status_code} - {res.text}")

        if res.status_code == 200:
            data = res.json()
            if data.get("status") == "success":
                order_data = data.get("data", {})
                # Upstox v3 place order returns 'order_id' directly for single orders
                if "order_id" in order_data:
                    print(f"Order placed successfully. Order ID: {order_data['order_id']}")
                    return order_data["order_id"]
                elif "order_ids" in order_data and order_data["order_ids"]:
                    # This might be from a multi-order endpoint, but v3/order/place is typically single.
                    # Handling for completeness if it returns an array of one ID.
                    print(f"Order placed successfully. Order ID: {order_data['order_ids'][0]}")
                    return order_data["order_ids"][0]
                else:
                    print(f"Unexpected successful response format: {data}")
                    return None
            else:
                print(f"Order failed with status 'failure' or unexpected: {data}")
                return None
        elif res.status_code == 400:
            data = res.json()
            # Upstox API error structure can vary, common are 'errors' or 'detail'
            error_message = data.get("message", "Unknown error")
            if "errors" in data and isinstance(data["errors"], list):
                for err in data["errors"]:
                    if err.get("errorCode") == "UDAPI100060":
                        print(f"Order failed: Insufficient funds for {instrument_key}")
                        return None
                    error_message = err.get('message', error_message)
                    print(f"Order failed: {error_message} for {instrument_key}")
            else:
                print(f"Order failed (400 Bad Request): {error_message} for {instrument_key}")
            return None
        else:
            print(f"Error placing order (HTTP {res.status_code}): {res.text}")
            return None
    except requests.exceptions.RequestException as req_e:
        print(f"Network or API request error in place_order: {req_e}")
        return None
    except Exception as e:
        print(f"Unexpected exception in place_order: {e}")
        return None

# --- NEW: Margin and Fund Management ---

def calculate_strategy_margin(config, orders_payload):
    """
    Calculates the estimated margin required for a basket of orders.
    :param config: Configuration dictionary.
    :param orders_payload: List of dictionaries, each representing an order leg.
                           Example: [{"instrument_key": "...", "quantity": 1, "transaction_type": "BUY", "product": "D"}]
    :return: Estimated used_margin (float) or None if calculation fails.
    """
    try:
        url = f"{config['base_url']}/charges/margin"
        instruments_data = []
        for order in orders_payload:
            instruments_data.append({
                "instrument_key": order["instrument_key"],
                "quantity": order["quantity"],
                "transaction_type": order["transaction_type"],
                "product": "D" # Assuming NRML for F&O margin calculation. VERIFY THIS.
            })

        payload = {"instruments": instruments_data}
        res = requests.post(url, headers=config['headers'], json=payload)

        if res.status_code == 200:
            data = res.json()["data"]
            # The schema suggests additionalProp1, additionalProp2 etc.
            # We need to find the total/aggregate margin. Assuming the first property holds the total.
            # You might need to sum 'used_margin' across all 'additionalPropX' if it's item-wise.
            # A safer bet is to assume the API returns a single aggregated margin for the basket.
            # Let's try to find a key that holds the total margin.
            # If "additionalProp1" etc. are the only keys, we'll try to sum them.
            total_used_margin = 0
            found_margin_data = False
            for prop_key in data:
                if isinstance(data[prop_key], dict) and 'used_margin' in data[prop_key]:
                    total_used_margin += data[prop_key]['used_margin']
                    found_margin_data = True
            if found_margin_data:
                return total_used_margin
            print(f"Could not find 'used_margin' in the response data properties for margin calculation: {data}")
            return None
        print(f"Error calculating margin: {res.status_code} - {res.text}")
        return None
    except requests.exceptions.RequestException as req_e:
        print(f"Network or API request error in calculate_strategy_margin: {req_e}")
        return None
    except Exception as e:
        print(f"Exception in calculate_strategy_margin: {e}")
        return None

def get_user_funds_and_margin(config, segment="SEC"):
    """
    Fetches the user's fund and margin details for a given segment.
    :param config: Configuration dictionary.
    :param segment: 'SEC' for Securities (Equity, F&O, Currency), 'COM' for Commodity.
    :return: A dictionary with 'used_margin' and 'available_margin' or None.
    """
    try:
        url = f"{config['base_url']}/user/get-funds-and-margin"
        params = {"segment": segment}
        res = requests.get(url, headers=config['headers'], params=params)

        if res.status_code == 200:
            data = res.json()["data"]
            # Assuming 'additionalProp1' or similar holds the consolidated data
            # Based on the schema, it looks like it might return multiple props, but
            # in practice, for user funds, it's usually a single consolidated object.
            # Let's try to extract from the first available property.
            if data:
                first_prop_key = list(data.keys())[0]
                margin_info = data.get(first_prop_key, {})
                used_margin = margin_info.get("used_margin")
                available_margin = margin_info.get("available_margin")
                if used_margin is not None and available_margin is not None:
                    return {
                        "used_margin": used_margin,
                        "available_margin": available_margin
                    }
                print(f"Could not find 'used_margin' or 'available_margin' in the response: {data}")
                return None
            print(f"No data returned for funds and margin: {res.text}")
            return None
        print(f"Error fetching funds and margin: {res.status_code} - {res.text}")
        return None
    except requests.exceptions.RequestException as req_e:
        print(f"Network or API request error in get_user_funds_and_margin: {req_e}")
        return None
    except Exception as e:
        print(f"Exception in get_user_funds_and_margin: {e}")
        return None

def get_order_book(config):
    """
    Fetches the current order book (all orders for the day).
    :param config: Configuration dictionary.
    :return: A list of order dictionaries or an empty list.
    """
    try:
        url = f"{config['base_url']}/order/retrieve-all"
        res = requests.get(url, headers=config['headers'])

        if res.status_code == 200:
            data = res.json().get("data", [])
            return data if isinstance(data, list) else [] # Ensure it's a list
        print(f"Error fetching order book: {res.status_code} - {res.text}")
        return []
    except requests.exceptions.RequestException as req_e:
        print(f"Network or API request error in get_order_book: {req_e}")
        return []
    except Exception as e:
        print(f"Exception in get_order_book: {e}")
        return []

# --- Existing Functions (Adjusted for new APIs) ---

def load_upcoming_events(config):
    try:
        df = pd.read_csv(config['event_url'])
        df["Datetime"] = df["Date"] + " " + df["Time"]
        df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d-%b %H:%M", errors="coerce")
        current_year = datetime.now().year
        # Correctly handle year for events that might be in next year but same month/day
        df["Datetime"] = df["Datetime"].apply(lambda dt: dt.replace(year=current_year) if pd.notnull(dt) and dt.month >= datetime.now().month else (dt.replace(year=current_year + 1) if pd.notnull(dt) else dt))
        now = datetime.now()
        expiry_dt = datetime.strptime(config['expiry_date'], "%Y-%m-%d")
        mask = (df["Datetime"] >= now) & (df["Datetime"] <= expiry_dt + timedelta(days=1)) # Add a day to include expiry day
        filtered = df.loc[mask, ["Datetime", "Event", "Classification", "Forecast", "Prior"]]
        return filtered.sort_values("Datetime").reset_index(drop=True)
    except Exception as e:
        print(f"⚠️ Failed to load upcoming events: {e}")
        return pd.DataFrame(columns=["Datetime", "Event", "Classification", "Forecast", "Prior"])

def load_ivp(config, avg_iv):
    try:
        iv_df = pd.read_csv(config['ivp_url'])
        iv_df.dropna(subset=["ATM_IV"], inplace=True)
        iv_df = iv_df.tail(30) # Consider last 30 data points for IVP
        if len(iv_df) == 0:
            return 0
        ivp = round((iv_df["ATM_IV"] < avg_iv).sum() / len(iv_df) * 100, 2)
        return ivp
    except Exception as e:
        print(f"Exception in load_ivp: {e}")
        return 0

def extract_seller_metrics(option_chain, spot_price):
    try:
        # Filter out options without market_data or greeks for robustness
        valid_options = [opt for opt in option_chain if opt.get("call_options") and opt.get("put_options") and
                         opt["call_options"].get("market_data") and opt["put_options"].get("market_data") and
                         opt["call_options"].get("option_greeks") and opt["put_options"].get("option_greeks")]
        if not valid_options:
            print("No valid options with market data or greeks found in chain.")
            return {}

        atm = min(valid_options, key=lambda x: abs(x["strike_price"] - spot_price))

        call = atm["call_options"]
        put = atm["put_options"]

        # Safely get LTP and Greeks, defaulting if not present
        call_ltp = call["market_data"].get("ltp", 0)
        put_ltp = put["market_data"].get("ltp", 0)

        call_iv = call["option_greeks"].get("iv", 0)
        put_iv = put["option_greeks"].get("iv", 0)

        call_theta = call["option_greeks"].get("theta", 0)
        put_theta = put["option_greeks"].get("theta", 0)

        call_vega = call["option_greeks"].get("vega", 0)
        put_vega = put["option_greeks"].get("vega", 0)

        call_delta = call["option_greeks"].get("delta", 0)
        put_delta = put["option_greeks"].get("delta", 0)

        call_gamma = call["option_greeks"].get("gamma", 0)
        put_gamma = put["option_greeks"].get("gamma", 0)

        call_pop = call["option_greeks"].get("pop", 0)
        put_pop = put["option_greeks"].get("pop", 0)


        return {
            "strike": atm["strike_price"],
            "straddle_price": call_ltp + put_ltp,
            "avg_iv": (call_iv + put_iv) / 2,
            "theta": call_theta + put_theta,
            "vega": call_vega + put_vega,
            "delta": call_delta + put_delta,
            "gamma": call_gamma + put_gamma,
            "pop": (call_pop + put_pop) / 2 * 100 # POP is often 0-1, so multiply by 100 for percentage
        }
    except Exception as e:
        print(f"Exception in extract_seller_metrics: {e}")
        return {}

def full_chain_table(option_chain, spot_price):
    try:
        chain_data = []
        for opt in option_chain:
            strike = opt.get("strike_price")
            call = opt.get("call_options")
            put = opt.get("put_options")

            # Ensure all necessary keys exist before accessing
            if strike is None or not call or not put or \
               not call.get("market_data") or not put.get("market_data") or \
               not call.get("option_greeks") or not put.get("option_greeks"):
                continue

            call_iv = call["option_greeks"].get("iv", 0)
            put_iv = put["option_greeks"].get("iv", 0)
            call_theta = call["option_greeks"].get("theta", 0)
            put_theta = put["option_greeks"].get("theta", 0)
            call_vega = call["option_greeks"].get("vega", 0)
            put_vega = put["option_greeks"].get("vega", 0)
            call_ltp = call["market_data"].get("ltp", 0)
            put_ltp = put["market_data"].get("ltp", 0)
            call_oi = call["market_data"].get("oi", 0)
            put_oi = put["market_data"].get("oi", 0)

            if abs(strike - spot_price) <= 300:
                chain_data.append({
                    "Strike": strike,
                    "Call IV": call_iv,
                    "Put IV": put_iv,
                    "IV Skew": call_iv - put_iv,
                    "Total Theta": call_theta + put_theta,
                    "Total Vega": call_vega + put_vega,
                    "Straddle Price": call_ltp + put_ltp,
                    "Total OI": call_oi + put_oi
                })
        return pd.DataFrame(chain_data)
    except Exception as e:
        print(f"Exception in full_chain_table: {e}")
        return pd.DataFrame()

def market_metrics(option_chain, expiry_date):
    try:
        expiry_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        days_to_expiry = (expiry_dt - datetime.now()).days
        
        # Filter for options with valid OI data
        valid_options = [opt for opt in option_chain if opt.get("call_options") and opt.get("put_options") and
                         opt["call_options"].get("market_data") and opt["put_options"].get("market_data")]

        call_oi = sum(opt["call_options"]["market_data"].get("oi", 0) for opt in valid_options)
        put_oi = sum(opt["put_options"]["market_data"].get("oi", 0) for opt in valid_options)
        
        pcr = put_oi / call_oi if call_oi != 0 else 0

        # Calculate Max Pain
        strikes_in_chain = [opt["strike_price"] for opt in valid_options]
        if not strikes_in_chain:
            return {"days_to_expiry": days_to_expiry, "pcr": round(pcr, 2), "max_pain": 0}

        min_strike = min(strikes_in_chain)
        max_strike = max(strikes_in_chain)
        
        # Ensure step is appropriate for Nifty (e.g., 50)
        possible_strikes = range(int(min_strike), int(max_strike) + 1, 50) # Assuming 50 point strikes for Nifty

        max_pain = 0
        min_loss = float('inf')

        for strike in possible_strikes:
            current_loss = 0
            for opt in valid_options:
                option_strike = opt["strike_price"]
                call_oi_at_strike = opt["call_options"]["market_data"].get("oi", 0)
                put_oi_at_strike = opt["put_options"]["market_data"].get("oi", 0)

                # Loss for Call writers at this strike
                current_loss += max(0, strike - option_strike) * call_oi_at_strike
                # Loss for Put writers at this strike
                current_loss += max(0, option_strike - strike) * put_oi_at_strike
            
            if current_loss < min_loss:
                min_loss = current_loss
                max_pain = strike
        
        return {"days_to_expiry": days_to_expiry, "pcr": round(pcr, 2), "max_pain": max_pain}
    except Exception as e:
        print(f"Exception in market_metrics: {e}")
        return {"days_to_expiry": 0, "pcr": 0, "max_pain": 0}

def calculate_volatility(config, seller):
    try:
        # Using a fixed CSV for now, will replace with API call later for historical data
        df = pd.read_csv(config['nifty_url'])
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%Y")
        df = df.sort_values('Date')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df.dropna(inplace=True)
        if df.empty or len(df) < 7: # Ensure enough data for 7-day HV
            print("Not enough historical data for volatility calculations.")
            return 0, 0, 0
        hv_7 = np.std(df["Log_Returns"][-7:]) * np.sqrt(252) * 100
        
        # Ensure enough data for GARCH model
        if len(df["Log_Returns"]) < 2: # GARCH needs more than 1 observation
            print("Not enough log returns for GARCH model.")
            return hv_7, 0, round(seller['avg_iv'] - hv_7, 2) if hv_7 != 0 else 0

        model = arch_model(df["Log_Returns"], vol="Garch", p=1, q=1)
        res = model.fit(disp="off")
        forecast = res.forecast(horizon=7)
        # Check if variance data is available for forecast
        if not forecast.variance.empty and forecast.variance.iloc[-1] is not None:
            garch_7d = np.mean(np.sqrt(forecast.variance.iloc[-1]) * np.sqrt(252) * 100)
        else:
            garch_7d = 0 # Default to 0 if forecast fails
            print("GARCH forecast variance not available.")

        iv_rv_spread = round(seller['avg_iv'] - hv_7, 2) if hv_7 != 0 else 0
        return hv_7, garch_7d, iv_rv_spread
    except Exception as e:
        print(f"Exception in calculate_volatility: {e}")
        return 0, 0, 0

def calculate_iv_skew_slope(full_chain_df):
    try:
        if full_chain_df.empty or len(full_chain_df) < 2:
            return 0
        from scipy.stats import linregress
        # Filter out NaN or infinite values from IV Skew
        df_filtered = full_chain_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["IV Skew", "Strike"])
        if len(df_filtered) < 2:
            return 0 # Need at least two points for regression
        slope, _, _, _, _ = linregress(df_filtered["Strike"], df_filtered["IV Skew"])
        return slope
    except Exception as e:
        print(f"Exception in calculate_iv_skew_slope: {e}")
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
        print(f"Exception in calculate_regime: {e}")
        return 0, "Unknown", "Error calculating regime.", ""

def suggest_strategy(regime_label, ivp, iv_minus_rv, days_to_expiry, event_df, expiry_date, straddle_price, spot_price):
    strategies = []
    rationale = []
    event_warning = None
    event_window = 3 if ivp > 80 else 2
    high_impact_event_near = False
    event_impact_score = 0
    
    expiry_dt_obj = datetime.strptime(expiry_date, "%Y-%m-%d")

    for _, row in event_df.iterrows():
        dt, level = row["Datetime"], row["Classification"]
        if pd.isna(dt): # Skip if Datetime is NaT
            continue

        if (level == "High") and (0 <= (expiry_dt_obj - dt).days <= event_window):
            high_impact_event_near = True
        
        if level == "High" and pd.notnull(row.get("Forecast")) and pd.notnull(row.get("Prior")):
            try:
                # Robust parsing of Forecast and Prior, handling non-numeric data
                forecast_val = str(row["Forecast"]).strip().replace('%', '')
                prior_val = str(row["Prior"]).strip().replace('%', '')
                
                # Convert to numeric, handle errors
                forecast = float(forecast_val) if forecast_val.replace('.', '', 1).isdigit() else None
                prior = float(prior_val) if prior_val.replace('.', '', 1).isdigit() else None
                
                if forecast is not None and prior is not None and abs(forecast - prior) > 0.5:
                    event_impact_score += 1
            except Exception as e:
                # print(f"Error parsing event forecast/prior: {row['Event']} - {e}")
                continue # Skip this event if parsing fails

    if high_impact_event_near:
        event_warning = f"⚠️ High-impact event within {event_window} days of expiry. Prefer defined-risk strategies."
    if event_impact_score > 0:
        rationale.append(f"High-impact events with significant forecast deviations ({event_impact_score} events).")
    
    expected_move_pct = (straddle_price / spot_price) * 100

    if regime_label == "🔥 High Vol Trend":
        if high_impact_event_near or event_impact_score > 0:
            strategies = ["Iron Fly", "Wide Strangle"] # Wide strangle for wider range
            rationale.append("High volatility with major event — use defined-risk structures.")
        else:
            strategies = ["Iron Fly", "Wide Strangle"]
            rationale.append("Strong IV premium — neutral strategies for premium capture.")
    elif regime_label == "🚧 Elevated Volatility":
        strategies = ["Iron Condor", "Jade Lizard"]
        rationale.append("Volatility above average — range-bound strategies offer favorable reward-risk.")
    elif regime_label == "😴 Neutral Volatility":
        if days_to_expiry >= 3:
            strategies = ["Jade Lizard", "Bull Put Spread"] # Slight directional bias
            rationale.append("Market balanced — slight directional bias strategies offer edge.")
        else: # Shorter expiry in neutral vol
            strategies = ["Iron Fly", "ATM Strangle"]
            rationale.append("Tight expiry — quick theta-based capture via short Iron Fly/ATM Strangle.")
    elif regime_label == "💤 Low Volatility":
        if days_to_expiry > 7:
            strategies = ["Straddle", "Calendar Spread"] # Benefit from potential IV increase
            rationale.append("Low IV with longer expiry — benefit from potential IV increase or directional move.")
        else:
            strategies = ["Straddle", "ATM Strangle"] # Still premium collection, but monitor for breakout
            rationale.append("Low IV — premium collection favorable but monitor for breakout risk.")

    # Refine strategies if events have significant impact but not too close to expiry
    if event_impact_score > 0 and not high_impact_event_near:
        # Prioritize defined-risk strategies if there are volatile events, even if not super close to expiry
        strategies_for_events = [s for s in strategies if "Iron" in s or "Lizard" in s or "Spread" in s]
        if strategies_for_events: # Only replace if there are defined-risk options
            strategies = strategies_for_events 
            rationale.append("Prioritizing defined-risk due to potential impact from upcoming events.")

    if ivp > 85 and iv_minus_rv > 5:
        rationale.append(f"Volatility overpriced (IVP: {ivp}%, IV-RV: {iv_minus_rv}%) — ideal for selling premium.")
    elif ivp < 30:
        rationale.append(f"Volatility underpriced (IVP: {ivp}%) — consider avoiding unhedged selling or look for long vega plays.")
    
    rationale.append(f"Expected move: ±{expected_move_pct:.2f}% based on straddle price.")
    return strategies, " | ".join(rationale), event_warning

def fetch_trade_data(config, full_chain_df):
    try:
        url_positions = f"{config['base_url']}/portfolio/short-term-positions"
        res_positions = requests.get(url_positions, headers=config['headers'])
        url_trades = f"{config['base_url']}/order/trades/get-trades-for-day"
        res_trades = requests.get(url_trades, headers=config['headers'])

        if res_positions.status_code == 200 and res_trades.status_code == 200:
            positions = res_positions.json()["data"] if res_positions.json().get("data") else []
            trades = res_trades.json()["data"] if res_trades.json().get("data") else []

            trade_counts = {}
            for trade in trades:
                # Basic heuristic: identify strategy from instrument type. Refine as needed.
                instrument_key = trade.get("instrument_key", "")
                if "NIFTY" in instrument_key and ("CE" in instrument_key or "PE" in instrument_key):
                    # This is a very rough way to map to a strategy.
                    # Ideally, you'd track the strategy for each order you place.
                    strat = "Option Strategy" # Generic for now, can be refined
                else:
                    strat = "Equity Trade"
                trade_counts[strat] = trade_counts.get(strat, 0) + 1
            
            trades_df_list = [] # Renamed to avoid conflict with trades variable
            for pos in positions:
                instrument = pos.get("instrument_key")
                quantity = pos.get("quantity")
                average_price = pos.get("average_price")
                pnl = pos.get("pnl")
                
                if not instrument or quantity is None or average_price is None or pnl is None:
                    continue # Skip incomplete position data

                # Determine strategy type based on instrument for current positions
                if "NIFTY" in instrument and ("CE" in instrument or "PE" in instrument):
                    strat = "Option Strategy" # Placeholder, refine based on your actual strategy tracking
                else:
                    strat = "Equity Holding" # Or other types

                capital = abs(quantity) * average_price # Abs for quantity, as it can be negative for short positions

                # Fetch brokerage for this position
                # Note: Brokerage API params need to reflect exiting the position.
                # Here, we're assuming exiting the current position.
                transaction_type_for_brokerage = "SELL" if quantity > 0 else "BUY" # If current position is long, you sell to exit
                
                url_brokerage = f"{config['base_url']}/charges/brokerage"
                params_brokerage = {
                    "instrument_token": instrument,
                    "quantity": abs(quantity), # Always use positive quantity for brokerage calculation
                    "product": pos.get("product", "NRML"), # Use product from position if available
                    "transaction_type": transaction_type_for_brokerage,
                    "price": average_price # Use current average price, or current LTP for exit
                }
                res_brokerage = requests.get(url_brokerage, headers=config['headers'], params=params_brokerage)
                brokerage = res_brokerage.json()["data"]["charges"]["total"] if res_brokerage.status_code == 200 and res_brokerage.json().get("data") else 0

                # Vega for the position from full_chain_df (assuming it matches)
                # This is a rough estimation, ideally you'd track Vega per leg
                position_vega = 0
                matching_strikes = full_chain_df[full_chain_df["Strike"].apply(lambda x: abs(x - pos.get("strike_price", 0)) < 1)] # Find nearest strike
                if not matching_strikes.empty:
                    position_vega = matching_strikes["Total Vega"].iloc[0] * abs(quantity) / config["lot_size"] # Vega per lot

                trades_df_list.append({
                    "strategy": strat,
                    "capital_used": capital,
                    "potential_loss": capital * 0.1, # Placeholder, needs refinement based on actual strategy max loss
                    "realized_pnl": pnl - brokerage, # pnl from API includes unrealized. This is just for demonstration.
                    "trades_today": trade_counts.get(strat, 0),
                    "sl_hit": False, # This logic needs actual stop-loss tracking
                    "vega": position_vega # Per position vega
                })
            return pd.DataFrame(trades_df_list) if trades_df_list else pd.DataFrame()
        print(f"Error fetching positions: {res_positions.status_code} - {res_positions.text}")
        print(f"Error fetching trades: {res_trades.status_code} - {res_trades.text}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Exception in fetch_trade_data: {e}")
        return pd.DataFrame()


def evaluate_full_risk(trades_df, config, regime_label, current_available_capital):
    try:
        daily_risk_limit = config['daily_risk_limit_pct'] * config['total_capital'] # Use total_capital from config, updated by API if needed
        weekly_risk_limit = config['weekly_risk_limit_pct'] * config['total_capital']
        
        strategy_summary = []
        total_cap_used = 0
        total_risk_used = 0
        total_realized_pnl = 0
        total_vega = 0
        flags = []

        # If trades_df is empty, return initial summary with current capital
        if trades_df.empty:
            portfolio_summary = {
                "Total Capital": config['total_capital'],
                "Capital Deployed": 0,
                "Exposure %": 0,
                "Risk on Table": 0,
                "Risk %": 0,
                "Available Funds": current_available_capital, # Use actual available capital
                "Daily Risk Limit": daily_risk_limit,
                "Weekly Risk Limit": weekly_risk_limit,
                "Realized P&L": 0,
                "Drawdown ₹": 0,
                "Drawdown %": 0,
                "Portfolio Vega": 0,
                "Flags": ["No trades or positions found."]
            }
            return pd.DataFrame(), portfolio_summary

        for _, row in trades_df.iterrows():
            strat = row["strategy"]
            capital_used = row["capital_used"]
            potential_loss = row["potential_loss"] # Use potential_loss directly for risk
            pnl = row["realized_pnl"]
            sl_hit = row["sl_hit"] # Placeholder, needs actual SL tracking
            trades_today = row["trades_today"] # Placeholder, needs actual trade counting per strategy
            vega = row["vega"] # Per position vega

            cfg = config['risk_config'].get(strat, {"capital_pct": 0.05, "risk_per_trade_pct": 0.01}) # Default values if strat not in config
            
            # Adjust risk factor based on regime
            risk_factor = 0.8 if regime_label == "🔥 High Vol Trend" else 1.1 if regime_label == "💤 Low Volatility" else 1.0
            
            max_cap = cfg["capital_pct"] * config['total_capital']
            max_risk = cfg["risk_per_trade_pct"] * max_cap * risk_factor

            risk_ok = potential_loss <= max_risk

            strategy_summary.append({
                "Strategy": strat,
                "Capital Used": capital_used,
                "Cap Limit": round(max_cap, 2),
                "% Used": round(capital_used / max_cap * 100, 2) if max_cap != 0 else 0,
                "Potential Risk": potential_loss,
                "Risk Limit": round(max_risk, 2),
                "P&L": pnl,
                "Vega": vega,
                "Risk OK?": "✅" if risk_ok else "❌"
            })
            total_cap_used += capital_used
            total_risk_used += potential_loss
            total_realized_pnl += pnl
            total_vega += vega # Sum up position vegas for portfolio vega

            if not risk_ok:
                flags.append(f"❌ {strat} exceeded risk limit (Risk: {potential_loss:.2f}, Limit: {max_risk:.2f})")
            if sl_hit and trades_today > 3: # Example condition for potential revenge trading
                flags.append(f"⚠️ {strat} shows possible revenge trading (SL hit + {trades_today} trades)")
        
        net_dd = -total_realized_pnl if total_realized_pnl < 0 else 0
        exposure_pct = round(total_cap_used / config['total_capital'] * 100, 2) if config['total_capital'] != 0 else 0
        risk_pct = round(total_risk_used / config['total_capital'] * 100, 2) if config['total_capital'] != 0 else 0
        dd_pct = round(net_dd / config['total_capital'] * 100, 2) if config['total_capital'] != 0 else 0
        
        portfolio_summary = {
            "Total Capital": config['total_capital'],
            "Capital Deployed": total_cap_used,
            "Exposure %": exposure_pct,
            "Risk on Table": total_risk_used,
            "Risk %": risk_pct,
            "Available Funds": current_available_capital, # Use actual available capital
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
        print(f"Exception in evaluate_full_risk: {e}")
        return pd.DataFrame(), {}

def find_option_by_strike(option_chain, strike, option_type):
    try:
        for opt in option_chain:
            if abs(opt.get("strike_price", 0) - strike) < 0.01:
                if option_type == "CE":
                    return opt.get("call_options")
                elif option_type == "PE":
                    return opt.get("put_options")
        print(f"No option found for strike {strike} {option_type}")
        return None
    except Exception as e:
        print(f"Exception in find_option_by_strike: {e}")
        return None

# --- Strategy Definitions (Updated to include 'product' type) ---

def iron_fly(option_chain, spot_price, config, lots=1):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    wing = 100 # Example wing distance
    
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike, "PE")
    ce_long_opt = find_option_by_strike(option_chain, strike + wing, "CE")
    pe_long_opt = find_option_by_strike(option_chain, strike - wing, "PE")

    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        print("Error: Missing options for Iron Fly")
        return None

    instrument_keys = [ce_short_opt["instrument_key"], pe_short_opt["instrument_key"],
                      ce_long_opt["instrument_key"], pe_long_opt["instrument_key"]]
    
    greeks_data = get_option_greeks(config, instrument_keys)
    
    ce_short_price = greeks_data.get(ce_short_opt["instrument_key"], {}).get("last_price", ce_short_opt["market_data"]["ltp"])
    pe_short_price = greeks_data.get(pe_short_opt["instrument_key"], {}).get("last_price", pe_short_opt["market_data"]["ltp"])
    ce_long_price = greeks_data.get(ce_long_opt["instrument_key"], {}).get("last_price", ce_long_opt["market_data"]["ltp"])
    pe_long_price = greeks_data.get(pe_long_opt["instrument_key"], {}).get("last_price", pe_long_opt["market_data"]["ltp"])

    premium = (ce_short_price + pe_short_price - ce_long_price - pe_long_price) * lots
    max_loss = (wing - (premium / (config["lot_size"] * lots))) * config["lot_size"] * lots

    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "NRML"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "NRML"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "NRML"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "NRML"}
    ]
    
    pricing = {
        ce_short_opt["instrument_key"]: {"last_price": ce_short_price},
        pe_short_opt["instrument_key"]: {"last_price": pe_short_price},
        ce_long_opt["instrument_key"]: {"last_price": ce_long_price},
        pe_long_opt["instrument_key"]: {"last_price": pe_long_price}
    }

    return {"strategy": "Iron Fly", "strikes": [strike, strike, strike + wing, strike - wing],
            "premium": premium, "max_loss": max_loss, "max_profit": premium, "orders": orders, "pricing": pricing}

def iron_condor(option_chain, spot_price, config, lots=1):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    short_wing_distance = 150 # Distance for short strikes
    long_wing_distance = 250  # Distance for long (hedging) strikes from ATM

    # Calculate short and long strike prices
    ce_short_strike = atm["strike_price"] + short_wing_distance
    pe_short_strike = atm["strike_price"] - short_wing_distance
    ce_long_strike = atm["strike_price"] + long_wing_distance
    pe_long_strike = atm["strike_price"] - long_wing_distance

    # Find options
    ce_short_opt = find_option_by_strike(option_chain, ce_short_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, pe_short_strike, "PE")
    ce_long_opt = find_option_by_strike(option_chain, ce_long_strike, "CE")
    pe_long_opt = find_option_by_strike(option_chain, pe_long_strike, "PE")

    if not all([ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]):
        print("Error: Missing options for Iron Condor")
        return None
    
    instrument_keys = [opt["instrument_key"] for opt in [ce_short_opt, pe_short_opt, ce_long_opt, pe_long_opt]]
    greeks_data = get_option_greeks(config, instrument_keys)

    # Get prices from Greeks API, fallback to LTP if not available
    ce_short_price = greeks_data.get(ce_short_opt["instrument_key"], {}).get("last_price", ce_short_opt["market_data"]["ltp"])
    pe_short_price = greeks_data.get(pe_short_opt["instrument_key"], {}).get("last_price", pe_short_opt["market_data"]["ltp"])
    ce_long_price = greeks_data.get(ce_long_opt["instrument_key"], {}).get("last_price", ce_long_opt["market_data"]["ltp"])
    pe_long_price = greeks_data.get(pe_long_opt["instrument_key"], {}).get("last_price", pe_long_opt["market_data"]["ltp"])

    premium = (ce_short_price + pe_short_price - ce_long_price - pe_long_price) * lots
    
    # Max loss is difference between wings minus premium received
    max_loss = ((long_wing_distance - short_wing_distance) - (premium / (config["lot_size"] * lots))) * config["lot_size"] * lots

    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "NRML"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "NRML"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "NRML"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "NRML"}
    ]

    pricing = {
        ce_short_opt["instrument_key"]: {"last_price": ce_short_price},
        pe_short_opt["instrument_key"]: {"last_price": pe_short_price},
        ce_long_opt["instrument_key"]: {"last_price": ce_long_price},
        pe_long_opt["instrument_key"]: {"last_price": pe_long_price}
    }

    return {"strategy": "Iron Condor", "strikes": [ce_short_strike, pe_short_strike, ce_long_strike, pe_long_strike],
            "premium": premium, "max_loss": max_loss, "max_profit": premium, "orders": orders, "pricing": pricing}

def jade_lizard(option_chain, spot_price, config, lots=1):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    
    # Sell OTM Call (e.g., ATM + 100)
    call_short_strike = atm["strike_price"] + 100
    # Sell ATM Put
    put_short_strike = atm["strike_price"]
    # Buy OTM Put (hedge for short put)
    put_long_strike = atm["strike_price"] - 100

    ce_short_opt = find_option_by_strike(option_chain, call_short_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_short_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, put_long_strike, "PE")

    if not all([ce_short_opt, pe_short_opt, pe_long_opt]):
        print("Error: Missing options for Jade Lizard")
        return None

    instrument_keys = [opt["instrument_key"] for opt in [ce_short_opt, pe_short_opt, pe_long_opt]]
    greeks_data = get_option_greeks(config, instrument_keys)

    ce_short_price = greeks_data.get(ce_short_opt["instrument_key"], {}).get("last_price", ce_short_opt["market_data"]["ltp"])
    pe_short_price = greeks_data.get(pe_short_opt["instrument_key"], {}).get("last_price", pe_short_opt["market_data"]["ltp"])
    pe_long_price = greeks_data.get(pe_long_opt["instrument_key"], {}).get("last_price", pe_long_opt["market_data"]["ltp"])

    premium = (ce_short_price + pe_short_price - pe_long_price) * lots
    
    # Max loss is difference between put strikes minus premium received for the put spread side
    # The call side has unlimited risk without a hedge, which is typical for Jade Lizard.
    # Here, we define max_loss for the put spread part.
    max_loss = ((put_short_strike - put_long_strike) - (pe_short_price - pe_long_price)) * config["lot_size"] * lots
    # Note: Max profit is typically the premium received. Max loss on call side is unlimited.
    # The 'max_loss' returned here is only for the hedged put spread.
    # For a true Jade Lizard, if the market moves significantly up, the short call has unlimited loss.
    # A safer representation might be 'float("inf")' for the max_loss to signify open-ended risk on one side.
    # I'll keep it as the put spread loss for now for consistency with other defined-risk calculations,
    # but acknowledge the open risk on the call side.

    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "NRML"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "NRML"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "NRML"}
    ]

    pricing = {
        ce_short_opt["instrument_key"]: {"last_price": ce_short_price},
        pe_short_opt["instrument_key"]: {"last_price": pe_short_price},
        pe_long_opt["instrument_key"]: {"last_price": pe_long_price}
    }

    return {"strategy": "Jade Lizard", "strikes": [call_short_strike, put_short_strike, put_long_strike],
            "premium": premium, "max_loss": max_loss, "max_profit": premium, "orders": orders, "pricing": pricing}

def straddle(option_chain, spot_price, config, lots=1):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]
    
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, strike, "PE")

    if not all([ce_short_opt, pe_short_opt]):
        print("Error: Missing options for Straddle")
        return None
    
    instrument_keys = [opt["instrument_key"] for opt in [ce_short_opt, pe_short_opt]]
    greeks_data = get_option_greeks(config, instrument_keys)

    ce_short_price = greeks_data.get(ce_short_opt["instrument_key"], {}).get("last_price", ce_short_opt["market_data"]["ltp"])
    pe_short_price = greeks_data.get(pe_short_opt["instrument_key"], {}).get("last_price", pe_short_opt["market_data"]["ltp"])

    premium = (ce_short_price + pe_short_price) * lots
    max_loss = float("inf") # Unlimited loss potential for short straddle

    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "NRML"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "NRML"}
    ]

    pricing = {
        ce_short_opt["instrument_key"]: {"last_price": ce_short_price},
        pe_short_opt["instrument_key"]: {"last_price": pe_short_price}
    }

    return {"strategy": "Straddle", "strikes": [strike, strike],
            "premium": premium, "max_loss": max_loss, "max_profit": premium, "orders": orders, "pricing": pricing}

def calendar_spread(option_chain, spot_price, config, lots=1):
    # This simplified calendar spread assumes trading on the same strike but different expiries.
    # The current `find_option_by_strike` only works on current expiry.
    # To implement this correctly, you'd need to fetch option chain for a *different* expiry.
    # For now, I'll provide a placeholder. This will NOT work with current `find_option_by_strike`.
    print("Warning: Calendar Spread requires fetching options from multiple expiries. This implementation is a placeholder.")
    
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    strike = atm["strike_price"]

    # This part needs to be updated to find options for NEAR and FAR expiry
    # Example: ce_short_opt_near_exp = find_option_by_strike(option_chain_near_exp, strike, "CE")
    #          ce_long_opt_far_exp = find_option_by_strike(option_chain_far_exp, strike, "CE")

    # Placeholder logic (will currently try to get same option, which is not a calendar spread)
    ce_short_opt = find_option_by_strike(option_chain, strike, "CE") # This is assumed near expiry
    ce_long_opt = find_option_by_strike(option_chain, strike, "CE") # This should be far expiry, but uses same chain

    if not all([ce_short_opt, ce_long_opt]): # This check will likely fail or return same opt
        print("Error: Missing options for Calendar Spread. Ensure multi-expiry data is fetched.")
        return None

    # In a real calendar spread, instrument_keys would be different (different expiries)
    instrument_keys = [ce_short_opt["instrument_key"], ce_long_opt["instrument_key"]]
    greeks_data = get_option_greeks(config, instrument_keys)

    ce_short_price = greeks_data.get(ce_short_opt["instrument_key"], {}).get("last_price", ce_short_opt["market_data"]["ltp"])
    ce_long_price = greeks_data.get(ce_long_opt["instrument_key"], {}).get("last_price", ce_long_opt["market_data"]["ltp"])

    # For a typical calendar spread (buy long expiry, sell short expiry), premium is usually negative (cost)
    premium = (ce_short_price - ce_long_price) * config["lot_size"] * lots # Short premium - Long premium (should be negative if long costs more)
    
    # Max loss is typically the net debit paid
    max_loss = abs(premium) # Initial debit paid
    max_profit = float("inf") # Theoretically unlimited, but capped by time decay difference and IV expansion

    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "NRML"},
        {"instrument_key": ce_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "NRML"}
    ]
    
    pricing = {
        ce_short_opt["instrument_key"]: {"last_price": ce_short_price},
        ce_long_opt["instrument_key"]: {"last_price": ce_long_price}
    }

    return {"strategy": "Calendar Spread", "strikes": [strike, strike],
            "premium": premium, "max_loss": max_loss, "max_profit": max_profit, "orders": orders, "pricing": pricing}

def bull_put_spread(option_chain, spot_price, config, lots=1):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    
    # Short OTM Put
    short_strike = atm["strike_price"] - 100 
    # Long further OTM Put (hedge)
    long_strike = atm["strike_price"] - 200

    pe_short_opt = find_option_by_strike(option_chain, short_strike, "PE")
    pe_long_opt = find_option_by_strike(option_chain, long_strike, "PE")

    if not all([pe_short_opt, pe_long_opt]):
        print("Error: Missing options for Bull Put Spread")
        return None
    
    instrument_keys = [opt["instrument_key"] for opt in [pe_short_opt, pe_long_opt]]
    greeks_data = get_option_greeks(config, instrument_keys)

    pe_short_price = greeks_data.get(pe_short_opt["instrument_key"], {}).get("last_price", pe_short_opt["market_data"]["ltp"])
    pe_long_price = greeks_data.get(pe_long_opt["instrument_key"], {}).get("last_price", pe_long_opt["market_data"]["ltp"])

    premium = (pe_short_price - pe_long_price) * lots # Net credit received
    # Max loss = difference in strikes - premium received
    max_loss = ((short_strike - long_strike) - (premium / (config["lot_size"] * lots))) * config["lot_size"] * lots

    orders = [
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "NRML"},
        {"instrument_key": pe_long_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "BUY", "product": "NRML"}
    ]

    pricing = {
        pe_short_opt["instrument_key"]: {"last_price": pe_short_price},
        pe_long_opt["instrument_key"]: {"last_price": pe_long_price}
    }

    return {"strategy": "Bull Put Spread", "strikes": [short_strike, long_strike],
            "premium": premium, "max_loss": max_loss, "max_profit": premium, "orders": orders, "pricing": pricing}

def wide_strangle(option_chain, spot_price, config, lots=1):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    
    # Wide OTM Call & Put
    call_strike = atm["strike_price"] + 200
    put_strike = atm["strike_price"] - 200

    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")

    if not all([ce_short_opt, pe_short_opt]):
        print("Error: Missing options for Wide Strangle")
        return None
    
    instrument_keys = [opt["instrument_key"] for opt in [ce_short_opt, pe_short_opt]]
    greeks_data = get_option_greeks(config, instrument_keys)

    ce_short_price = greeks_data.get(ce_short_opt["instrument_key"], {}).get("last_price", ce_short_opt["market_data"]["ltp"])
    pe_short_price = greeks_data.get(pe_short_opt["instrument_key"], {}).get("last_price", pe_short_opt["market_data"]["ltp"])

    premium = (ce_short_price + pe_short_price) * lots
    max_loss = float("inf") # Unlimited loss potential for short strangle

    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "NRML"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "NRML"}
    ]

    pricing = {
        ce_short_opt["instrument_key"]: {"last_price": ce_short_price},
        pe_short_opt["instrument_key"]: {"last_price": pe_short_price}
    }

    return {"strategy": "Wide Strangle", "strikes": [call_strike, put_strike],
            "premium": premium, "max_loss": max_loss, "max_profit": premium, "orders": orders, "pricing": pricing}

def atm_strangle(option_chain, spot_price, config, lots=1):
    atm = min(option_chain, key=lambda x: abs(x["strike_price"] - spot_price))
    
    # Closer OTM Call & Put
    call_strike = atm["strike_price"] + 50
    put_strike = atm["strike_price"] - 50

    ce_short_opt = find_option_by_strike(option_chain, call_strike, "CE")
    pe_short_opt = find_option_by_strike(option_chain, put_strike, "PE")

    if not all([ce_short_opt, pe_short_opt]):
        print("Error: Missing options for ATM Strangle")
        return None
    
    instrument_keys = [opt["instrument_key"] for opt in [ce_short_opt, pe_short_opt]]
    greeks_data = get_option_greeks(config, instrument_keys)

    ce_short_price = greeks_data.get(ce_short_opt["instrument_key"], {}).get("last_price", ce_short_opt["market_data"]["ltp"])
    pe_short_price = greeks_data.get(pe_short_opt["instrument_key"], {}).get("last_price", pe_short_opt["market_data"]["ltp"])

    premium = (ce_short_price + pe_short_price) * lots
    max_loss = float("inf") # Unlimited loss potential for short strangle

    orders = [
        {"instrument_key": ce_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "NRML"},
        {"instrument_key": pe_short_opt["instrument_key"], "quantity": lots * config["lot_size"], "transaction_type": "SELL", "product": "NRML"}
    ]

    pricing = {
        ce_short_opt["instrument_key"]: {"last_price": ce_short_price},
        pe_short_opt["instrument_key"]: {"last_price": pe_short_price}
    }

    return {"strategy": "ATM Strangle", "strikes": [call_strike, put_strike],
            "premium": premium, "max_loss": max_loss, "max_profit": premium, "orders": orders, "pricing": pricing}

# --- Dashboard & Plotting Functions ---

def print_dashboard_metrics(config, spot_price, seller, hv_7, garch_7d, iv_rv_spread, ivp, market, iv_skew_slope, regime_score, regime, regime_note, regime_explanation, event_df, strategies, strategy_rationale, event_warning, strategy_df, portfolio_summary, strategy_details):
    expiry_dt = datetime.strptime(config['expiry_date'], "%Y-%m-%d")
    print("\n" + "="*50)
    print("📊 OPTION SELLER DASHBOARD SUMMARY")
    print("="*50)
    print(f"📍 Spot Price           : {spot_price:.0f}")
    print(f"🎯 ATM Strike           : {seller['strike']:.0f}")
    print(f"💰 Straddle Price       : ₹{seller['straddle_price']:.2f}")
    print(f"📊 Breakeven Range      : {seller['strike'] - seller['straddle_price']:.0f} – {seller['strike'] + seller['straddle_price']:.0f}")
    print(f"📉 ATM IV               : {seller['avg_iv']:.2f}%")
    print(f"📉 Realized Vol (7D)    : {hv_7:.2f}%")
    print(f"🔮 GARCH Vol (7D)       : {garch_7d:.2f}%")
    print(f"🧮 IV - RV Spread       : {iv_rv_spread:+.2f}%")
    print(f"📊 IV Percentile (IVP)  : {ivp}%")
    print(f"⏳ Theta (Total)        : ₹{seller['theta']:.2f}")
    print(f"🌪️ Vega (IV Risk)       : ₹{seller['vega']:.2f}")
    print(f"📐 Delta                : {seller['delta']:.4f}")
    print(f"⚡ Gamma                : {seller['gamma']:.6f}")
    print(f"🎯 POP (Avg)            : {seller['pop']:.2f}%")
    print(f"📆 Days to Expiry       : {market['days_to_expiry']} days")
    print(f"🔁 PCR                  : {market['pcr']:.2f}")
    print(f"🎯 Max Pain             : {market['max_pain']:.0f}")
    print(f"📉 IV Skew Slope        : {iv_skew_slope:.4f}")
    print(f"🧠 Volatility Regime    : {regime} (Score: {regime_score:.2f})")
    print(f"📝 Note                : {regime_note}")
    print(f"📋 Details             : {regime_explanation}")
    print("\n📅 Events (Before Expiry):")
    if not event_df.empty:
        print(event_df.to_string(index=False))
        if any((expiry_dt - dt).days <= 3 and impact == "High" for dt, impact in zip(event_df["Datetime"], event_df["Classification"])):
            print("⚠️ High-impact event within 3 days of expiry.")
    else:
        print("No events found.")
    print("\n📈 Strategies:")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Rationale: {strategy_rationale}")
    if event_warning:
        print(event_warning)
    print("\n📊 Strategy Details:")
    if strategy_details:
        for detail in strategy_details:
            print(f"\nStrategy: {detail['strategy']}")
            print(f"  Strikes: {detail['strikes']}")
            print(f"  Premium: ₹{detail['premium']:.2f}")
            print(f"  Max Profit: ₹{detail['max_profit']:.2f}")
            print(f"  Max Loss: ₹{detail['max_loss']:.2f}" if detail['max_loss'] != float('inf') else "  Max Loss: Unlimited")
            # New: Estimated Margin
            if 'estimated_margin' in detail and detail['estimated_margin'] is not None:
                print(f"  Estimated Margin: ₹{detail['estimated_margin']:.2f}")
            else:
                print("  Estimated Margin: N/A (failed to calculate)")
    else:
        print("No valid strategies found for current market conditions.")

    print("\n📊 Risk Summary:")
    print(strategy_df.to_string(index=False))
    print("\n📦 Portfolio:")
    for k, v in portfolio_summary.items():
        if k != "Flags":
            if isinstance(v, (float, int)):
                # Adjust formatting for percentages vs currency
                if "%" in k:
                     print(f"{k:<20}: {v:.2f}%")
                elif "Capital" in k or "Risk" in k or "P&L" in k or "Drawdown" in k or "Funds" in k:
                    print(f"{k:<20}: ₹{v:,.2f}")
                else: # Generic float/int
                    print(f"{k:<20}: {v}")
            else: # Other types like strings
                print(f"{k:<20}: {v}")
    if portfolio_summary.get("Flags"):
        print("\n🚨 Risks/Warnings:")
        for flag in portfolio_summary["Flags"]:
            print(flag)
    else:
        print("\n✅ No risk violations.")
    print("="*50 + "\n")

def plot_payoff_diagram(strategy_details, spot_price):
    plt.figure(figsize=(10, 6))
    strikes = np.linspace(spot_price - 300, spot_price + 300, 200) # More points for smoother curve

    if not strategy_details:
        print("No strategy details to plot payoff diagram.")
        plt.close()
        return

    for detail in strategy_details:
        payoffs = np.zeros_like(strikes, dtype=float) # Ensure float array
        
        # Calculate for each order in the strategy
        for order in detail["orders"]:
            instrument_key = order["instrument_key"]
            qty = order["quantity"]
            transaction_type = order["transaction_type"]
            
            # Extract strike from instrument_key (rough method, better to store in detail)
            # Find the strike from detail['strikes'] list that corresponds to this instrument_key
            # This requires matching instrument_key back to its original strike in `detail['strikes']`
            # For simplicity, I'll use the 'strikes' list directly based on index for now,
            # but a robust solution would map instrument_key to its strike.
            try:
                # Find the option object in the original option_chain for this instrument_key
                # This needs access to the global option_chain or passing it.
                # A more robust way would be to store strike price in `orders` dict during strategy creation.
                # For now, let's assume `detail['strikes']` is ordered correctly.
                order_index = detail["orders"].index(order)
                strike = detail["strikes"][order_index] # This is a weak assumption.
            except (ValueError, IndexError):
                print(f"Warning: Could not determine strike for instrument {instrument_key}. Skipping payoff for this leg.")
                continue

            is_buy = (transaction_type == "BUY")
            is_call = ("CE" in instrument_key) # Check for "CE" in instrument_key
            
            # Get the price from the 'pricing' dict, fall back to 0 if not found
            price = detail["pricing"].get(instrument_key, {}).get("last_price", 0)

            if is_call:
                payoff_per_share = (strikes - strike) - price if is_buy else price - (strikes - strike)
                payoff_per_share = np.maximum(0, payoff_per_share) if is_buy else np.minimum(0, payoff_per_share)
            else: # Put option
                payoff_per_share = (strike - strikes) - price if is_buy else price - (strike - strikes)
                payoff_per_share = np.maximum(0, payoff_per_share) if is_buy else np.minimum(0, payoff_per_share)
            
            # Total payoff for this leg (per unit, then multiplied by quantity and lot size)
            payoffs += payoff_per_share * (abs(qty) / config["lot_size"]) # qty is lots * lot_size

        plt.plot(strikes, payoffs, label=detail["strategy"])

    plt.axvline(spot_price, linestyle="--", color="gray", label="Current Spot Price")
    plt.axhline(0, linestyle="--", color="black", label="Zero P&L")
    plt.title("📊 Payoff Diagram for Suggested Strategies")
    plt.xlabel("Underlying Price at Expiry")
    plt.ylabel("P&L (₹)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_vol_comparison(seller, hv_7, garch_7d):
    labels = ['ATM IV', 'Realized Vol (7D)', 'GARCH Vol (7D)']
    values = [seller["avg_iv"], hv_7, garch_7d]
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4, f"{bar.get_height():.2f}%", ha='center')
    plt.title("📊 Volatility Comparison: IV vs RV vs GARCH")
    plt.ylabel("Annualized Volatility (%)")
    plt.grid(axis='y', linestyle='--')
    plt.show()

def plot_chain_analysis(full_chain_df):
    if full_chain_df.empty:
        print("Full chain data is empty, cannot plot chain analysis.")
        return

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # IV Skew
    sns.lineplot(data=full_chain_df, x="Strike", y="IV Skew", ax=axes[0, 0], marker="o", color="purple")
    axes[0, 0].set_title("IV Skew Across Strikes")
    axes[0, 0].axhline(0, linestyle='--', color="gray")

    # Total Theta
    sns.lineplot(data=full_chain_df, x="Strike", y="Total Theta", ax=axes[0, 1], marker="o", color="green")
    axes[0, 1].set_title("Total Theta Across Strikes")

    # Straddle Price
    sns.lineplot(data=full_chain_df, x="Strike", y="Straddle Price", ax=axes[1, 0], marker="o", color="orange")
    axes[1, 0].set_title("Straddle Price Across Strikes")

    # Total OI
    # Use `full_chain_df["Strike"].astype(str)` to treat strikes as categorical for bar plot x-axis if preferred
    sns.barplot(x=full_chain_df["Strike"].astype(str), y=full_chain_df["Total OI"], ax=axes[1, 1], palette="Blues_d")
    axes[1, 1].set_title("Total OI Across Strikes")
    axes[1, 1].tick_params(axis='x', rotation=45) # Rotate x-axis labels for readability

    plt.tight_layout()
    plt.show()

# --- Main Execution Flow ---

def main():
    print("Initializing VolGuard system...")
    
    # 1. Fetch live market data
    option_chain = fetch_option_chain(config)
    if not option_chain:
        print("Failed to fetch option chain data. Exiting.")
        return

    # Use underlying_spot_price from the first option in the chain
    # It's more reliable to get the spot price directly from a dedicated endpoint if available
    # For now, stick to this as it's part of your existing flow.
    spot_price = option_chain[0].get("underlying_spot_price")
    if not spot_price:
        print("Could not determine spot price from option chain. Exiting.")
        return

    vix, nifty_spot = get_indices_quotes(config) # Renamed to nifty_spot for clarity
    if vix is None or nifty_spot is None:
        print("Failed to fetch VIX or Nifty Spot data. Exiting.")
        return
    
    # Update config's total_capital with current available funds if desired
    # This makes the risk evaluation more dynamic based on real trading capital
    funds_info = get_user_funds_and_margin(config, segment="SEC")
    current_available_funds = 0
    current_used_margin = 0
    if funds_info:
        current_available_funds = funds_info['available_margin']
        current_used_margin = funds_info['used_margin']
        config['total_capital'] = current_available_funds + current_used_margin # Re-evaluate total capital based on actual account
        print(f"💰 Account Total Capital (Approx): ₹{config['total_capital']:,.2f}")
        print(f"💰 Available Funds: ₹{current_available_funds:,.2f}")
        print(f"💰 Used Margin: ₹{current_used_margin:,.2f}")
    else:
        print("⚠️ Could not fetch real-time funds and margin. Using configured total capital.")
        current_available_funds = config['total_capital'] # Fallback

    # 2. Extract key metrics
    seller = extract_seller_metrics(option_chain, spot_price)
    if not seller:
        print("Failed to extract seller metrics. Exiting.")
        return

    full_chain_df = full_chain_table(option_chain, spot_price)
    if full_chain_df.empty:
        print("Failed to create full chain table. Exiting.")
        return

    market = market_metrics(option_chain, config['expiry_date'])
    if not market["pcr"]: # Check for a key metric to ensure success
        print("Failed to calculate market metrics. Exiting.")
        return
    
    ivp = load_ivp(config, seller["avg_iv"])
    if ivp == 0:
        print("Warning: IVP calculation failed or returned zero.")
    
    hv_7, garch_7d, iv_rv_spread = calculate_volatility(config, seller)
    if hv_7 == 0 and garch_7d == 0:
        print("Warning: Volatility calculations failed or returned zero.")
    
    iv_skew_slope = calculate_iv_skew_slope(full_chain_df)

    # 3. Determine volatility regime
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
    
    # 4. Load upcoming events
    event_df = load_upcoming_events(config)

    # 5. Suggest strategies
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

    # 6. Generate strategy details with estimated margin
    strategy_details = []
    print("\nCalculating estimated margins for suggested strategies...")
    for strat in strategies:
        strat_clean = strat.replace("(hedged)", "").replace("with strict stop", "").replace("short ", "").strip()
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
        if strat_clean in func_map:
            detail = func_map[strat_clean](option_chain, spot_price, config)
            if detail:
                # Calculate estimated margin for the strategy
                estimated_margin = calculate_strategy_margin(config, detail["orders"])
                detail["estimated_margin"] = estimated_margin
                strategy_details.append(detail)
            else:
                print(f"Could not generate details for strategy: {strat_clean}")
    
    if not strategy_details:
        print("No valid strategies with calculable details available.")
        return
    
    # 7. Evaluate current risk and portfolio
    trades_df = fetch_trade_data(config, full_chain_df)
    strategy_df, portfolio_summary = evaluate_full_risk(trades_df, config, regime, current_available_funds)
    
    pd.set_option("display.float_format", "{:.2f}".format)

    # 8. Print and plot dashboard
    print("\n📘 ATM ±300 Chain Table:")
    print(full_chain_df)
    
    eff_df = full_chain_df.copy()
    # Handle division by zero for Theta/Vega if Total Vega is 0
    eff_df["Theta/Vega"] = eff_df.apply(lambda row: row["Total Theta"] / row["Total Vega"] if row["Total Vega"] != 0 else np.nan, axis=1)
    eff_df = eff_df[["Strike", "Total Theta", "Total Vega", "Theta/Vega"]].sort_values("Theta/Vega", ascending=False).dropna()
    print("\n📊 Theta/Vega Ranking (Higher is better for sellers):")
    print(eff_df)

    print_dashboard_metrics(config, spot_price, seller, hv_7, garch_7d, iv_rv_spread, ivp, market, iv_skew_slope,
                            regime_score, regime, regime_note, regime_explanation, event_df, strategies,
                            strategy_rationale, event_warning, strategy_df, portfolio_summary, strategy_details)
    
    plot_vol_comparison(seller, hv_7, garch_7d)
    plot_chain_analysis(full_chain_df)
    plot_payoff_diagram(strategy_details, spot_price)

    # 9. Order Placement Logic
    print("\n🚀 Order Placement:")
    print("Available Strategies:")
    for idx, detail in enumerate(strategy_details):
        margin_str = f"Estimated Margin: ₹{detail['estimated_margin']:.2f}" if detail['estimated_margin'] is not None else "Estimated Margin: N/A"
        print(f"{idx + 1}. {detail['strategy']}: Premium ₹{detail['premium']:.2f}, Max Loss ₹{detail['max_loss']:.2f} ({margin_str})")
    
    while True:
        try:
            choice = input("\nEnter strategy number to place order (0 to skip, -1 to show strategies, -2 to get live order book): ")
            choice = int(choice)

            if choice == -2: # New option to fetch live order book
                print("\n--- Live Order Book ---")
                live_orders = get_order_book(config)
                if live_orders:
                    for order in live_orders:
                        print(f"Order ID: {order.get('order_id')}, "
                              f"Instrument: {order.get('trading_symbol')}, "
                              f"Type: {order.get('transaction_type')}, "
                              f"Quantity: {order.get('quantity')}, "
                              f"Filled: {order.get('filled_quantity')}, "
                              f"Status: {order.get('status')}, "
                              f"Message: {order.get('status_message')}")
                else:
                    print("No live orders found or error fetching order book.")
                print("-----------------------")
                continue # Go back to menu

            if choice == -1:
                print("\nAvailable Strategies:")
                for idx, detail in enumerate(strategy_details):
                    margin_str = f"Estimated Margin: ₹{detail['estimated_margin']:.2f}" if detail['estimated_margin'] is not None else "Estimated Margin: N/A"
                    print(f"{idx + 1}. {detail['strategy']}: Premium ₹{detail['premium']:.2f}, Max Loss ₹{detail['max_loss']:.2f} ({margin_str})")
                continue

            if choice == 0:
                print("Order placement skipped.")
                break

            if 0 < choice <= len(strategy_details):
                detail = strategy_details[choice - 1]
                
                # Pre-order margin check
                if detail['estimated_margin'] is not None:
                    # Refresh funds before placing order
                    current_funds_info = get_user_funds_and_margin(config, segment="SEC")
                    current_available_funds_for_check = current_funds_info['available_margin'] if current_funds_info else 0

                    if detail['estimated_margin'] > current_available_funds_for_check:
                        print(f"❌ Insufficient funds. Estimated margin required: ₹{detail['estimated_margin']:.2f}, Available: ₹{current_available_funds_for_check:.2f}")
                        confirm = input("Do you still want to attempt placing the order? (yes/no): ").lower()
                        if confirm != "yes":
                            print("Order placement cancelled due to insufficient funds.")
                            continue # Go back to menu
                    else:
                        print(f"✅ Funds sufficient. Estimated margin required: ₹{detail['estimated_margin']:.2f}, Available: ₹{current_available_funds_for_check:.2f}")
                else:
                    print("⚠️ Cannot verify margin, estimated margin not available for this strategy.")
                    confirm = input("Do you still want to proceed without margin verification? (yes/no): ").lower()
                    if confirm != "yes":
                        print("Order placement cancelled.")
                        continue


                confirm = input(f"Confirm placing {detail['strategy']} order? (yes/no): ").lower()
                if confirm == "yes":
                    order_ids = []
                    failed_orders = []
                    
                    # Separate buy and sell legs for sequential execution
                    buy_legs = [order for order in detail["orders"] if order["transaction_type"] == "BUY"]
                    sell_legs = [order for order in detail["orders"] if order["transaction_type"] == "SELL"]

                    # Step 1: Place BUY legs first
                    print(f"\nPlacing BUY legs for {detail['strategy']}...")
                    for order in buy_legs:
                        order_id = place_order(config, order["instrument_key"], order["quantity"], order["transaction_type"])
                        if order_id:
                            order_ids.append(order_id)
                            print(f"Buy Order placed: {order_id} for {order['instrument_key']} (Quantity: {order['quantity']})")
                        else:
                            failed_orders.append(order["instrument_key"])
                    
                    if buy_legs and not order_ids: # All buy orders failed
                        print(f"All BUY orders failed for {detail['strategy']}. Cancelling strategy placement.")
                        break # Exit order placement loop

                    # Step 2: Poll order book to confirm BUY legs are filled
                    if buy_legs:
                        print("\nWaiting for BUY legs to fill (polling order book every 5 seconds)...")
                        all_buy_orders_filled = False
                        max_wait_time = 120 # seconds
                        start_time = time.time()

                        while not all_buy_orders_filled and (time.time() - start_time) < max_wait_time:
                            current_orders = get_order_book(config)
                            all_buy_orders_filled = True
                            for buy_order_id in order_ids:
                                found_order = next((o for o in current_orders if o.get('order_id') == buy_order_id), None)
                                if not found_order or found_order.get('status') not in ["COMPLETE", "FILLED"]:
                                    all_buy_orders_filled = False
                                    if found_order and found_order.get('status') in ["CANCELLED", "REJECTED"]:
                                        print(f"Warning: Buy order {buy_order_id} was {found_order.get('status')}: {found_order.get('status_message')}")
                                        # Decide if to proceed or abort
                                        # For now, if any buy leg fails, assume we cannot proceed with sell legs
                                        print("Aborting sell leg placement due to failed buy leg.")
                                        break
                            if not all_buy_orders_filled:
                                time.sleep(5) # Wait before polling again
                            else:
                                print("All BUY legs confirmed filled!")
                                break
                        
                        if not all_buy_orders_filled:
                            print(f"Timeout or some BUY orders not filled after {max_wait_time} seconds. Review order book manually. Aborting SELL leg placement.")
                            # You might want to cancel pending buy orders here if not filled
                            break

                        # Step 3: Refresh funds to check margin benefit after buy legs are filled
                        print("\nRefreshing funds to check margin benefit...")
                        current_funds_info = get_user_funds_and_margin(config, segment="SEC")
                        current_available_funds_after_buy = current_funds_info['available_margin'] if current_funds_info else 0
                        print(f"Available Funds after BUY legs: ₹{current_available_funds_after_buy:,.2f}")
                        
                        # Re-check margin (simplified, assuming sell legs alone don't exceed available_margin after buy benefit)
                        # A more precise check would be to calculate margin for remaining sell legs and ensure sufficient funds.
                        # For now, just a general check.
                        if current_available_funds_after_buy < 0: # Or below a safe threshold
                            print("Warning: Available funds are low after BUY legs. Proceeding with SELL legs may lead to margin calls/rejections.")
                            proceed_sell = input("Proceed with SELL legs despite low funds? (yes/no): ").lower()
                            if proceed_sell != "yes":
                                print("SELL leg placement aborted.")
                                break


                    # Step 4: Place SELL legs
                    print(f"\nPlacing SELL legs for {detail['strategy']}...")
                    for order in sell_legs:
                        order_id = place_order(config, order["instrument_key"], order["quantity"], order["transaction_type"])
                        if order_id:
                            order_ids.append(order_id)
                            print(f"Sell Order placed: {order_id} for {order['instrument_key']} (Quantity: {order['quantity']})")
                        else:
                            failed_orders.append(order["instrument_key"])
                    
                    if order_ids:
                        print(f"Successfully placed {len(order_ids)} orders for {detail['strategy']}.")
                        if failed_orders:
                            print(f"Warning: {len(failed_orders)} orders failed: {failed_orders}")
                    else:
                        print(f"All orders failed for {detail['strategy']}.")
                    break # Exit order placement loop
                else:
                    print("Order placement cancelled.")
                    break
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(strategy_details)}, or -1 to show strategies, -2 to get live order book.")
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"Error in order placement: {e}")
            break

# --- Run the main function ---
if __name__ == "__main__":
    main()
