import requests 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from backend import (
    get_config,
    fetch_option_chain,
    get_indices_quotes,
    load_upcoming_events,
    extract_seller_metrics,
    full_chain_table,
    market_metrics,
    load_ivp,
    calculate_volatility,
    calculate_iv_skew_slope,
    calculate_regime,
    suggest_strategy,
    fetch_trade_data,
    get_funds_and_margin,
    calculate_sharpe_ratio,
    evaluate_full_risk,
    get_strategy_details
)

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
        background-color: #333330;
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
        padding: 10px 15px;
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

if 'access_token' not in st.session_state:
    st.session_state.access_token = ""
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

access_token = st.sidebar.text_input("Enter Upstox Access Token", type="password", value=st.session_state.access_token)

if st.sidebar.button("Login"):
    if access_token:
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
    from backend import logout
    if logout(config):
        st.session_state.access_token = ""
        st.session_state.logged_in = False
        st.experimental_rerun()

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
        st.markdown(f"<div class='metric-box'><h3>📈 Nifty 50 Spot</h3><div class='value'>{nifty:.2f}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-box'><h3>🌡️ India VIX</h3><div class='value'>{vix:.2f}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-box'><h3>🎯 ATM Strike</h3><div class='value'>{seller['strike']:.0f}</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-box'><h3>💰 Straddle Price</h3><div class='value'>₹{seller['straddle_price']:.2f}</div></div>", unsafe_allow_html=True)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown(f"<div class='metric-box'><h3>📉 ATM IV</h3><div class='value'>{seller['avg_iv']:.2f}%</div></div>", unsafe_allow_html=True)
    with col6:
        st.markdown(f"<div class='metric-box'><h3>📊 IVP</h3><div class='value'>{ivp}%</div></div>", unsafe_allow_html=True)
    with col7:
        st.markdown(f"<div class='metric-box'><h3>⏳ Days to Expiry</h3><div class='value'>{market['days_to_expiry']}</div></div>", unsafe_allow_html=True)
    with col8:
        st.markdown(f"<div class='metric-box'><h3>🔁 PCR</h3><div class='value'>{market['pcr']:.2f}</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Dashboard", "⛓️ Option Chain Analysis", "💡 Strategy Suggestions", "📈 Risk & Portfolio", "🚀 Place Orders", "🛡️ Risk Management Dashboard"])

    with tab1:
        st.markdown("<h2 style='color: #1E90FF;'>Market Overview</h2>", unsafe_allow_html=True)
        col_t1_1, col_t1_2 = st.columns([0.6, 0.4])
        with col_t1_1:
            st.subheader("Volatility Landscape")
            fig, ax = plt.subplots(figsize=(8, 5))
            labels = ['ATM IV', 'Realized Vol (7D)', 'GARCH Vol (7D)']
            values = [seller["avg_iv"], hv_7, garch_7d]
            colors = ['#00BFFF', '#32CD32', '#FF4500']
            bars = ax.bar(labels, values, color=colors)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                        f"{bar.get_height():.2f}%", ha='center', va='bottom', color='white')
            ax.set_title("📊 Volatility Comparison: IV vs RV vs GARCH", color="white")
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
            st.markdown(f"<div class='small-metric-box'><h4>🧮 IV - RV Spread:</h4> {iv_rv_spread:+.2f}%</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-metric-box'><h4>📉 IV Skew Slope:</h4> {iv_skew_slope:.4f}</div>", unsafe_allow_html=True)
            st.subheader("Breakeven & Max Pain")
            st.markdown(f"<div class='small-metric-box'><h4>📊 Breakeven Range:</h4> {seller['strike'] - seller['straddle_price']:.0f} – {seller['strike'] + seller['straddle_price']:.0f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-metric-box'><h4>🎯 Max Pain:</h4> {market['max_pain']:.0f}</div>", unsafe_allow_html=True)
        with col_t1_2:
            st.subheader("Greeks at ATM")
            st.markdown(f"<div class='small-metric-box'><h4>⏳ Theta (Total):</h4> ₹{seller['theta']:.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-metric-box'><h4>🌪️ Vega (IV Risk):</h4> ₹{seller['vega']:.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-metric-box'><h4>📐 Delta:</h4> {seller['delta']:.4f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-metric-box'><h4>⚡ Gamma:</h4> {seller['gamma']:.6f}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='small-metric-box'><h4>🎯 POP (Avg):</h4> {seller['pop']:.2f}%</div>", unsafe_allow_html=True)
            st.subheader("Upcoming Events")
            if not event_df.empty:
                st.dataframe(event_df.style.set_properties(**{'background-color': '#1A1C24', 'color': 'white'}), use_container_width=True)
                if event_warning:
                    st.warning(event_warning)
            else:
                st.info("No upcoming events before expiry.")

    with tab2:
        st.markdown("<h2 style='color: #1E90FF;'>Option Chain Analysis</h2>", unsafe_allow_html=True)
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
        st.subheader("ATM ±300 Chain Table")
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
            f"<div class='metric-box'><h3>🧠 Volatility Regime: {regime}</h3><p style='color: #6495ED;'>Score: {regime_score:.2f}</p><p>{regime_note}</p><p><i>{regime_explanation}</i></p>",
            unsafe_allow_html=True,
        )
        st.subheader("All Available Strategies")
        all_strategies = ["Iron Fly", "Iron Condor", "Jade Lizard", "Straddle", "Calendar Spread", "Bull Put Spread", "Wide Strangle", "ATM Strangle"]
        selected_strategies = st.multiselect("Select Strategies", options=all_strategies, default=strategies)
        st.info(f"**Suggested:** {', '.join(strategies)}")
        st.info(f"**Rationale:** {strategy_rationale}")
        if event_warning:
            st.warning(event_warning)
        for strat in selected_strategies:
            st.markdown(f"### {strat}")
            detail = get_strategy_details(strat, option_chain, spot_price, config, lots=1)
            if detail:
                st.markdown(f"**Strategy:** {detail['strategy']}")
                st.markdown(f"**Strikes:** {', '.join(map(str, detail['strikes']))}")
                st.markdown(f"**Premium Collected:** ₹{detail['premium_total']:.2f}")
                st.markdown(f"**Max Profit:** ₹{detail['max_profit']:.2f}" if detail['max_profit'] != float('inf') else "**Max Profit:** Unlimited")
                st.markdown(f"**Max Loss:** ₹{detail['max_loss']:.2f}" if detail['max_loss'] != float('inf') else "**Max Loss:** Unlimited")
                order_df = pd.DataFrame({
                    "Instrument": [order["instrument_key"] for order in detail["orders"]],
                    "Type": [order["transaction_type"] for order in detail["orders"]],
                    "Option Type": ["CE" if "CE" in order["instrument_key"] else "PE" for order in detail["orders"]],
                    "Strike": detail["strikes"],
                    "Quantity (per lot)": [config["lot_size"] for _ in detail["orders"]],
                    "LTP": [order.get("current_price", 0) for order in detail["orders"]],
                })
                st.dataframe(order_df.style.format(precision=2).set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
                st.subheader("Payoff Diagram")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.set_style("darkgrid")
                min_strike = min(min(d["strikes"]) for d in strategy_details if d["strikes"]) - 200
                max_strike = max(max(d["strikes"]) for d in strategy_details if d["strikes"]) + 200
                strikes = np.linspace(min_strike, max_strike, 200)
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
                ax.set_title("📊 Payoff Diagram", color="white")
                ax.set_xlabel("Underlying Price", color="white")
                ax.set_ylabel("Payoff (₹)", color="white")
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
                st.subheader("Risk Metrics")
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.markdown(f"<div class='small-metric-box'><h4>💰 Premium:</h4> ₹{detail['premium_total']:.2f}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='small-metric-box'><h4>📉 Max Loss:</h4> ₹{detail['max_loss']:.2f}</div>" if detail['max_loss'] != float('inf') else "<div class='small-metric-box'><h4>📉 Max Loss:</h4> Unlimited</div>", unsafe_allow_html=True)
                with col_r2:
                    st.markdown(f"<div class='small-metric-box'><h4>📈 Max Profit:</h4> ₹{detail['max_profit']:.2f}</div>" if detail['max_profit'] != float('inf') else "<div class='small-metric-box'><h4>📈 Max Profit:</h4> Unlimited</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='small-metric-box'><h4>🎯 Risk/Reward:</h4> {detail['max_loss'] / detail['max_profit']:.2f}</div>" if detail['max_profit'] > 0 and detail['max_loss'] != float('inf') else "<div class='small-metric-box'><h4>🎯 Risk/Reward:</h4> Undefined</div>", unsafe_allow_html=True)
                with col_r3:
                    capital_required = detail['premium_total'] * 10
                    st.markdown(f"<div class='small-metric-box'><h4>💸 Capital Required:</h4> ₹{capital_required:.2f}</div>", unsafe_allow_html=True)
                lots = st.number_input(f"Lots for {strat}", min_value=1, value=1, step=1, key=f"lots_{strat}")
                if st.button(f"Place {strat} Order", key=f"place_{strat}"):
                    updated_detail = get_strategy_details(strat, option_chain, spot_price, config, lots)
                    if updated_detail:
                        success = True
                        for order in updated_detail["orders"]:
                            order_id = place_order(
                                config,
                                order["instrument_key"],
                                order["quantity"],
                                order["transaction_type"],
                            )
                            if not order_id:
                                st.error(f"Failed to place order for {order['instrument_key']}")
                                success = False
                        if success:
                            st.success(f"Placed {strat} order with {lots} lots!")

    with tab4:
        st.markdown("<h2 style='color: #1E90FF;'>Risk & Portfolio Analysis</h2>", unsafe_allow_html=True)
        st.subheader("Portfolio Summary")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            st.markdown(f"<div class='metric-box'><h3>💰 Total Capital</h3><div class='value'>₹{portfolio_summary['Total Capital']:.2f}</div></div>", unsafe_allow_html=True)
        with col_p2:
            st.markdown(f"<div class='metric-box'><h3>📈 Capital Deployed</h3><div class='value'>₹{portfolio_summary['Capital Deployed']:.2f}</div></div>", unsafe_allow_html=True)
        with col_p3:
            st.markdown(f"<div class='metric-box'><h3>⚖️ Exposure %</h3><div class='value'>{portfolio_summary['Exposure %']:.2f}%</div></div>", unsafe_allow_html=True)
        with col_p4:
            st.markdown(f"<div class='metric-box'><h3>📉 Drawdown %</h3><div class='value'>{portfolio_summary['Drawdown %']:.2f}%</div></div>", unsafe_allow_html=True)
        st.subheader("Strategy Risk Table")
        if not strategy_df.empty:
            st.dataframe(strategy_df.style.format(precision=2).set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
            st.subheader("Capital Allocation")
            fig, ax = plt.subplots(figsize=(8, 8))
            labels = strategy_df["Strategy"]
            sizes = strategy_df["Capital Used"]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, textprops={'color': 'white'})
            ax.set_title("Capital Allocation by Strategy", color="white")
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#0E1117')
            st.pyplot(fig)
            if portfolio_summary.get("Flags"):
                st.subheader("Risk Alerts")
                for flag in portfolio_summary["Flags"]:
                    st.error(flag)

    with tab5:
        st.markdown("<h2 style='color: #1E90FF;'>Place Orders</h2>", unsafe_allow_html=True)
        st.subheader("Select Strategy to Place Order")
        all_strategies = ["Iron Fly", "Iron Condor", "Jade Lizard", "Straddle", "Calendar Spread", "Bull Put Spread", "Wide Strangle", "ATM Strangle"]
        selected_strategy = st.selectbox("Choose Strategy", all_strategies, key="order_strategy")
        if selected_strategy:
            lots = st.number_input("Number of Lots", min_value=1, value=1, step=1, key="order_lots")
            detail = get_strategy_details(selected_strategy, option_chain, spot_price, config, lots)
            if detail:
                st.subheader(f"{selected_strategy} Order Details")
                order_df = pd.DataFrame({
                    "Instrument": [order["instrument_key"] for order in detail["orders"]],
                    "Type": [order["transaction_type"] for order in detail["orders"]],
                    "Option Type": ["CE" if "CE" in order["instrument_key"] else "PE" for order in detail["orders"]],
                    "Strike": detail["strikes"],
                    "Quantity": [order["quantity"] for order in detail["orders"]],
                    "LTP": [order.get("current_price", 0) for order in detail["orders"]],
                })
                st.dataframe(order_df.style.format(precision=2).set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
                st.markdown(f"<div class='metric-box'><h3>💰 Total Premium:</h3><div class='value'>₹{detail['premium_total']:.2f}</div></div>", unsafe_allow_html=True)
                if st.button("Confirm and Place Order", key="confirm_order"):
                    success = True
                    for order in detail["orders"]:
                        order_id = place_order(
                            config,
                            order["instrument_key"],
                            order["quantity"],
                            order["transaction_type"],
                        )
                        if not order_id:
                            st.error(f"Failed to place order for {order['instrument_key']}")
                            success = False
                    if success:
                        st.success(f"Placed {selected_strategy} order with {lots} lots!")
            else:
                st.error(f"Unable to fetch details for {selected_strategy}.")
    
    with tab6:
        st.markdown("<h2 style='color: #1E90FF;'>Risk Management Dashboard</h2>", unsafe_allow_html=True)
        st.subheader("Portfolio Risk Overview")
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.markdown(f"<div class='metric-box'><h3>📉 Total Risk</h3><div class='value'>₹{portfolio_summary['Risk on Table']:.2f}</div></div>", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"<div class='metric-box'><h3>📊 Sharpe Ratio</h3><div class='value'>{sharpe_ratio:.2f}</div></div>", unsafe_allow_html=True)
        with col_r3:
            margin_pct = (funds_data["used_margin"] / funds_data["total_funds"] * 100) if funds_data["total_funds"] > 0 else 0
            st.markdown(f"<div class='metric-box'><h3>💸 Margin Utilization</h3><div class='value'>{margin_pct:.2f}%</div></div>", unsafe_allow_html=True)
        with col_r4:
            st.markdown(f"<div class='metric-box'><h3>🪙 Portfolio Vega</h3><div class='value'>₹{portfolio_summary['Portfolio Vega']:.2f}</div></div>", unsafe_allow_html=True)
        st.subheader("All Positions")
        if 'positions' in locals():
            pos_df = pd.DataFrame(positions)
            pos_df = pos_df[["instrument_token", "quantity", "average_price", "pnl"]]
            pos_df.columns = ["Instrument", "Quantity", "Avg Price", "P&L"]
            st.dataframe(pos_df.style.format({"Avg Price": "{:.2f}", "P&L": "{:.2f}", "Quantity": "{:.0f}"}).set_properties(**{"background-color": "#1A1C24", "color": "white"}), use_container_width=True)
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

else:
    st.markdown("<h2 style='text-align: center; color: #1E90FF;'>Welcome to Volguard</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #6495ED;'>To begin, please log in using your Upstox access token.</h4>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: #6495ED;'>You can generate your access token from the Upstox developer portal.</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: #6495ED;'>Ensure you have enabled Futures & Options permissions in your profile.</h5>", unsafe_allow_html=True)
