import streamlit as st
import streamlit.components.v1 as components
import backend
import pandas as pd
import matplotlib.pyplot as plt
import base64
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="VolGuard Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and improved visuals
st.markdown("""
<style>
    .reportview-container {
        background-color: #0e1117;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        width: 100%;
    }
    .stTextInput input {
        background-color: #1a1f28;
        color: #ffffff;
        border: 1px solid #262d3d;
    }
    .stSelectbox select {
        background-color: #1a1f28;
        color: #ffffff;
        border: 1px solid #262d3d;
    }
    .stButton button {
        background-color: #1e90ff;
        color: white;
        border-radius: 5px;
        height: 3em;
        width: 100%;
    }
    .metric-card {
        background-color: #1a1f28;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .metric-header {
        font-weight: bold;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.5em;
        font-weight: bold;
    }
    .metric-subtext {
        font-size: 0.8em;
        color: #aaaaaa;
    }
    .tab {
        background-color: #1a1f28;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }
    .card {
        background-color: #1a1f28;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .card-title {
        font-weight: bold;
        margin-bottom: 10px;
    }
    .flag-warning {
        background-color: #2d1b0f;
        color: #ffcc99;
        padding: 10px;
        border-left: 4px solid #ff9933;
        margin: 10px 0;
    }
    .flag-success {
        background-color: #1b262d;
        color: #cceeff;
        padding: 10px;
        border-left: 4px solid #3399ff;
        margin: 10px 0;
    }
    .strategy-card {
        background-color: #1f242f;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #262d3d;
    }
    .strategy-header {
        font-weight: bold;
        font-size: 1.2em;
        margin-bottom: 10px;
    }
    .strategy-detail {
        margin-left: 15px;
        margin-bottom: 5px;
    }
    .strategy-strike {
        color: #cccccc;
        font-size: 0.9em;
    }
    .strategy-premium {
        color: #33cc99;
        font-weight: bold;
    }
    .strategy-max-loss {
        color: #ff6666;
        font-weight: bold;
    }
    .strategy-margin {
        color: #ffff99;
    }
    .table-container {
        overflow-x: auto;
        border-radius: 10px;
        border: 1px solid #262d3d;
        padding: 10px;
        margin: 10px 0;
    }
    .dataframe thead tr th {
        background-color: #1a1f28 !important;
        color: #ffffff !important;
        border: 1px solid #262d3d !important;
    }
    .dataframe tbody tr td {
        color: #ffffff !important;
        border: 1px solid #262d3d !important;
    }
    .dataframe {
        width: 100% !important;
        border: none !important;
    }
</style>""", unsafe_allow_html=True)

def display_metric_card(title, value, subtext, col=None):
    """Display a visually appealing metric card"""
    card_html = f"""
    <div class="metric-card">
        <div class="metric-header">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtext">{subtext}</div>
    </div>
    """
    if col:
        with col:
            st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.markdown(card_html, unsafe_allow_html=True)

def display_flag(flag, col=None):
    """Display a flag with appropriate styling"""
    if flag.startswith("✅"):
        html = f'<div class="flag-success">{flag}</div>'
    elif flag.startswith("❌") or flag.startswith("⚠️"):
        html = f'<div class="flag-warning">{flag}</div>'
    else:
        html = f'<div>{flag}</div>'
    
    if col:
        with col:
            st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)

def display_strategy_card(detail):
    """Display a visually appealing strategy card"""
    strike_text = ", ".join([f"{strike}" for strike in detail["strikes"]])
    
    max_loss_value = detail["max_loss"]
    if max_loss_value == float('inf'):
        max_loss_display = "Unlimited"
    else:
        max_loss_display = f"₹{detail['max_loss']:.2f}"
    
    margin_display = f"₹{detail['estimated_margin']:.2f}" if detail[
        'estimated_margin'] is not None else "N/A"
    
    card_html = f"""
    <div class="strategy-card">
        <div class="strategy-header">{detail['strategy']}</div>
        <div class="strategy-detail">🎯 <span class="strategy-strike">Strikes: {strike_text}</span></div>
        <div class="strategy-detail">💰 Premium: <span class="strategy-premium">₹{detail['premium']:.2f}</span></div>
        <div class="strategy-detail">📉 Max Loss: <span class="strategy-max-loss">{max_loss_display}</span></div>
        <div class="strategy-detail">🛡️ Estimated Margin: <span class="strategy-margin">{margin_display}</span></div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def display_strategy_details(strategy_details):
    """Display strategy details in a visually appealing way"""
    cols = st.columns(len(strategy_details))
    for i, detail in enumerate(strategy_details):
        with cols[i]:
            display_strategy_card(detail)

def main():
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'backend_data' not in st.session_state:
        st.session_state.backend_data = None

    # Sidebar
    with st.sidebar:
        st.markdown("### 🛡️ VolGuard Pro\n#### Option Strategy Advisor")
        st.markdown("---")
        
        if not st.session_state.logged_in:
            access_token = st.text_input("🔑 Enter Access Token", type="password", 
                                      help="Enter your Upstox API access token")
            login_button = st.button("🔓 Login")
            
            if login_button and access_token:
                st.session_state.config = backend.get_config(access_token)
                st.session_state.logged_in = True
                st.rerun()
        else:
            st.success("✅ Token Accepted")
            st.info("🔄 Processing Data...")
            
            if st.button("🔁 Refresh Data"):
                st.session_state.data_loaded = False
            
            if not st.session_state.data_loaded:
                with st.spinner("Fetching market data..."):
                    (option_chain, spot_price, seller, hv_7, garch_7d, iv_rv_spread,
                     ivp, market, iv_skew_slope, regime_score, regime, regime_note,
                     regime_explanation, event_df, strategies, strategy_rationale,
                     event_warning, strategy_df, portfolio_summary, strategy_details,
                     full_chain_df) = backend.main(st.session_state.config)
                    
                    if option_chain:
                        st.session_state.data_loaded = True
                        st.session_state.backend_data = {
                            "option_chain": option_chain,
                            "spot_price": spot_price,
                            "seller": seller,
                            "hv_7": hv_7,
                            "garch_7d": garch_7d,
                            "iv_rv_spread": iv_rv_spread,
                            "ivp": ivp,
                            "market": market,
                            "iv_skew_slope": iv_skew_slope,
                            "regime_score": regime_score,
                            "regime": regime,
                            "regime_note": regime_note,
                            "regime_explanation": regime_explanation,
                            "event_df": event_df,
                            "strategies": strategies,
                            "strategy_rationale": strategy_rationale,
                            "event_warning": event_warning,
                            "strategy_df": strategy_df,
                            "portfolio_summary": portfolio_summary,
                            "strategy_details": strategy_details,
                            "full_chain_df": full_chain_df
                        }
                        st.rerun()
                    else:
                        st.error("Failed to fetch market data. Please try again.")
            else:
                st.success("Data loaded ✅")

    # Main content area
    if not st.session_state.logged_in:
        st.markdown("# 🛡️ VolGuard Pro - Option Strategy Advisor")
        st.markdown("## Welcome to VolGuard Pro")
        st.markdown("Please enter your access token in the sidebar to begin.")
        st.image("https://via.placeholder.com/1200x400?text=VolGuard+Pro+Dashboard+Preview")
        return

    if not st.session_state.data_loaded:
        st.markdown("# 🛡️ VolGuard Pro - Option Strategy Advisor")
        st.markdown("## Waiting for data to load...")
        with st.spinner("Loading data... This may take a few moments."):
            time.sleep(2)
        return

    # Retrieve data from session state
    data = st.session_state.backend_data
    option_chain = data["option_chain"]
    spot_price = data["spot_price"]
    seller = data["seller"]
    hv_7 = data["hv_7"]
    garch_7d = data["garch_7d"]
    iv_rv_spread = data["iv_rv_spread"]
    ivp = data["ivp"]
    market = data["market"]
    iv_skew_slope = data["iv_skew_slope"]
    regime_score = data["regime_score"]
    regime = data["regime"]
    regime_note = data["regime_note"]
    regime_explanation = data["regime_explanation"]
    event_df = data["event_df"]
    strategies = data["strategies"]
    strategy_rationale = data["strategy_rationale"]
    event_warning = data["event_warning"]
    strategy_df = data["strategy_df"]
    portfolio_summary = data["portfolio_summary"]
    strategy_details = data["strategy_details"]
    full_chain_df = data["full_chain_df"]

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", 
        "📈 Market Analysis", 
        "💼 Strategy Builder", 
        "🧮 Order Manager"
    ])

    # Dashboard Tab
    with tab1:
        st.markdown("## 📊 Real-Time Dashboard")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        display_metric_card("📍 Spot Price", f"{spot_price:.0f}", "Current Nifty Level", col1)
        display_metric_card("🎯 ATM Strike", f"{seller['strike']:.0f}", "At The Money Option", col2)
        display_metric_card("💰 Straddle Price", f"₹{seller['straddle_price']:.2f}", "Combined CE + PE Price", col3)
        breakeven_range = f"{seller['strike'] - seller['straddle_price']:.0f} – {seller['strike'] + seller['straddle_price']:.0f}"
        display_metric_card("📉 Breakeven Range", breakeven_range, "Straddle Breakpoints", col4)
        
        col5, col6, col7, col8 = st.columns(4)
        display_metric_card("📉 ATM IV", f"{seller['avg_iv']:.2f}%", "Implied Volatility", col5)
        display_metric_card("📉 Realized Vol (7D)", f"{hv_7:.2f}%", "Historical Volatility", col6)
        display_metric_card("🔮 GARCH Vol (7D)", f"{garch_7d:.2f}%", "Volatility Forecast", col7)
        display_metric_card("🧮 IV - RV Spread", f"{iv_rv_spread:+.2f}%", "Premium/Risk Indicator", col8)
        
        col9, col10, col11, col12 = st.columns(4)
        display_metric_card("📊 IV Percentile", f"{ivp}%", "IV Valuation", col9)
        display_metric_card("🌪️ Vega", f"₹{seller['vega']:.2f}", "Volatility Exposure", col10)
        display_metric_card("📐 Delta", f"{seller['delta']:.4f}", "Directional Bias", col11)
        display_metric_card("🧠 Vol Regime", regime, regime_note, col12)
        
        col13, col14, col15, col16 = st.columns(4)
        display_metric_card("📆 Days to Expiry", f"{market['days_to_expiry']}", "Contract Days Remaining", col13)
        display_metric_card("🔁 PCR", f"{market['pcr']:.2f}", "Put Call Ratio", col14)
        display_metric_card("🎯 Max Pain", f"{market['max_pain']:.0f}", "Pain Point", col15)
        display_metric_card("📉 IV Skew Slope", f"{iv_skew_slope:.4f}", "Skew Indicator", col16)
        
        # Volatility comparison chart
        st.markdown("### 📈 Volatility Analysis")
        fig = backend.plot_vol_comparison(seller, hv_7, garch_7d)
        st.pyplot(fig)
        plt.close()
        
        # Portfolio overview
        st.markdown("### 🧾 Portfolio Overview")
        col17, col18, col19, col20 = st.columns(4)
        display_metric_card("💰 Total Capital", f"₹{portfolio_summary['Total Capital']:,}", "Account Size", col17)
        display_metric_card("💸 Capital Deployed", f"₹{portfolio_summary['Capital Deployed']:,} ({portfolio_summary['Exposure %']:.2f}%)", "Used Portion", col18)
        display_metric_card("🛑 Daily Risk Limit", f"₹{portfolio_summary['Daily Risk Limit']:,}", "Maximum Daily Risk", col19)
        display_metric_card("📆 Weekly Risk Limit", f"₹{portfolio_summary['Weekly Risk Limit']:,}", "Maximum Weekly Risk", col20)
        
        # Drawdown info
        drawdown_col = st.columns(1)[0]
        drawdown_color = "red" if portfolio_summary["Drawdown ₹"] > 0 else "green"
        drawdown_text = f"🔴 Drawdown: ₹{portfolio_summary['Drawdown ₹']:,} ({portfolio_summary['Drawdown %']:.2f}%)"
        display_metric_card("📉 Portfolio Drawdown", drawdown_text, 
                          "Since Start of Period", drawdown_col)
        
        # Flags section
        st.markdown("### 🚩 Risk Warnings")
        if portfolio_summary.get("Flags"):
            for flag in portfolio_summary["Flags"]:
                display_flag(flag)
        else:
            st.success("✅ No active risk violations detected")
            
    # Market Analysis Tab
    with tab2:
        st.markdown("## 📈 Market Analysis")
        
        # Regime analysis
        st.markdown(f"### 🔍 Volatility Regime: **{regime}** (Score: {regime_score:.2f})")
        st.markdown(f"**Note:** {regime_note}")
        st.markdown(f"**Details:** {regime_explanation}")
        
        # Event analysis
        st.markdown("### 🗓️ Upcoming Events")
        if not event_df.empty:
            st.dataframe(event_df, use_container_width=True)
            if event_warning:
                display_flag(event_warning)
        else:
            st.info("No upcoming events affecting the market")
        
        # Chain analysis
        st.markdown("### 🔄 Options Chain Analysis")
        fig = backend.plot_chain_analysis(full_chain_df)
        st.pyplot(fig)
        plt.close()
        
        # Efficiency analysis
        st.markdown("### ⚙️ Strategy Efficiency")
        eff_df = full_chain_df.copy()
        eff_df["Theta/Vega"] = eff_df.apply(
            lambda row: row["Total Theta"] / row["Total Vega"] if row[
                                                                 "Total Vega"] != 0 else float('nan'),
            axis=1
        )
        eff_df = eff_df[["Strike", "Total Theta", "Total Vega", "Theta/Vega"]].sort_values(
            "Theta/Vega", ascending=False).dropna()
        
        st.dataframe(eff_df, use_container_width=True)
        
        # Regime graph
        st.markdown("### 📊 Volatility Comparison")
        fig = backend.plot_vol_comparison(seller, hv_7, garch_7d)
        st.pyplot(fig)
        plt.close()
        
    # Strategy Builder Tab
    with tab3:
        st.markdown("## 💼 Strategy Builder")
        
        # Strategy suggestion
        st.markdown("### 🎯 Recommended Strategies")
        st.markdown(f"**Rationale:** {strategy_rationale}")
        if event_warning:
            display_flag(event_warning)
        
        # Display strategy cards
        st.markdown("### 📦 Strategy Details")
        display_strategy_details(strategy_details)
        
        # All strategies
        st.markdown("### 📋 Full Strategy Menu")
        strategy_cols = st.columns(4)
        strategy_names = ["Iron Fly", "Iron Condor", "Jade Lizard", "Straddle", 
                         "Calendar Spread", "Bull Put Spread", "Wide Strangle", "ATM Strangle"]
        
        # Create cards for all strategies
        for i, name in enumerate(strategy_names):
            with strategy_cols[i % 4]:
                st.markdown(f"**{name}**")
                st.markdown("• Defined risk" if "Iron" in name or "Lizard" in name else "• Undefined risk")
                st.markdown("• Directional" if "Bull" in name or "Bear" in name else "• Neutral")
                st.markdown("• Complex" if "Fly" in name or "Condor" in name else "• Simple")
                st.markdown("---")
        
        # Strategy parameters
        st.markdown("### ⚙️ Strategy Parameters")
        strategy_select = st.selectbox("Select Strategy", strategy_names)
        
        col1, col2 = st.columns(2)
        lots = col1.number_input("Number of Lots", 1, 10, 1)
        expiry_date = col2.date_input("Expiry Date", 
                                    value=datetime.strptime(st.session_state.config['expiry_date'], "%Y-%m-%d"),
                                    disabled=True)
        
        # Generate strategy
        if st.button("🔍 Generate Strategy"):
            # Here you would actually generate the strategy based on inputs
            st.success(f"Generated {strategy_select} with {lots} lots")
            # Show some generated strategy details
            st.markdown("#### Generated Strategy Details")
            st.markdown("• Entry: XYZ")
            st.markdown("• Exit: ABC")
            st.markdown("• Profit Target: 75%")
            st.markdown("• Stop Loss: 25%")
            
        # Strategy visualizations
        st.markdown("### 📊 Strategy Visualization")
        if strategy_details:
            fig = backend.plot_payoff_diagram(strategy_details, spot_price)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("No strategy details available for visualization")
    
    # Order Manager Tab
    with tab4:
        st.markdown("## 📝 Order Manager")
        
        # Current order book
        st.markdown("### 🕒 Live Orders")
        try:
            live_orders = backend.get_order_book(st.session_state.config)
            if live_orders:
                order_df = pd.DataFrame(live_orders)
                st.dataframe(order_df, use_container_width=True)
            else:
                st.info("No live orders found")
        except:
            st.warning("Could not retrieve order book")
        
        # Positions
        st.markdown("### 🧾 Current Positions")
        try:
            url_positions = f"{st.session_state.config['base_url']}/portfolio/short-term-positions"
            res_positions = requests.get(url_positions, headers=st.session_state.config['headers'])
            if res_positions.status_code == 200:
                positions = res_positions.json()["data"]
                if positions:
                    positions_df = pd.DataFrame(positions)
                    st.dataframe(positions_df, use_container_width=True)
                else:
                    st.info("No current positions")
            else:
                st.warning("Could not retrieve positions")
        except Exception as e:
            st.error(f"Error retrieving positions: {str(e)}")
        
        # Place order form
        st.markdown("### 📥 Place New Order")
        order_form = st.form("order_form")
        strategy_name = order_form.selectbox("Strategy", strategies)
        strategy_index = strategies.index(strategy_name) if strategy_name in strategies else 0
        
        # Display selected strategy details
        if strategy_details and strategy_index < len(strategy_details):
            selected_detail = strategy_details[strategy_index]
            display_strategy_card(selected_detail)
        
        # Quantity and order type
        col1, col2 = order_form.columns(2)
        lots = col1.number_input("Lots", 1, 10, 1)
        order_type = col2.selectbox("Order Type", ["Limit", "Market"])
        
        # Submit button
        submit = order_form.form_submit_button("🚀 Place Order")
        if submit:
            # Here you would actually place the order
            st.success(f"Placed {order_type} order for {lots} lots of {strategy_name}")

if __name__ == "__main__":
    main() 
