import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime
import App  # Import your backend code as a module
import json

# Set page configuration
st.set_page_config(
    page_title="VolGuard - Your Trading Copilot",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS for dark theme enhancements and trading app aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #252526;
    }
    .stButton>button {
        background-color: #3c8dbc;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #2a6b9c;
    }
    .metric-card {
        background-color: #2c2c2c;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
    .metric-title {
        font-size: 14px;
        color: #cccccc;
    }
    .metric-value {
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #252526;
    }
    .stTabs [data-baseweb="tab"] {
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3c8dbc;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'access_token' not in st.session_state:
    st.session_state['access_token'] = None
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
if 'config' not in st.session_state:
    st.session_state['config'] = None
if 'data_fetched' not in st.session_state:
    st.session_state['data_fetched'] = False
if 'option_chain' not in st.session_state:
    st.session_state['option_chain'] = None
if 'spot_price' not in st.session_state:
    st.session_state['spot_price'] = None
if 'vix' not in st.session_state:
    st.session_state['vix'] = None
if 'nifty_spot' not in st.session_state:
    st.session_state['nifty_spot'] = None
if 'current_available_funds' not in st.session_state:
    st.session_state['current_available_funds'] = 0
if 'current_used_margin' not in st.session_state:
    st.session_state['current_used_margin'] = 0
if 'seller' not in st.session_state:
    st.session_state['seller'] = {}
if 'full_chain_df' not in st.session_state:
    st.session_state['full_chain_df'] = pd.DataFrame()
if 'market' not in st.session_state:
    st.session_state['market'] = {}
if 'ivp' not in st.session_state:
    st.session_state['ivp'] = 0
if 'hv_7' not in st.session_state:
    st.session_state['hv_7'] = 0
if 'garch_7d' not in st.session_state:
    st.session_state['garch_7d'] = 0
if 'iv_rv_spread' not in st.session_state:
    st.session_state['iv_rv_spread'] = 0
if 'iv_skew_slope' not in st.session_state:
    st.session_state['iv_skew_slope'] = 0
if 'regime_score' not in st.session_state:
    st.session_state['regime_score'] = 0
if 'regime' not in st.session_state:
    st.session_state['regime'] = ""
if 'regime_note' not in st.session_state:
    st.session_state['regime_note'] = ""
if 'regime_explanation' not in st.session_state:
    st.session_state['regime_explanation'] = ""
if 'event_df' not in st.session_state:
    st.session_state['event_df'] = pd.DataFrame()
if 'strategies' not in st.session_state:
    st.session_state['strategies'] = []
if 'strategy_rationale' not in st.session_state:
    st.session_state['strategy_rationale'] = ""
if 'event_warning' not in st.session_state:
    st.session_state['event_warning'] = None
if 'strategy_df' not in st.session_state:
    st.session_state['strategy_df'] = pd.DataFrame()
if 'portfolio_summary' not in st.session_state:
    st.session_state['portfolio_summary'] = {}
if 'strategy_details' not in st.session_state:
    st.session_state['strategy_details'] = []
if 'all_strategy_details' not in st.session_state:
    st.session_state['all_strategy_details'] = []

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #3c8dbc;'>VolGuard - Your Trading Copilot</h2>", unsafe_allow_html=True)
    
    # Logo (optional, if you have one)
    # st.image("assets/volguard_logo.png", use_column_width=True)
    
    if not st.session_state['authenticated']:
        st.subheader("Login")
        access_token = st.text_input("Enter Upstox API Access Token", type="password")
        if st.button("Login"):
            with st.spinner("Validating access token..."):
                # Update config with the provided token
                config = App.get_config()
                config['access_token'] = access_token
                config['headers']['Authorization'] = f"Bearer {access_token}"
                
                # Test token with a simple API call
                try:
                    url = f"{config['base_url']}/user/get-funds-and-margin"
                    res = requests.get(url, headers=config['headers'], params={"segment": "SEC"})
                    if res.status_code == 200:
                        st.session_state['access_token'] = access_token
                        st.session_state['authenticated'] = True
                        st.session_state['config'] = config
                        st.success("Access token validated! Fetching data...")
                        # Trigger data fetch
                        st.session_state['data_fetched'] = False
                    else:
                        st.error(f"Invalid access token: {res.status_code} - {res.text}")
                except Exception as e:
                    st.error(f"Error validating token: {e}")
    else:
        st.subheader("Account")
        st.write(f"Status: ✅ Connected")
        st.write(f"Token: {st.session_state['access_token'][:4]}...{st.session_state['access_token'][-4:]}")
        if st.button("Logout"):
            st.session_state['authenticated'] = False
            st.session_state['access_token'] = None
            st.session_state['config'] = None
            st.session_state['data_fetched'] = False
            st.success("Logged out successfully.")
            st.experimental_rerun()

    if st.session_state['authenticated']:
        st.subheader("Navigation")
        tab_selection = st.radio(
            "Go to:",
            ["Dashboard", "Chain Analysis", "Strategy Details", "Order Placement", "Order Book", "Portfolio"]
        )

# Function to fetch and update all data
def fetch_all_data():
    with st.spinner("Fetching market data..."):
        try:
            # Fetch option chain
            st.session_state['option_chain'] = App.fetch_option_chain(st.session_state['config'])
            if not st.session_state['option_chain']:
                st.error("Failed to fetch option chain data.")
                return False
            
            # Fetch spot price and indices
            st.session_state['spot_price'] = st.session_state['option_chain'][0].get("underlying_spot_price")
            if not st.session_state['spot_price']:
                st.error("Could not determine spot price.")
                return False
            
            st.session_state['vix'], st.session_state['nifty_spot'] = App.get_indices_quotes(st.session_state['config'])
            if st.session_state['vix'] is None or st.session_state['nifty_spot'] is None:
                st.error("Failed to fetch VIX or Nifty Spot data.")
                return False
            
            # Update total capital
            funds_info = App.get_user_funds_and_margin(st.session_state['config'], segment="SEC")
            if funds_info:
                st.session_state['current_available_funds'] = funds_info['available_margin']
                st.session_state['current_used_margin'] = funds_info['used_margin']
                st.session_state['config']['total_capital'] = st.session_state['current_available_funds'] + st.session_state['current_used_margin']
            else:
                st.warning("Could not fetch funds info. Using default total capital.")
            
            # Calculate metrics
            st.session_state['seller'] = App.extract_seller_metrics(st.session_state['option_chain'], st.session_state['spot_price'])
            st.session_state['full_chain_df'] = App.full_chain_table(st.session_state['option_chain'], st.session_state['spot_price'])
            st.session_state['market'] = App.market_metrics(st.session_state['option_chain'], st.session_state['config']['expiry_date'])
            st.session_state['ivp'] = App.load_ivp(st.session_state['config'], st.session_state['seller']["avg_iv"])
            st.session_state['hv_7'], st.session_state['garch_7d'], st.session_state['iv_rv_spread'] = App.calculate_volatility(st.session_state['config'], st.session_state['seller'])
            st.session_state['iv_skew_slope'] = App.calculate_iv_skew_slope(st.session_state['full_chain_df'])
            
            # Volatility regime
            st.session_state['regime_score'], st.session_state['regime'], st.session_state['regime_note'], st.session_state['regime_explanation'] = App.calculate_regime(
                atm_iv=st.session_state['seller']["avg_iv"],
                ivp=st.session_state['ivp'],
                realized_vol=st.session_state['hv_7'],
                garch_vol=st.session_state['garch_7d'],
                straddle_price=st.session_state['seller']["straddle_price"],
                spot_price=st.session_state['spot_price'],
                pcr=st.session_state['market']['pcr'],
                vix=st.session_state['vix'],
                iv_skew_slope=st.session_state['iv_skew_slope']
            )
            
            # Events
            st.session_state['event_df'] = App.load_upcoming_events(st.session_state['config'])
            
            # Strategies
            st.session_state['strategies'], st.session_state['strategy_rationale'], st.session_state['event_warning'] = App.suggest_strategy(
                regime_label=st.session_state['regime'],
                ivp=st.session_state['ivp'],
                iv_minus_rv=st.session_state['iv_rv_spread'],
                days_to_expiry=st.session_state['market']['days_to_expiry'],
                event_df=st.session_state['event_df'],
                expiry_date=st.session_state['config']['expiry_date'],
                straddle_price=st.session_state['seller']["straddle_price"],
                spot_price=st.session_state['spot_price']
            )
            
            # Strategy details (recommended)
            st.session_state['strategy_details'] = []
            func_map = {
                "Iron Fly": App.iron_fly,
                "Iron Condor": App.iron_condor,
                "Jade Lizard": App.jade_lizard,
                "Straddle": App.straddle,
                "Calendar Spread": App.calendar_spread,
                "Bull Put Spread": App.bull_put_spread,
                "Wide Strangle": App.wide_strangle,
                "ATM Strangle": App.atm_strangle
            }
            for strat in st.session_state['strategies']:
                strat_clean = strat.replace("(hedged)", "").replace("with strict stop", "").replace("short ", "").strip()
                if strat_clean in func_map:
                    detail = func_map[strat_clean](st.session_state['option_chain'], st.session_state['spot_price'], st.session_state['config'])
                    if detail:
                        detail['estimated_margin'] = App.calculate_strategy_margin(st.session_state['config'], detail["orders"])
                        st.session_state['strategy_details'].append(detail)
            
            # All possible strategies
            st.session_state['all_strategy_details'] = []
            for strat_name, func in func_map.items():
                detail = func(st.session_state['option_chain'], st.session_state['spot_price'], st.session_state['config'])
                if detail:
                    detail['estimated_margin'] = App.calculate_strategy_margin(st.session_state['config'], detail["orders"])
                    st.session_state['all_strategy_details'].append(detail)
            
            # Risk and portfolio
            trades_df = App.fetch_trade_data(st.session_state['config'], st.session_state['full_chain_df'])
            st.session_state['strategy_df'], st.session_state['portfolio_summary'] = App.evaluate_full_risk(
                trades_df, st.session_state['config'], st.session_state['regime'], st.session_state['current_available_funds']
            )
            
            st.session_state['data_fetched'] = True
            st.success("Data fetched successfully!")
            return True
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return False

# Trigger data fetch if authenticated and not yet fetched
if st.session_state['authenticated'] and not st.session_state['data_fetched']:
    fetch_all_data()

# Tabs
if st.session_state['authenticated']:
    tabs = st.tabs(["Dashboard", "Chain Analysis", "Strategy Details", "Order Placement", "Order Book", "Portfolio"])
    
    # Dashboard Tab
    with tabs[0]:
        st.header("📊 Dashboard")
        if st.button("Refresh Data", key="refresh_dashboard"):
            st.session_state['data_fetched'] = False
            fetch_all_data()
            st.experimental_rerun()
        
        if st.session_state['data_fetched']:
            col1, col2, col3 = st.columns(3)
            
            # Market Metrics
            with col1:
                st.markdown("<div class='metric-card'><div class='metric-title'>Spot Price</div><div class='metric-value'>₹{:.0f}</div></div>".format(st.session_state['spot_price']), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>ATM Strike</div><div class='metric-value'>₹{:.0f}</div></div>".format(st.session_state['seller']['strike']), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>Straddle Price</div><div class='metric-value'>₹{:.2f}</div></div>".format(st.session_state['seller']['straddle_price']), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>Breakeven Range</div><div class='metric-value'>{:.0f} – {:.0f}</div></div>".format(
                    st.session_state['seller']['strike'] - st.session_state['seller']['straddle_price'],
                    st.session_state['seller']['strike'] + st.session_state['seller']['straddle_price']
                ), unsafe_allow_html=True)
            
            # Volatility Metrics
            with col2:
                st.markdown("<div class='metric-card'><div class='metric-title'>ATM IV</div><div class='metric-value'>{:.2f}%</div></div>".format(st.session_state['seller']['avg_iv']), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>Realized Vol (7D)</div><div class='metric-value'>{:.2f}%</div></div>".format(st.session_state['hv_7']), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>GARCH Vol (7D)</div><div class='metric-value'>{:.2f}%</div></div>".format(st.session_state['garch_7d']), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>IV - RV Spread</div><div class='metric-value'>{:+.2f}%</div></div>".format(st.session_state['iv_rv_spread']), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>IV Percentile (IVP)</div><div class='metric-value'>{:.2f}%</div></div>".format(st.session_state['ivp']), unsafe_allow_html=True)
            
            # Greeks and Market
            with col3:
                st.markdown("<div class='metric-card'><div class='metric-title'>Theta (Total)</div><div class='metric-value'>₹{:.2f}</div></div>".format(st.session_state['seller']['theta']), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>Vega (IV Risk)</div><div class='metric-value'>₹{:.2f}</div></div>".format(st.session_state['seller']['vega']), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>Days to Expiry</div><div class='metric-value'>{} days</div></div>".format(st.session_state['market']['days_to_expiry']), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>PCR</div><div class='metric-value'>{:.2f}</div></div>".format(st.session_state['market']['pcr']), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>Max Pain</div><div class='metric-value'>₹{:.0f}</div></div>".format(st.session_state['market']['max_pain']), unsafe_allow_html=True)
            
            # Volatility Regime
            st.subheader("🧠 Volatility Regime")
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Regime</div><div class='metric-value'>{st.session_state['regime']} (Score: {st.session_state['regime_score']:.2f})</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Note</div><div class='metric-value'>{st.session_state['regime_note']}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Details</div><div class='metric-value'>{st.session_state['regime_explanation']}</div></div>", unsafe_allow_html=True)
            
            # Volatility Comparison Plot
            st.subheader("📊 Volatility Comparison")
            fig_vol = go.Figure(data=[
                go.Bar(
                    x=['ATM IV', 'Realized Vol (7D)', 'GARCH Vol (7D)'],
                    y=[st.session_state['seller']['avg_iv'], st.session_state['hv_7'], st.session_state['garch_7d']],
                    marker_color=['#1f77b4', '#2ca02c', '#d62728']
                )
            ])
            fig_vol.update_layout(
                title="Volatility Comparison: IV vs RV vs GARCH",
                yaxis_title="Annualized Volatility (%)",
                template="plotly_dark",
                showlegend=False
            )
            for i, v in enumerate([st.session_state['seller']['avg_iv'], st.session_state['hv_7'], st.session_state['garch_7d']]):
                fig_vol.add_annotation(
                    x=i, y=v, text=f"{v:.2f}%", showarrow=False, yshift=10
                )
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Chain Analysis Plot
            st.subheader("📈 Chain Analysis")
            if not st.session_state['full_chain_df'].empty:
                fig_chain = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("IV Skew Across Strikes", "Total Theta Across Strikes", "Straddle Price Across Strikes", "Total OI Across Strikes"),
                    specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "bar"}]]
                )
                
                # IV Skew
                fig_chain.add_trace(
                    go.Scatter(x=st.session_state['full_chain_df']['Strike'], y=st.session_state['full_chain_df']['IV Skew'], mode='lines+markers', name='IV Skew', line=dict(color='purple')),
                    row=1, col=1
                )
                fig_chain.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
                
                # Total Theta
                fig_chain.add_trace(
                    go.Scatter(x=st.session_state['full_chain_df']['Strike'], y=st.session_state['full_chain_df']['Total Theta'], mode='lines+markers', name='Total Theta', line=dict(color='green')),
                    row=1, col=2
                )
                
                # Straddle Price
                fig_chain.add_trace(
                    go.Scatter(x=st.session_state['full_chain_df']['Strike'], y=st.session_state['full_chain_df']['Straddle Price'], mode='lines+markers', name='Straddle Price', line=dict(color='orange')),
                    row=2, col=1
                )
                
                # Total OI
                fig_chain.add_trace(
                    go.Bar(x=st.session_state['full_chain_df']['Strike'], y=st.session_state['full_chain_df']['Total OI'], name='Total OI', marker_color='blue'),
                    row=2, col=2
                )
                
                fig_chain.update_layout(
                    template="plotly_dark",
                    height=600,
                    showlegend=False
                )
                st.plotly_chart(fig_chain, use_container_width=True)
            
            # Payoff Diagram
            st.subheader("📊 Payoff Diagram")
            if st.session_state['strategy_details']:
                fig_payoff = go.Figure()
                strikes = [st.session_state['spot_price'] + i for i in range(-300, 301, 3)]
                
                for detail in st.session_state['strategy_details']:
                    payoffs = [0] * len(strikes)
                    for order_idx, order in enumerate(detail["orders"]):
                        instrument_key = order["instrument_key"]
                        qty = order["quantity"]
                        transaction_type = order["transaction_type"]
                        try:
                            strike = detail["strikes"][order_idx]
                        except IndexError:
                            continue
                        is_buy = (transaction_type == "BUY")
                        is_call = ("CE" in instrument_key)
                        price = detail["pricing"].get(instrument_key, {}).get("last_price", 0)
                        
                        for i, s in enumerate(strikes):
                            if is_call:
                                payoff = (s - strike) - price if is_buy else price - (s - strike)
                                payoff = max(0, payoff) if is_buy else min(0, payoff)
                            else:
                                payoff = (strike - s) - price if is_buy else price - (strike - s)
                                payoff = max(0, payoff) if is_buy else min(0, payoff)
                            payoffs[i] += payoff * (abs(qty) / st.session_state['config']["lot_size"])
                    
                    fig_payoff.add_trace(
                        go.Scatter(x=strikes, y=payoffs, mode='lines', name=detail["strategy"])
                    )
                
                fig_payoff.add_vline(x=st.session_state['spot_price'], line_dash="dash", line_color="gray", annotation_text="Spot Price")
                fig_payoff.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Zero P&L")
                fig_payoff.update_layout(
                    title="Payoff Diagram for Suggested Strategies",
                    xaxis_title="Underlying Price at Expiry",
                    yaxis_title="P&L (₹)",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_payoff, use_container_width=True)
            
            # Upcoming Events
            st.subheader("📅 Upcoming Events")
            if not st.session_state['event_df'].empty:
                st.dataframe(st.session_state['event_df'])
                if st.session_state['event_warning']:
                    st.warning(st.session_state['event_warning'])
            else:
                st.info("No upcoming events found.")
            
            # Strategies
            st.subheader("📈 Recommended Strategies")
            st.write(f"**Strategies**: {', '.join(st.session_state['strategies'])}")
            st.write(f"**Rationale**: {st.session_state['strategy_rationale']}")
    
    # Chain Analysis Tab
    with tabs[1]:
        st.header("📈 Chain Analysis")
        if st.button("Refresh Data", key="refresh_chain"):
            st.session_state['data_fetched'] = False
            fetch_all_data()
            st.experimental_rerun()
        
        if not st.session_state['full_chain_df'].empty:
            st.subheader("ATM ±300 Chain Table")
            st.dataframe(st.session_state['full_chain_df'])
            
            eff_df = st.session_state['full_chain_df'].copy()
            eff_df["Theta/Vega"] = eff_df.apply(
                lambda row: row["Total Theta"] / row["Total Vega"] if row["Total Vega"] != 0 else float('nan'), axis=1
            )
            eff_df = eff_df[["Strike", "Total Theta", "Total Vega", "Theta/Vega"]].sort_values("Theta/Vega", ascending=False).dropna()
            st.subheader("Theta/Vega Ranking")
            st.dataframe(eff_df)
        else:
            st.warning("No chain data available.")
    
    # Strategy Details Tab
    with tabs[2]:
        st.header("📊 Strategy Details")
        if st.button("Refresh Data", key="refresh_strategies"):
            st.session_state['data_fetched'] = False
            fetch_all_data()
            st.experimental_rerun()
        
        st.subheader("Recommended Strategies")
        for detail in st.session_state['strategy_details']:
            with st.expander(f"{detail['strategy']}"):
                st.write(f"**Strikes**: {detail['strikes']}")
                st.write(f"**Premium**: ₹{detail['premium']:.2f}")
                st.write(f"**Max Profit**: ₹{detail['max_profit']:.2f}")
                st.write(f"**Max Loss**: {'Unlimited' if detail['max_loss'] == float('inf') else f'₹{detail["max_loss"]:.2f}'}")
                st.write(f"**Estimated Margin**: {'N/A' if detail['estimated_margin'] is None else f'₹{detail["estimated_margin"]:.2f}'}")
        
        st.subheader("All Available Strategies")
        for detail in st.session_state['all_strategy_details']:
            with st.expander(f"{detail['strategy']}"):
                st.write(f"**Strikes**: {detail['strikes']}")
                st.write(f"**Premium**: ₹{detail['premium']:.2f}")
                st.write(f"**Max Profit**: ₹{detail['max_profit']:.2f}")
                st.write(f"**Max Loss**: {'Unlimited' if detail['max_loss'] == float('inf') else f'₹{detail["max_loss"]:.2f}'}")
                st.write(f"**Estimated Margin**: {'N/A' if detail['estimated_margin'] is None else f'₹{detail["estimated_margin"]:.2f}'}")
    
    # Order Placement Tab
    with tabs[3]:
        st.header("🚀 Order Placement")
        if st.button("Refresh Data", key="refresh_orders"):
            st.session_state['data_fetched'] = False
            fetch_all_data()
            st.experimental_rerun()
        
        st.subheader("Select Strategy")
        strategy_options = [detail['strategy'] for detail in st.session_state['all_strategy_details']]
        selected_strategy = st.selectbox("Choose a strategy:", strategy_options)
        
        if selected_strategy:
            detail = next((d for d in st.session_state['all_strategy_details'] if d['strategy'] == selected_strategy), None)
            if detail:
                st.write(f"**Premium**: ₹{detail['premium']:.2f}")
                st.write(f"**Max Profit**: ₹{detail['max_profit']:.2f}")
                st.write(f"**Max Loss**: {'Unlimited' if detail['max_loss'] == float('inf') else f'₹{detail['max_loss']:.2f}'}")
                st.write(f"**Estimated Margin**: {'N/A' if detail['estimated_margin'] is None else f'₹{detail['estimated_margin']:.2f}'}")
                
                if detail['estimated_margin'] is not None and detail['estimated_margin'] > st.session_state['current_available_funds']:
                    st.warning(f"Insufficient funds. Required: ₹{detail['estimated_margin']:.2f}, Available: ₹{st.session_state['current_available_funds']:.2f}")
                    proceed_anyway = st.checkbox("Proceed anyway?")
                else:
                    proceed_anyway = True
                
                if st.button("Place Order"):
                    if not proceed_anyway:
                        st.error("Order placement cancelled due to insufficient funds.")
                    else:
                        with st.spinner("Placing order..."):
                            order_ids = []
                            failed_orders = []
                            
                            buy_legs = [order for order in detail["orders"] if order["transaction_type"] == "BUY"]
                            sell_legs = [order for order in detail["orders"] if order["transaction_type"] == "SELL"]
                            
                            # Place BUY legs
                            for order in buy_legs:
                                order_id = App.place_order(
                                    st.session_state['config'],
                                    order["instrument_key"],
                                    order["quantity"],
                                    order["transaction_type"]
                                )
                                if order_id:
                                    order_ids.append(order_id)
                                    st.write(f"Buy Order placed: {order_id} for {order['instrument_key']} (Quantity: {order['quantity']})")
                                else:
                                    failed_orders.append(order["instrument_key"])
                            
                            if buy_legs and not order_ids:
                                st.error(f"All BUY orders failed for {detail['strategy']}.")
                            else:
                                if buy_legs:
                                    st.write("Waiting for BUY legs to fill...")
                                    all_buy_orders_filled = False
                                    max_wait_time = 120
                                    start_time = time.time()
                                    
                                    while not all_buy_orders_filled and (time.time() - start_time) < max_wait_time:
                                        current_orders = App.get_order_book(st.session_state['config'])
                                        all_buy_orders_filled = True
                                        for buy_order_id in order_ids:
                                            found_order = next((o for o in current_orders if o.get('order_id') == buy_order_id), None)
                                            if not found_order or found_order.get('status') not in ["COMPLETE", "FILLED"]:
                                                all_buy_orders_filled = False
                                                if found_order and found_order.get('status') in ["CANCELLED", "REJECTED"]:
                                                    st.error(f"Buy order {buy_order_id} {found_order.get('status')}: {found_order.get('status_message')}")
                                                    break
                                        if not all_buy_orders_filled:
                                            time.sleep(5)
                                        else:
                                            st.success("All BUY legs filled!")
                                            break
                                    
                                    if not all_buy_orders_filled:
                                        st.error("Timeout or some BUY orders not filled. Aborting SELL legs.")
                                            return
                                    
                                    # Refresh funds
                                    funds_info = App.get_user_funds_and_margin(st.session_state['config'], segment="SEC")
                                    current_available_funds_after_buy = funds_info['available_margin'] if funds_info else 0
                                    if current_available_funds_after_buy < 0:
                                        st.warning("Low funds after BUY legs. SELL legs may fail.")
                                        if not st.checkbox("Proceed with SELL legs?"):
                                            st.error("Order placement cancelled.")
                                            return
                            
                            # Place SELL legs
                            for order in sell_legs:
                                order_id = App.place_order(
                                    st.session_state['config'],
                                    order["instrument_key"],
                                    order["quantity"],
                                    order["transaction_type"]
                                )
                                if order_id:
                                    order_ids.append(order_id)
                                    st.write(f"Sell Order placed: {order_id} for {order['instrument_key']} (Quantity: {order['quantity']})")
                                else:
                                    failed_orders.append(order["instrument_key"])
                            
                            if order_ids:
                                st.success(f"Placed {len(order_ids)} orders for {detail['strategy']}.")
                                if failed_orders:
                                    st.warning(f"{len(failed_orders)} orders failed: {failed_orders}")
                            else:
                                st.error(f"All orders failed for {detail['strategy']}.")
    
    # Order Book Tab
    with tabs[4]:
        st.header("📋 Order Book")
        if st.button("Refresh Order Book"):
            with st.spinner("Fetching order book..."):
                orders = App.get_order_book(st.session_state['config'])
                if orders:
                    order_data = []
                    for order in orders:
                        order_data.append({
                            "Order ID": order.get('order_id', 'N/A'),
                            "Instrument": order.get('trading_symbol', 'N/A'),
                            "Type": order.get('transaction_type', 'N/A'),
                            "Quantity": order.get('quantity', 0),
                            "Filled": order.get('filled_quantity', 0),
                            "Status": order.get('status', 'N/A'),
                            "Message": order.get('status_message', '')
                        })
                    st.dataframe(pd.DataFrame(order_data))
                else:
                    st.info("No orders found or error fetching order book.")
    
    # Portfolio Tab
    with tabs[5]:
        st.header("📦 Portfolio")
        if st.button("Refresh Data", key="refresh_portfolio"):
            st.session_state['data_fetched'] = False
            fetch_all_data()
            st.experimental_rerun()
        
        if st.session_state['portfolio_summary']:
            st.subheader("Portfolio Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='metric-card'><div class='metric-title'>Total Capital</div><div class='metric-value'>₹{:.2f}</div></div>".format(st.session_state['portfolio_summary'].get('Total Capital', 0)), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>Capital Deployed</div><div class='metric-value'>₹{:.2f}</div></div>".format(st.session_state['portfolio_summary'].get('Capital Deployed', 0)), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>Exposure %</div><div class='metric-value'>{:.2f}%</div></div>".format(st.session_state['portfolio_summary'].get('Exposure %', 0)), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>Available Funds</div><div class='metric-value'>₹{:.2f}</div></div>".format(st.session_state['portfolio_summary'].get('Available Funds', 0)), unsafe_allow_html=True)
            with col2:
                st.markdown("<div class='metric-card'><div class='metric-title'>Risk on Table</div><div class='metric-value'>₹{:.2f}</div></div>".format(st.session_state['portfolio_summary'].get('Risk on Table', 0)), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>Realized P&L</div><div class='metric-value'>₹{:.2f}</div></div>".format(st.session_state['portfolio_summary'].get('Realized P&L', 0)), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>Drawdown ₹</div><div class='metric-value'>₹{:.2f}</div></div>".format(st.session_state['portfolio_summary'].get('Drawdown ₹', 0)), unsafe_allow_html=True)
                st.markdown("<div class='metric-card'><div class='metric-title'>Portfolio Vega</div><div class='metric-value'>{:.2f}</div></div>".format(st.session_state['portfolio_summary'].get('Portfolio Vega', 0)), unsafe_allow_html=True)
            
            st.subheader("Risk Summary")
            if not st.session_state['strategy_df'].empty:
                st.dataframe(st.session_state['strategy_df'])
            else:
                st.info("No risk data available.")
            
            if st.session_state['portfolio_summary'].get('Flags'):
                st.subheader("🚨 Risks/Warnings")
                for flag in st.session_state['portfolio_summary']['Flags']:
                    st.warning(flag)
            else:
                st.success("✅ No risk violations.")
else:
    st.warning("Please log in to access the dashboard.")
