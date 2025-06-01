import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime
import App  # Import your backend code

# Page configuration
st.set_page_config(
    page_title="VolGuard - Your Trading Copilot",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS for dark theme and trading app aesthetics
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
        box-shadow: 1px 1px 5px rgba(0,0,0,0.3);
    }
    .metric-title {
        font-size: 14px;
        color: #cccccc;
    }
    .metric-value {
        font-size: 18px;
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
    .st-expander {
        background-color: #2c2c2c;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    defaults = {
        'access_token': None,
        'authenticated': False,
        'config': {},
        'data_fetched': False,
        'option_chain': [],
        'spot_price': 0.0,
        'vix': 0.0,
        'nifty_spot': 0.0,
        'current_available_funds': 0.0,
        'current_used_margin': 0.0,
        'seller': {},
        'full_chain_df': pd.DataFrame(),
        'market': {'days_to_expiry': 0, 'pcr': 0.0, 'max_pain': 0.0},
        'ivp': 0.0,
        'hv_7': 0.0,
        'garch_7d': 0.0,
        'iv_rv_spread': 0.0,
        'iv_skew_slope': 0.0,
        'regime_score': 0.0,
        'regime': "",
        'regime_note': "",
        'regime_explanation': "",
        'event_df': pd.DataFrame(),
        'strategies': [],
        'strategy_rationale': "",
        'event_warning': None,
        'strategy_df': pd.DataFrame(),
        'portfolio_summary': {},
        'strategy_details': [],
        'all_strategy_details': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #3c8dbc;'>VolGuard</h2>", unsafe_allow_html=True)
    
    if not st.session_state['authenticated']:
        st.subheader("Login")
        access_token = st.text_input("Upstox API Access Token", type="password")
        if st.button("Login"):
            with st.spinner("Validating..."):
                try:
                    config = App.get_config()
                    config['access_token'] = access_token
                    config['headers']['Authorization'] = f"Bearer {access_token}"
                    url = f"{config['base_url']}/user/get-funds-and-margin"
                    res = requests.get(url, headers=config['headers'], params={"segment": "SEC"})
                    if res.status_code == 200:
                        st.session_state['access_token'] = access_token
                        st.session_state['authenticated'] = True
                        st.session_state['config'] = config
                        st.session_state['data_fetched'] = False
                        st.success("Login successful!")
                    else:
                        st.error(f"Invalid token: {res.status_code} - {res.text}")
                except Exception as e:
                    st.error(f"Error validating token: {e}")
    else:
        st.subheader("Account")
        st.write(f"Status: ✅ Connected")
        st.write(f"Token: {st.session_state['access_token'][:4]}...{st.session_state['access_token'][-4:]}")
        if st.button("Logout"):
            for key in st.session_state.keys():
                st.session_state[key] = None
            initialize_session_state()
            st.success("Logged out.")
            st.rerun()

    if st.session_state['authenticated']:
        st.subheader("Navigation")
        st.radio(
            "Go to:",
            ["Dashboard", "Chain Analysis", "Strategy Details", "Order Placement", "Order Book", "Portfolio"],
            key="nav_radio"
        )

# Data fetch function
def fetch_all_data():
    try:
        with st.spinner("Fetching data..."):
            # Option chain
            st.session_state['option_chain'] = App.fetch_option_chain(st.session_state['config']) or []
            if not st.session_state['option_chain']:
                st.error("Failed to fetch option chain.")
                return False
            
            # Spot price
            st.session_state['spot_price'] = st.session_state['option_chain'][0].get("underlying_spot_price", 0.0)
            if not st.session_state['spot_price']:
                st.error("Could not determine spot price.")
                return False
            
            # Indices
            vix, nifty_spot = App.get_indices_quotes(st.session_state['config'])
            st.session_state['vix'] = vix or 0.0
            st.session_state['nifty_spot'] = nifty_spot or 0.0
            if st.session_state['vix'] == 0 or st.session_state['nifty_spot'] == 0:
                st.error("Failed to fetch VIX or Nifty Spot.")
                return False
            
            # Funds
            funds_info = App.get_user_funds_and_margin(st.session_state['config'], segment="SEC") or {}
            st.session_state['current_available_funds'] = funds_info.get('available_margin', 0.0)
            st.session_state['current_used_margin'] = funds_info.get('used_margin', 0.0)
            st.session_state['config']['total_capital'] = st.session_state['current_available_funds'] + st.session_state['current_used_margin']
            
            # Metrics
            st.session_state['seller'] = App.extract_seller_metrics(st.session_state['option_chain'], st.session_state['spot_price']) or {}
            st.session_state['full_chain_df'] = App.full_chain_table(st.session_state['option_chain'], st.session_state['spot_price']) or pd.DataFrame()
            st.session_state['market'] = App.market_metrics(st.session_state['option_chain'], st.session_state['config']['expiry_date']) or {'days_to_expiry': 0, 'pcr': 0.0, 'max_pain': 0.0}
            st.session_state['ivp'] = App.load_ivp(st.session_state['config'], st.session_state['seller'].get("avg_iv", 0.0)) or 0.0
            st.session_state['hv_7'], st.session_state['garch_7d'], st.session_state['iv_rv_spread'] = App.calculate_volatility(st.session_state['config'], st.session_state['seller']) or (0.0, 0.0, 0.0)
            st.session_state['iv_skew_slope'] = App.calculate_iv_skew_slope(st.session_state['full_chain_df']) or 0.0
            
            # Regime
            st.session_state['regime_score'], st.session_state['regime'], st.session_state['regime_note'], st.session_state['regime_explanation'] = App.calculate_regime(
                atm_iv=st.session_state['seller'].get("avg_iv", 0.0),
                ivp=st.session_state['ivp'],
                realized_vol=st.session_state['hv_7'],
                garch_vol=st.session_state['garch_7d'],
                straddle_price=st.session_state['seller'].get("straddle_price", 0.0),
                spot_price=st.session_state['spot_price'],
                pcr=st.session_state['market'].get('pcr', 0.0),
                vix=st.session_state['vix'],
                iv_skew_slope=st.session_state['iv_skew_slope']
            ) or (0.0, "", "", "")
            
            # Events
            st.session_state['event_df'] = App.load_upcoming_events(st.session_state['config']) or pd.DataFrame()
            
            # Strategies
            st.session_state['strategies'], st.session_state['strategy_rationale'], st.session_state['event_warning'] = App.suggest_strategy(
                regime_label=st.session_state['regime'],
                ivp=st.session_state['ivp'],
                iv_minus_rv=st.session_state['iv_rv_spread'],
                days_to_expiry=st.session_state['market'].get('days_to_expiry', 0),
                event_df=st.session_state['event_df'],
                expiry_date=st.session_state['config']['expiry_date'],
                straddle_price=st.session_state['seller'].get("straddle_price", 0.0),
                spot_price=st.session_state['spot_price']
            ) or ([], "", None)
            
            # Strategy details
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
            st.session_state['strategy_details'] = []
            for strat in st.session_state['strategies']:
                strat_clean = strat.replace("(hedged)", "").replace("with strict stop", "").replace("short ", "").strip()
                if strat_clean in func_map:
                    detail = func_map[strat_clean](st.session_state['option_chain'], st.session_state['spot_price'], st.session_state['config'])
                    if detail:
                        detail['estimated_margin'] = App.calculate_strategy_margin(st.session_state['config'], detail["orders"]) or None
                        st.session_state['strategy_details'].append(detail)
            
            # All strategies
            st.session_state['all_strategy_details'] = []
            for strat_name, func in func_map.items():
                detail = func(st.session_state['option_chain'], st.session_state['spot_price'], st.session_state['config'])
                if detail:
                    detail['estimated_margin'] = App.calculate_strategy_margin(st.session_state['config'], detail["orders"]) or None
                    st.session_state['all_strategy_details'].append(detail)
            
            # Risk and portfolio
            trades_df = App.fetch_trade_data(st.session_state['config'], st.session_state['full_chain_df']) or pd.DataFrame()
            st.session_state['strategy_df'], st.session_state['portfolio_summary'] = App.evaluate_full_risk(
                trades_df, st.session_state['config'], st.session_state['regime'], st.session_state['current_available_funds']
            ) or (pd.DataFrame(), {})
            
            st.session_state['data_fetched'] = True
            st.success("Data fetched!")
            return True
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return False

# Fetch data if authenticated
if st.session_state['authenticated'] and not st.session_state['data_fetched']:
    fetch_all_data()

# Tabs
if st.session_state['authenticated']:
    tabs = st.tabs(["Dashboard", "Chain Analysis", "Strategy Details", "Order Placement", "Order Book", "Portfolio"])
    
    # Dashboard
    with tabs[0]:
        st.header("📊 Dashboard")
        if st.button("Refresh", key="refresh_dashboard"):
            st.session_state['data_fetched'] = False
            fetch_all_data()
            st.rerun()
        
        if st.session_state['data_fetched']:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Spot Price</div><div class='metric-value'>₹{st.session_state['spot_price']:.0f}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>ATM Strike</div><div class='metric-value'>₹{st.session_state['seller'].get('strike', 0):.0f}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Straddle Price</div><div class='metric-value'>₹{st.session_state['seller'].get('straddle_price', 0):.2f}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Breakeven Range</div><div class='metric-value'>{st.session_state['seller'].get('strike', 0) - st.session_state['seller'].get('straddle_price', 0):.0f} – {st.session_state['seller'].get('strike', 0) + st.session_state['seller'].get('straddle_price', 0):.0f}</div></div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<div class='metric-card'><div class='metric-title'>ATM IV</div><div class='metric-value'>{st.session_state['seller'].get('avg_iv', 0):.2f}%</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Realized Vol (7D)</div><div class='metric-value'>{st.session_state['hv_7']:.2f}%</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>GARCH Vol (7D)</div><div class='metric-value'>{st.session_state['garch_7d']:.2f}%</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>IV - RV Spread</div><div class='metric-value'>{st.session_state['iv_rv_spread']:+.2f}%</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>IV Percentile</div><div class='metric-value'>{st.session_state['ivp']:.2f}%</div></div>", unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Theta (Total)</div><div class='metric-value'>₹{st.session_state['seller'].get('theta', 0):.2f}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Vega (IV Risk)</div><div class='metric-value'>₹{st.session_state['seller'].get('vega', 0):.2f}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Days to Expiry</div><div class='metric-value'>{st.session_state['market']['days_to_expiry']} days</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>PCR</div><div class='metric-value'>{st.session_state['market']['pcr']:.2f}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Max Pain</div><div class='metric-value'>₹{st.session_state['market']['max_pain']:.0f}</div></div>", unsafe_allow_html=True)
            
            st.subheader("🧠 Volatility Regime")
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Regime</div><div class='metric-value'>{st.session_state['regime']} (Score: {st.session_state['regime_score']:.2f})</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Note</div><div class='metric-value'>{st.session_state['regime_note']}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-title'>Details</div><div class='metric-value'>{st.session_state['regime_explanation']}</div></div>", unsafe_allow_html=True)
            
            st.subheader("📊 Volatility Comparison")
            fig_vol = go.Figure(data=[
                go.Bar(
                    x=['ATM IV', 'Realized Vol (7D)', 'GARCH Vol (7D)'],
                    y=[st.session_state['seller'].get('avg_iv', 0), st.session_state['hv_7'], st.session_state['garch_7d']],
                    marker_color=['#1f77b4', '#2ca02c', '#d62728']
                )
            ])
            fig_vol.update_layout(
                title="Volatility Comparison",
                yaxis_title="Annualized Volatility (%)",
                template="plotly_dark",
                showlegend=False
            )
            for i, v in enumerate([st.session_state['seller'].get('avg_iv', 0), st.session_state['hv_7'], st.session_state['garch_7d']]):
                fig_vol.add_annotation(x=i, y=v, text=f"{v:.2f}%", showarrow=False, yshift=10)
            st.plotly_chart(fig_vol, use_container_width=True)
            
            st.subheader("📈 Chain Analysis")
            if not st.session_state['full_chain_df'].empty:
                fig_chain = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("IV Skew", "Total Theta", "Straddle Price", "Total OI"),
                    specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "bar"}]]
                )
                fig_chain.add_trace(go.Scatter(x=st.session_state['full_chain_df']['Strike'], y=st.session_state['full_chain_df']['IV Skew'], mode='lines+markers', name='IV Skew', line=dict(color='purple')), row=1, col=1)
                fig_chain.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
                fig_chain.add_trace(go.Scatter(x=st.session_state['full_chain_df']['Strike'], y=st.session_state['full_chain_df']['Total Theta'], mode='lines+markers', name='Total Theta', line=dict(color='green')), row=1, col=2)
                fig_chain.add_trace(go.Scatter(x=st.session_state['full_chain_df']['Strike'], y=st.session_state['full_chain_df']['Straddle Price'], mode='lines+markers', name='Straddle Price', line=dict(color='orange')), row=2, col=1)
                fig_chain.add_trace(go.Bar(x=st.session_state['full_chain_df']['Strike'], y=st.session_state['full_chain_df']['Total OI'], name='Total OI', marker_color='blue'), row=2, col=2)
                fig_chain.update_layout(template="plotly_dark", height=600, showlegend=False)
                st.plotly_chart(fig_chain, use_container_width=True)
            
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
                    fig_payoff.add_trace(go.Scatter(x=strikes, y=payoffs, mode='lines', name=detail["strategy"]))
                fig_payoff.add_vline(x=st.session_state['spot_price'], line_dash="dash", line_color="gray", annotation_text="Spot Price")
                fig_payoff.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Zero P&L")
                fig_payoff.update_layout(title="Payoff Diagram", xaxis_title="Price at Expiry", yaxis_title="P&L (₹)", template="plotly_dark")
                st.plotly_chart(fig_payoff, use_container_width=True)
            
            st.subheader("📅 Upcoming Events")
            if not st.session_state['event_df'].empty:
                st.dataframe(st.session_state['event_df'])
                if st.session_state['event_warning']:
                    st.warning(st.session_state['event_warning'])
            else:
                st.info("No events found.")
            
            st.subheader("📈 Recommended Strategies")
            st.write(f"**Strategies**: {', '.join(st.session_state['strategies'])}")
            st.write(f"**Rationale**: {st.session_state['strategy_rationale']}")
    
    # Chain Analysis
    with tabs[1]:
        st.header("📈 Chain Analysis")
        if st.button("Refresh", key="refresh_chain"):
            st.session_state['data_fetched'] = False
            fetch_all_data()
            st.rerun()
        
        if not st.session_state['full_chain_df'].empty:
            st.subheader("ATM ±300 Chain")
            st.dataframe(st.session_state['full_chain_df'])
            eff_df = st.session_state['full_chain_df'].copy()
            eff_df["Theta/Vega"] = eff_df.apply(lambda row: row["Total Theta"] / row["Total Vega"] if row["Total Vega"] != 0 else 0.0, axis=1)
            eff_df = eff_df[["Strike", "Total Theta", "Total Vega", "Theta/Vega"]].sort_values("Theta/Vega", ascending=False).dropna()
            st.subheader("Theta/Vega Ranking")
            st.dataframe(eff_df)
        else:
            st.warning("No chain data.")
    
    # Strategy Details
    with tabs[2]:
        st.header("Strategies")
        if st.button("Refresh", key="refresh_strategies"):
            st.session_state['data_fetched'] = False
            fetch_all_data()
            st.rerun()
        
        st.subheader("Recommended Strategies")
        for detail in st.session_state['strategy_details']:
            with st.expander(f"{detail['strategy']}"):
                st.write(f"**Strikes**: {detail['strikes']}")
                st.write(f"**Premium**: ₹{detail['premium']:.2f}")
                st.write(f"**Max Profit**: ₹{detail['max_profit']:.2f}")
                st.write(f"**Max Loss**: {'Unlimited' if detail['max_loss'] == float('inf') else f'₹{detail['max_loss']:.2f}'}")
                st.write(f"**Margin**: {'N/A' if not detail['estimated_margin'] else f'₹{detail['estimated_margin']:.2f}'}")
        
        st.subheader("All Strategies")
        for detail in st.session_state['all_strategy_details']:
            with st.expander(f"{detail['strategy']}"):
                st.write(f"**Strikes**: {detail['strikes']}")
                st.write(f"**Premium**: ₹{detail['premium']:.2f}")
                st.write(f"**Max Profit**: ₹{detail['max_profit']:.2f}")
                st.write(f"**Max Loss**: {'Unlimited' if detail['max_loss'] == float('inf') else f'₹{detail['max_loss']:.2f}'}")
                st.write(f"**Margin**: {'N/A' if not detail['estimated_margin'] else f'₹{detail['estimated_margin']:.2f}'}")
    
    # Order Placement
    with tabs[3]:
        st.header("Place Order")
        if st.button("Refresh", key="refresh_orders"):
            st.session_state['data_fetched'] = False
            fetch_all_data()
            st.rerun()
        
        st.subheader("Select Strategy")
        strategy_options = [detail['strategy'] for detail in st.session_state['all_strategy_details']]
        selected_strategy = st.selectbox("Strategy:", strategy_options, key="strategy_select")
        
        if selected_strategy:
            detail = next((d for d in st.session_state['all_strategy_details'] if d['strategy'] == selected_strategy), None)
            if detail:
                st.write(f"**Strikes**: {detail['strikes']}")
                st.write(f"**Premium**: ₹{detail['premium']:.2f}")
                st.write(f"**Max Profit**: ₹{detail['max_profit']:.2f}")
                st.write(f"**Max Loss**: {'Unlimited' if detail['max_loss'] == float('inf') else f'₹{detail['max_loss']:.2f}'}")
                st.write(f"**Margin**: {'N/A' if not detail['estimated_margin'] else f'₹{detail['estimated_margin']:.2f}'}")
                
                proceed = True
                if detail['estimated_margin'] and detail['estimated_margin'] > st.session_state['current_available_funds']:
                    st.warning(f"Insufficient funds: Required ₹{detail['estimated_margin']:.2f}, Available: ₹{st.session_state['current_available_funds']:.2f}")
                    proceed = st.checkbox("Proceed anyway?", key="proceed_check")
                
                if st.button("Place Order", key="place_order"):
                    if not proceed:
                        st.error("Order cancelled due to insufficient funds.")
                    else:
                        with st.spinner("Placing order..."):
                            order_ids = []
                            failed_orders = []
                            buy_legs = [order for order in detail["orders"] if order["transaction_type"] == "BUY"]
                            for order in buy_legs:
                                order_id = App.place_order(
                                    st.session_state['config'],
                                    order["instrument_key"],
                                    order["quantity"],
                                    order["transaction_type"]
                                )
                                if order_id:
                                    order_ids.append(order_id)
                                    st.write(f"Buy Order: {order_id} for {order['instrument_key']} (Qty: {order['quantity']})")
                                else:
                                    failed_orders.append(order["instrument_key"])
                            if buy_legs and not order_ids:
                                st.error(f"All BUY orders failed for {detail['strategy']}.")
                            else:
                                if buy_legs:
                                    st.write("Waiting for BUY legs to fill...")
                                    all_filled = False
                                    start_time = time.time()
                                    while not all_filled and (time.time() - start_time) < 120:
                                        orders = App.get_order_book(st.session_state['config']) or []
                                        all_filled = True
                                        for oid in order_ids:
                                            order = next((o for o in orders if o.get('order_id') == oid), None)
                                            if not order or order.get('status') not in ["COMPLETE", "FILLED"]:
                                                all_filled = False
                                                if order and order.get('status') in ["CANCELLED", "REJECTED"]:
                                                    st.error(f"Buy order {oid} {order.get('status')}: {order.get('status_message', '')}")
                                                    all_filled = False
                                                    break
                                        time.sleep(1)
                                    if not all_filled:
                                        st.error("Timeout waiting for BUY legs. Aborting.")
                                    else:
                                        st.success("All BUY legs filled!")
                                        funds_info = App.get_user_funds_and_margin(st.session_state['config'], segment="SEC") or {}
                                        current_funds = funds_info.get('available_margin', 0.0)
                                        if current_funds < 0:
                                            st.warning("Low funds after BUY legs.")
                                            if not st.checkbox("Proceed with SELL legs?", key="sell_proceed"):
                                                st.error("Order cancelled.")
                                                sell_legs = []
                            
                            sell_legs = [order for order in detail["orders"] if order["transaction_type"] == "SELL"]
                            for order in sell_legs:
                                order_id = App.place_order(
                                    st.session_state['config'],
                                    order["instrument_key"],
                                    order["quantity"],
                                    order["transaction_type"]
                                )
                                if order_id:
                                    order_ids.append(order_id)
                                    st.write(f"Sell Order: {order_id} for {order['instrument_key']} (Qty: {order['quantity']})")
                                else:
                                    failed_orders.append(order["instrument_key"])
                            
                            if order_ids:
                                st.success(f"Placed {len(order_ids)} orders for {detail['strategy']}.")
                                if failed_orders:
                                    st.warning(f"{len(failed_orders)} orders failed: {failed_orders}")
                            else:
                                st.error(f"All orders failed for {detail['strategy']}.")
    
    # Order Book
    with tabs[4]:
        st.header("Order Book")
        if st.button("Refresh", key="refresh_order_book"):
            with st.spinner("Fetching orders..."):
                orders = App.get_order_book(st.session_state['config']) or []
                if orders:
                    order_data = [{
                        "Order ID": o.get('order_id', 'N/A'),
                        "Instrument": o.get('trading_symbol', 'N/A'),
                        "Type": o.get('transaction_type', 'N/A'),
                        "Quantity": o.get('quantity', 0),
                        "Status": o.get('status', 'N/A'),
                        "Message": o.get('status_message', '')
                    } for o in orders]
                    st.dataframe(pd.DataFrame(order_data))
                else:
                    st.warning("No orders found.")
    
    # Portfolio
    with tabs[5]:
        st.header("Portfolio")
        if st.button("Refresh", key="refresh_portfolio"):
            st.session_state['data_fetched'] = False
            fetch_all_data()
            st.rerun()
        
        if st.session_state['portfolio_summary']:
            st.subheader("Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Total Capital</div><div class='metric-value'>₹{st.session_state['portfolio_summary'].get('Total Capital', 0):.2f}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Capital Deployed</div><div class='metric-value'>₹{st.session_state['portfolio_summary'].get('Capital Deployed', 0):.2f}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Exposure %</div><div class='metric-value'>{st.session_state['portfolio_summary'].get('Exposure %', 0):.2f}%</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Available Funds</div><div class='metric-value'>₹{st.session_state['portfolio_summary'].get('Available Funds', 0):.2f}</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Risk on Table</div><div class='metric-value'>₹{st.session_state['portfolio_summary'].get('Risk on Table', 0):.2f}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Realized P&L</div><div class='metric-value'>₹{st.session_state['portfolio_summary'].get('Realized P&L', 0):.2f}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Drawdown</div><div class='metric-value'>₹{st.session_state['portfolio_summary'].get('Drawdown ₹', 0):.2f}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-card'><div class='metric-title'>Portfolio Vega</div><div class='metric-value'>{st.session_state['portfolio_summary'].get('Portfolio Vega', 0):.2f}</div></div>", unsafe_allow_html=True)
            
            st.subheader("Risk Summary")
            if not st.session_state['strategy_df'].empty:
                st.dataframe(st.session_state['strategy_df'])
            else:
                st.warning("No risk data.")
            
            if st.session_state['portfolio_summary'].get('Flags'):
                st.subheader("🚨 Warnings")
                for flag in st.session_state['portfolio_summary']['Flags']:
                    st.warning(flag)
            else:
                st.success("✅ No risks.")
else:
    st.warning("Please log in.")
