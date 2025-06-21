import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from streamlit_option_menu import option_menu

# --- UTILS ---
# Assuming dcf and other dependencies will be available in the path
# from fundamentals.dcf import discounted_cash_flow

# --- MOCK DATA & FUNCTIONS ---

def get_mock_data():
    """Returns mock data for charts and tables."""
    return {
        "price_data": pd.DataFrame(
            np.random.randn(90, 1).cumsum() + 130,
            columns=['price'],
            index=pd.to_datetime(pd.date_range('2023-01-01', periods=90, freq='D'))
        ),
        "watchlist": {
            "AAPL": "+1.24%", "GOOGL": "-0.54%", "BTC-USD": "+2.11%", "TSLA": "-1.03%"
        },
        "headlines": [
            ("Fed hints at rate pause — markets surge.", "Reuters"),
            ("AI chip demand lifts semiconductor sector.", "Bloomberg"),
            ("Crypto ETF chatter reignites bull run.", "CoinDesk"),
        ],
        "executives": [
            ("Sundar Pichai", "CEO / Director"),
            ("Ruth Porat", "CFO"),
            ("Prabhakar Raghavan", "SVP, Search"),
        ],
        "trades": pd.DataFrame({
            'Symbol': ['AAPL', 'GOOGL', 'TSLA', 'MSFT'], 'Type': ['Call', 'Put', 'Call', 'Put'],
            'Strike': [150, 2800, 700, 300], 'Premium': [2.5, 15.2, 8.1, 5.5], 'Quantity': [10, 5, 12, 8]
        })
    }

def mock_dcf(symbol):
    """Mock DCF function."""
    return f"""
    ### DCF Valuation for {symbol}
    - **Estimated Value:** ${np.random.randint(150, 250)}
    - **Growth Rate:** {np.random.uniform(0.04, 0.06):.2%}
    - **Margin of Safety:** 20%
    """

# --- STYLING & CUSTOM COMPONENTS ---

def load_css():
    """Loads custom CSS for styling the dashboard."""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
            
            /* General Styles */
            body { font-family: 'Inter', sans-serif; }
            .stApp { background-color: #0A0A0A; color: #EAEAEA; }
            .main .block-container { padding: 0; }
            
            /* Custom Header */
            .app-header {
                position: fixed; top: 0; left: 0; right: 0; z-index: 999;
                display: flex; align-items: center; justify-content: space-between;
                padding: 0.5rem 2rem;
                background-color: rgba(10, 10, 10, 0.8);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid #1A1A1A;
            }
            .app-header .title { display: flex; align-items: center; gap: 0.75rem; }
            .app-header .title h1 { font-size: 1.25rem; font-weight: 600; margin: 0; }
            .app-header .search-bar input {
                background-color: #1A1A1A; border: 1px solid #262626; border-radius: 6px;
                color: #EAEAEA; padding: 0.5rem 0.75rem; width: 300px;
            }
            
            /* Main Content Padding */
            .stApp > div:first-child > div:first-child > div:first-child { padding-top: 5rem; }

            /* Card Component */
            .card {
                background-color: #141414; border: 1px solid #262626;
                border-radius: 8px; padding: 1.5rem; height: 100%;
            }
            .card h3 {
                font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem;
                display: flex; align-items: center; gap: 0.5rem;
            }

            /* Custom Tables */
            .custom-table { width: 100%; font-size: 0.875rem; }
            .custom-table tr td:last-child { text-align: right; color: #8C8C8C; }
            
            /* Utility Classes */
            .text-green { color: #34D399; }
            .text-red { color: #F87171; }
            .font-mono { font-family: 'IBM Plex Mono', monospace; }
        </style>
    """, unsafe_allow_html=True)

def icon(icon_name):
    """Returns an SVG for a Lucide icon."""
    return f'<img src="https://lucide.dev/icons/{icon_name}.svg" width="20" height="20" style="vertical-align: middle;">'

# --- UI LAYOUT COMPONENTS ---

def display_header():
    """Displays the custom header."""
    st.markdown(f"""
        <div class="app-header">
            <div class="title">
                {icon('activity')}
                <h1>Terminal X</h1>
            </div>
            <div class="search-bar">
                <input type="text" placeholder="Search symbol, news, or functions...">
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_sidebar(data):
    """Displays the sidebar with watchlist and headlines."""
    with st.sidebar:
        st.subheader("Watchlist")
        for symbol, change in data["watchlist"].items():
            color = "text-green" if change.startswith('+') else "text-red"
            st.markdown(f"**{symbol}** <span class='{color} font-mono'>{change}</span>", unsafe_allow_html=True)

        st.subheader("Headlines")
        for (headline, source) in data["headlines"]:
            st.markdown(f"_{headline}_ <br><span style='color:#8C8C8C; font-size:0.8em;'>{source}</span>", unsafe_allow_html=True)

def display_price_chart(data):
    """Displays the interactive price chart."""
    chart = alt.Chart(data.reset_index()).mark_area(
        line={'color': '#84cc16'},
        gradient='linear',
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='#84cc16', offset=0), alt.GradientStop(color='rgba(10,10,10,0)', offset=0.8)],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x=alt.X('index:T', axis=alt.Axis(format='%b %d', labelAngle=-45, grid=False)),
        y=alt.Y('price:Q', axis=alt.Axis(title=None, grid=True, gridColor='#1A1A1A'), scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip('index:T', format='%Y-%m-%d'), alt.Tooltip('price:Q', format='$.2f')]
    ).properties(height=400).interactive()
    st.altair_chart(chart, use_container_width=True)

# --- TAB COMPONENTS ---

def profile_tab(data):
    st.markdown("### GOOGL - Alphabet Inc.")
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.container():
            st.markdown(f'<div class="card">{icon("bar-chart-2")} <h3>Price Chart</h3></div>', unsafe_allow_html=True)
            display_price_chart(data["price_data"])
    with col2:
        st.markdown(f'<div class="card">{icon("users")} <h3>Key Executives</h3><table class="custom-table">{"".join(f"<tr><td>{name}</td><td>{title}</td></tr>" for name, title in data["executives"])}</table></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card" style="margin-top:1rem;">{icon("info")} <h3>Company Info</h3><p>Alphabet Inc. is a holding company...</p></div>', unsafe_allow_html=True)


def trades_tab(data):
    st.markdown(f'<div class="card">{icon("book-open")}<h3>Trade History</h3></div>', unsafe_allow_html=True)
    st.dataframe(data["trades"], use_container_width=True)

def dcf_tab():
    st.markdown(f'<div class="card">{icon("calculator")}<h3>Discounted Cash Flow (DCF)</h3></div>', unsafe_allow_html=True)
    symbol = st.text_input("Enter a symbol", "AAPL")
    if st.button("Run DCF"):
        st.markdown(mock_dcf(symbol))

# --- MAIN APP ---

def run():
    st.set_page_config(layout="wide", page_title="Terminal X", page_icon="⚡")
    load_css()
    mock_data = get_mock_data()

    display_header()
    display_sidebar(mock_data)

    # Main content area
    with st.container():
        st.markdown('<div style="padding: 0 2rem;">', unsafe_allow_html=True) # Main content padding
        
        selected = option_menu(
            menu_title=None,
            options=["Profile", "Financials", "News", "Trades", "DCF"],
            icons=["user", "briefcase", "newspaper", "book-open", 'calculator'],
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#0A0A0A", "border-bottom": "1px solid #1A1A1A"},
                "icon": {"color": "#8C8C8C", "font-size": "16px"},
                "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px", "--hover-color": "#262626", "color": "#EAEAEA"},
                "nav-link-selected": {"background-color": "#1A1A1A", "color": "#84cc16"},
            }
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Tab content
        st.markdown('<div style="padding: 1rem 2rem;">', unsafe_allow_html=True)
        if selected == "Profile":
            profile_tab(mock_data)
        elif selected == "Trades":
            trades_tab(mock_data)
        elif selected == "DCF":
            dcf_tab()
        else:
            st.markdown(f'<div class="card"><h3>{selected}</h3><p>This section is under construction.</p></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    run() 