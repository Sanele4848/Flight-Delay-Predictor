import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib
import base64
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.prediction_pipeline import FlightDelayPredictor

# Page Configuration
st.set_page_config(
    page_title="Flight Delay Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper Functions
MONTH_NAMES = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'
}


def get_month_name(month_number):
    return MONTH_NAMES.get(month_number, 'Unknown')


def get_image_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def get_risk_profile(probability):
    if probability < 0.15:
        return {
            "level": "Very Low",
            "color": "#137333",
            "bg_color": "#e6f4ea",
            "message": "This route demonstrates exceptional reliability with minimal historical delays. Optimal for time-critical travel.",
            "toast_class": "toast-very-low",
            "icon": "✓",
            "recommendation": "Highly Recommended",
            "recommendation_detail": "Excellent choice! This route has outstanding on-time performance."
        }
    elif probability < 0.25:
        return {
            "level": "Low",
            "color": "#1967d2",
            "bg_color": "#e8f0fe",
            "message": "Above-average performance with strong on-time metrics. Minor delays possible but statistically uncommon.",
            "toast_class": "toast-low",
            "icon": "✓",
            "recommendation": "Recommended",
            "recommendation_detail": "Good choice! This route shows reliable performance with low delay rates."
        }
    elif probability < 0.35:
        return {
            "level": "Moderate",
            "color": "#ea8600",
            "bg_color": "#fef7e0",
            "message": "Delays occur with moderate frequency. Recommend flexible booking options and connection buffers.",
            "toast_class": "toast-moderate",
            "icon": "⚠",
            "recommendation": "Proceed with Caution",
            "recommendation_detail": "Consider adding buffer time and flexible booking options for this route."
        }
    else:
        return {
            "level": "High",
            "color": "#c5221f",
            "bg_color": "#fce8e6",
            "message": "Elevated delay risk based on historical patterns. Travel insurance and flexible arrangements strongly advised.",
            "toast_class": "toast-high",
            "icon": "!",
            "recommendation": "Not Recommended",
            "recommendation_detail": "High delay risk. We suggest exploring alternative routes or dates."
        }


# Caching
@st.cache_resource
def load_predictor():
    return FlightDelayPredictor()


@st.cache_data
def load_ui_lookup():
    models_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    ui_lookup_path = os.path.join(models_path, 'ui_lookup_table.pkl')
    return joblib.load(ui_lookup_path)


@st.cache_data
def get_carrier_airports(_ui_lookup_df, carrier):
    carrier_airports = _ui_lookup_df[_ui_lookup_df['carrier'] == carrier]['airport'].unique().tolist()
    return sorted(carrier_airports)


@st.cache_data
def get_airport_carriers(_ui_lookup_df, airport):
    airport_carriers = _ui_lookup_df[_ui_lookup_df['airport'] == airport]['carrier'].unique().tolist()
    return sorted(airport_carriers)


# Initialize session state for loading
if 'app_loaded' not in st.session_state:
    st.session_state.app_loaded = False

# Initialize session state for toast
if 'show_toast' not in st.session_state:
    st.session_state.show_toast = False

# Custom CSS - Google Material Design 3
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@300;400;500&display=swap');

    /* Simplified Loading Screen */
    .loading-screen {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: #fafafa;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 99999;
        animation: fadeOut 0.8s ease-in-out 2s forwards;
    }

    @keyframes fadeOut {
        from {
            opacity: 1;
            visibility: visible;
        }
        to {
            opacity: 0;
            visibility: hidden;
        }
    }

    .loading-logo {
        width: 200px;
        height: auto;
        animation: logoFloat 2.5s ease-in-out infinite;
    }

    @keyframes logoFloat {
        0%, 100% {
            transform: translateY(0px);
            opacity: 0.9;
        }
        50% {
            transform: translateY(-15px);
            opacity: 1;
        }
    }

    /* Reset and Base */
    .stApp {
        background: #fafafa;
        font-family: 'Roboto', sans-serif;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Typography */
    h1, h2, h3 {
        font-family: 'Google Sans', sans-serif !important;
        font-weight: 500 !important;
        color: #1f1f1f !important;
        letter-spacing: -0.02em;
    }

    h1 {
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        font-size: 2rem !important;
        margin-top: 2rem !important;
    }

    h3 {
        font-size: 1.5rem !important;
    }

    p, label, div {
        color: #3c4043;
        font-size: 1rem;
    }

    label {
        font-weight: 500 !important;
        color: #1f1f1f !important;
        font-size: 0.95rem !important;
    }

    /* Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 28px;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
        transition: box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
    }

    .metric-card:hover {
        box-shadow: 0 1px 3px 0 rgba(60,64,67,0.3), 0 4px 8px 3px rgba(60,64,67,0.15);
    }

    /* Input Fields - Material Design */
    .stSelectbox > div > div {
        background: #ffffff !important;
        border: 1.5px solid #dadce0 !important;
        border-radius: 8px !important;
        transition: all 0.2s !important;
        font-size: 1rem !important;
        color: #1f1f1f !important;
    }

    .stSelectbox > div > div:hover {
        border-color: #1f1f1f !important;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3) !important;
        background: #ffffff !important;
    }

    .stSelectbox > div > div:focus-within {
        border-color: #1a73e8 !important;
        border-width: 2px !important;
        box-shadow: 0 0 0 1px #1a73e8 !important;
        background: #ffffff !important;
    }

    /* Text Input Styling */
    .stTextInput > div > div > input {
        background: #ffffff !important;
        border: 1.5px solid #dadce0 !important;
        border-radius: 8px !important;
        transition: all 0.2s !important;
        font-size: 1rem !important;
        color: #1f1f1f !important;
        padding: 10px 14px !important;
    }

    .stTextInput > div > div > input:hover {
        border-color: #1f1f1f !important;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3) !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #1a73e8 !important;
        border-width: 2px !important;
        box-shadow: 0 0 0 1px #1a73e8 !important;
        outline: none !important;
    }

    /* Dropdown options */
    [data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #1f1f1f !important;
        font-size: 1rem !important;
    }

    /* Dropdown menu */
    [data-baseweb="popover"] {
        background: #ffffff !important;
    }

    [role="option"] {
        background: #ffffff !important;
        color: #1f1f1f !important;
    }

    [role="option"]:hover {
        background: #f1f3f4 !important;
        color: #1f1f1f !important;
    }

    /* Selected option in dropdown */
    [data-baseweb="select"] [data-baseweb="input"] {
        color: #1f1f1f !important;
    }

    /* Input text while typing in selectbox */
    [data-baseweb="select"] input {
        color: #1f1f1f !important;
        caret-color: #1a73e8 !important;
    }

    /* Ensure all text in select is visible */
    [data-baseweb="select"] * {
        color: #1f1f1f !important;
    }

    /* Search input in dropdown */
    [data-baseweb="menu"] input {
        color: #1f1f1f !important;
        background: #ffffff !important;
    }

    /* Multi-select specific */
    .stMultiSelect > div > div {
        background: #ffffff !important;
        border: 1.5px solid #dadce0 !important;
        color: #1f1f1f !important;
    }

    .stMultiSelect > div > div:hover {
        border-color: #1f1f1f !important;
        background: #ffffff !important;
    }

    /* Buttons */
    .stButton > button {
        background: #1a73e8 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 100px !important;
        padding: 14px 36px !important;
        font-family: 'Google Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        letter-spacing: 0.0107em !important;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        height: auto !important;
    }

    .stButton > button:hover {
        background: #1557b0 !important;
        color: #ffffff !important;
        box-shadow: 0 1px 3px 0 rgba(60,64,67,0.3), 0 4px 8px 3px rgba(60,64,67,0.15) !important;
    }

    .stButton > button:active {
        background: #174ea6 !important;
        color: #ffffff !important;
    }

    /* Force white text color on button text specifically */
    .stButton > button p,
    .stButton > button span,
    .stButton > button div {
        color: #ffffff !important;
    }

    /* Disabled button - Material Design style */
    .stButton > button:disabled,
    .stButton > button:disabled:hover,
    .stButton > button:disabled:active,
    button[disabled],
    button[disabled]:hover {
        background: #e8eaed !important;
        color: #5f6368 !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
        opacity: 1 !important;
        border: 1px solid #dadce0 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 1px solid #dadce0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        color: #3c4043;
        font-family: 'Google Sans', sans-serif;
        font-weight: 500;
        padding: 14px 18px;
        border-radius: 8px 8px 0 0;
        font-size: 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: white;
        color: #1a73e8;
        border-bottom: 3px solid #1a73e8;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Google Sans', sans-serif;
        font-size: 2.25rem;
        font-weight: 500;
        color: #1f1f1f;
    }

    [data-testid="stMetricLabel"] {
        font-family: 'Roboto', sans-serif;
        font-size: 0.85rem;
        color: #3c4043;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #dadce0;
        overflow: hidden;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 8px;
        border: 1px solid #dadce0;
        font-family: 'Google Sans', sans-serif;
        color: #1f1f1f;
        font-size: 1rem;
        padding: 16px !important;
    }

    .streamlit-expanderHeader:hover {
        border-color: #1f1f1f;
    }

    /* Info boxes */
    .stAlert {
        background: white;
        border: 1px solid #dadce0;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #1a73e8;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #dadce0;
        margin: 2rem 0;
    }

    /* Container spacing */
    .block-container {
        padding-top: 5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Navigation Bar - ENHANCED */
    .nav-bar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 64px;
        background: #ffffff;
        border-bottom: 1px solid #dadce0;
        z-index: 1000;
        display: flex;
        align-items: center;
        padding: 0 24px;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
    }

    .nav-content {
        max-width: 1400px;
        width: 100%;
        margin: 0 auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        text-decoration: none;
    }

    .nav-logo {
        width: auto;
        height: 40px;
        object-fit: contain;
    }

    .nav-logo-emoji {
        font-size: 28px;
    }

    .nav-title {
        font-family: 'Google Sans', sans-serif;
        font-size: 1.5rem;
        font-weight: 500;
        color: #1f1f1f;
        letter-spacing: -0.02em;
    }

    .nav-links {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .nav-link {
        font-family: 'Google Sans', sans-serif;
        font-size: 0.95rem;
        font-weight: 500;
        color: #5f6368;
        padding: 8px 16px;
        border-radius: 24px;
        text-decoration: none;
        transition: all 0.2s;
        cursor: pointer;
    }

    .nav-link:hover {
        background: #f1f3f4;
        color: #1f1f1f;
    }

    /* ENHANCED: Darker background for active nav item */
    .nav-link.active {
        background: #d2e3fc;
        color: #1a73e8;
        font-weight: 500;
    }

    .nav-button {
        background: #1a73e8;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 24px;
        font-family: 'Google Sans', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 1px 2px 0 rgba(60,64,67,0.3);
    }

    .nav-button:hover {
        background: #1557b0;
        box-shadow: 0 1px 3px 0 rgba(60,64,67,0.3), 0 2px 4px 2px rgba(60,64,67,0.15);
    }

    /* Mobile responsive */
    @media (max-width: 768px) {
        .nav-links {
            display: none;
        }

        .block-container {
            padding-top: 4rem;
        }
    }

    /* Smooth scroll */
    html {
        scroll-behavior: smooth;
    }

    /* Section anchors */
    .section-anchor {
        scroll-margin-top: 80px;
        display: block;
        height: 0;
        visibility: hidden;
    }

    /* Remove box border from containers */
    [data-testid="stVerticalBlock"] > div {
        background: transparent;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #1a73e8 !important;
    }

    /* Toast Notification Styles */
    @keyframes slideInDown {
        from {
            transform: translateY(-100%);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    @keyframes slideOutUp {
        from {
            transform: translateY(0);
            opacity: 1;
        }
        to {
            transform: translateY(-100%);
            opacity: 0;
        }
    }

    @keyframes progressBar {
        from {
            width: 0%;
        }
        to {
            width: 100%;
        }
    }

    .toast-notification {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 9999;
        width: 100%;
        padding: 15px 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
        animation: slideInDown 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        font-family: 'Google Sans', sans-serif;
        overflow: hidden;
    }

    .toast-notification.hide {
        animation: slideOutUp 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .toast-progress-bar {
        position: absolute;
        bottom: 0;
        left: 0;
        height: 4px;
        background: rgba(255, 255, 255, 0.9);
        animation: progressBar 4s linear;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    }

    .toast-very-low {
        background: linear-gradient(135deg, #34a853 0%, #2d9248 100%);
        color: white;
    }

    .toast-very-low .toast-progress-bar {
        background: rgba(255, 255, 255, 0.9);
    }

    .toast-low {
        background: linear-gradient(135deg, #4285f4 0%, #3367d6 100%);
        color: white;
    }

    .toast-low .toast-progress-bar {
        background: rgba(255, 255, 255, 0.9);
    }

    .toast-moderate {
        background: linear-gradient(135deg, #fbbc04 0%, #f9ab00 100%);
        color: white;
    }

    .toast-moderate .toast-progress-bar {
        background: rgba(255, 255, 255, 0.9);
    }

    .toast-high {
        background: linear-gradient(135deg, #ea4335 0%, #d93025 100%);
        color: white;
    }

    .toast-high .toast-progress-bar {
        background: rgba(255, 255, 255, 0.9);
    }

    .toast-container {
        max-width: 1400px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        position: relative;
    }

    .toast-icon {
        font-size: 32px;
        margin-right: 16px;
        display: inline-block;
        vertical-align: middle;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2));
    }

    .toast-content {
        display: inline-block;
        vertical-align: middle;
        flex: 1;
    }

    .toast-title {
        font-family: 'Google Sans', sans-serif;
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 4px;
        color: white;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }

    .toast-message {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.95);
        line-height: 1.4;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }

    .toast-close {
        background: rgba(255, 255, 255, 0.2);
        border: none;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        cursor: pointer;
        font-size: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
        margin-left: 16px;
        flex-shrink: 0;
    }

    .toast-close:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: scale(1.1);
    }

    .toast-close:active {
        transform: scale(0.95);
    }
</style>
""", unsafe_allow_html=True)

# Loading Screen
if not st.session_state.app_loaded:
    logo_path = "assets/img/FlightCAST_loading (1).png"
    logo_base64_loading = get_image_base64(logo_path)

    if logo_base64_loading:
        logo_html_loading = f'<img src="data:image/png;base64,{logo_base64_loading}" class="loading-logo" alt="FlightCAST Logo">'
    else:
        logo_html_loading = '<div style="font-size: 80px;">✈️</div>'

    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown(f"""
        <div class="loading-screen">
            {logo_html_loading}
        </div>
        <script>
            setTimeout(function() {{
                var loadingScreen = document.querySelector('.loading-screen');
                if (loadingScreen && loadingScreen.parentElement) {{
                    loadingScreen.parentElement.remove();
                }}
            }}, 2800);
        </script>
        """, unsafe_allow_html=True)

    time.sleep(2.2)
    st.session_state.app_loaded = True
    loading_placeholder.empty()
    st.rerun()

# Initialize
predictor = load_predictor()
ui_lookup = load_ui_lookup()
ui_lookup['carrier_name'] = ui_lookup['carrier'].map(predictor.carrier_names)
ui_lookup['airport_name'] = ui_lookup['airport'].map(predictor.airport_names)

# Initialize session state for navigation
if 'active_nav' not in st.session_state:
    st.session_state.active_nav = 'home'

# Toast placeholder at the top
toast_placeholder = st.empty()

# Navigation Bar with logo
logo_path = "assets/img/FlightCAST.png"
logo_base64 = get_image_base64(logo_path)

if logo_base64:
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="nav-logo" alt="FlightCAST Logo">'
else:
    logo_html = '<span class="nav-logo-emoji">✈️</span>'

nav_id = int(time.time() * 1000)

st.markdown(f"""
<div class="nav-bar">
    <div class="nav-content">
        <div class="nav-brand">
            {logo_html}
            <span class="nav-title">Flight Delay Analytics</span>
        </div>
        <div class="nav-links">
            <a class="nav-link nav-link-home" href="#home" data-section="home">Home</a>
            <a class="nav-link nav-link-analytics" href="#analytics" data-section="analytics">Analytics</a>
            <a class="nav-link nav-link-insights" href="#insights" data-section="insights">Insights</a>
            <a class="nav-link nav-link-about" href="#about" data-section="about">About</a>
            <button class="nav-button" onclick="scrollToSection('analytics')">Get Started</button>
        </div>
    </div>
</div>
<script id="nav-script-{nav_id}">
    (function() {{
        const oldScripts = document.querySelectorAll('[id^="nav-script-"]');
        oldScripts.forEach(script => {{
            if (script.id !== 'nav-script-{nav_id}') {{
                script.remove();
            }}
        }});

        function scrollToSection(sectionId) {{
            const section = document.getElementById(sectionId);
            if (section) {{
                const navHeight = 80;
                const targetPosition = section.getBoundingClientRect().top + window.pageYOffset - navHeight;

                window.scrollTo({{
                    top: targetPosition,
                    behavior: 'smooth'
                }});

                updateActiveNav(sectionId);
            }}
        }}

        function updateActiveNav(activeSection) {{
            const navLinks = document.querySelectorAll('.nav-link');
            navLinks.forEach(link => {{
                link.classList.remove('active');
                if (link.getAttribute('data-section') === activeSection) {{
                    link.classList.add('active');
                }}
            }});
        }}

        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {{
            const newLink = link.cloneNode(true);
            link.parentNode.replaceChild(newLink, link);
        }});

        document.querySelectorAll('.nav-link').forEach(link => {{
            link.addEventListener('click', function(e) {{
                e.preventDefault();
                const sectionId = this.getAttribute('data-section');
                scrollToSection(sectionId);
            }});
        }});

        window.scrollToSection = scrollToSection;

        const sections = document.querySelectorAll('.section-anchor');
        const observerOptions = {{
            root: null,
            rootMargin: '-100px 0px -70% 0px',
            threshold: 0
        }};

        const observer = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting) {{
                    const sectionId = entry.target.id;
                    updateActiveNav(sectionId);
                }}
            }});
        }}, observerOptions);

        sections.forEach(section => {{
            observer.observe(section);
        }});

        setTimeout(() => {{
            updateActiveNav('home');
        }}, 100);
    }})();
</script>
""", unsafe_allow_html=True)

# Header
st.markdown('<div id="home" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown("<h1>Flight Delay Analytics</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='font-size: 1.15rem; color: #3c4043; margin-bottom: 2rem;'>Machine learning-powered insights for intelligent travel planning. Delay threshold: 15 minutes from scheduled arrival.</p>",
    unsafe_allow_html=True)

# Input Section - Horizontal Layout
st.markdown('<div id="analytics" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown("### Configure Flight Parameters")
st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

input_col1, input_col2, input_col3 = st.columns([3, 3, 2])


def format_carrier_name(carrier_code):
    name = predictor.carrier_names.get(carrier_code, "Unknown Carrier")
    return f"{name} ({carrier_code})"


with input_col1:
    carrier = st.selectbox(
        'Airline',
        predictor.stats['carriers'],
        format_func=format_carrier_name,
        label_visibility="visible"
    )

available_airports = get_carrier_airports(ui_lookup, carrier)


def format_airport_name(airport_code):
    name = predictor.airport_names.get(airport_code, "Unknown Airport")
    return f"{airport_code} - {name}"


with input_col2:
    if not available_airports:
        st.selectbox('Destination Airport', ['No data available'], disabled=True)
        airport = None
    else:
        airport = st.selectbox(
            'Destination Airport',
            available_airports,
            format_func=format_airport_name
        )

with input_col3:
    month = st.selectbox('Travel Month', options=list(MONTH_NAMES.keys()), format_func=get_month_name)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

predict_btn = st.button('Analyze Route', disabled=(airport is None))

st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

# Prediction Results
if predict_btn and airport:
    with st.spinner('Processing analysis...'):
        result = predictor.predict(carrier, airport, month, 100)
        risk_profile = get_risk_profile(result['delay_probability'])

    # Show toast notification
    import random

    toast_id = f"toast_{random.randint(100000, 999999)}"
    stay_duration = 3.0
    fade_duration = 0.6

    with toast_placeholder.container():
        st.markdown(f"""
        <style>
            @keyframes delayedFadeOut {{
                0% {{ opacity: 1; transform: translateY(0); }}
                {(stay_duration / (stay_duration + fade_duration)) * 100}% {{ opacity: 1; transform: translateY(0); }}
                100% {{ opacity: 0; transform: translateY(-20px); }}
            }}

            #{toast_id} {{
                animation: delayedFadeOut {stay_duration + fade_duration}s ease-in-out forwards;
            }}

            #{toast_id} .toast-progress-bar {{
                animation: progressBar {stay_duration}s linear;
            }}
        </style>
        <div id="{toast_id}" class="toast-notification {risk_profile['toast_class']}" style="position: fixed; top: 0; left: 0; right: 0; z-index: 999999; margin: 0;">
            <div class="toast-container">
                <span class="toast-icon">{risk_profile['icon']}</span>
                <div class="toast-content">
                    <div class="toast-title">{risk_profile['recommendation']}</div>
                    <div class="toast-message">{risk_profile['recommendation_detail']}</div>
                </div>
            </div>
            <div class="toast-progress-bar"></div>
        </div>
        <script>
            setTimeout(function() {{
                var toast = document.getElementById('{toast_id}');
                if(toast && toast.parentElement) {{
                    toast.parentElement.remove();
                }}
            }}, {int((stay_duration + fade_duration) * 1000)});
        </script>
        """, unsafe_allow_html=True)

    st.markdown("### Predictive Analysis Results")
    st.markdown(
        f"<p style='color: #3c4043; margin-bottom: 24px; font-size: 1.05rem;'>{result['carrier']} to {result['airport']} • {get_month_name(result['month'])}</p>",
        unsafe_allow_html=True)

    # Enhanced metrics display with visual hierarchy
    metric_col1, metric_col2 = st.columns([2, 1])

    with metric_col1:
        # Main probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result['delay_probability'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Delay Probability", 'font': {'size': 20, 'family': 'Google Sans'}},
            number={'suffix': "%", 'font': {'size': 48, 'family': 'Google Sans'}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#dadce0"},
                'bar': {'color': risk_profile['color']},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#dadce0",
                'steps': [
                    {'range': [0, 15], 'color': '#e6f4ea'},
                    {'range': [15, 25], 'color': '#e8f0fe'},
                    {'range': [25, 35], 'color': '#fef7e0'},
                    {'range': [35, 100], 'color': '#fce8e6'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': predictor.stats['overall_delay_rate'] * 100
                }
            }
        ))

        fig_gauge.update_layout(
            paper_bgcolor="white",
            font={'color': "#1f1f1f", 'family': "Roboto"},
            height=300,
            margin=dict(l=20, r=20, t=80, b=20)
        )

        st.plotly_chart(fig_gauge, use_container_width=True)

    with metric_col2:
        # Risk classification card
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {risk_profile['color']}15 0%, {risk_profile['color']}05 100%); 
                    border: 2px solid {risk_profile['color']}; 
                    border-radius: 16px; 
                    padding: 24px; 
                    height: 300px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    position: relative;
                    overflow: hidden;">
            <div style="color: {risk_profile['color']}; font-size: 0.9rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 12px;">Risk Level</div>
            <div style="font-size: 2.5rem; font-weight: 600; color: {risk_profile['color']}; font-family: 'Google Sans', sans-serif; margin-bottom: 16px; position: relative; z-index: 1;">{risk_profile['level']}</div>
            <div style="color: #1f1f1f; font-size: 0.95rem; line-height: 1.5; position: relative; z-index: 1;">{risk_profile['message'][:120]}...</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

    # Secondary metrics in a data grid
    metric_col3, metric_col4, metric_col5 = st.columns(3)

    with metric_col3:
        st.markdown(f"""
        <div style="background: white; 
                    border-left: 4px solid #1a73e8; 
                    border-radius: 8px; 
                    padding: 20px 24px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <div style="color: #5f6368; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500;">Avg Delay</div>
            </div>
            <div style="font-size: 2rem; font-weight: 600; color: #1f1f1f; font-family: 'Google Sans', sans-serif;">{result['avg_delay_minutes']:.0f}<span style="font-size: 1rem; color: #5f6368; font-weight: 400;"> min</span></div>
            <div style="color: #5f6368; font-size: 0.85rem; margin-top: 8px;">When delays occur</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_col4:
        st.markdown(f"""
        <div style="background: white; 
                    border-left: 4px solid #ea8600; 
                    border-radius: 8px; 
                    padding: 20px 24px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <div style="color: #5f6368; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500;">Expected</div>
            </div>
            <div style="font-size: 2rem; font-weight: 600; color: #1f1f1f; font-family: 'Google Sans', sans-serif;">{result['expected_delays_per_100']:.0f}<span style="font-size: 1rem; color: #5f6368; font-weight: 400;">/100</span></div>
            <div style="color: #5f6368; font-size: 0.85rem; margin-top: 8px;">Delayed flights</div>
        </div>
        """, unsafe_allow_html=True)

    with metric_col5:
        system_avg = predictor.stats['overall_delay_rate'] * 100
        comparison = result['delay_probability'] * 100 - system_avg
        comparison_color = "#34a853" if comparison < 0 else "#ea4335"
        comparison_text = "better" if comparison < 0 else "worse"

        st.markdown(f"""
        <div style="background: white; 
                    border-left: 4px solid {comparison_color}; 
                    border-radius: 8px; 
                    padding: 20px 24px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.06);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <div style="color: #5f6368; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500;">vs Average</div>
            </div>
            <div style="font-size: 2rem; font-weight: 600; color: {comparison_color}; font-family: 'Google Sans', sans-serif;">{abs(comparison):.1f}<span style="font-size: 1rem; font-weight: 400;">%</span></div>
            <div style="color: #5f6368; font-size: 0.85rem; margin-top: 8px;">{comparison_text} than avg</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Advisory message
    st.markdown(f"""
    <div style="background: {risk_profile['bg_color']}; border-left: 4px solid {risk_profile['color']}; padding: 20px; border-radius: 8px; margin-bottom: 32px;">
        <div style="color: {risk_profile['color']}; font-weight: 500; margin-bottom: 6px; font-size: 1rem;">Advisory</div>
        <div style="color: #1f1f1f; font-size: 1rem;">{risk_profile['message']}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Understanding the Metrics", expanded=False):
        st.markdown(f"""
        <div style='font-size: 1rem; line-height: 1.6; color: #3c4043;'>
        <p><strong>Delay Probability ({result['delay_probability']:.1%}):</strong> Statistical likelihood that a flight on this route will experience a delay of 15 minutes or more. Based on historical data, approximately {result['expected_delays_per_100']:.0f} out of 100 flights are affected.</p>

        <p><strong>Expected Delay Duration ({result['avg_delay_minutes']:.0f} minutes):</strong> When delays occur on this specific route, the average duration is {result['avg_delay_minutes']:.0f} minutes.</p>

        <p><strong>Practical Application:</strong> For this booking, there is a {result['delay_probability']:.1%} chance of a 15+ minute delay. If delayed, expect approximately {result['avg_delay_minutes']:.0f} minutes of additional wait time.</p>
        </div>
        """, unsafe_allow_html=True)

    # Analysis Tabs
    st.markdown('<div id="insights" class="section-anchor"></div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(['Temporal Patterns', 'Comparative Metrics', 'Route Intelligence'])

    with tab1:
        st.markdown("#### Seasonal Performance Analysis")
        st.markdown(
            f"<p style='color: #3c4043; margin-bottom: 24px; font-size: 1rem;'>{carrier} operations at {airport}</p>",
            unsafe_allow_html=True)

        monthly_data = [predictor.predict(carrier, airport, m, 100) for m in range(1, 13)]
        df_monthly = pd.DataFrame(monthly_data)
        df_monthly['Month_Name'] = df_monthly['month'].map(MONTH_NAMES)
        df_monthly['Delay_Probability_Pct'] = df_monthly['delay_probability'] * 100

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=df_monthly['Month_Name'],
                y=df_monthly['Delay_Probability_Pct'],
                mode='lines+markers',
                line=dict(color='#1a73e8', width=3),
                marker=dict(size=10, color='#1a73e8'),
                name='Delay Rate'
            ))
            fig_line.add_hline(
                y=predictor.stats['overall_delay_rate'] * 100,
                line_dash="dash",
                line_color="#ea8600",
                line_width=2,
                annotation_text="System Average",
                annotation_position="right",
                annotation_font_size=13,
                annotation_font_color="#1f1f1f"
            )
            fig_line.update_layout(
                title='Monthly Delay Probability',
                xaxis_title='',
                yaxis_title='Probability (%)',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Roboto', size=14, color='#1f1f1f'),
                title_font=dict(family='Google Sans', size=17, color='#1f1f1f', weight=500),
                height=400,
                margin=dict(l=60, r=40, t=60, b=40),
                xaxis=dict(showgrid=False, showline=True, linewidth=1, linecolor='#dadce0',
                           tickfont=dict(size=13, color='#1f1f1f')),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e8eaed', showline=True, linewidth=1,
                           linecolor='#dadce0', tickfont=dict(size=13, color='#1f1f1f'),
                           title_font=dict(size=14, color='#1f1f1f'))
            )
            st.plotly_chart(fig_line, use_container_width=True)

        with chart_col2:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=df_monthly['Month_Name'],
                y=df_monthly['avg_delay_minutes'],
                marker=dict(
                    color=df_monthly['avg_delay_minutes'],
                    colorscale='Reds',
                    showscale=False,
                    line=dict(width=0)
                )
            ))
            fig_bar.update_layout(
                title='Average Delay Duration',
                xaxis_title='',
                yaxis_title='Minutes',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family='Roboto', size=14, color='#1f1f1f'),
                title_font=dict(family='Google Sans', size=17, color='#1f1f1f', weight=500),
                height=400,
                margin=dict(l=60, r=40, t=60, b=40),
                xaxis=dict(showgrid=False, showline=True, linewidth=1, linecolor='#dadce0',
                           tickfont=dict(size=13, color='#1f1f1f')),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e8eaed', showline=True, linewidth=1,
                           linecolor='#dadce0', tickfont=dict(size=13, color='#1f1f1f'),
                           title_font=dict(size=14, color='#1f1f1f'))
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        best_month = df_monthly.loc[df_monthly['delay_probability'].idxmin()]
        worst_month = df_monthly.loc[df_monthly['delay_probability'].idxmax()]

        insight_col1, insight_col2 = st.columns(2)
        with insight_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #137333; font-weight: 500; margin-bottom: 10px; font-size: 1rem;">Optimal Travel Period</div>
                <div style="font-size: 1.5rem; font-weight: 500; color: #1f1f1f; margin-bottom: 6px;">{best_month['Month_Name']}</div>
                <div style="color: #3c4043; font-size: 1rem;">{best_month['delay_probability']:.1%} delay rate • {best_month['avg_delay_minutes']:.0f} min avg delay</div>
            </div>
            """, unsafe_allow_html=True)

        with insight_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="color: #ea8600; font-weight: 500; margin-bottom: 10px; font-size: 1rem;">Challenging Period</div>
                <div style="font-size: 1.5rem; font-weight: 500; color: #1f1f1f; margin-bottom: 6px;">{worst_month['Month_Name']}</div>
                <div style="color: #3c4043; font-size: 1rem;">{worst_month['delay_probability']:.1%} delay rate • {worst_month['avg_delay_minutes']:.0f} min avg delay</div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("#### Performance Benchmarking")
        st.markdown(
            f"<p style='color: #3c4043; margin-bottom: 24px; font-size: 1rem;'>Analysis for {get_month_name(month)}</p>",
            unsafe_allow_html=True)

        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            st.markdown(f"**Airline Performance at {airport}**")
            airport_carriers = get_airport_carriers(ui_lookup, airport)
            carrier_perf = [predictor.predict(c, airport, month, 100) for c in airport_carriers]
            df_carriers = pd.DataFrame(carrier_perf).sort_values('delay_probability')
            df_carriers['Delay_Rate'] = df_carriers['delay_probability'] * 100

            fig_carriers = go.Figure()
            fig_carriers.add_trace(go.Bar(
                y=df_carriers['carrier'],
                x=df_carriers['Delay_Rate'],
                orientation='h',
                marker=dict(
                    color=df_carriers['Delay_Rate'],
                    colorscale='RdYlGn_r',
                    showscale=False,
                    line=dict(width=0)
                ),
                text=df_carriers['Delay_Rate'].round(1).astype(str) + '%',
                textposition='outside',
                textfont=dict(size=13, color='#1f1f1f')
            ))
            fig_carriers.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_title='Delay Rate (%)',
                yaxis_title='',
                font=dict(family='Roboto', size=14, color='#1f1f1f'),
                height=400,
                margin=dict(l=60, r=60, t=20, b=40),
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e8eaed', showline=True, linewidth=1,
                           linecolor='#dadce0', tickfont=dict(size=13, color='#1f1f1f'),
                           title_font=dict(size=14, color='#1f1f1f')),
                yaxis=dict(showgrid=False, showline=True, linewidth=1, linecolor='#dadce0',
                           tickfont=dict(size=13, color='#1f1f1f'))
            )
            st.plotly_chart(fig_carriers, use_container_width=True)

        with comp_col2:
            st.markdown(f"**Destination Performance for {carrier}**")
            carrier_airports_comp = get_carrier_airports(ui_lookup, carrier)
            airport_perf = [predictor.predict(carrier, a, month, 100) for a in carrier_airports_comp]
            df_airports = pd.DataFrame(airport_perf).sort_values('delay_probability')
            df_airports['display_name'] = df_airports['airport'].apply(
                lambda x: f"{x} - {predictor.airport_names.get(x, '')[:20]}")
            df_airports['Delay_Rate'] = df_airports['delay_probability'] * 100

            fig_airports = go.Figure()
            fig_airports.add_trace(go.Bar(
                y=df_airports.head(15)['display_name'],
                x=df_airports.head(15)['Delay_Rate'],
                orientation='h',
                marker=dict(
                    color=df_airports.head(15)['Delay_Rate'],
                    colorscale='RdYlGn_r',
                    showscale=False,
                    line=dict(width=0)
                ),
                text=df_airports.head(15)['Delay_Rate'].round(1).astype(str) + '%',
                textposition='outside',
                textfont=dict(size=13, color='#1f1f1f')
            ))
            fig_airports.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_title='Delay Rate (%)',
                yaxis_title='',
                font=dict(family='Roboto', size=14, color='#1f1f1f'),
                height=400,
                margin=dict(l=60, r=60, t=20, b=40),
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e8eaed', showline=True, linewidth=1,
                           linecolor='#dadce0', tickfont=dict(size=13, color='#1f1f1f'),
                           title_font=dict(size=14, color='#1f1f1f')),
                yaxis=dict(categoryorder='total ascending', showgrid=False, showline=True, linewidth=1,
                           linecolor='#dadce0', tickfont=dict(size=13, color='#1f1f1f'))
            )
            st.plotly_chart(fig_airports, use_container_width=True)

    with tab3:
        st.markdown("#### Historical Performance Dashboard")
        st.markdown(
            "<p style='color: #3c4043; margin-bottom: 24px; font-size: 1rem;'>Aggregate insights across all routes and carriers</p>",
            unsafe_allow_html=True)

        # Airline Performance Leaderboard
        st.markdown("**Airline Reliability Rankings**")

        # Calculate airline-level statistics
        carrier_stats = []
        for c in predictor.stats['carriers']:
            carrier_data = ui_lookup[ui_lookup['carrier'] == c]
            if not carrier_data.empty:
                carrier_stats.append({
                    'carrier': c,
                    'carrier_name': predictor.carrier_names.get(c, c),
                    'avg_delay_prob': carrier_data['delay_probability'].mean(),
                    'avg_delay_mins': carrier_data['avg_delay_minutes'].mean(),
                    'routes_count': len(carrier_data)
                })

        df_carrier_stats = pd.DataFrame(carrier_stats).sort_values('avg_delay_prob')
        df_carrier_stats['Delay_Rate_Pct'] = df_carrier_stats['avg_delay_prob'] * 100
        df_carrier_stats['Risk_Level'] = df_carrier_stats['avg_delay_prob'].apply(
            lambda p: get_risk_profile(p)['level']
        )

        # Top 10 Best Airlines
        perf_col1, perf_col2 = st.columns(2)

        with perf_col1:
            st.markdown("**Most Reliable Airlines**")
            top_carriers = df_carrier_stats.head(10)

            fig_top_carriers = go.Figure()
            fig_top_carriers.add_trace(go.Bar(
                y=top_carriers['carrier_name'],
                x=top_carriers['Delay_Rate_Pct'],
                orientation='h',
                marker=dict(
                    color=['#34a853', '#34a853', '#34a853', '#4285f4', '#4285f4',
                           '#4285f4', '#fbbc04', '#fbbc04', '#fbbc04', '#fbbc04'],
                    line=dict(width=0)
                ),
                text=top_carriers['Delay_Rate_Pct'].round(1).astype(str) + '%',
                textposition='outside',
                textfont=dict(size=13, color='#1f1f1f')
            ))
            fig_top_carriers.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_title='Average Delay Rate (%)',
                yaxis_title='',
                font=dict(family='Roboto', size=14, color='#1f1f1f'),
                height=400,
                margin=dict(l=60, r=60, t=20, b=40),
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e8eaed',
                           showline=True, linewidth=1, linecolor='#dadce0',
                           tickfont=dict(size=13, color='#1f1f1f'),
                           title_font=dict(size=14, color='#1f1f1f')),
                yaxis=dict(categoryorder='total ascending', showgrid=False,
                           showline=True, linewidth=1, linecolor='#dadce0',
                           tickfont=dict(size=13, color='#1f1f1f'))
            )
            st.plotly_chart(fig_top_carriers, use_container_width=True)

        with perf_col2:
            st.markdown("**Challenging Airlines**")
            bottom_carriers = df_carrier_stats.tail(10)

            fig_bottom_carriers = go.Figure()
            fig_bottom_carriers.add_trace(go.Bar(
                y=bottom_carriers['carrier_name'],
                x=bottom_carriers['Delay_Rate_Pct'],
                orientation='h',
                marker=dict(
                    color=['#fbbc04', '#fbbc04', '#fbbc04', '#ea4335', '#ea4335',
                           '#ea4335', '#ea4335', '#ea4335', '#ea4335', '#ea4335'],
                    line=dict(width=0)
                ),
                text=bottom_carriers['Delay_Rate_Pct'].round(1).astype(str) + '%',
                textposition='outside',
                textfont=dict(size=13, color='#1f1f1f')
            ))
            fig_bottom_carriers.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_title='Average Delay Rate (%)',
                yaxis_title='',
                font=dict(family='Roboto', size=14, color='#1f1f1f'),
                height=400,
                margin=dict(l=60, r=60, t=20, b=40),
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#e8eaed',
                           showline=True, linewidth=1, linecolor='#dadce0',
                           tickfont=dict(size=13, color='#1f1f1f'),
                           title_font=dict(size=14, color='#1f1f1f')),
                yaxis=dict(categoryorder='total descending', showgrid=False,
                           showline=True, linewidth=1, linecolor='#dadce0',
                           tickfont=dict(size=13, color='#1f1f1f'))
            )
            st.plotly_chart(fig_bottom_carriers, use_container_width=True)

        st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

        # Airport Performance Analysis
        st.markdown("**Airport Reliability Analysis**")

        airport_stats = []
        for a in ui_lookup['airport'].unique():
            airport_data = ui_lookup[ui_lookup['airport'] == a]
            if not airport_data.empty:
                airport_stats.append({
                    'airport': a,
                    'airport_name': predictor.airport_names.get(a, a),
                    'avg_delay_prob': airport_data['delay_probability'].mean(),
                    'avg_delay_mins': airport_data['avg_delay_minutes'].mean(),
                    'carriers_count': len(airport_data['carrier'].unique())
                })

        df_airport_stats = pd.DataFrame(airport_stats).sort_values('avg_delay_prob')
        df_airport_stats['Delay_Rate_Pct'] = df_airport_stats['avg_delay_prob'] * 100

        airport_col1, airport_col2 = st.columns(2)

        with airport_col1:
            st.markdown("**Top 15 Most Reliable Destinations**")
            st.dataframe(
                df_airport_stats.head(15)[
                    ['airport', 'airport_name', 'Delay_Rate_Pct', 'avg_delay_mins', 'carriers_count']],
                column_config={
                    "airport": st.column_config.TextColumn("Code", width="small"),
                    "airport_name": st.column_config.TextColumn("Airport"),
                    "Delay_Rate_Pct": st.column_config.NumberColumn(
                        "Delay Rate",
                        format="%.1f%%"
                    ),
                    "avg_delay_mins": st.column_config.NumberColumn(
                        "Avg Delay",
                        format="%.0f min"
                    ),
                    "carriers_count": st.column_config.NumberColumn(
                        "Airlines",
                        format="%d"
                    )
                },
                use_container_width=True,
                hide_index=True,
                height=400
            )

        with airport_col2:
            st.markdown("**15 Most Challenging Destinations**")
            st.dataframe(
                df_airport_stats.tail(15)[
                    ['airport', 'airport_name', 'Delay_Rate_Pct', 'avg_delay_mins', 'carriers_count']],
                column_config={
                    "airport": st.column_config.TextColumn("Code", width="small"),
                    "airport_name": st.column_config.TextColumn("Airport"),
                    "Delay_Rate_Pct": st.column_config.NumberColumn(
                        "Delay Rate",
                        format="%.1f%%"
                    ),
                    "avg_delay_mins": st.column_config.NumberColumn(
                        "Avg Delay",
                        format="%.0f min"
                    ),
                    "carriers_count": st.column_config.NumberColumn(
                        "Airlines",
                        format="%d"
                    )
                },
                use_container_width=True,
                hide_index=True,
                height=400
            )

# About Section
st.markdown('<div id="about" class="section-anchor"></div>', unsafe_allow_html=True)
st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)
st.markdown("## About This Platform")
st.markdown("""
    <h3 style="margin-top: 0; color: #1f1f1f;">Flight Delay Analytics Platform</h3>
    <p style="font-size: 1.05rem; line-height: 1.7; color: #3c4043; margin-bottom: 16px;">
        Our advanced analytics platform leverages machine learning to provide accurate flight delay predictions, 
        helping travelers make informed decisions about their journeys.
    </p>
    <p style="font-size: 1.05rem; line-height: 1.7; color: #3c4043; margin-bottom: 16px;">
        <strong>Key Features:</strong>
    </p>
    <ul style="font-size: 1rem; line-height: 1.8; color: #3c4043; padding-left: 20px;">
        <li><strong>Predictive Analytics:</strong> Machine learning models trained on historical flight data to predict delay probabilities</li>
        <li><strong>Risk Assessment:</strong> Four-tier risk classification system (Very Low, Low, Moderate, High)</li>
        <li><strong>Seasonal Insights:</strong> Month-by-month performance analysis for optimal travel planning</li>
        <li><strong>Comparative Intelligence:</strong> Benchmark airlines and destinations to find the best routes</li>
        <li><strong>Route Optimization:</strong> Discover the most reliable flight combinations</li>
    </ul>
    <p style="font-size: 1.05rem; line-height: 1.7; color: #3c4043; margin-top: 24px;">
        <strong>Methodology:</strong> Our predictions are based on a delay threshold of 15 minutes from scheduled arrival time. 
        The system analyzes carrier performance, destination characteristics, and seasonal patterns to generate accurate forecasts.
    </p>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; color: #3c4043; padding: 24px; border-top: 1px solid #dadce0; margin-top: 48px;'>
    <p style='margin: 0; font-size: 1rem;'><strong>Flight Delay Analytics Platform</strong></p>
    <p style='margin: 8px 0 0 0; font-size: 0.9rem;'>Delay threshold: 15+ minutes from scheduled arrival</p>
</div>
""", unsafe_allow_html=True)