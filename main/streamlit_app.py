from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "SFinDSet.csv"
MODEL_PATH = BASE_DIR / "model" / "xgboost_transaction_amount_model.joblib"
LOGO_PATH = BASE_DIR / "assets" / "finova_logo.png"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FINOVA | FinTech Intelligence",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
## ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN APP
       ======================================================== */

    .stApp {
        background-color: #F8FAFC;
    }


    /* ========================================================
       SIDEBAR - PURE BLACK
       ======================================================== */

    [data-testid="stSidebar"] {
        background-color: #000000 !important;
    }

    [data-testid="stSidebarContent"] {
        background-color: #000000 !important;
    }

    [data-testid="stSidebarUserContent"] {
        background-color: #000000 !important;
    }


    /* ========================================================
       SIDEBAR WRITTEN TEXT - WHITE
       ======================================================== */

    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] p {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] strong {
        color: #FFFFFF !important;
    }


    /* ========================================================
       NAVIGATION RADIO TEXT
       ======================================================== */

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }


    /* ========================================================
       DATE INPUTS
       ======================================================== */

    [data-testid="stSidebar"] [data-testid="stDateInput"] label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] [data-testid="stDateInput"] input {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 1px solid #555555 !important;
        border-radius: 7px !important;
        min-height: 40px !important;
    }

    [data-testid="stSidebar"] [data-testid="stDateInput"] input::placeholder {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] [data-testid="stDateInput"] button {
        color: #FFFFFF !important;
    }


    /* ========================================================
       SELECTBOXES
       ======================================================== */

    [data-testid="stSidebar"] [data-testid="stSelectbox"] label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #111111 !important;
        border: 1px solid #555555 !important;
        border-radius: 7px !important;
        min-height: 40px !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: #FFFFFF !important;
    }


    /* ========================================================
       NUMBER INPUT
       ======================================================== */

    [data-testid="stSidebar"] [data-testid="stNumberInput"] label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        background-color: #111111 !important;
        color: #FFFFFF !important;
        border: 1px solid #555555 !important;
        border-radius: 7px !important;
        min-height: 40px !important;
    }


    /* ========================================================
       SIDEBAR CAPTION
       ======================================================== */

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #CBD5E1 !important;
    }


    /* ========================================================
       SIDEBAR DIVIDERS
       ======================================================== */

    [data-testid="stSidebar"] hr {
        border-color: #333333 !important;
    }


    /* ========================================================
       MAIN PAGE TITLES
       ======================================================== */

    .finova-title {
        font-size: 42px;
        font-weight: 800;
        color: #0B1220;
        margin-bottom: 0px;
    }

    .finova-subtitle {
        font-size: 18px;
        color: #64748B;
        margin-top: 0px;
        margin-bottom: 25px;
    }


    /* ========================================================
       KPI CARDS
       ======================================================== */

    .kpi-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        min-height: 110px;
    }

    .kpi-title {
        font-size: 14px;
        color: #64748B;
        font-weight: 600;
    }

    .kpi-value {
        font-size: 28px;
        color: #0B1220;
        font-weight: 800;
        margin-top: 5px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #0B1220;
        margin-top: 25px;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv(DATA_PATH)

    # Convert transaction amount
    data["Transaction_Amount"] = pd.to_numeric(
        data["Transaction_Amount"],
        errors="coerce"
    )

    # Convert transaction date
    data["Transaction_Date"] = pd.to_datetime(
        data["Transaction_Date"],
        errors="coerce"
    )

    # Remove rows with invalid essential values
    data = data.dropna(
        subset=[
            "Transaction_Amount",
            "Transaction_Date"
        ]
    ).copy()

    # Model time features
    data["Year"] = data["Transaction_Date"].dt.year.astype(int)
    data["Month"] = data["Transaction_Date"].dt.month.astype(int)
    data["Hour"] = data["Transaction_Date"].dt.hour.astype(int)

    return data


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()


# ============================================================
# MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "Transaction_Mode",
    "Transaction_Type",
    "Transaction_Source",
    "Transaction_Destination",
    "Year",
    "Month",
    "Hour"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_money(value, currency, exchange_rate):

    if pd.isna(value):
        value = 0

    value = float(value)

    if currency == "NGN":
        value = value * exchange_rate
        symbol = "₦"
    else:
        symbol = "$"

    if abs(value) >= 1_000_000_000_000:
        return f"{symbol}{value / 1_000_000_000_000:.2f}T"

    elif abs(value) >= 1_000_000_000:
        return f"{symbol}{value / 1_000_000_000:.2f}B"

    elif abs(value) >= 1_000_000:
        return f"{symbol}{value / 1_000_000:.2f}M"

    elif abs(value) >= 1_000:
        return f"{symbol}{value / 1_000:.2f}K"

    return f"{symbol}{value:,.2f}"


def convert_amount_series(series, currency, exchange_rate):

    converted = series.copy()

    if currency == "NGN":
        converted = converted * exchange_rate

    return converted


def kpi_card(title, value):

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # LOGO
    # --------------------------------------------------------

    if LOGO_PATH.exists():

        st.image(
            str(LOGO_PATH),
            width=180
        )

    else:

        st.markdown(
            """
            <div style="
                font-size:32px;
                font-weight:800;
                color:white;
                margin-bottom:0px;
            ">
            FINOVA
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div style="
            font-size:13px;
            color:#94A3B8;
            margin-top:-10px;
            margin-bottom:25px;
        ">
        Financial Transaction Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --------------------------------------------------------
    # NAVIGATION
    # --------------------------------------------------------

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Overview",
            "💳 Transaction Analysis",
            "👤 Account Analysis",
            "🔍 Transaction Search",
            "🤖 ML Prediction",
            "📊 Model Performance"
        ]
    )

    st.markdown("---")

    # --------------------------------------------------------
    # GLOBAL FILTERS
    # --------------------------------------------------------

    st.markdown("### Filters")

    valid_dates = df["Transaction_Date"].dropna()

    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    start_date = st.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )

    end_date = st.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

    mode_options = ["All"] + sorted(
        df["Transaction_Mode"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_mode = st.selectbox(
        "Transaction Mode",
        mode_options
    )

    type_options = ["All"] + sorted(
        df["Transaction_Type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_type = st.selectbox(
        "Transaction Type",
        type_options
    )

    st.markdown("---")

    # --------------------------------------------------------
    # CURRENCY FILTER
    # --------------------------------------------------------

    st.markdown("### Currency")

    display_currency = st.selectbox(
        "Display Currency",
        ["USD", "NGN"]
    )

    usd_to_ngn = st.number_input(
        "USD → NGN Exchange Rate",
        min_value=1.0,
        value=1500.0,
        step=10.0,
        help="Used only to convert dashboard values and predictions for display."
    )

    if display_currency == "NGN":

        st.caption(
            f"Using: $1 = ₦{usd_to_ngn:,.2f}"
        )

    else:

        st.caption(
            f"Using: $1 = ₦{usd_to_ngn:,.2f}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # ABOUT
    # --------------------------------------------------------

    st.markdown(
        """
        **FINOVA**

        Financial transaction intelligence platform built with:

        Python  
        Pandas  
        Scikit-learn  
        XGBoost  
        Streamlit
        """
    )


# ============================================================
# VALIDATE DATE RANGE
# ============================================================

if start_date > end_date:

    st.error(
        "Start Date cannot be later than End Date."
    )

    st.stop()


# ============================================================
# APPLY GLOBAL FILTERS
# ============================================================

filtered_df = df[
    (df["Transaction_Date"].dt.date >= start_date)
    &
    (df["Transaction_Date"].dt.date <= end_date)
].copy()


if selected_mode != "All":

    filtered_df = filtered_df[
        filtered_df["Transaction_Mode"].astype(str)
        == selected_mode
    ]


if selected_type != "All":

    filtered_df = filtered_df[
        filtered_df["Transaction_Type"].astype(str)
        == selected_type
    ]


# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    col_logo, col_title = st.columns([1, 5])

    with col_logo:

        if LOGO_PATH.exists():

            st.image(
                str(LOGO_PATH),
                width=140
            )

    with col_title:

        st.markdown(
            '<div class="finova-title">FINOVA</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="finova-subtitle">'
            'Financial Transaction Intelligence Platform'
            '</div>',
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # FILTER STATUS
    # --------------------------------------------------------

    st.info(
        f"Showing {len(filtered_df):,} transactions "
        f"from {start_date} to {end_date} "
        f"• Currency: {display_currency}"
    )

    # --------------------------------------------------------
    # KPI VALUES
    # --------------------------------------------------------

    total_transactions = len(filtered_df)

    total_value = filtered_df[
        "Transaction_Amount"
    ].sum()

    average_transaction = (
        filtered_df["Transaction_Amount"].mean()
        if not filtered_df.empty
        else 0
    )

    unique_accounts = filtered_df[
        "Account_Number"
    ].nunique()

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        kpi_card(
            "TOTAL TRANSACTIONS",
            f"{total_transactions:,}"
        )

    with col2:

        kpi_card(
            "TOTAL TRANSACTION VALUE",
            format_money(
                total_value,
                display_currency,
                usd_to_ngn
            )
        )

    with col3:

        kpi_card(
            "AVERAGE TRANSACTION",
            format_money(
                average_transaction,
                display_currency,
                usd_to_ngn
            )
        )

    with col4:

        kpi_card(
            "UNIQUE ACCOUNTS",
            f"{unique_accounts:,}"
        )

    # --------------------------------------------------------
    # TRANSACTION ACTIVITY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Transaction Activity</div>',
        unsafe_allow_html=True
    )

    if not filtered_df.empty:

        daily_data = (
            filtered_df
            .set_index("Transaction_Date")
            .resample("D")
            .agg(
                Transactions=("Transaction_ID", "count"),
                Transaction_Value=("Transaction_Amount", "sum")
            )
            .reset_index()
        )

        daily_data["Transaction_Value_Display"] = (
            convert_amount_series(
                daily_data["Transaction_Value"],
                display_currency,
                usd_to_ngn
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            fig = px.line(
                daily_data,
                x="Transaction_Date",
                y="Transactions",
                markers=True,
                title="Transaction Volume Over Time"
            )

            fig.update_layout(
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title="Transactions"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        with col2:

            fig = px.line(
                daily_data,
                x="Transaction_Date",
                y="Transaction_Value_Display",
                markers=True,
                title=f"Transaction Value Over Time ({display_currency})"
            )

            fig.update_layout(
                template="plotly_white",
                xaxis_title="Date",
                yaxis_title=f"Transaction Value ({display_currency})"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        # ----------------------------------------------------
        # TRANSACTION MODE / TYPE
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            mode_data = (
                filtered_df
                .groupby("Transaction_Mode")
                .agg(
                    Transactions=("Transaction_ID", "count"),
                    Value=("Transaction_Amount", "sum")
                )
                .reset_index()
            )

            fig = px.bar(
                mode_data,
                x="Transaction_Mode",
                y="Transactions",
                title="Transactions by Mode"
            )

            fig.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        with col2:

            type_data = (
                filtered_df
                .groupby("Transaction_Type")
                .agg(
                    Transactions=("Transaction_ID", "count"),
                    Value=("Transaction_Amount", "sum")
                )
                .reset_index()
            )

            fig = px.bar(
                type_data,
                x="Transaction_Type",
                y="Transactions",
                title="Transactions by Type"
            )

            fig.update_layout(
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        # ----------------------------------------------------
        # HOURLY ACTIVITY
        # ----------------------------------------------------

        hourly_data = (
            filtered_df
            .groupby("Hour")
            .size()
            .reset_index(name="Transactions")
        )

        fig = px.line(
            hourly_data,
            x="Hour",
            y="Transactions",
            markers=True,
            title="Transaction Activity by Hour"
        )

        fig.update_layout(
            template="plotly_white",
            xaxis=dict(dtick=1)
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    else:

        st.warning(
            "No transactions match the selected filters."
        )


# ============================================================
# TRANSACTION ANALYSIS
# ============================================================

elif page == "💳 Transaction Analysis":

    st.markdown(
        '<div class="finova-title">Transaction Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="finova-subtitle">'
        'Explore transaction behavior and monetary activity'
        '</div>',
        unsafe_allow_html=True
    )

    if filtered_df.empty:

        st.warning(
            "No transactions match the selected filters."
        )

    else:

        analysis_df = filtered_df.copy()

        analysis_df["Display_Amount"] = convert_amount_series(
            analysis_df["Transaction_Amount"],
            display_currency,
            usd_to_ngn
        )

        # ----------------------------------------------------
        # AMOUNT DISTRIBUTION
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            fig = px.histogram(
                analysis_df,
                x="Display_Amount",
                nbins=50,
                title=f"Transaction Amount Distribution ({display_currency})"
            )

            fig.update_layout(
                template="plotly_white",
                xaxis_title=f"Transaction Amount ({display_currency})"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        with col2:

            mode_value = (
                analysis_df
                .groupby("Transaction_Mode")[
                    "Display_Amount"
                ]
                .sum()
                .reset_index()
            )

            fig = px.bar(
                mode_value,
                x="Transaction_Mode",
                y="Display_Amount",
                title=f"Transaction Value by Mode ({display_currency})"
            )

            fig.update_layout(
                template="plotly_white",
                yaxis_title=f"Transaction Value ({display_currency})"
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

        # ----------------------------------------------------
        # TRANSACTION TYPE
        # ----------------------------------------------------

        type_value = (
            analysis_df
            .groupby("Transaction_Type")[
                "Display_Amount"
            ]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            type_value,
            names="Transaction_Type",
            values="Display_Amount",
            title=f"Transaction Value by Type ({display_currency})"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        # ----------------------------------------------------
        # SUMMARY TABLE
        # ----------------------------------------------------

        st.markdown(
            "### Transaction Summary"
        )

        summary = (
            analysis_df
            .groupby("Transaction_Type")
            .agg(
                Transactions=("Transaction_ID", "count"),
                Total_Value=("Display_Amount", "sum"),
                Average_Value=("Display_Amount", "mean")
            )
            .reset_index()
        )

        summary["Total_Value"] = summary[
            "Total_Value"
        ].map(
            lambda x: format_money(
                x,
                display_currency,
                1
            )
        )

        summary["Average_Value"] = summary[
            "Average_Value"
        ].map(
            lambda x: format_money(
                x,
                display_currency,
                1
            )
        )

        st.dataframe(
            summary,
            width="stretch",
            hide_index=True
        )


# ============================================================
# ACCOUNT ANALYSIS
# ============================================================

elif page == "👤 Account Analysis":

    st.markdown(
        '<div class="finova-title">Account Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="finova-subtitle">'
        'Analyze transaction behavior across accounts'
        '</div>',
        unsafe_allow_html=True
    )

    if filtered_df.empty:

        st.warning(
            "No accounts match the selected filters."
        )

    else:

        account_summary = (
            filtered_df
            .groupby(
                [
                    "Account_Number",
                    "Account_Name"
                ]
            )
            .agg(
                Transactions=("Transaction_ID", "count"),
                Total_Value=("Transaction_Amount", "sum"),
                Average_Transaction=(
                    "Transaction_Amount",
                    "mean"
                )
            )
            .reset_index()
        )

        account_summary = account_summary.sort_values(
            "Total_Value",
            ascending=False
        )

        display_accounts = account_summary.copy()

        display_accounts["Total_Value"] = (
            display_accounts["Total_Value"]
            * (
                usd_to_ngn
                if display_currency == "NGN"
                else 1
            )
        )

        display_accounts["Average_Transaction"] = (
            display_accounts["Average_Transaction"]
            * (
                usd_to_ngn
                if display_currency == "NGN"
                else 1
            )
        )

        st.markdown(
            "### Top Accounts by Transaction Value"
        )

        st.dataframe(
            display_accounts.head(20),
            width="stretch",
            hide_index=True
        )

        top_accounts = display_accounts.head(10)

        fig = px.bar(
            top_accounts,
            x="Account_Name",
            y="Total_Value",
            title=f"Top 10 Accounts by Transaction Value ({display_currency})"
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_tickangle=-45,
            yaxis_title=f"Transaction Value ({display_currency})"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# ============================================================
# TRANSACTION SEARCH
# ============================================================

elif page == "🔍 Transaction Search":

    st.markdown(
        '<div class="finova-title">Transaction Search</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="finova-subtitle">'
        'Search transactions using their ID or account number'
        '</div>',
        unsafe_allow_html=True
    )

    search_type = st.radio(
        "Search by",
        [
            "Transaction ID",
            "Account Number"
        ],
        horizontal=True
    )

    search_value = st.text_input(
        f"Enter {search_type}"
    )

    if search_value:

        if search_type == "Transaction ID":

            result = df[
                df["Transaction_ID"]
                .astype(str)
                .str.strip()
                == search_value.strip()
            ].copy()

        else:

            result = df[
                df["Account_Number"]
                .astype(str)
                .str.strip()
                == search_value.strip()
            ].copy()

        if result.empty:

            st.warning(
                "No matching transaction found."
            )

        else:

            st.success(
                f"{len(result):,} matching transaction(s) found."
            )

            display_result = result.copy()

            display_result["Transaction_Amount"] = (
                convert_amount_series(
                    display_result["Transaction_Amount"],
                    display_currency,
                    usd_to_ngn
                )
            )

            st.dataframe(
                display_result,
                width="stretch",
                hide_index=True
            )


# ============================================================
# ML PREDICTION
# ============================================================

elif page == "🤖 ML Prediction":

    st.markdown(
        '<div class="finova-title">ML Transaction Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="finova-subtitle">'
        'Predict transaction amount using the trained XGBoost pipeline'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "The prediction uses the trained XGBoost pipeline directly. "
        "FastAPI does not need to be running for this dashboard prediction."
    )

    st.markdown("### Prediction Inputs")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # TRANSACTION MODE
    # --------------------------------------------------------

    with col1:

        prediction_mode = st.selectbox(
            "Transaction Mode",
            sorted(
                df["Transaction_Mode"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

        prediction_type = st.selectbox(
            "Transaction Type",
            sorted(
                df["Transaction_Type"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        )

        # Use number inputs instead of huge selectboxes
        # because source/destination IDs can contain many values.

        source_default = int(
            df["Transaction_Source"]
            .dropna()
            .median()
        )

        prediction_source = st.number_input(
            "Transaction Source",
            min_value=0,
            value=source_default,
            step=1
        )

        destination_default = int(
            df["Transaction_Destination"]
            .dropna()
            .median()
        )

        prediction_destination = st.number_input(
            "Transaction Destination",
            min_value=0,
            value=destination_default,
            step=1
        )

    # --------------------------------------------------------
    # DATE/TIME
    # --------------------------------------------------------

    with col2:

        prediction_date = st.date_input(
            "Prediction Date",
            value=max_date
        )

        prediction_hour = st.slider(
            "Transaction Hour",
            min_value=0,
            max_value=23,
            value=12
        )

        prediction_year = prediction_date.year

        prediction_month = prediction_date.month

        st.info(
            f"Model features generated: "
            f"Year={prediction_year}, "
            f"Month={prediction_month}, "
            f"Hour={prediction_hour}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Transaction Amount",
        type="primary",
        width="stretch"
    ):

        # EXACT FEATURES USED DURING MODEL TRAINING
        prediction_input = pd.DataFrame(
            [
                {
                    "Transaction_Mode": prediction_mode,
                    "Transaction_Type": prediction_type,
                    "Transaction_Source": int(prediction_source),
                    "Transaction_Destination": int(
                        prediction_destination
                    ),
                    "Year": int(prediction_year),
                    "Month": int(prediction_month),
                    "Hour": int(prediction_hour)
                }
            ]
        )

        try:

            # Make absolutely sure column order matches
            prediction_input = prediction_input[
                MODEL_FEATURES
            ]

            # ------------------------------------------------
            # DIRECT MODEL PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                prediction_input
            )

            predicted_amount = float(
                prediction[0]
            )

            # ------------------------------------------------
            # DISPLAY CURRENCY CONVERSION
            # ------------------------------------------------

            displayed_prediction = predicted_amount

            if display_currency == "NGN":

                displayed_prediction = (
                    predicted_amount
                    * usd_to_ngn
                )

            st.success(
                "Prediction completed successfully."
            )

            st.markdown(
                f"""
                <div class="kpi-card">

                    <div class="kpi-title">
                        PREDICTED TRANSACTION AMOUNT
                    </div>

                    <div class="kpi-value">
                        {format_money(
                            predicted_amount,
                            display_currency,
                            usd_to_ngn
                        )}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # PREDICTION DETAILS
            # ------------------------------------------------

            st.markdown("### Prediction Details")

            details_col1, details_col2 = st.columns(2)

            with details_col1:

                st.write(
                    "**Transaction Mode:**",
                    prediction_mode
                )

                st.write(
                    "**Transaction Type:**",
                    prediction_type
                )

                st.write(
                    "**Transaction Source:**",
                    int(prediction_source)
                )

                st.write(
                    "**Transaction Destination:**",
                    int(prediction_destination)
                )

            with details_col2:

                st.write(
                    "**Year:**",
                    prediction_year
                )

                st.write(
                    "**Month:**",
                    prediction_month
                )

                st.write(
                    "**Hour:**",
                    prediction_hour
                )

                st.write(
                    "**Display Currency:**",
                    display_currency
                )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown(
        '<div class="finova-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="finova-subtitle">'
        'Evaluation of transaction amount prediction models'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    performance = pd.DataFrame(
        {
            "Model": [
                "Linear Regression",
                "XGBoost"
            ],

            "MAE": [
                13075509.50,
                13128583.22
            ],

            "RMSE": [
                15159973.95,
                15199648.18
            ],

            "R²": [
                0.0049,
                -0.0003
            ]
        }
    )

    st.markdown(
        "### Model Comparison"
    )

    st.dataframe(
        performance,
        width="stretch",
        hide_index=True
    )

    # --------------------------------------------------------
    # MODEL CARDS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        kpi_card(
            "BEST MAE",
            "Linear Regression"
        )

    with col2:

        kpi_card(
            "BEST RMSE",
            "Linear Regression"
        )

    with col3:

        kpi_card(
            "BEST R²",
            "Linear Regression"
        )

    # --------------------------------------------------------
    # MODEL INTERPRETATION
    # --------------------------------------------------------

    st.markdown(
        "### Model Interpretation"
    )

    st.warning(
        """
        The current models show limited predictive power for
        individual transaction amounts.

        XGBoost did not outperform Linear Regression, and its
        R² is slightly negative.

        This indicates that the current transaction and temporal
        features contain limited information for predicting
        transaction amount.

        Additional behavioral, account-level, or historical
        features could potentially improve predictive performance.
        """
    )

    # --------------------------------------------------------
    # METRICS EXPLANATION
    # --------------------------------------------------------

    st.markdown(
        "### Metrics"
    )

    st.markdown(
        """
        **MAE — Mean Absolute Error**

        The average absolute difference between the actual
        and predicted transaction amounts.


        **RMSE — Root Mean Squared Error**

        Similar to MAE but gives greater weight to larger
        prediction errors.


        **R² — Coefficient of Determination**

        Measures how much variation in transaction amount
        is explained by the model.
        """
    )