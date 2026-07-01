import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

ALL_TICKERS = [
    "JPM", "GS", "MS", "BAC", "WFC", "C", "SCHW", "V", "MA",
    "AXP", "BLK", "SPGI", "ICE", "CME", "PNC", "USB", "TFC", "COF",
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "CRM", "ORCL",
    "ADBE", "INTC", "CSCO", "IBM", "CRWD", "PANW", "FTNT", "NOW",
    "INTU", "AMD", "QCOM", "TXN",
    "XOM", "CVX", "OXY", "F", "GM", "DAL", "AAL", "CCL",
    "FCX", "X", "TSLA", "UBER",
]

st.set_page_config(page_title="FilingPulse", layout="wide")
st.title("FilingPulse — Management Tone Tracker")

left, right = st.columns(2)

with left:
    st.subheader("Tone Shift History")
    ticker = st.selectbox("Select ticker", ALL_TICKERS)

    if st.button("Load chart", key="load_chart"):
        resp = requests.get(f"{API_URL}/sentiment", params={"ticker": ticker})
        if resp.ok:
            history = resp.json()["history"]
            if history:
                dates = [row["filing_date"] for row in history]
                scores = [row["avg_sentiment_score"] for row in history]
                shift_dates = [row["filing_date"] for row in history if row["material_shift"]]
                shift_scores = [row["avg_sentiment_score"] for row in history if row["material_shift"]]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, y=scores, mode="lines+markers", name="Avg sentiment"
                ))
                fig.add_trace(go.Scatter(
                    x=shift_dates, y=shift_scores, mode="markers",
                    marker=dict(color="red", size=12), name="Material shift"
                ))
                fig.update_layout(
                    xaxis_title="Filing date", yaxis_title="Avg sentiment score"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment history found for this ticker yet.")
        else:
            st.error("Failed to load sentiment history.")

with right:
    st.subheader("Ask a question")
    question = st.text_input("Question")
    filter_ticker = st.selectbox("Filter by ticker (optional)", ["Any"] + ALL_TICKERS)

    if st.button("Submit", key="submit_query") and question:
        payload = {
            "question": question,
            "ticker": None if filter_ticker == "Any" else filter_ticker,
        }
        resp = requests.post(f"{API_URL}/query", json=payload)
        if resp.ok:
            result = resp.json()
            st.markdown(f"**Answer:** {result['answer']}")
            for citation in result["citations"]:
                with st.expander(f"{citation['ticker']} {citation['form']} {citation['filing_date']}"):
                    st.write(citation["excerpt"])
        else:
            st.error("Query failed.")

st.divider()
st.subheader("Subscribe to tone shift alerts")
sub_col1, sub_col2, sub_col3 = st.columns([2, 2, 1])
with sub_col1:
    sub_ticker = st.selectbox("Ticker", ALL_TICKERS, key="sub_ticker")
with sub_col2:
    sub_email = st.text_input("Email")
with sub_col3:
    st.write("")
    st.write("")
    if st.button("Subscribe"):
        resp = requests.post(
            f"{API_URL}/subscribe", json={"email": sub_email, "ticker": sub_ticker}
        )
        if resp.ok:
            st.success(f"Subscribed {sub_email} to {sub_ticker} alerts.")
        else:
            st.error("Subscription failed.")
