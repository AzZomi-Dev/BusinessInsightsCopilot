import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import matplotlib.pyplot as plt

# Configure OpenAI
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

st.set_page_config(page_title="Business Insights Copilot", layout="wide")

st.title("📊 Business Insights Copilot")
st.markdown("Upload your **Sales** and **Support** data to receive AI-powered insights.")

# Upload CSVs
sales_file = st.file_uploader("Upload Sales Data CSV", type="csv")
support_file = st.file_uploader("Upload Support Tickets CSV", type="csv")

if sales_file and support_file:
    sales_df = pd.read_csv(sales_file, parse_dates=["date"])
    support_df = pd.read_csv(support_file, parse_dates=["date"])

    st.subheader("📈 Sales Overview")
    st.dataframe(sales_df.head())

    sales_trend = sales_df.groupby("date")["sales_amount"].sum().reset_index()
    st.line_chart(sales_trend.rename(columns={"date": "index"}).set_index("index"))

    st.subheader("🛠️ Support Overview")
    st.dataframe(support_df.head())

    tickets_trend = support_df.groupby("date")["ticket_id"].count().reset_index()
    tickets_trend.rename(columns={"ticket_id": "ticket_count"}, inplace=True)
    st.line_chart(tickets_trend.rename(columns={"date": "index"}).set_index("index"))

    from scipy.stats import zscore

    # --- Sales Anomaly Detection ---
    sales_trend["z_score"] = zscore(sales_trend["sales_amount"])
    sales_trend["anomaly"] = sales_trend["z_score"].abs() > 2

    # --- Support Anomaly Detection ---
    tickets_trend["z_score"] = zscore(tickets_trend["ticket_count"])
    tickets_trend["anomaly"] = tickets_trend["z_score"].abs() > 2

    # --- Plot Sales Anomalies ---
    st.subheader("📉 Sales Anomalies")
    fig1, ax1 = plt.subplots()
    ax1.plot(sales_trend["date"], sales_trend["sales_amount"], label="Sales")
    ax1.scatter(
        sales_trend[sales_trend["anomaly"]]["date"],
        sales_trend[sales_trend["anomaly"]]["sales_amount"],
        color='red', label="Anomaly"
    )
    ax1.legend()
    st.pyplot(fig1)

    # --- Plot Support Ticket Anomalies ---
    st.subheader("⚠️ Support Ticket Anomalies")
    fig2, ax2 = plt.subplots()
    ax2.plot(tickets_trend["date"], tickets_trend["ticket_count"], label="Tickets")
    ax2.scatter(
        tickets_trend[tickets_trend["anomaly"]]["date"],
        tickets_trend[tickets_trend["anomaly"]]["ticket_count"],
        color='orange', label="Anomaly"
    )
    ax2.legend()
    st.pyplot(fig2)


    # GPT: Generate Insights
    if st.button("🔍 Generate AI Insights"):
        with st.spinner("Thinking..."):
            prompt = f"""
            You are a business analyst AI. Analyze the following data trends and anomalies:

            Sales Trend:
            {sales_trend.tail(10).to_string(index=False)}

            Support Tickets Trend:
            {tickets_trend.tail(10).to_string(index=False)}

            Flagged Sales Anomalies:
            {sales_trend[sales_trend['anomaly']].tail(5).to_string(index=False)}

            Flagged Ticket Anomalies:
            {tickets_trend[tickets_trend['anomaly']].tail(5).to_string(index=False)}

            Provide a 3-bullet insight summary about unusual patterns, potential causes, and business suggestions.
            """


            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a business insights assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            insight = response.choices[0].message.content
            st.markdown("### 🤖 AI-Powered Insights")
            st.markdown(insight)
            import io
import contextlib

# Ask Your Data Copilot
st.subheader("💬 Ask Your Data Copilot")

user_question = st.text_input("Ask a question about your data (e.g., 'Average resolution time in April?')")

if user_question and sales_file and support_file:
    with st.spinner("AI is thinking..."):
        question_prompt = f"""
You are a data analyst AI. Answer this question using the data provided.

Sales Data Sample:
{sales_df.head(5).to_string(index=False)}

Support Data Sample:
{support_df.head(5).to_string(index=False)}

Full Question: {user_question}

Respond in this JSON format:
{{
  "thought": "How you will approach this",
  "code": "Python code using pandas to answer it",
  "answer": "A short, plain-English answer or summary"
}}
"""

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You're a helpful Python data analyst."},
                    {"role": "user", "content": question_prompt}
                ],
                temperature=0.3
            )

            reply = response.choices[0].message.content
            st.markdown("### 🧠 GPT Reasoning + Answer")
            st.code(reply, language='json')

            # Parse code from reply
            import json, re
            code_block = re.search(r'"code":\s*"(.*?)"', reply, re.DOTALL)
            if code_block:
                try:
                    python_code = bytes(code_block.group(1), "utf-8").decode("unicode_escape")

                    with contextlib.redirect_stdout(io.StringIO()) as f:
                        exec(python_code, {"sales_df": sales_df, "support_df": support_df})
                    result = f.getvalue()

                    st.success("✅ Answer Output:")
                    st.code(result)

                except Exception as e:
                    st.error(f"Error executing code: {e}")
            else:
                st.warning("GPT didn't return usable Python code.")

        except openai.NotFoundError:
            st.error("❌ Expired, try later.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Please upload both CSV files to begin.")
