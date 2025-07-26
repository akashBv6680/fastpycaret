# === app.py ===
import streamlit as st
import pandas as pd
import os
from utils.eda_generator import generate_eda_report
from utils.model_trainer import train_model_and_report
from utils.report_sender import send_email_with_attachments, is_valid_email
from utils.presentation import create_presentation
from utils.email_ai_responder import fetch_latest_email, generate_ai_reply
from utils.qa_agent import answer_query_about_df

st.set_page_config(page_title="🤖 Agentic Data Science AI", layout="wide")
st.title("🤖 Agentic Data Science Automation")

# --- Dataset Upload ---
st.header("📁 Upload Dataset")
file = st.file_uploader("Upload a CSV, Excel, or JSON file", type=["csv", "xlsx", "xls", "json"])

def load_data(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".json"):
        return pd.read_json(uploaded_file)

# --- File Processing ---
if file:
    df = load_data(file)
    st.session_state.df = df
    st.success("✅ Dataset loaded successfully.")
    st.dataframe(df.head())

    target_column = st.selectbox("🎯 Select Target Column for Modeling", df.columns)

    # Run EDA + Modeling
    if st.button("🚀 Run Full Data Science Pipeline"):
        with st.spinner("Generating EDA Report..."):
            eda_pdf = generate_eda_report(df)
            st.session_state.eda_pdf = eda_pdf
            st.success("EDA Report Generated ✅")

        with st.spinner("Training ML Model and Generating Report..."):
            model_pdf, task_type = train_model_and_report(df, target_column)
            st.session_state.model_pdf = model_pdf
            st.session_state.task_type = task_type
            st.success("Model Report Generated ✅")

        with st.spinner("Generating PowerPoint Presentation..."):
            pptx_path = create_presentation(df, target_column, task_type)
            st.session_state.presentation = pptx_path
            st.success("Presentation Created ✅")

    # --- NotebookLM-style Q&A ---
    st.markdown("---")
    st.subheader("🧠 Ask Questions About Your Dataset")
    user_question = st.text_input("What would you like to know?")
    if user_question:
        with st.spinner("Thinking..."):
            answer = answer_query_about_df(df, user_question)
        st.success("✅ Answer:")
        st.write(answer)

# --- Email Reports ---
st.markdown("---")
st.header("📧 Share Reports with Client")
recipient_email = st.text_input("Enter Client's Email Address:")

if st.button("📤 Send Reports to Client"):
    if not recipient_email:
        st.warning("Please enter a client email address.")
    elif not is_valid_email(recipient_email):
        st.warning("⚠️ Invalid email address format. Please check and try again.")
    else:
        attachments = []
        for key in ['eda_pdf', 'model_pdf', 'presentation']:
            if key in st.session_state:
                attachments.append(st.session_state[key])

        subject = "Your Data Science Reports Are Ready!"
        body = f"""
Dear Client,

Your data science analysis is complete. Please find attached:

1. EDA Report
2. Model Performance Report
3. PowerPoint Summary

Task Type: {st.session_state.get('task_type', 'N/A').title()}

Best regards,
Your AutoML Agent
"""
        if send_email_with_attachments(recipient_email, subject, body, attachments):
            st.success("✅ Reports sent successfully!")
        else:
            st.error("❌ Failed to send email. Check server logs.")

# --- AI Email Responder ---
st.markdown("---")
st.header("📬 Automated Email Responder")
if st.button("📥 Check & Auto-Reply to Latest Unread Email"):
    from_email, subject, body = fetch_latest_email()
    if from_email:
        st.markdown(f"**From:** `{from_email}`")
        st.markdown(f"**Subject:** `{subject}`")
        st.text_area("📨 Message", body, height=200)

        with st.spinner("Generating reply..."):
            reply = generate_ai_reply(body)
        st.text_area("🤖 AI Reply", reply, height=200)

        if st.button("Send AI Reply"):
            if send_email_with_attachments(from_email, f"RE: {subject}", reply):
                st.success("✅ Reply sent!")
            else:
                st.error("❌ Failed to send reply.")
    else:
        st.info("📭 No new unread emails found.")

st.markdown("---")
st.markdown("Made with ❤️ by your Agentic AI. Powered by PyCaret + Streamlit")

# === utils/qa_agent.py ===
def answer_query_about_df(df, question):
    import together
    together.api_key = os.getenv("TOGETHER_API_KEY")
    prompt = f"""
You are a smart data science assistant. The user uploaded this dataset:

{df.head(5).to_string()}

Answer the following question based on the dataset:

{question}
"""
    response = together.Complete.create(
        prompt=prompt,
        model="togethercomputer/llama-2-70b-chat",
        max_tokens=250
    )
    return response['output']['choices'][0]['text'].strip()
