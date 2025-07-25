import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import smtplib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import ssl
import imaplib
import email

# Import PyCaret modules
# We import specific functions to avoid name clashes and keep code clear
from pycaret.classification import setup as class_setup, compare_models as class_compare, tune_model as class_tune, \
                                  evaluate_model as class_evaluate, plot_model as class_plot, save_model as class_save, \
                                  load_model as class_load, predict_model as class_predict, interpret_model as class_interpret
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, tune_model as reg_tune, \
                               evaluate_model as reg_evaluate, plot_model as reg_plot, save_model as reg_save, \
                               load_model as reg_load, predict_model as reg_predict, interpret_model as reg_interpret
from pycaret.clustering import setup as cluster_setup, create_model as cluster_create, assign_model as cluster_assign, \
                               plot_model as cluster_plot, save_model as cluster_save, load_model as cluster_load
from pycaret.anomaly import setup as anomaly_setup, create_model as anomaly_create, assign_model as anomaly_assign, \
                            plot_model as anomaly_plot, save_model as anomaly_save, load_model as anomaly_load
from pycaret.nlp import setup as nlp_setup, create_model as nlp_create, assign_model as nlp_assign, \
                        plot_model as nlp_plot, save_model as nlp_save, load_model as nlp_load
from pycaret.arules import setup as arules_setup, create_model as arules_create, predict_model as arules_predict


# === Email Config (using st.secrets for security) ===
# Ensure these are set in your Streamlit Cloud secrets:
# EMAIL_ADDRESS = "your_email@gmail.com"
# EMAIL_PASSWORD = "your_app_password" # Use an App Password for Gmail!
# TOGETHER_API_KEY = "your_together_api_key"

EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465 # For SSL
IMAP_SERVER = "imap.gmail.com"

# === Streamlit Config ===
st.set_page_config(page_title="⚡ Fast AutoML Agent", layout="wide")
st.title("🤖 Fast AutoML + Email Agent")
st.markdown("---")

# --- Helper Functions ---

# Function to validate email format
def is_valid_email(email_str):
    return re.match(r"[^@\s]+@[^@\s]+\.[a-zA-Z0-9]+$", email_str)

# Function to generate basic EDA visualizations and save to PDF
def generate_eda_visualizations(df):
    pdf_name = "visual_report.pdf"
    from matplotlib.backends.backend_pdf import PdfPages # Import here to avoid circular dependency if placed at top

    with PdfPages(pdf_name) as pdf:
        for col in df.columns:
            plt.figure(figsize=(8, 6)) # Increased figure size for better clarity
            if df[col].dtype == "object" or df[col].nunique() < 10:
                # For categorical or low-cardinality numerical columns
                if df[col].nunique() <= 5: # Use pie chart for very few unique categories
                    df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='Pastel1')
                    plt.ylabel('') # Hide default ylabel for pie chart
                    plt.title(f"Pie Chart of {col}")
                else: # Use countplot for more categories
                    sns.countplot(y=col, data=df, palette='viridis')
                    plt.title(f"Bar Chart of {col}")
            elif np.issubdtype(df[col].dtype, np.number):
                # For numerical columns
                if df[col].nunique() < 20 and df[col].nunique() > 2: # Histogram for discrete numerical
                    sns.histplot(df[col], kde=False, bins=df[col].nunique(), color='skyblue')
                    plt.title(f"Histogram of {col}")
                else: # KDE plot for continuous numerical
                    sns.kdeplot(df[col], fill=True, color='salmon')
                    plt.title(f"Distribution of {col}")
            else:
                continue # Skip unsupported column types
            plt.tight_layout()
            pdf.savefig()
            plt.close()
    return pdf_name

# Function to send email with attachments
def send_email_with_attachments(to_email, subject, body, attachment_paths=None):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    if attachment_paths:
        for file_path in attachment_paths:
            if os.path.exists(file_path):
                try:
                    with open(file_path, "rb") as attachment:
                        part = MIMEBase("application", "octet-stream")
                        part.set_payload(attachment.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            "Content-Disposition",
                            f"attachment; filename= {os.path.basename(file_path)}",
                        )
                        msg.attach(part)
                except Exception as e:
                    st.warning(f"Could not attach {os.path.basename(file_path)}: {e}")
            else:
                st.warning(f"Attachment file not found: {os.path.basename(file_path)}")

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# Function to fetch the latest unread email
def fetch_latest_email():
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        mail.select("inbox")
        result, data = mail.search(None, 'UNSEEN')
        ids = data[0].split()

        if not ids:
            return None, None, None

        latest_id = ids[-1]
        result, msg_data = mail.fetch(latest_id, "(RFC822)")
        raw_email = msg_data[0][1]
        email_message = email.message_from_bytes(raw_email)

        from_email = email_message["From"]
        subject = email_message["Subject"]
        body = ""

        if email_message.is_multipart():
            for part in email_message.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition'))
                # Look for plain text parts, avoiding attachments
                if ctype == 'text/plain' and 'attachment' not in cdispo:
                    body = part.get_payload(decode=True).decode(errors='ignore')
                    break
        else:
            body = email_message.get_payload(decode=True).decode(errors='ignore')

        return from_email, subject, body

    except Exception as e:
        st.error(f"❌ Error fetching email: {e}")
        return None, None, None

# Function to generate AI reply using Together AI
def generate_ai_reply(message_content):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are a helpful business assistant. Reply to emails in simple, non-technical English, focusing on clarity and action items. Keep it concise and professional."},
            {"role": "user", "content": message_content}
        ],
        "max_tokens": 200 # Limit response length
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err} - {response.text}")
        return "Sorry, I couldn't generate a reply right now due to an API error."
    except Exception as err:
        st.error(f"An error occurred: {err}")
        return "Sorry, I couldn't generate a reply right now."

# --- Streamlit Session State Initialization ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = None
if 'pycaret_exp' not in st.session_state:
    st.session_state.pycaret_exp = None
if 'best_model_pycaret' not in st.session_state:
    st.session_state.best_model_pycaret = None
if 'tuned_model_pycaret' not in st.session_state:
    st.session_state.tuned_model_pycaret = None
if 'model_saved_path' not in st.session_state:
    st.session_state.model_saved_path = None
if 'loaded_model_pycaret' not in st.session_state:
    st.session_state.loaded_model_pycaret = None
if 'model_report_pdf' not in st.session_state: # Ensure this is initialized
    st.session_state.model_report_pdf = None

# === Upload and Detect Task ===
st.sidebar.header("📊 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a dataset", type=["csv", "xlsx"])

if uploaded_file:
    ext = uploaded_file.name.split(".")[-1]
    try:
        if ext == "csv":
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)

        st.write("✅ Data Preview")
        st.dataframe(st.session_state.df.head())

        # Allow user to select target column
        st.session_state.target_col = st.sidebar.selectbox("Select Target Column", st.session_state.df.columns, key='target_select')

        # Auto-detect task type
        # Improved detection: check for numeric type and low unique values for potential classification
        if (st.session_state.df[st.session_state.target_col].nunique() <= 20 and st.session_state.df[st.session_state.target_col].dtype == 'object') or \
           (st.session_state.df[st.session_state.target_col].nunique() < 10 and np.issubdtype(st.session_state.df[st.session_state.target_col].dtype, np.number)):
            st.session_state.task_type = "classification"
        else:
            st.session_state.task_type = "regression"
        st.info(f"🔍 Detected Task Type: **{st.session_state.task_type.title()}**")

    except Exception as e:
        st.error(f"Error loading file or detecting task: {e}")
        st.session_state.df = None # Reset dataframe on error

# --- PyCaret AutoML Workflow ---
if st.session_state.df is not None and st.session_state.target_col is not None:
    st.header("⚙️ PyCaret AutoML Process")

    if st.button("🚀 Run PyCaret Setup"):
        with st.spinner("Initializing PyCaret environment and preprocessing data..."):
            if st.session_state.task_type == "classification":
                st.session_state.pycaret_exp = class_setup(data=st.session_state.df, target=st.session_state.target_col,
                                                            session_id=123, silent=True, verbose=False, html=False)
            elif st.session_state.task_type == "regression":
                st.session_state.pycaret_exp = reg_setup(data=st.session_state.df, target=st.session_state.target_col,
                                                          session_id=123, silent=True, verbose=False, html=False)
            # Unsupervised tasks
            elif st.session_state.task_type == "clustering":
                st.session_state.pycaret_exp = cluster_setup(data=st.session_state.df, session_id=123, silent=True, verbose=False, html=False)
            elif st.session_state.task_type == "anomaly":
                st.session_state.pycaret_exp = anomaly_setup(data=st.session_state.df, session_id=123, silent=True, verbose=False, html=False)
            elif st.session_state.task_type == "nlp":
                text_col_options = [col for col in st.session_state.df.columns if st.session_state.df[col].dtype == 'object']
                if not text_col_options:
                    st.warning("No text columns found for NLP. Please ensure your dataset has text data.")
                    st.session_state.pycaret_exp = None # Prevent further steps if no text column
                else:
                    text_col = st.sidebar.selectbox("Select Text Column for NLP", text_col_options, key='nlp_text_col')
                    st.session_state.pycaret_exp = nlp_setup(data=st.session_state.df, target=text_col, session_id=123, silent=True, verbose=False, html=False)
            elif st.session_state.task_type == "arules":
                # For Association Rules, need transaction and item columns
                st.warning("Association Rule Mining requires specific data format (transaction ID, item ID). Ensure your data is suitable.")
                transaction_col_options = st.session_state.df.columns.tolist()
                item_col_options = st.session_state.df.columns.tolist()
                
                transaction_col = st.sidebar.selectbox("Select Transaction ID Column", transaction_col_options, key='arules_trans_col')
                item_col = st.sidebar.selectbox("Select Item Column", item_col_options, key='arules_item_col')
                st.session_state.pycaret_exp = arules_setup(data=st.session_state.df, transaction_id=transaction_col, item_id=item_col, session_id=123, silent=True, verbose=False, html=False)

        if st.session_state.pycaret_exp is not None:
            st.success("PyCaret Setup Complete! Data is ready for modeling.")
        else:
            st.error("PyCaret Setup failed. Please check your data and selections.")


    if st.session_state.pycaret_exp is not None:
        if st.session_state.task_type in ["classification", "regression"]:
            st.subheader("Automated Model Selection and Tuning")
            st.info("To speed up execution, PyCaret will compare a limited number of models and tune the best one with fewer iterations by default. For a more exhaustive search, consider running this process offline.")

            if st.button("📊 Compare & Select Best Model (Fast)"):
                with st.spinner("Comparing various machine learning models (fast mode)... This might take a moment."):
                    if st.session_state.task_type == "classification":
                        # Use n_select to quickly get the top model
                        st.session_state.best_model_pycaret = class_compare(n_select=1, turbo=True, verbose=False)
                    else: # regression
                        st.session_state.best_model_pycaret = reg_compare(n_select=1, turbo=True, verbose=False)
                
                if st.session_state.best_model_pycaret is not None:
                    st.success(f"Best model identified: **{st.session_state.best_model_pycaret.__class__.__name__}**")
                    st.write("Here's a summary of the models compared:")
                    comparison_df = st.session_state.pycaret_exp.pull()
                    st.dataframe(comparison_df)
                    
                    # Display core metrics of the best model
                    best_model_row = comparison_df.iloc[0]
                    st.markdown(f"**Best Model ({best_model_row.name}):**")
                    if st.session_state.task_type == "classification":
                        st.markdown(f"- Accuracy: `{best_model_row['Accuracy']:.4f}`")
                        st.markdown(f"- AUC: `{best_model_row['AUC']:.4f}`")
                        st.markdown(f"- F1 Score: `{best_model_row['F1']:.4f}`")
                    else: # regression
                        st.markdown(f"- R2 Score: `{best_model_row['R2']:.4f}`")
                        st.markdown(f"- MAE: `{best_model_row['MAE']:.4f}`")
                        st.markdown(f"- RMSE: `{best_model_row['RMSE']:.4f}`")
                else:
                    st.warning("Could not identify a best model. Check your data or try again.")


            if st.session_state.best_model_pycaret is not None:
                if st.button("✨ Tune Best Model (Fast)"):
                    with st.spinner("Fine-tuning the best model for optimal performance (limited iterations)..."):
                        if st.session_state.task_type == "classification":
                            st.session_state.tuned_model_pycaret = class_tune(st.session_state.best_model_pycaret, optimize='Accuracy', n_iter=5, verbose=False) # Reduced n_iter for speed
                        else: # regression
                            st.session_state.tuned_model_pycaret = reg_tune(st.session_state.best_model_pycaret, optimize='R2', n_iter=5, verbose=False) # Reduced n_iter for speed
                    st.success(f"Model tuned: **{st.session_state.tuned_model_pycaret.__class__.__name__}**")
                    st.write("Here's a summary of the tuning results:")
                    st.dataframe(st.session_state.pycaret_exp.pull()) # Display the tuning results table

                if st.session_state.tuned_model_pycaret is not None:
                    st.subheader("📈 Model Evaluation & Interpretation for Presentation")

                    # Generate and save PyCaret plots to a single PDF
                    if st.button("Generate Model Performance Report (PDF for Client)"):
                        model_report_pdf_name = "pycaret_model_performance_report.pdf"
                        from matplotlib.backends.backend_pdf import PdfPages # Import here to ensure it's available for this function

                        with PdfPages(model_report_pdf_name) as pdf:
                            plot_functions = {
                                "classification": [
                                    ("Area Under Curve (AUC)", "auc"),
                                    ("Confusion Matrix", "confusion_matrix"),
                                    ("Class Prediction Error", "error"),
                                    ("Feature Importance", "feature"),
                                    ("Decision Boundary", "boundary") # Added for classification
                                ],
                                "regression": [
                                    ("Residuals Plot", "residuals"),
                                    ("Prediction Error", "error"),
                                    ("Feature Importance", "feature"),
                                    ("Cooks Distance", "cooks"),
                                    ("Learning Curve", "learning") # Added for regression
                                ]
                            }
                            
                            current_plot_funcs = plot_functions.get(st.session_state.task_type, [])

                            for plot_title, plot_type in current_plot_funcs:
                                try:
                                    st.write(f"Generating '{plot_title}' plot...")
                                    plt.figure(figsize=(10, 7)) # Adjust figure size for PDF
                                    
                                    # PyCaret's plot_model saves to file, then we load and add to PDF
                                    if st.session_state.task_type == "classification":
                                        class_plot(st.session_state.tuned_model_pycaret, plot=plot_type, save=True, verbose=False, display_format='streamlit')
                                    else: # regression
                                        reg_plot(st.session_state.tuned_model_pycaret, plot=plot_type, save=True, verbose=False, display_format='streamlit')
                                    
                                    # PyCaret saves plots with specific names (e.g., 'auc.png', 'confusion_matrix.png')
                                    # For 'feature' plot, it's 'Feature Importance.png'
                                    # For 'boundary' plot, it's 'Decision Boundary.png'
                                    # For 'learning' plot, it's 'Learning Curve.png'
                                    
                                    img_filename_map = {
                                        "auc": "AUC.png",
                                        "confusion_matrix": "Confusion Matrix.png",
                                        "error": "Prediction Error.png",
                                        "feature": "Feature Importance.png",
                                        "boundary": "Decision Boundary.png",
                                        "residuals": "Residuals.png",
                                        "cooks": "Cooks Distance.png",
                                        "learning": "Learning Curve.png"
                                    }
                                    img_path = img_filename_map.get(plot_type, f"{plot_type}.png") # Fallback to default naming

                                    if os.path.exists(img_path):
                                        img = plt.imread(img_path)
                                        fig, ax = plt.subplots(figsize=(10, 7))
                                        ax.imshow(img)
                                        ax.axis('off') # Hide axes
                                        ax.set_title(plot_title)
                                        pdf.savefig(fig)
                                        plt.close(fig)
                                        os.remove(img_path) # Clean up temp image
                                    else:
                                        st.warning(f"Plot file not found for '{plot_title}' ({img_path}). Skipping.")

                                except Exception as e:
                                    st.warning(f"Could not generate '{plot_title}' plot: {e}. Skipping this plot.")
                                    plt.close('all') # Ensure all plots are closed

                            # Interpret model (SHAP) - Summary Plot
                            try:
                                st.write("Generating SHAP interpretation (Feature Importance Summary)...")
                                plt.figure(figsize=(10, 7))
                                if st.session_state.task_type == "classification":
                                    class_interpret(st.session_state.tuned_model_pycaret, plot='summary', save=True, verbose=False, display_format='streamlit')
                                else: # regression
                                    reg_interpret(st.session_state.tuned_model_pycaret, plot='summary', save=True, verbose=False, display_format='streamlit')
                                
                                shap_img_path = "SHAP Summary (Bar).png" # Default name for SHAP summary plot
                                if os.path.exists(shap_img_path):
                                    img = plt.imread(shap_img_path)
                                    fig, ax = plt.subplots(figsize=(10, 7))
                                    ax.imshow(img)
                                    ax.axis('off')
                                    ax.set_title("SHAP Feature Importance Summary")
                                    pdf.savefig(fig)
                                    plt.close(fig)
                                    os.remove(shap_img_path)
                                else:
                                    st.warning(f"SHAP plot file not found: {shap_img_path}. Skipping.")

                            except Exception as e:
                                st.warning(f"Error generating SHAP interpretation: {e}. Please ensure 'shap' is installed (`pip install shap`) and model supports interpretation. Skipping SHAP plot.")
                                plt.close('all')

                        st.success(f"📄 Model Performance Report generated as '{model_report_pdf_name}'.")
                        st.session_state.model_report_pdf = model_report_pdf_name
                        st.info("You can now send this report to your client using the 'Share Reports with Client' section below.")

        elif st.session_state.task_type in ["clustering", "anomaly", "nlp", "arules"]:
            if st.button(f"⚙️ Create & Assign {st.session_state.task_type.title()} Model"):
                with st.spinner(f"Creating and assigning {st.session_state.task_type} model..."):
                    if st.session_state.task_type == "clustering":
                        # Example: create KMeans model with 3 clusters
                        model = cluster_create('kmeans', num_clusters=3, verbose=False)
                        st.session_state.tuned_model_pycaret = cluster_assign(model, verbose=False)
                        st.write("Preview of data with assigned clusters:")
                        st.dataframe(st.session_state.tuned_model_pycaret.head())
                        cluster_plot(model, plot='elbow', save=True, verbose=False) # Example plot
                        st.success("Clustering model created and assigned!")
                    elif st.session_state.task_type == "anomaly":
                        # Example: create Isolation Forest model
                        model = anomaly_create('iforest', verbose=False)
                        st.session_state.tuned_model_pycaret = anomaly_assign(model, verbose=False)
                        st.write("Preview of data with anomaly labels:")
                        st.dataframe(st.session_state.tuned_model_pycaret.head())
                        anomaly_plot(model, plot='ts_lines', save=True, verbose=False) # Example plot
                        st.success("Anomaly Detection model created and assigned!")
                    elif st.session_state.task_type == "nlp":
                        # Example: create LDA model with 5 topics
                        model = nlp_create('lda', num_topics=5, verbose=False)
                        st.session_state.tuned_model_pycaret = nlp_assign(model, verbose=False)
                        st.write("Preview of data with topic assignments:")
                        st.dataframe(st.session_state.tuned_model_pycaret.head())
                        nlp_plot(model, plot='topic_model', save=True, verbose=False) # Example plot
                        st.success("NLP model created and assigned!")
                    elif st.session_state.task_type == "arules":
                        # Association Rules Mining
                        st.session_state.tuned_model_pycaret = arules_create(min_support=0.05, min_confidence=0.1, verbose=False)
                        st.write("Generated Association Rules:")
                        st.dataframe(st.session_state.tuned_model_pycaret)
                        st.success("Association Rules generated!")
                
                # For unsupervised tasks, we might not have a 'model_report_pdf' in the same way,
                # but the output is directly displayed or saved by PyCaret functions.
                st.session_state.model_report_pdf = None # Reset for unsupervised if no specific PDF is generated

# --- Save/Load Model ---
if st.session_state.tuned_model_pycaret is not None:
    st.header("💾 Save & Load Model")
    model_filename = st.text_input("Enter a filename to save/load your model:", value="my_automl_model")

    if st.button("Save Model"):
        with st.spinner(f"Saving '{model_filename}.pkl' (includes preprocessing pipeline)..."):
            if st.session_state.task_type == "classification":
                class_save(st.session_state.tuned_model_pycaret, model_filename)
            elif st.session_state.task_type == "regression":
                reg_save(st.session_state.tuned_model_pycaret, model_filename)
            elif st.session_state.task_type == "clustering":
                cluster_save(st.session_state.tuned_model_pycaret, model_filename)
            elif st.session_state.task_type == "anomaly":
                anomaly_save(st.session_state.tuned_model_pycaret, model_filename)
            elif st.session_state.task_type == "nlp":
                nlp_save(st.session_state.tuned_model_pycaret, model_filename)
            # arules doesn't typically save a 'model' in the same way, its output is the rules dataframe
            st.session_state.model_saved_path = f"{model_filename}.pkl"
        st.success(f"Model saved successfully as '{model_filename}.pkl'!")

    if st.session_state.model_saved_path:
        if st.button("Load Saved Model"):
            with st.spinner(f"Loading '{model_filename}.pkl'..."):
                if st.session_state.task_type == "classification":
                    st.session_state.loaded_model_pycaret = class_load(model_filename)
                elif st.session_state.task_type == "regression":
                    st.session_state.loaded_model_pycaret = reg_load(model_filename)
                elif st.session_state.task_type == "clustering":
                    st.session_state.loaded_model_pycaret = cluster_load(model_filename)
                elif st.session_state.task_type == "anomaly":
                    st.session_state.loaded_model_pycaret = anomaly_load(model_filename)
                elif st.session_state.task_type == "nlp":
                    nlp_load(model_filename) # NLP load doesn't return a model object directly for predict
                    st.session_state.loaded_model_pycaret = "NLP Model Loaded (for inference)" # Placeholder
                # arules doesn't have a load_model function in the same way
            st.success("Model loaded for predictions!")
            if st.session_state.task_type in ["classification", "regression"]:
                st.write(f"Loaded model: **{st.session_state.loaded_model_pycaret.__class__.__name__}**")

# --- Make Predictions ---
if st.session_state.loaded_model_pycaret is not None and st.session_state.task_type in ["classification", "regression"]:
    st.header("🔮 Make New Predictions")
    st.write("Enter values for the features to get a prediction:")

    # Create dynamic input fields based on the original DataFrame's columns (excluding target)
    input_data_for_prediction = {}
    if st.session_state.df is not None and st.session_state.target_col is not None:
        features_for_input = [col for col in st.session_state.df.columns if col != st.session_state.target_col]
        for feature in features_for_input:
            sample_value = st.session_state.df[feature].iloc[0]
            if pd.api.types.is_numeric_dtype(st.session_state.df[feature]):
                input_data_for_prediction[feature] = st.number_input(f"Value for '{feature}'", value=float(sample_value), key=f"pred_input_{feature}")
            else: # Treat as string for categorical input
                input_data_for_prediction[feature] = st.text_input(f"Value for '{feature}'", value=str(sample_value), key=f"pred_input_{feature}")

        new_data_df = pd.DataFrame([input_data_for_prediction])

        if st.button("Get Prediction"):
            with st.spinner("Generating prediction..."):
                if st.session_state.task_type == "classification":
                    predictions = class_predict(st.session_state.loaded_model_pycaret, data=new_data_df, verbose=False)
                    st.success("Prediction Result:")
                    st.write(f"Predicted Category: **{predictions['prediction_label'].iloc[0]}**")
                    st.write(f"Confidence Score: **{predictions['prediction_score'].iloc[0]:.2f}**")
                else: # regression
                    predictions = reg_predict(st.session_state.loaded_model_pycaret, data=new_data_df, verbose=False)
                    st.success("Prediction Result:")
                    st.write(f"Predicted Value: **{predictions['prediction_label'].iloc[0]:.2f}**")
    else:
        st.info("Please upload a dataset and run PyCaret Setup first to enable predictions.")

# --- Email Client Notification ---
st.markdown("---")
st.header("📧 Share Reports with Client")
recipient_email = st.text_input("Enter Client's Email Address for Reports:")

if st.button("📤 Send Reports to Client"):
    if not recipient_email:
        st.warning("Please enter a client email address.")
    elif not is_valid_email(recipient_email):
        st.warning("⚠️ Invalid email address format. Please check and try again.")
        # Auto-reply with clarification message
        auto_reply_body = "Hi,\n\nIt seems the email address you provided for sending reports was not valid. Please double-check the email address and try sending the reports again.\n\nThank you,\nYour AutoML Agent"
        send_email_with_attachments(recipient_email, "Issue with Email Address for Reports", auto_reply_body)
    else:
        eda_pdf_path = None
        model_pdf_path = None
        
        # Generate EDA report if data is available
        if st.session_state.df is not None:
            with st.spinner("Generating data visualization report..."):
                eda_pdf_path = generate_eda_visualizations(st.session_state.df)
            st.success("Data visualization report generated.")
        else:
            st.warning("No data uploaded to generate visual report.")

        # Use the pre-generated model report PDF path if available
        if 'model_report_pdf' in st.session_state and st.session_state.model_report_pdf:
            model_pdf_path = st.session_state.model_report_pdf
        else:
            st.warning("Model performance report not yet generated. Please click 'Generate Model Performance Report (PDF)' first.")

        attachments_to_send = []
        if eda_pdf_path:
            attachments_to_send.append(eda_pdf_path)
        if model_pdf_path:
            attachments_to_send.append(model_pdf_path)

        if attachments_to_send:
            email_subject = "Your Automated Machine Learning Report is Ready!"
            email_body = f"""
Dear Client,

I'm excited to share the results of the automated machine learning analysis on your dataset.

Attached you will find two reports:
1.  **Data Insights Report:** This PDF contains various charts and graphs that help you understand your data at a glance. It shows things like how your data is distributed and the common values in your columns.
2.  **Model Performance Report:** This PDF provides details on the machine learning model built for your task ({st.session_state.task_type.title()}). It includes important charts that show how well the model learned from your data and how reliable its predictions are.

Our system automatically chose the best model for your specific problem and fine-tuned it to achieve the best possible results.

We hope these insights are helpful for your business decisions. Please let us know if you have any questions.

Best regards,

Your AutoML Agent
"""
            if send_email_with_attachments(recipient_email, email_subject, email_body, attachments_to_send):
                st.success(f"✅ Both Visual & Model Reports sent to {recipient_email}.")
            else:
                st.error("Failed to send reports. Please check the logs for details.")
        else:
            st.info("No reports to send. Please ensure data is uploaded and reports are generated.")


# === Email Auto-Responder ===
st.markdown("---")
st.header("📬 Automated Email Responder")
st.write("This agent can check your Gmail inbox for new messages and automatically generate simple, non-technical replies using AI.")

if st.button("📥 Check & Auto-Reply to Latest Unread Email"):
    from_email, subject, body = fetch_latest_email()
    if from_email:
        st.subheader("📨 Incoming Email Details")
        st.markdown(f"**From:** `{from_email}`")
        st.markdown(f"**Subject:** `{subject}`")
        st.text_area("Original Message Content", value=body, height=150, key="incoming_email_body")

        with st.spinner("Generating AI reply..."):
            ai_reply_content = generate_ai_reply(body)
        st.text_area("🤖 AI Generated Reply", value=ai_reply_content, height=180, key="ai_reply_content")

        if st.button("Send AI Reply"):
            if send_email_with_attachments(from_email, f"RE: {subject}", ai_reply_content):
                st.success(f"✅ AI reply sent successfully to `{from_email}`.")
            else:
                st.error("Failed to send AI reply.")
    else:
        st.info("📭 No new unread emails found in your inbox.")

st.markdown("---")
st.markdown("This AutoML Agent is powered by PyCaret and Streamlit.")
