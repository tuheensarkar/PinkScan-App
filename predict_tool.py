import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import os
from deep_translator import GoogleTranslator
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

# Ensure set_page_config is the very first Streamlit command
st.set_page_config(page_title="PinkScan - Predict Tool", page_icon=":hospital:", layout="wide")

# Cache model loading
@st.cache_resource
def load_model():
    return joblib.load('breast_cancer_model.pkl')

# Load model
model = load_model()

# Healthy profile
healthy_profile = [
    12.0, 18.0, 78.0, 450.0, 0.09, 0.07, 0.03, 0.02, 0.16, 0.06,
    0.4, 1.0, 2.5, 20.0, 0.005, 0.01, 0.015, 0.005, 0.015, 0.002,
    13.5, 22.0, 88.0, 550.0, 0.11, 0.14, 0.08, 0.04, 0.22, 0.07
]

# Feature names and ranges
feature_names = [
    "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", "Mean Smoothness",
    "Mean Compactness", "Mean Concavity", "Mean Concave Points", "Mean Symmetry", "Mean Fractal Dimension",
    "Radius Error", "Texture Error", "Perimeter Error", "Area Error", "Smoothness Error",
    "Compactness Error", "Concavity Error", "Concave Points Error", "Symmetry Error", "Fractal Dimension Error",
    "Worst Radius", "Worst Texture", "Worst Perimeter", "Worst Area", "Worst Smoothness",
    "Worst Compactness", "Worst Concavity", "Worst Concave Points", "Worst Symmetry", "Worst Fractal Dimension"
]

feature_ranges = {
    "Mean Radius": (6.0, 28.0), "Mean Texture": (9.0, 39.0), "Mean Perimeter": (43.0, 188.0), "Mean Area": (143.0, 2501.0),
    "Mean Smoothness": (0.05, 0.16), "Mean Compactness": (0.02, 0.35), "Mean Concavity": (0.0, 0.43),
    "Mean Concave Points": (0.0, 0.2), "Mean Symmetry": (0.1, 0.3), "Mean Fractal Dimension": (0.05, 0.1),
    "Radius Error": (0.1, 2.5), "Texture Error": (0.36, 4.9), "Perimeter Error": (0.76, 21.98),
    "Area Error": (6.8, 542.2), "Smoothness Error": (0.001, 0.03), "Compactness Error": (0.002, 0.14),
    "Concavity Error": (0.0, 0.4), "Concave Points Error": (0.0, 0.05), "Symmetry Error": (0.008, 0.08),
    "Fractal Dimension Error": (0.001, 0.03), "Worst Radius": (7.9, 36.04), "Worst Texture": (12.02, 49.54),
    "Worst Perimeter": (50.41, 251.2), "Worst Area": (185.2, 4254.0), "Worst Smoothness": (0.07, 0.22),
    "Worst Compactness": (0.02, 1.06), "Worst Concavity": (0.0, 1.25), "Worst Concave Points": (0.0, 0.29),
    "Worst Symmetry": (0.16, 0.66), "Worst Fractal Dimension": (0.06, 0.21)
}

LANGUAGES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'bn': 'Bengali', 'hi': 'Hindi'
}

# Apply light theme CSS after set_page_config
st.markdown("""
<style>
.stApp {
    background-color: #fff;
    color: #000000;
}
.stButton>button {
    background-color: #ff69b4;
    color: #fff;
    border-radius: 25px;
    padding: 10px 20px;
    border: none;
    font-weight: bold;
    font-size: 1rem;
}
.stButton>button:hover {
    background-color: #ff4d9e;
    color: #fff;
}
.stHeader {
    color: #ff69b4;
    font-size: 2.5rem;
    text-align: center;
    font-weight: bold;
}
.stSubheader {
    color: #000000;
    font-size: 1.5rem;
}
.stText {
    color: #808080;
}
.stWarning {
    background-color: #ffebee;
    border: 1px solid #ffcdd2;
    border-radius: 5px;
    padding: 10px;
    color: #000000;
}
.stSuccess {
    background-color: #e8f5e9;
    border: 1px solid #c8e6c9;
    border-radius: 5px;
    padding: 10px;
    color: #000000;
}
.stError {
    background-color: #ffebee;
    border: 1px solid #ffcdd2;
    border-radius: 5px;
    padding: 10px;
    color: #000000;
}
.chatbot-container {
    border: 1px solid #ff69b4;
    border-radius: 10px;
    padding: 15px;
    margin-top: 20px;
    background-color: #fff;
}
.chatbot-message {
    margin: 10px 0;
    padding: 10px;
    border-radius: 5px;
    background-color: #f8f9fa;
    color: #000000;
}
.chatbot-user {
    background-color: #e9ecef;
    text-align: right;
    color: #000000;
}
.chatbot-bot {
    background-color: #fff;
    text-align: left;
    color: #000000;
}
.chatbot-message strong {
    color: #ff69b4;
    font-weight: bold;
    margin-right: 5px;
}
.sidebar .sidebar-content {
    background-color: #f8f9fa;
    color: #000000;
}
.sidebar .sidebar-content h2, .sidebar .sidebar-content h3 {
    color: #ff69b4;
}
.sidebar .sidebar-content p, .sidebar .sidebar-content li {
    color: #000000;
}
</style>
""", unsafe_allow_html=True)

# Translation functions
@st.cache_data
def translate_text(text, lang):
    translator = GoogleTranslator(source='auto', target=lang)
    try:
        return translator.translate(text)
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

def translate(_text):
    language = st.session_state.get('language', 'English')
    return translate_text(_text, list(LANGUAGES.keys())[list(LANGUAGES.values()).index(language)])

@st.cache_data
def predict_breast_cancer(input_features):
    input_features = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_features)
    probability = model.predict_proba(input_features)[0]
    return "Malignant" if prediction[0] == 0 else "Benign", probability[0] * 100

@st.cache_data
def generate_pdf_report(patient_id, result, input_features, healthy_profile, risk_score):
    pdf = FPDF()
    pdf.add_page()
    
    left_margin = 20
    top_margin = 20
    right_margin = 20
    pdf.set_margins(left=left_margin, top=top_margin, right=right_margin)
    pdf.set_auto_page_break(auto=True, margin=20)
    
    pdf.set_line_width(0.5)
    pdf.rect(10, 10, 190, 277)
    
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Breast Cancer Prediction Report", 0, 1, "C")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 8, f"Patient ID: {patient_id}", 0, 1)
    pdf.cell(0, 8, f"Diagnosis: {result}", 0, 1)
    pdf.cell(0, 8, f"Risk Score: {risk_score:.2f}% likelihood of malignant tumor", 0, 1)
    pdf.ln(10)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Feature Comparison", 0, 1)
    pdf.set_font("Helvetica", size=10)
    
    col_widths = [60, 50, 50]
    headers = ["Feature Name", "Patient Value", "Healthy Value"]
    pdf.set_fill_color(220, 220, 220)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 8, header, 1, 0, "C", 1)
    pdf.ln(8)
    
    pdf.set_fill_color(255, 255, 255)
    for i, (name, patient_val, healthy_val) in enumerate(zip(feature_names, input_features, healthy_profile)):
        pdf.cell(col_widths[0], 8, name, 1)
        pdf.cell(col_widths[1], 8, f"{patient_val:.2f}", 1, 0, "C")
        pdf.cell(col_widths[2], 8, f"{healthy_val:.2f}", 1, 0, "C")
        pdf.ln(8)
    
    pdf.add_page()
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Doctor's Recommendations", 0, 1)
    pdf.set_font("Helvetica", size=10)
    tips = [
        "Schedule regular mammograms as recommended by your doctor",
        "Perform monthly breast self-examinations",
        "Maintain a healthy weight through diet and exercise",
        "Limit alcohol consumption",
        "Quit smoking and avoid secondhand smoke"
    ]
    for tip in tips:
        pdf.cell(0, 6, tip.encode('latin-1', 'replace').decode('latin-1'), 0, 1)
    
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Recommended Dietary Routine", 0, 1)
    pdf.set_font("Helvetica", size=10)
    diet = [
        "Eat plenty of fruits and vegetables (5+ servings daily)",
        "Choose whole grains over refined grains",
        "Include lean proteins (fish, poultry, beans)",
        "Limit processed and red meats",
        "Use healthy fats (olive oil, avocados, nuts)"
    ]
    for item in diet:
        pdf.cell(0, 6, item.encode('latin-1', 'replace').decode('latin-1'), 0, 1)
    
    pdf.ln(10)
    
    histogram_path = f"histogram_{patient_id}.png"
    pie_chart_path = f"pie_chart_{patient_id}.png"
    
    try:
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        sns.histplot(input_features, bins=10, kde=False, color="blue", label="Patient", ax=ax1)
        sns.histplot(healthy_profile, bins=10, kde=False, color="green", label="Healthy", ax=ax1)
        ax1.legend(fontsize=8)
        ax1.tick_params(axis='both', which='major', labelsize=8)
        plt.title("Feature Distribution", fontsize=10)
        plt.tight_layout()
        fig1.savefig(histogram_path, bbox_inches='tight', dpi=100)
        plt.close(fig1)

        labels = ["Malignant" if result == "Malignant" else "", "Benign" if result == "Benign" else "", "Healthy"]
        sizes = [33.33 if result == "Malignant" else 0, 33.33 if result == "Benign" else 0, 33.33]
        colors = ["#FF6347", "#4682B4", "#32CD32"]
        explode = (0.1 if result == "Malignant" else 0, 0.1 if result == "Benign" else 0, 0)
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', 
                                          shadow=True, startangle=90, textprops={'fontsize': 8, 'color': 'black'})
        ax2.axis("equal")
        plt.setp(autotexts, size=8, weight="bold", color="white")
        plt.title("Prediction Breakdown", fontsize=10)
        plt.tight_layout()
        fig2.savefig(pie_chart_path, bbox_inches='tight', dpi=100)
        plt.close(fig2)

        y_position = pdf.get_y()
        pdf.image(histogram_path, x=left_margin, y=y_position, w=80)
        pdf.image(pie_chart_path, x=left_margin + 90, y=y_position, w=80)
    except Exception as e:
        st.error(f"Error generating graphs for PDF: {e}")
    finally:
        for file_path in [histogram_path, pie_chart_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError as e:
                    st.warning(f"Could not remove temporary file {file_path}: {e}")

    pdf_file = f"report_{patient_id}.pdf"
    pdf.output(pdf_file)
    return pdf_file

@st.cache_data
def save_patient_history(patient_id, result, risk_score):
    history_file = 'patient_history.csv'
    if os.path.exists(history_file):
        try:
            history_df = pd.read_csv(history_file, encoding='utf-8')
        except UnicodeDecodeError:
            history_df = pd.read_csv(history_file, encoding='latin-1')
    else:
        history_df = pd.DataFrame(columns=['patient_id', 'prediction', 'risk_score', 'date'])

    new_entry = pd.DataFrame({
        'patient_id': [patient_id],
        'prediction': [result],
        'risk_score': [risk_score],
        'date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
    })
    history_df = pd.concat([history_df, new_entry], ignore_index=True)
    history_df.to_csv(history_file, index=False, encoding='utf-8')

@st.cache_data
def get_patient_history(patient_id=None):
    history_file = 'patient_history.csv'
    if os.path.exists(history_file):
        try:
            history_df = pd.read_csv(history_file, encoding='utf-8')
        except UnicodeDecodeError:
            history_df = pd.read_csv(history_file, encoding='latin-1')
        if patient_id:
            return history_df[history_df['patient_id'] == patient_id]
        return history_df
    return pd.DataFrame(columns=['patient_id', 'prediction', 'risk_score', 'date'])

@st.cache_data
def send_email(email, patient_id, result, risk_score, pdf_file):
    sender_email = "tuheensarkarofficial@gmail.com"  # Replace with your Gmail address
    app_password = "edwb arob jqis cwjc"  # Replace with your App Password (16-character code)

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = email
    msg['Subject'] = f"Breast Cancer Prediction Report for {patient_id}"
    msg.attach(MIMEText(f"Prediction: {result}\nRisk Score: {risk_score:.2f}% likelihood of malignant tumor"))

    try:
        with open(pdf_file, "rb") as f:
            part = MIMEApplication(f.read(), Name=pdf_file)
            part['Content-Disposition'] = f'attachment; filename="{pdf_file}"'
            msg.attach(part)

        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, app_password)
            server.send_message(msg)
            st.success("Email sent successfully using port 587!")
            server.quit()
        except Exception as e:
            st.error(f"Failed to send email using port 587: {str(e)}")
            try:
                server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
                server.ehlo()
                server.login(sender_email, app_password)
                server.send_message(msg)
                st.success("Email sent successfully using port 465!")
                server.quit()
            except Exception as e2:
                st.error(f"Failed to send email using port 465: {str(e2)}")
                st.error("Please check your Gmail credentials, App Password, and network settings.")
    except Exception as e:
        st.error(f"Error preparing email or attaching file: {str(e)}")

def main():
    # Language selection
    if 'language' not in st.session_state:
        st.session_state.language = 'English'
    language = st.sidebar.selectbox("Select Language", options=list(LANGUAGES.values()), 
                                   index=list(LANGUAGES.values()).index(st.session_state.language))
    st.session_state.language = language

    st.title(translate("PinkScan - Breast Cancer Prediction Tool"))
    st.write(translate(
        "Stay ahead with early detection and smart decisions. This AI-powered tool predicts whether a breast tumor is malignant or benign based on input features and compares it with a typical healthy (benign) profile."
    ))

    st.sidebar.header(translate("About PinkScan"))
    st.sidebar.write(translate(
        "PinkScan uses a Random Forest Classifier trained on the Breast Cancer Wisconsin (Diagnostic) Dataset. Malignant: Cancerous tumor. Benign: Non-cancerous tumor."
    ))

    with st.sidebar.expander(translate("Learn More About Breast Cancer")):
        st.markdown("""
            <h2 style='color: #FF69B4;'>Understanding Breast Cancer</h2>
            <p style='color: #333333;'>
                Breast cancer is a type of cancer that develops in the cells of the breasts. It is the most common cancer among women worldwide, but it can also affect men.
            </p>
            <h3 style='color: #FF69B4;'>Symptoms</h3>
            <ul style='color: #333333;'>
                <li>A lump or thickening in the breast or underarm.</li>
                <li>Changes in the size, shape, or appearance of the breast.</li>
                <li>Nipple discharge or inversion.</li>
                <li>Redness or pitting of the breast skin (like an orange peel).</li>
            </ul>
            <h3 style='color: #FF69B4;'>Prevention</h3>
            <ul style='color: #333333;'>
                <li>Maintain a healthy weight.</li>
                <li>Exercise regularly.</li>
                <li>Limit alcohol consumption.</li>
                <li>Avoid smoking.</li>
            </ul>
        """, unsafe_allow_html=True)

    patient_id = st.text_input(translate("Enter Patient ID"), placeholder="e.g., PAT12345")

    feature_groups = {
        translate("Mean Features"): feature_names[:10],
        translate("Error Features"): feature_names[10:20],
        translate("Worst Features"): feature_names[20:]
    }

    input_features = []
    for group_name, features in feature_groups.items():
        with st.expander(f"{group_name}"):
            for feature in features:
                min_val, max_val = feature_ranges[feature]
                value = st.number_input(f"{feature}", min_value=float(min_val), max_value=float(max_val), 
                                       value=float(min_val), help=f"Enter a value between {min_val} and {max_val}")
                if value < min_val or value > max_val:
                    st.warning(f"Value for {feature} is outside typical range ({min_val}-{max_val}).")
                input_features.append(value)

    if st.button(translate("Predict")):
        if not patient_id:
            st.warning(translate("Please enter a Patient ID."))
        else:
            try:
                result, risk_score = predict_breast_cancer(input_features)
                if result == "Malignant":
                    st.error(f"Patient ID: {patient_id}\n{translate('Prediction')}: {result}")
                else:
                    st.success(f"Patient ID: {patient_id}\n{translate('Prediction')}: {result}")

                st.write(f"{translate('Risk Score')}: {risk_score:.2f}% {translate('likelihood of malignant tumor')}")

                current_patient = pd.DataFrame({
                    'patient_id': [patient_id],
                    'prediction': [result],
                    'risk_score': [risk_score],
                    'date': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
                })
                st.subheader(translate("Current Patient Details"))
                st.write(current_patient)

                st.header(translate("Data Analytics"))
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(translate("Feature Distribution"))
                    @st.cache_data
                    def plot_histogram(patient_data, healthy_data):
                        fig, ax = plt.subplots(figsize=(4, 2))
                        sns.histplot(patient_data, bins=10, kde=False, color="blue", label="Patient", ax=ax)
                        sns.histplot(healthy_data, bins=10, kde=False, color="green", label="Healthy", ax=ax)
                        ax.legend(fontsize=6)
                        ax.tick_params(axis='both', which='major', labelsize=6)
                        plt.tight_layout()
                        return fig
                    fig1 = plot_histogram(input_features, healthy_profile)
                    st.pyplot(fig1)
                    plt.close(fig1)

                with col2:
                    st.subheader(translate("Prediction Comparison"))
                    @st.cache_data
                    def plot_pie(result):
                        labels = [translate("Malignant"), translate("Benign"), translate("Healthy")]
                        sizes = [1 if result == "Malignant" else 0, 1 if result == "Benign" else 0, 1]
                        colors = ["#FF6347", "#4682B4", "#32CD32"]
                        fig, ax = plt.subplots(figsize=(4, 2))
                        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", 
                               startangle=90, textprops={'fontsize': 6, 'color': 'black'})
                        ax.axis("equal")
                        plt.tight_layout()
                        return fig
                    fig2 = plot_pie(result)
                    st.pyplot(fig2)
                    plt.close(fig2)

                st.subheader(translate("Feature Importance"))
                @st.cache_data
                def plot_feature_importance():
                    importances = model.feature_importances_
                    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(5)
                    fig, ax = plt.subplots(figsize=(4, 2))
                    sns.barplot(x='importance', y='feature', data=feature_importance_df, ax=ax, color="#FF69B4")
                    plt.tight_layout()
                    return fig
                fig3 = plot_feature_importance()
                st.pyplot(fig3)
                plt.close(fig3)

                st.subheader(translate("Download PDF Report"))
                pdf_file = generate_pdf_report(patient_id, result, input_features, healthy_profile, risk_score)
                with open(pdf_file, "rb") as file:
                    st.download_button(
                        label=translate("Download Report"),
                        data=file,
                        file_name=pdf_file,
                        mime="application/pdf",
                        key="download_button"
                    )

                save_patient_history(patient_id, result, risk_score)
                st.write(translate("Prediction saved to patient history."))

                email = st.text_input(translate("Enter your email for results (optional)"))
                if st.button(translate("Send Results via Email")) and email:
                    send_email(email, patient_id, result, risk_score, pdf_file)

            except Exception as e:
                st.error(f"Prediction error: {e}")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.header(translate("Patient History"))
    with col2:
        if st.button(translate("View")):
            history_df = get_patient_history()
            if not history_df.empty:
                st.subheader(translate("Full Patient History"))
                st.write(history_df)
            else:
                st.write(translate("No patient history available."))

    if patient_id:
        history = get_patient_history(patient_id)
        if not history.empty:
            st.write(translate("Recent History for This Patient:"))
            st.write(history)
        else:
            st.write(translate("No history found for this patient."))

    # Chatbot Section (Restored previous version with full header and history)
    st.header(translate("Ask About Breast Cancer"))
    st.write(translate("Interact with PinkScan Bot for instant answers about breast cancer. Ask anything!"))

    # Chat history and input
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    st.subheader(translate("Chat History"))
    chatbot_container = st.container()
    with chatbot_container:
        for message in st.session_state.chat_history:
            username = "You" if message["user"] else "PinkScan Bot"
            message_class = "chatbot-user" if message["user"] else "chatbot-bot"
            st.markdown(f'<div class="chatbot-message {message_class}"><strong style="color: #ff69b4;">{username}:</strong> {message["text"]}</div>', unsafe_allow_html=True)

    # Chat input
    user_input = st.text_input(translate("Type your question here"), key="chat_input", help="Ask any question about breast cancer")
    if st.button(translate("Send"), key="chat_send"):
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"text": user_input, "user": True})
            # Get chatbot response
            response = get_chatbot_response(user_input)
            # Add chatbot response to chat history
            st.session_state.chat_history.append({"text": response, "user": False})
            # Clear input and rerun
            st.rerun()

    st.header(translate("Batch Prediction"))
    uploaded_file = st.file_uploader(translate("Upload a CSV file with feature values"), type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Normalize column names: strip quotes, extra spaces, and standardize case
            df.columns = [col.strip("'").strip().title() for col in df.columns]
            # Convert expected feature names to title case for consistency
            feature_columns = [col.title() for col in feature_names]
            st.write(translate("Uploaded Data (Normalized Column Names):"))
            st.write(df)

            if st.button(translate("Predict Batch")):
                if "Patient Id" not in df.columns:  # Check for title case
                    st.error(translate("The uploaded file must contain a 'Patient ID' column (case-insensitive)."))
                else:
                    missing_columns = [col for col in feature_columns if col not in df.columns]
                    if missing_columns:
                        st.error(translate(f"The uploaded file is missing the following required columns: {missing_columns}"))
                        st.write(translate("Required columns (case-normalized):"))
                        st.write(feature_columns)
                        st.write(translate("Actual columns in uploaded file:"))
                        st.write(list(df.columns))
                    else:
                        # Check for non-numeric values or NaNs
                        if not df[feature_columns].apply(lambda x: pd.to_numeric(x, errors='coerce').notnull().all()).all():
                            st.error(translate("All feature values must be numeric. Please check the CSV for invalid entries."))
                        else:
                            predictions = model.predict(df[feature_columns].values)
                            probabilities = model.predict_proba(df[feature_columns].values)[:, 0] * 100
                            df["Prediction"] = ["Malignant" if p == 0 else "Benign" for p in predictions]
                            df["Risk Score (%)"] = probabilities
                            st.write(translate("Predictions:"))
                            st.write(df)
                            # Option to download batch results as CSV
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label=translate("Download Batch Predictions"),
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv",
                                key="batch_download"
                            )
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.write(translate("Please ensure the CSV is properly formatted with all required columns."))

    with st.form(translate("feedback_form")):
        feedback = st.text_area(translate("Provide feedback or suggestions"))
        if st.form_submit_button(translate("Submit Feedback")):
            with open("feedback.txt", "a") as f:
                f.write(f"Patient ID: {patient_id}, Feedback: {feedback}, Date: {pd.Timestamp.now()}\n")
            st.success(translate("Thank you for your feedback!"))

# Simple rule-based chatbot responses (restored previous version)
def get_chatbot_response(user_input):
    user_input = user_input.lower().strip()
    responses = {
        "hello": "Hello! How can I assist you with breast cancer information today?",
        "hi": "Hi there! What would you like to know about breast cancer?",
        "treatment": "Common treatments include surgery, radiation, chemotherapy, and hormone therapy. Please consult your doctor for personalized advice.",
        "prevention": "Prevention includes regular screenings, maintaining a healthy diet, exercising regularly, and avoiding smoking/alcohol.",
        "symptoms": "Symptoms include a lump in the breast or underarm, changes in breast size/shape, nipple discharge, or skin redness/pitting. Consult a doctor if you notice these.",
        "cure": "There’s no universal cure, but early detection and treatment (e.g., surgery, therapy) can manage or eliminate breast cancer. Consult a healthcare professional.",
        "what is breast cancer": "Breast cancer is a type of cancer that forms in the cells of the breasts, often as a lump or tumor. It’s common among women but can affect men too.",
        "help": "I can help with information on breast cancer symptoms, prevention, treatment, and more. What would you like to know?",
        "default": "I'm sorry, I don’t understand that. Please ask about breast cancer symptoms, prevention, treatment, or say 'help' for options."
    }
    for keyword in responses:
        if keyword in user_input:
            return responses[keyword]
    return responses["default"]

if __name__ == "__main__":
    main()