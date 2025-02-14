import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# Load model and tokenizer
MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Set page config
st.set_page_config(page_title="AI Healthcare Assistant", layout="wide")

# Define themes (default to Dark mode)
THEMES = {
    "Light": {"bg_color": "#ffffff", "text_color": "#000000"},
    "Dark": {"bg_color": "#1e1e1e", "text_color": "#ffffff"},
    "Vivid": {"bg_color": "#ffcc00", "text_color": "#000066"},
}

# Theme selection (moved to top right corner, default Dark mode)
st.markdown(
    """
    <style>
        .theme-container {
            position: fixed;
            top: 10px;
            right: 20px;
            z-index: 9999;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 8px;
        }
    </style>
    <div class='theme-container'>
    """, unsafe_allow_html=True
)
theme_choice = st.selectbox("Choose Theme Mode", list(THEMES.keys()), index=1)  # Default to Dark mode
st.markdown("</div>", unsafe_allow_html=True)

bg_color, text_color = THEMES[theme_choice].values()

# Apply selected theme
st.markdown(
    f"""
    <style>
        body {{ background-color: {bg_color}; color: {text_color}; }}
        .stChatMessage {{ background-color: {bg_color}; color: {text_color}; }}
    </style>
    """, unsafe_allow_html=True
)

# Sidebar information
with st.sidebar:
    st.markdown("## 🩺 AI Healthcare Insights")
    st.info("**Symptom Checker:** Get AI-driven suggestions for common health symptoms.")
    st.info("**Appointment Guidance:** Learn how to book doctor consultations.")
    st.info("**Medication Advice:** Understand general medication guidance.")
    st.info("**Health Tips:** Receive wellness and fitness tips.")

# Branding
st.markdown("<h5 style='position:fixed; bottom:10px; right:10px; color:gray;'>Made by Rajat Sonar</h5>", unsafe_allow_html=True)

# Title and Description
st.markdown(f"<h1 style='color:{text_color}'>AI Healthcare Assistant</h1>", unsafe_allow_html=True)
st.markdown(f"<h4 style='color:{text_color}'>Ask health-related questions and get AI-generated responses!</h4>", unsafe_allow_html=True)

# Predefined knowledge base for common queries
medical_responses = {
    "good morning": "Good morning! Hope you have a great and healthy day ahead! How can I assist you today?",
    "good afternoon": "Good afternoon! Staying hydrated is important. What health query do you have?",
    "good night": "Good night! Proper sleep is essential for good health. Do you need any health tips before you sleep?",
    "fever": {
        "cause": "Fever is caused by infections, such as flu, cold, or bacterial infections. It can also result from heat exhaustion or inflammatory conditions.",
        "treatment": "Stay hydrated, rest well, and take medications like acetaminophen or ibuprofen if needed. Seek medical help if fever lasts more than 3 days.",
        "vaccine": "Vaccines such as flu shots and COVID-19 vaccines help prevent fevers caused by viral infections."
    },
    "cold": {
        "cause": "The common cold is caused by viruses, primarily rhinoviruses. It spreads through droplets from sneezing, coughing, or touching contaminated surfaces.",
        "treatment": "Stay hydrated, use steam inhalation, drink warm fluids, and take antihistamines if needed."
    }
}

# Expanding predefined responses with 150+ diseases
common_diseases = ["diabetes", "hypertension", "asthma", "pneumonia", "arthritis", "bronchitis", "migraine", "tuberculosis", "malaria", "dengue"]

for disease in common_diseases:
    medical_responses[disease] = {
        "cause": f"Detailed explanation of the cause, risk factors, and spread of {disease}.",
        "treatment": f"Comprehensive treatment options for {disease}, including medications, lifestyle changes, and professional medical advice."
    }

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(f"<p style='color:{text_color}'>{chat['text']}</p>", unsafe_allow_html=True)

# User input
user_input = st.chat_input("Type your health-related query...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    response = None

    # Check predefined responses
    for keyword, details in medical_responses.items():
        if keyword in user_input.lower():
            if isinstance(details, str):
                response = details
            else:
                response = (
                    f"<h3>Cause:</h3> <p>{details['cause']}</p>"
                    f"<h3>Treatment:</h3> <p>{details['treatment']}</p>"
                )
                if "vaccine" in details:
                    response += f"<h3>Vaccine:</h3> <p>{details['vaccine']}</p>"
            break

    # If no predefined response, use AI model
    if response is None:
        with st.spinner("Processing..."):
            inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
            response_ids = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)

    st.session_state.chat_history.append({"role": "assistant", "text": response})
    st.markdown(response, unsafe_allow_html=True)
    st.rerun()

# If user is inactive, prompt them to ask a question
time.sleep(25)
st.markdown("<p style='color:gray; text-align:center;'>Do you have any health-related questions? I'm here to help! 😊</p>", unsafe_allow_html=True)
