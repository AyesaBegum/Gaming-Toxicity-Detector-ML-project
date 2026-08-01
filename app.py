import streamlit as st
import joblib
import re
import string
import numpy as np
import time

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Gaming Toxicity Detector",
    page_icon="🎮",
    layout="centered"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.stApp{
    background:
    linear-gradient(
        135deg,
        #0f172a 0%,
        #111827 50%,
        #1e293b 100%
    );
}

.main-header{
    text-align:center;
    padding:2rem;
    border-radius:25px;
    background:
    linear-gradient(
        135deg,
        #667eea 0%,
        #764ba2 100%
    );
    box-shadow:0 10px 35px rgba(0,0,0,0.4);
    margin-bottom:2rem;
}

.main-header h1{
    color:white;
    font-size:2.8rem;
    margin-bottom:0.5rem;
}

.main-header p{
    color:#e5e7eb;
    font-size:1rem;
}

.info-box{
    background:rgba(255,255,255,0.08);
    padding:1rem;
    border-radius:15px;
    border-left:5px solid #667eea;
    margin-bottom:1.5rem;
}

.info-box p{
    color:white;
    margin:0.4rem 0;
}

.stTextArea textarea{
    background:white !important;
    color:black !important;
    caret-color:black !important;
    border-radius:15px !important;
    border:2px solid #667eea !important;
    font-size:1rem !important;
    font-weight:500 !important;
}

.stTextArea label{
    color:white !important;
    font-size:1.1rem !important;
    font-weight:700 !important;
}

.stButton > button{
    background:
    linear-gradient(
        135deg,
        #667eea 0%,
        #764ba2 100%
    ) !important;
    color:white !important;
    border:none !important;
    border-radius:50px !important;
    padding:0.9rem 2rem !important;
    font-size:1rem !important;
    font-weight:700 !important;
    width:100% !important;
    transition:0.3s ease !important;
}

.stButton > button:hover{
    transform:translateY(-3px);
    box-shadow:0 8px 25px rgba(102,126,234,0.5);
}

.result-clean{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding:1.5rem; border-radius:20px; text-align:center; margin-top:1rem; animation:fadeIn 0.5s ease; }
.result-mild{ background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); padding:1.5rem; border-radius:20px; text-align:center; margin-top:1rem; animation:fadeIn 0.5s ease; }
.result-moderate{ background: linear-gradient(135deg, #ff9966 0%, #ff5e62 100%); padding:1.5rem; border-radius:20px; text-align:center; margin-top:1rem; animation:fadeIn 0.5s ease; }
.result-severe{ background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); padding:1.5rem; border-radius:20px; text-align:center; margin-top:1rem; animation:fadeIn 0.5s ease; }
.result-extreme{ background: linear-gradient(135deg, #4b0000 0%, #ff0000 100%); padding:1.5rem; border-radius:20px; text-align:center; margin-top:1rem; animation:fadeIn 0.5s ease; box-shadow:0 0 25px rgba(255,0,0,0.7); }

.result-clean h3, .result-mild h3, .result-moderate h3, .result-severe h3, .result-extreme h3{ color:white; margin:0; font-size:1.8rem; }
.result-clean p, .result-mild p, .result-moderate p, .result-severe p, .result-extreme p{ color:white; margin-top:0.7rem; }

@keyframes fadeIn{
    from{ opacity:0; transform:translateY(20px); }
    to{ opacity:1; transform:translateY(0); }
}

.meter-label{ display:flex; justify-content:space-between; margin-top:1rem; color:white; font-size:0.95rem; }
.meter-bar{ width:100%; height:25px; background:rgba(255,255,255,0.2); border-radius:20px; overflow:hidden; margin-top:0.5rem; }
.meter-fill{ height:100%; border-radius:20px; text-align:right; padding-right:10px; color:white; font-weight:bold; line-height:25px; }

.footer{ text-align:center; color:#d1d5db; margin-top:2rem; padding:1rem; font-size:0.9rem; }

</style>
""", unsafe_allow_html=True)


# LOAD MODELS (FIXED: Removed Naive Bayes)


@st.cache_resource
def load_models():
    # Only loading the files found in your folder screenshot
    rf_model = joblib.load('random_forest_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return rf_model, vectorizer, label_encoder

try:
    rf_model, vectorizer, label_encoder = load_models()
except Exception as e:
    st.error(f"❌ Model files not found. Error: {e}")
    st.stop()

# TEXT CLEANING


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# PREDICTION SYSTEM (FIXED: Random Forest Engine)


def predict_message(text):
    cleaned = clean_text(text)

    # 1. SAFETY OVERRIDE
    extreme_keywords = ["kill you", "i will kill", "bomb", "rape", "murder", "die", "terrorist", "hang yourself", "shoot you"]
    if any(word in cleaned for word in extreme_keywords):
        return "extreme", 0.99, "Rule-Based Safety Filter"

    # 2. RANDOM FOREST PREDICTION
    vectorized = vectorizer.transform([cleaned])
    rf_probs = rf_model.predict_proba(vectorized)[0]
    confidence = np.max(rf_probs)
    rf_pred = rf_model.predict(vectorized)[0]

    label = label_encoder.inverse_transform([rf_pred])[0]
    return label, confidence, "Random Forest Ensemble"


# UI CONTENT


st.markdown("""
<div class="main-header">
<h1>🎮 Gaming Toxicity Detector</h1>
<p>AI-powered toxicity moderation for gaming communities</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<p>📌 Detects: Clean, Mild, Moderate, Severe & Extreme Toxicity</p>
<p>🤖 Engine: Random Forest Classifier + Rule-Based Safety Override</p>
</div>
""", unsafe_allow_html=True)

user_input = st.text_area(
    "📝 ENTER GAMING CHAT MESSAGE",
    placeholder="Example: 'Great game everyone!' or 'You are so bad at this'",
    height=150
)

col1, col2, col3 = st.columns([1,2,1])
with col2:
    analyze = st.button("🔍 ANALYZE MESSAGE")

if analyze:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        with st.spinner("Analyzing toxicity..."):
            time.sleep(0.5)
            label, confidence, model_used = predict_message(user_input)

        # Style Logic
        styles = {
            "clean": ("result-clean", "🟢", "Safe gaming message detected", "#38ef7d"),
            "mild": ("result-mild", "🟡", "Minor toxic language detected", "#f2c94c"),
            "moderate": ("result-moderate", "🟠", "Moderate toxicity detected", "#ff9966"),
            "severe": ("result-severe", "🔴", "Severe toxic content detected", "#f45c43"),
            "extreme": ("result-extreme", "🚨", "Extreme dangerous toxicity detected", "#ff0000")
        }
        
        res_class, icon, msg, bar_color = styles.get(label.lower(), styles["clean"])

        st.markdown(f"""
        <div class="{res_class}">
        <h3>{icon} {label.upper()}</h3>
        <p>{msg}</p>
        <p>🤖 Model Used: <strong>{model_used}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="meter-label"><span>📊 Confidence Score</span><span>{confidence*100:.2f}%</span></div>
        <div class="meter-bar"><div class="meter-fill" style="width:{confidence*100}%; background:{bar_color};">{confidence*100:.1f}%</div></div>
        """, unsafe_allow_html=True)

st.markdown("""<div class="footer">🎮 Built for safer gaming communities<br><br>🟢 Clean | 🟡 Mild | 🟠 Moderate | 🔴 Severe | 🚨 Extreme</div>""", unsafe_allow_html=True)