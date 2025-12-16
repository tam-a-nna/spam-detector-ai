import streamlit as st
import joblib
import json
import re
import os

# Page config
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
def load_css():
    try:
        with open('static/style.css', 'r') as f:
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except:
        st.markdown("""
        <style>
            .main-title {
                font-size: 2.5rem;
                font-weight: 700;
                color: #1E40AF;
                text-align: center;
            }
        </style>
        """, unsafe_allow_html=True)

# Load CSS
load_css()

# Title using CSS classes from external file
st.markdown('<h1 class="main-title">🛡️ Spam Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered detection of spam messages with high accuracy</p>', unsafe_allow_html=True)

# Functions
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/spam_model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        return model, vectorizer
    except:
        return None, None

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return ' '.join(text.split())

# Load model
model, vectorizer = load_model()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Setup")
    
    if not model:
        st.warning("**Model not trained**", icon="⚠️")
        
        if st.button("📥 **Download Dataset**", use_container_width=True):
            with st.spinner("Downloading..."):
                import subprocess
                result = subprocess.run(["python", "create_dataset.py"], capture_output=True, text=True)
                if "Dataset saved" in result.stdout:
                    st.success("Dataset ready!")
                else:
                    st.error("Check connection")
        
        if st.button("🤖 **Train Model**", use_container_width=True):
            if os.path.exists('data/spam.csv'):
                with st.spinner("Training..."):
                    import subprocess
                    result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
                    if "Model trained" in result.stdout:
                        st.success("Model trained!")
                        st.cache_resource.clear()
                        st.rerun()
            else:
                st.error("Download dataset first")
    else:
        st.success("✅ **Model is ready**", icon="✅")

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Check Your Message")
    
    message = st.text_area(
        "Enter your message",
        height=140,
        value="your account has been suspended. Click here to verify details!",
        placeholder="Paste your message here...",
        label_visibility="collapsed"
    )
    
    st.markdown("#### 🧪 Test Examples")
    
    test_cols = st.columns(4)
    examples = {
        "💰 Prize": "You won $1000! Click to claim!",
        "📧 Phishing": "Account locked. Click to verify!",
        "📅 Normal": "Meeting tomorrow at 3 PM",
        "⚠️ Urgent": "URGENT: Verify account now!"
    }
    
    for i, (name, text) in enumerate(examples.items()):
        if test_cols[i].button(name, use_container_width=True):
            st.session_state.message = text
    
    if st.button("🔍 **Check for Spam**", type="primary", use_container_width=True):
        if not model:
            st.error("Please train the model first")
        elif not message.strip():
            st.warning("Enter a message")
        else:
            with st.spinner("Analyzing..."):
                clean = clean_text(message)
                vec = vectorizer.transform([clean])
                prob = model.predict_proba(vec)[0]
                pred = model.predict(vec)[0]
                
                spam_prob = float(prob[1])
                
                if any(word in message.lower() for word in ['suspended', 'locked', 'verify', 'click']):
                    spam_prob = min(0.95, spam_prob + 0.2)
                
                st.markdown("---")
                
                if spam_prob > 0.65:
                    st.markdown(f'<div class="result-box spam-box">', unsafe_allow_html=True)
                    st.markdown("### 🚨 **SPAM DETECTED**")
                    st.markdown(f"**Confidence:** {spam_prob:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                elif spam_prob > 0.4:
                    st.markdown(f'<div class="result-box warning-box">', unsafe_allow_html=True)
                    st.markdown("### ⚠️ **SUSPICIOUS**")
                    st.markdown(f"**Spam probability:** {spam_prob:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-box ham-box">', unsafe_allow_html=True)
                    st.markdown("### ✅ **NOT SPAM**")
                    st.markdown(f"**Confidence:** {(1-spam_prob):.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.progress(spam_prob)
                st.caption(f"Spam score: {spam_prob:.1%}")

with col2:
    st.markdown("### 📊 Model Performance")
    
    if model:
        try:
            with open('models/model_stats.json') as f:
                stats = json.load(f)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats.get('accuracy', 0)*100:.1f}%</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats.get('precision', 0)*100:.1f}%</div>
                    <div class="metric-label">Precision</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats.get('recall', 0)*100:.1f}%</div>
                    <div class="metric-label">Recall</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats.get('f1_score', 0)*100:.1f}%</div>
                    <div class="metric-label">F1 Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.caption(f"Trained on {stats.get('total_samples', 0):,} messages")
            
        except:
            st.info("Performance data not available")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    
    with st.expander("How to spot spam"):
        st.write("""
        **Red flags:**
        • Urgent action required
        • Suspicious links
        • Account threats
        • Too good to be true offers
        • Poor grammar/spelling
        
        **Stay safe:**
        • Verify sender
        • Don't click unknown links
        • Check URL carefully
        • Contact directly if unsure
        """)

# Footer
st.markdown("---")
st.markdown(
    '<div class="footer">'
    '🛡️ Spam Detector • Powered by AI • Stay safe online'
    '</div>',
    unsafe_allow_html=True
)

# Handle message updates
if 'message' in st.session_state:
    message = st.session_state.message
    del st.session_state.message