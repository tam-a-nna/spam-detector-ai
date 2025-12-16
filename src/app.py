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

# Beautiful CSS
st.markdown("""
<style>
    /* Main title */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E40AF, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #6B7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s;
        margin-bottom: 0.5rem;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E40AF;
        margin-bottom: 0.2rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Text area */
    .stTextArea textarea {
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        font-size: 1rem;
        padding: 1rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .test-button {
        background: #F3F4F6 !important;
        color: #374151 !important;
        border: 1px solid #D1D5DB !important;
    }
    
    .test-button:hover {
        background: #E5E7EB !important;
        transform: translateY(-1px);
    }
    
    /* Result boxes */
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border-left: 6px solid;
        animation: fadeIn 0.5s;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .spam-box {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border-left-color: #DC2626;
    }
    
    .ham-box {
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        border-left-color: #16A34A;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left-color: #D97706;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3B82F6, #1E40AF);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Title
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

# Sidebar - Simple setup
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
    # Input section
    st.markdown("### 📝 Check Your Message")
    
    message = st.text_area(
        "",
        height=140,
        value="your account has been suspended. Click here to verify details!",
        placeholder="Paste your message here...",
        label_visibility="collapsed"
    )
    
    # Test buttons - smaller and cleaner
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
    
    # Check button
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
                
                # Keyword boost
                if any(word in message.lower() for word in ['suspended', 'locked', 'verify', 'click']):
                    spam_prob = min(0.95, spam_prob + 0.2)
                
                # Display result
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
                
                # Progress bar
                st.progress(spam_prob)
                st.caption(f"Spam score: {spam_prob:.1%}")

with col2:
    # Model performance - smaller font
    st.markdown("### 📊 Model Performance")
    
    if model:
        try:
            with open('models/model_stats.json') as f:
                stats = json.load(f)
            
            # Create metric cards with smaller font
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
            
            # Small dataset info
            st.caption(f"Trained on {stats.get('total_samples', 0):,} messages")
            
        except:
            st.info("Performance data not available")
    
    # Tips section
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
    '<div style="text-align: center; color: #6B7280; font-size: 0.9rem; padding: 1rem;">'
    '🛡️ Spam Detector • Powered by AI • Stay safe online'
    '</div>',
    unsafe_allow_html=True
)


if 'message' in st.session_state:
    message = st.session_state.message
    del st.session_state.message