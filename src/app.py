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
        with open('style.css', 'r') as f:
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    except:
        # Fallback minimal CSS if file not found
        st.markdown("""
        <style>
            .main-title { font-size: 2.5rem; font-weight: 700; text-align: center; }
            .metric-card { background: white; padding: 1rem; border-radius: 10px; }
            .result-box { padding: 1.5rem; border-radius: 12px; margin: 1.5rem 0; }
            .spam-box { background: #FEE2E2; border-left: 6px solid #DC2626; }
            .ham-box { background: #DCFCE7; border-left: 6px solid #16A34A; }
            .warning-box { background: #FEF3C7; border-left: 6px solid #D97706; }
        </style>
        """, unsafe_allow_html=True)

load_css()

# Title using CSS classes from external file
st.markdown('<h1 class="main-title">🛡️ Spam Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered detection of spam messages with high accuracy</p>', unsafe_allow_html=True)

# Functions
@st.cache_resource
def load_model():
    """Load trained model and vectorizer"""
    try:
        if os.path.exists('models/spam_model.pkl'):
            model = joblib.load('models/spam_model.pkl')
            vectorizer = joblib.load('models/vectorizer.pkl')
            return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return None, None

def enhance_text_for_detection(text):
    """Enhance text to help spam detection"""
    text = str(text).lower().strip()
    
    # Keep URLs for spam detection
    text = re.sub(r'https?://\S+|www\.\S+', ' http_url ', text)
    
    # Keep dollar signs and numbers
    text = re.sub(r'\$(\d+)', r'dollar_\1 ', text)
    
    # Count exclamation marks
    exclamation_count = text.count('!')
    if exclamation_count > 0:
        text = text + ' exclamation_' * min(exclamation_count, 5)
    
    # Spam keywords - FIXED: 'red' to 'reward'
    spam_keywords = [
        'win', 'winner', 'free', 'dollar', 'money', 'prize', 'cash',
        'urgent', 'claim', 'verify', 'congrat', 'won', 'offer', 'limited',
        'selected', 'click', 'call', 'suspended', 'locked', 'account',
        'payment', 'pay', 'fee', 'charge', 'cost', 'price', 'subscribe',
        'buy', 'purchase', 'order', 'transfer', 'deposit', 'wire', 'bank',
        'alert', 'bonus', 'reward', 'gift', 'wow', 'secret', 'exclusive'  # FIXED: 'red' → 'reward'
    ]
    
    # Meeting + payment = automatic spam
    if ('meeting' in text or 'meet' in text) and any(word in text for word in ['pay', 'payment', 'fee', 'dollar', 'money', 'cost']):
        text = text + ' meeting_payment_scam meeting_payment_scam meeting_payment_scam'
    
    # Enhance spam signals
    for keyword in spam_keywords:
        if keyword in text:
            text = text + f' {keyword}_spam {keyword}_spam'
    
    return text

# Load model
model, vectorizer = load_model()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Setup")
    
    if not model:
        st.warning("**Model not trained**", icon="⚠️")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 **Get Data**", use_container_width=True, key="get_data_btn"):
                with st.spinner("Creating dataset..."):
                    try:
                        from create_dataset import save_dataset
                        save_dataset()
                        st.success("Dataset ready!")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            if st.button("🤖 **Train Model**", use_container_width=True, key="train_model_btn"):
                if os.path.exists('data/spam.csv'):
                    with st.spinner("Training AI model..."):
                        import subprocess
                        result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True, shell=True)
                        if result.returncode == 0:
                            st.success("Model trained!")
                            st.cache_resource.clear()
                            st.rerun()
                        else:
                            st.error(f"Training failed:\n{result.stderr[:200]}")
                else:
                    st.error("Get data first")
    else:
        st.success("✅ **Model is ready**", icon="✅")
        
        # Show quick stats
        try:
            with open('models/model_stats.json') as f:
                stats = json.load(f)
            
            st.metric("Accuracy", f"{stats.get('accuracy', 0)*100:.1f}%")
        except:
            pass
    
    st.markdown("---")
    st.markdown("### 🧠 How it works")
    st.caption("""
    This system uses:
    • **TF-IDF** for text features
    • **Logistic Regression** for classification
    • **Enhanced preprocessing** for spam patterns
    • **Real-time** detection
    """)

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Check Your Message")
    
    # Use session state to preserve text
    if 'message' not in st.session_state:
        st.session_state.message = "hi, we have an meeting!!you need to pay 2 DOLLARs first"
    
    message = st.text_area(
        "Enter your message",
        height=140,
        value=st.session_state.message,
        placeholder="Paste your message here...",
        key="message_input",
        label_visibility="collapsed"
    )
    
    # Update session state
    st.session_state.message = message
    
    st.markdown("####  Test Examples")
    
    test_cols = st.columns(4)
    examples = {
        " Prize": "You won $1000! Click to claim!",
        " Urgent": "URGENT: Verify account now!",
        " Normal": "Meeting tomorrow at 3 PM",
        " Payment": "hi, we have an meeting!!you need to pay 2 DOLLARs first"
    }
    
    for i, (name, text) in enumerate(examples.items()):
        if test_cols[i].button(name, use_container_width=True, key=f"example_{name}"):
            st.session_state.message = text
            st.rerun()
    
    if st.button("🔍 **Check for Spam**", type="primary", use_container_width=True, key="check_spam_btn"):
        if not model or not vectorizer:
            st.error(" Please train the model first")
        elif not message.strip():
            st.warning("⚠️ Enter a message")
        else:
            with st.spinner("AI is analyzing..."):
                # Enhanced preprocessing
                enhanced_text = enhance_text_for_detection(message)
                
                # Transform and predict
                try:
                    vec = vectorizer.transform([enhanced_text])
                    prob = model.predict_proba(vec)[0]
                    pred = model.predict(vec)[0]
                    
                    spam_prob = float(prob[1])
                    ham_prob = float(prob[0])
                    
                    # Extra boost for obvious spam patterns
                    spam_boosters = [
                        ('meeting' in message.lower() and any(word in message.lower() for word in ['pay', 'dollar', 'money', 'fee']), 0.4),
                        ('!!!' in message or '!!' in message, 0.15),
                        ('$' in message, 0.2),
                        ('click here' in message.lower() or 'verify now' in message.lower(), 0.25),
                        ('won' in message.lower() and '$' in message, 0.3)
                    ]
                    
                    for condition, boost in spam_boosters:
                        if condition:
                            spam_prob = min(0.99, spam_prob + boost)
                    
                    st.markdown("---")
                    
                    # Display result
                    if spam_prob > 0.7:
                        st.markdown(f'<div class="result-box spam-box">', unsafe_allow_html=True)
                        st.markdown("## 🚨 **SPAM DETECTED**")  # FIXED: Added 🚨
                        st.markdown(f"**Spam confidence:** {spam_prob:.1%}")
                        st.markdown(f"**Ham confidence:** {ham_prob:.1%}")
                        st.markdown("""
                        ⚠️ **Warning:** This appears to be spam!
                        • Do not click links
                        • Do not provide personal info
                        • Consider reporting as spam
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    elif spam_prob > 0.4:
                        st.markdown(f'<div class="result-box warning-box">', unsafe_allow_html=True)
                        st.markdown("##  **SUSPICIOUS**")
                        st.markdown(f"**Spam probability:** {spam_prob:.1%}")
                        st.markdown("""
                        🔍 **Be cautious:**
                        • Verify the sender
                        • Check for odd requests
                        • Look for spelling errors
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-box ham-box">', unsafe_allow_html=True)
                        st.markdown("##  **NOT SPAM**")
                        st.markdown(f"**Legitimate confidence:** {ham_prob:.1%}")
                        st.markdown(f"**Spam probability:** {spam_prob:.1%}")
                        st.markdown("""
                         **Appears safe:** 
                        • No obvious spam indicators
                        • Normal communication patterns
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Progress bars
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Spam score:** {spam_prob:.1%}")
                        st.progress(spam_prob)
                    with col_b:
                        st.write(f"**Ham score:** {ham_prob:.1%}")
                        st.progress(ham_prob)
                    
                    # Show what triggered
                    with st.expander(" Analysis details"):
                        st.write(f"**Original text:** `{message}`")
                        st.write(f"**Enhanced text:** `{enhanced_text[:200]}...`")
                        st.write(f"**Prediction:** {'SPAM' if pred == 1 else 'HAM'}")
                        
                        # Show detected spam indicators
                        spam_indicators = []
                        if 'meeting' in message.lower() and any(word in message.lower() for word in ['pay', 'dollar', 'money']):
                            spam_indicators.append("Meeting + Payment")
                        if '$' in message:
                            spam_indicators.append("Dollar sign")
                        if '!!!' in message or '!!' in message:
                            spam_indicators.append("Exclamation marks")
                        if any(word in message.lower() for word in ['win', 'winner', 'prize', 'free']):
                            spam_indicators.append("Prize/free offer")
                        
                        if spam_indicators:
                            st.write(f"**Spam indicators:** {', '.join(spam_indicators)}")
                        else:
                            st.write("**Spam indicators:** None detected")
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {e}")  # FIXED: Added ❌

with col2:
    st.markdown("###  Model Performance")
    
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
                
                # F1 Score display - decide whether to keep or remove
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats.get('f1_score', 0)*100:.1f}%</div>
                    <div class="metric-label">F1 Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Simple status indicator
            accuracy = stats.get('accuracy', 0)
            if accuracy > 0.9:
                st.success("Excellent performance")
            elif accuracy > 0.8:
                st.info("⚡ Good performance")
            elif accuracy > 0.7:
                st.warning(" Moderate performance")
            else:
                st.error(" Needs improvement")
            
        except Exception as e:
            st.info("📊 Performance data not available")
            st.caption("Train model to see metrics")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    
    with st.expander("How to spot spam"):
        st.write("""
        **🚨 Red flags (likely spam):**
        • Urgent action required (VERIFY NOW!)
        • Requests for payment/money
        • "You won!" or "Free!" offers
        • Account suspension threats
        • Poor grammar/spelling
        • Multiple exclamation marks!!!
        • Suspicious links
        
        **✅ Normal messages (likely ham):**  # FIXED: Added ✅
        • Clear, professional language
        • Known sender/organization
        • Reasonable requests
        • Proper grammar
        • No pressure tactics
        
        **🔍 When in doubt:**
        1. Verify sender email address
        2. Don't click unknown links
        3. Check URL carefully
        4. Contact organization directly
        5. Use this spam detector!
        """)

# Footer
st.markdown("---")
st.markdown(
    '<div class="footer">'
    '🛡️ Spam Detector v2.0 • Powered by AI & NLP • Stay safe online'
    '</div>',
    unsafe_allow_html=True
)

# Add some space at bottom
st.markdown("<br><br>", unsafe_allow_html=True)