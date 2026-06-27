import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import time
import os

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Acoustic Shield | Smart City",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Enterprise UI, Benchmark Cards & Larger Tabs
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0;}
    .sub-header { font-size: 1.2rem; color: #64748B; margin-top: 0; margin-bottom: 2rem;}
    
    /* تكبير حجم التبويبات Tabs */
    button[data-baseweb="tab"] p {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #1E293B;
    }
    
    /* Benchmark Cards Styling */
    .benchmark-card {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #3B82F6;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s ease-in-out;
    }
    .benchmark-card:hover { transform: translateY(-3px); }
    .b-title { font-size: 1.1rem; font-weight: 600; color: #1E293B; margin-bottom: 5px;}
    .b-acc { font-size: 1.8rem; font-weight: 800; color: #10B981; margin: 10px 0;}
    .b-desc { font-size: 0.9rem; color: #64748B;}
    
    /* Highlighted Team Card */
    .team-card {
        border-left: 6px solid #8B5CF6;
        background: linear-gradient(145deg, #F8FAFC 0%, #F1F5F9 100%);
    }
    .team-acc { color: #8B5CF6; }
    
    /* Multiple Files Container */
    .file-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR (PROJECT IDENTITY)
# ==========================================
with st.sidebar:
    logo_path = "smart city logo.jpg"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.warning("⚠️ Logo file 'smart city logo.jpg' not found in directory.")
        
    st.title("🛡️ Acoustic Shield")
    st.caption("Privacy-First Urban Safety System")
    st.divider()
    
    st.subheader("👨‍💻 Engineering Team")
    st.markdown("- **Faris Alsulami**")
    st.markdown("- **Hussain Mash**")
    st.markdown("- **Abdulrahim Alharbi**")
    
    st.divider()
    st.info("🎓 **CS4507 - Pattern Recognition**\n\nUmm Al-Qura University")

# ==========================================
# 3. CORE LOGIC (CACHED FOR SPEED)
# ==========================================
@st.cache_resource
def load_acoustic_model():
    return load_model('robust_model.keras')

try:
    model = load_acoustic_model()
except Exception as e:
    st.error("⚠️ System Error: 'robust_model.keras' not found in the directory.")
    st.stop()

def extract_spatial_features(audio_path):
    y, sr = librosa.load(audio_path, res_type='kaiser_fast', duration=3.0)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    if mel_spec_db.shape[1] < 128:
        pad_width = 128 - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :128]
        
    return y, sr, mel_spec_db

# ==========================================
# 4. MAIN DASHBOARD UI (TABS)
# ==========================================
st.markdown('<p class="main-header">Intelligent Urban Acoustic Monitoring</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Driven Emergency Detection without Visual Surveillance.</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎙️ Live System Inference", "📊 Research Benchmarks & Methodology"])

# ------------------------------------------
# TAB 1: LIVE INFERENCE (The Testing Environment)
# ------------------------------------------
with tab1:
    st.markdown("### 🧪 Real-time Audio Diagnostics (Batch Processing)")
    uploaded_files = st.file_uploader("📂 Drop multiple .wav files from the urban environment...", type=["wav"], accept_multiple_files=True)

    if uploaded_files:
        st.success(f"✅ Successfully loaded {len(uploaded_files)} audio file(s). Ready for analysis.")
        
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(f'<div class="file-container">', unsafe_allow_html=True)
            st.markdown(f"#### 🎵 Audio Source: `{uploaded_file.name}`")
            
            # مشغل الصوت يظهر دائماً
            st.audio(uploaded_file, format='audio/wav')
            
            # زر التحليل (يجب أن يكون له key مميز لكل ملف)
            analyze_button = st.button(f"🔍 Run AI Analysis for {uploaded_file.name}", key=f"btn_{idx}")
            
            if analyze_button:
                with st.spinner(f'🔄 AI is analyzing spatial acoustic patterns...'):
                    time.sleep(0.8) # إضافة تشويق بسيط
                    
                    try:
                        audio, sr, mel_spec_db = extract_spatial_features(uploaded_file)
                        input_data = mel_spec_db.reshape(1, 128, 128, 1)
                        prediction = model.predict(input_data)
                        confidence = prediction[0][0] * 100
                        
                        col1, col2 = st.columns([1, 1.5])
                        
                        with col1:
                            st.subheader("📊 Threat Assessment")
                            if prediction[0][0] >= 0.5:
                                st.error("🚨 EMERGENCY DETECTED")
                                st.metric(label="Threat Confidence", value=f"{confidence:.2f}%", delta="High Priority", delta_color="inverse")
                                st.markdown("**Action:** Alerting nearest first responders.")
                            else:
                                st.success("✅ ENVIRONMENT SECURE")
                                st.metric(label="Safety Confidence", value=f"{100 - confidence:.2f}%", delta="Normal Routine")
                                st.markdown("**Action:** No action required.")
                            
                            with st.expander("⚙️ Technical Inference Details"):
                                st.write(f"- **Input Shape:** `(128, 128, 1)`")
                                st.write(f"- **Algorithm:** Robust CNN (Augmented)")
                                st.write(f"- **Raw Output Prob:** `{prediction[0][0]:.6f}`")

                        with col2:
                            st.subheader("👁️ AI Sensory Vision (Mel-Spectrogram)")
                            fig, ax = plt.subplots(figsize=(8, 4))
                            img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap='turbo')
                            fig.colorbar(img, ax=ax, format='%+2.0f dB')
                            ax.set_title(f'Extracted Features')
                            fig.patch.set_alpha(0.0)
                            ax.patch.set_alpha(0.0)
                            st.pyplot(fig)
                            plt.close(fig)
                            
                    except Exception as e:
                        st.error(f"Failed to process {uploaded_file.name}. Error: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")

# ------------------------------------------
# TAB 2: BENCHMARKS & EXPERIMENTS
# ------------------------------------------
with tab2:
    st.markdown("### 📚 Historical Experiments vs. Our Implementation")
    st.write("A comparative analysis of environmental sound classification breakthroughs using the UrbanSound8K dataset.")
    
    col_hist, col_team = st.columns(2)
    
    with col_hist:
        st.markdown("""
        <div class="benchmark-card">
            <div class="b-title">🧑‍🔬 Piczak's Baseline (2015)</div>
            <div class="b-desc">Pioneered the use of CNNs on environmental audio by treating Mel-Spectrograms as image data, moving away from traditional 1D analysis.</div>
            <div class="b-acc" style="color: #64748B;">73.1%</div>
            <div class="b-desc"><i>Model: Basic CNN without deep augmentations.</i></div>
        </div>
        
        <div class="benchmark-card">
            <div class="b-title">🧑‍🔬 Salamon & Bello (2017)</div>
            <div class="b-desc">Proved the necessity of Data Augmentation (time-stretching, pitch-shifting) to make models robust against urban noise variability.</div>
            <div class="b-acc" style="color: #3B82F6;">79.0%</div>
            <div class="b-desc"><i>Model: CNN with extensive data augmentation techniques.</i></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_team:
        st.markdown("""
        <div class="benchmark-card team-card">
            <div class="b-title">🎯 Our Privacy-Preserving Approach (Current)</div>
            <div class="b-desc">By focusing the architecture exclusively on <b>Emergency Isolation</b> (Sirens & Gunshots vs Background) for smart cities, combined with modern robust in-layer augmentations (RandomZoom, RandomTranslation).</div>
            <div class="b-acc team-acc">97.02%</div>
            <div class="b-desc"><i>Model: Deep Robust CNN + BatchNormalization.</i></div>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    st.subheader("🔬 Core System Pipeline")
    st.info("1️⃣ **Extraction:** Raw audio converted to 128x128 Mel-Spectrogram matrices.\n\n2️⃣ **Augmentation:** Dynamic pitch/time shifting to ensure environmental resilience.\n\n3️⃣ **Classification:** Deep CNN outputs binary probability (Emergency vs Normal) without capturing visual or spoken privacy data.")