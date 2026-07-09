import streamlit as st
import numpy as np
import pickle
import pandas as pd
import json

# Set page config for premium look
st.set_page_config(
    page_title="Mobile Price Intelligence Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for glassmorphic premium look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main Styles */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #0a0c10;
        color: #e6edf3;
    }
    
    /* Heading Styling */
    h1, h2, h3, .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
    
    /* Glassmorphic Container */
    .glass-card {
        background: rgba(22, 27, 34, 0.8);
        border-radius: 12px;
        border: 1px solid #30363d;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 20px;
    }
    
    /* Metric styling */
    .metric-title {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        color: #58a6ff;
    }
    
    /* Prediction Cards */
    .prediction-card {
        border-radius: 12px;
        padding: 25px;
        color: white;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .budget-card {
        background: linear-gradient(135deg, #1b4d3e 0%, #1e2922 100%);
        border: 1px solid #2ecc71;
    }
    .mid-card {
        background: linear-gradient(135deg, #4d3f1b 0%, #29261e 100%);
        border: 1px solid #f1c40f;
    }
    .premium-card {
        background: linear-gradient(135deg, #4d2b1b 0%, #29211e 100%);
        border: 1px solid #e67e22;
    }
    .flagship-card {
        background: linear-gradient(135deg, #4d1b1b 0%, #291e1e 100%);
        border: 1px solid #e74c3c;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load metadata and model
@st.cache_resource
def load_assets():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    feature_names = pickle.load(open("features.pkl", "rb"))
    
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
        
    return model, scaler, feature_names, metadata

try:
    model, scaler, feature_names, metadata = load_assets()
except Exception as e:
    st.error(f"Error loading model or metadata. Please make sure train_model.py has been run. Details: {e}")
    st.stop()

# Load Dataset for Comparative Stats
@st.cache_data
def load_dataset():
    return pd.read_csv("train.csv")

df_train = load_dataset()

# Price Predictor Engine Content
st.markdown("<h1>Mobile Price Predictor Engine</h1>", unsafe_allow_html=True)
st.write("Enter the key specifications of the mobile phone below to predict its price range tier.")

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Initialize input_data with metadata defaults
input_data = {}
for col in feature_names:
    input_data[col] = metadata["features"][col]["default"]
    
with col1:
    st.subheader("Core Performance")
    
    # RAM
    ram_meta = metadata["features"]["ram"]
    input_data["ram"] = st.number_input(
        "RAM (Memory) (MB)", 
        min_value=int(ram_meta["min"]), 
        max_value=int(ram_meta["max"]), 
        value=int(ram_meta["default"]), 
        step=1
    )
    
    # Internal Storage
    mem_meta = metadata["features"]["int_memory"]
    input_data["int_memory"] = st.number_input(
        "Internal Storage (GB)", 
        min_value=int(mem_meta["min"]), 
        max_value=int(mem_meta["max"]), 
        value=int(mem_meta["default"]), 
        step=1
    )
    
    # Processor Cores
    cores_meta = metadata["features"]["n_cores"]
    input_data["n_cores"] = st.number_input(
        "Processor Cores (1-8)", 
        min_value=int(cores_meta["min"]), 
        max_value=int(cores_meta["max"]), 
        value=int(cores_meta["default"]),
        step=1
    )
    
    # Weight
    wt_meta = metadata["features"]["mobile_wt"]
    input_data["mobile_wt"] = st.number_input(
        "Mobile Weight (grams)", 
        min_value=int(wt_meta["min"]), 
        max_value=int(wt_meta["max"]), 
        value=int(wt_meta["default"]),
        step=1
    )
    
with col2:
    st.subheader("Display, Battery & Camera")
    
    # Battery Power
    bat_meta = metadata["features"]["battery_power"]
    input_data["battery_power"] = st.number_input(
        "Battery Capacity (mAh)", 
        min_value=int(bat_meta["min"]), 
        max_value=int(bat_meta["max"]), 
        value=int(bat_meta["default"]), 
        step=1
    )
    
    # Pixel Resolution (Height & Width)
    px_h_meta = metadata["features"]["px_height"]
    input_data["px_height"] = st.number_input(
        "Pixel Height Resolution", 
        min_value=int(px_h_meta["min"]), 
        max_value=int(px_h_meta["max"]), 
        value=int(px_h_meta["default"]), 
        step=1
    )
    
    px_w_meta = metadata["features"]["px_width"]
    input_data["px_width"] = st.number_input(
        "Pixel Width Resolution", 
        min_value=int(px_w_meta["min"]), 
        max_value=int(px_w_meta["max"]), 
        value=int(px_w_meta["default"]), 
        step=1
    )
    
    # Primary Camera
    pc_meta = metadata["features"]["pc"]
    input_data["pc"] = st.number_input(
        "Primary Camera (Megapixels)", 
        min_value=int(pc_meta["min"]), 
        max_value=int(pc_meta["max"]), 
        value=int(pc_meta["default"]),
        step=1
    )
    
# Expander showing other hidden features
with st.expander("Advanced Details (Using Average Dataset Defaults Behind the Scenes)"):
    advanced_rows = []
    hidden_cols = [c for c in feature_names if c not in ["ram", "battery_power", "int_memory", "n_cores", "mobile_wt", "px_height", "px_width", "pc"]]
    for col in hidden_cols:
        col_val = input_data[col]
        display_val = "Yes" if col_val == 1 and metadata["features"][col]["type"] == "binary" else (
            "No" if col_val == 0 and metadata["features"][col]["type"] == "binary" else str(col_val)
        )
        col_name_clean = col.replace("_", " ").title()
        advanced_rows.append({"Specification": col_name_clean, "Applied Default Value": display_val})
    st.table(pd.DataFrame(advanced_rows))
    
st.markdown("</div>", unsafe_allow_html=True)

# Trigger Prediction
if st.button("Predict Mobile Price Class", use_container_width=True):
    # Ensure feature alignment
    input_df = pd.DataFrame([input_data])[feature_names]
    
    # Scale and Predict
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    
    # Visual styling based on predicted price tier
    classes = {
        0: {"title": "Low Cost Class", "class_name": "budget-card", "desc": "Budget-friendly tier. Equivalent to basic utility and entry-level specifications."},
        1: {"title": "Medium Cost Class", "class_name": "mid-card", "desc": "Mid-range tier. Offers balanced performance and specifications for everyday work/life."},
        2: {"title": "High Cost Class", "class_name": "premium-card", "desc": "Premium tier. Higher hardware capabilities, improved camera, and multi-tasking features."},
        3: {"title": "Very High Cost Class", "class_name": "flagship-card", "desc": "Flagship tier. Ultra-premium hardware specs, maximal RAM, battery capacity, and display quality."}
    }
    
    selected_class = classes[pred]
    
    st.markdown(f"""
    <div class="prediction-card {selected_class['class_name']}">
        <h2 style='color: white; margin: 0;'>Predicted Tier: {selected_class['title']}</h2>
        <p style='font-size: 1.1rem; color: #eceff4; margin-top: 10px; margin-bottom: 20px;'>
            {selected_class['desc']}
        </p>
        <hr style='border-color: rgba(255,255,255,0.1);'>
        <p style='font-size: 0.9rem; color: #d8dee9; margin: 0;'>
            <strong>Prediction Confidence Metric:</strong> Machine Learning Model reports XGBoost cross-validated test accuracy of ~90%.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparative Analysis with predicted class
    st.subheader("Comparative Intelligence")
    class_subset = df_train[df_train["price_range"] == pred]
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        avg_ram = class_subset["ram"].mean()
        st.metric("Your RAM vs Avg for this Tier", f"{input_data['ram']} MB", f"{input_data['ram'] - avg_ram:+.0f} MB relative to avg ({avg_ram:.0f} MB)")
    with col_c2:
        avg_bat = class_subset["battery_power"].mean()
        st.metric("Your Battery vs Avg for this Tier", f"{input_data['battery_power']} mAh", f"{input_data['battery_power'] - avg_bat:+.0f} mAh relative to avg ({avg_bat:.0f} mAh)")
    with col_c3:
        avg_wt = class_subset["mobile_wt"].mean()
        st.metric("Your Weight vs Avg for this Tier", f"{input_data['mobile_wt']} g", f"{input_data['mobile_wt'] - avg_wt:+.0f} g relative to avg ({avg_wt:.0f} g)", delta_color="inverse")
