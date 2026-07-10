# Mobile Price Classification System

A machine learning project that predicts the price tier of a mobile phone based on its hardware specifications.

## About
This was a team project built for our MDC (Minor Degree Course) subject. ML model developed by my teammate, presentation and Streamlit deployment by me.

## Price Tiers
- 🟢 Low Cost — Budget tier
- 🟡 Medium Cost — Mid-range tier  
- 🟠 High Cost — Premium tier
- 🔴 Very High Cost — Flagship tier

## Dataset
Mobile price classification dataset with 2000 phones and 20 features including RAM, battery, camera, storage, processor cores, display resolution and more.

## Model
- Algorithm: XGBoost Classifier
- Accuracy: ~90%
- Key feature: RAM is the most important predictor (40% feature importance)

## Tech Stack
- Python, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit

## How to Run
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## Live Demo
https://mobile-price-classification-system-44.streamlit.app/
