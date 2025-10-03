import os
import streamlit as st
import pandas as pd
import joblib


# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Page background */
    .stApp {
        background-color: #f0f8ff;
    }

    /* Title */
    .stTitle {
        color: #2c3e50;
        font-weight: bold;
        font-size: 36px;
    }

    
    /* Form container */
    .stForm {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Subheaders */
    .stSubheader {
        color: #34495e;
        font-weight: bold;
    }

    /* Expander */
    .stExpander{
        background-color: white;    
    }

    /* Success / Prediction box */
    .stAlert {
        background-color: #dff9fb !important;
        color: #130f40 !important;
        border-radius: 10px;
        padding: 10px;
    </style>
    """, unsafe_allow_html=True
)


# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "SVC.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "le.pkl")


# Load model + encoder (cached)
@st.cache_resource
def load_artifacts(model_path, encoder_path):
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        return None, None
    
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder

model, encoder = load_artifacts(MODEL_PATH, ENCODER_PATH)


# Streamlit Config
st.set_page_config(page_title="🌸 Iris Classifier", layout="centered")


# Title & Description
st.markdown("<h1 class='stTitle'>🌸 Iris Species Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='stSubheader'>A <b>Machine Learning App</b> to classify Iris flowers into Setosa, Versicolor, or Virginica.</p>", 
    unsafe_allow_html=True
)


# Error handling with friendly messages
if model is None or encoder is None:
    st.error("❌ Oops! The model or encoder is missing. Please make sure 'SVC.pkl' and 'label_encoder.pkl' are in the 'models/' folder.")
    st.stop()
else:
    st.success("🎉 Welcome! The model and encoder are loaded successfully. You are ready to make predictions!")


# User Input Form
with st.form("iris_form"):
    st.subheader("🔢 Enter Flower Measurements")
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1, value=5.1)
    sepal_width  = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1, value=3.5)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1, value=1.4)
    petal_width  = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1, value=0.2)

    submitted = st.form_submit_button("🔍 Predict")

# Prediction
if submitted:
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    pred_num = model.predict(input_data)[0]
    prediction = encoder.inverse_transform([pred_num])[0]

    st.subheader("🌿 Prediction Result")
    st.success(f"**Predicted Iris Species:** {prediction}")


    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
        prob_df = pd.DataFrame({
            "Species": encoder.inverse_transform(model.classes_),
            "Probability": proba
        }).sort_values(by="Probability", ascending=False)

        with st.expander("📊 Prediction Confidence"):   
            st.bar_chart(prob_df.set_index("Species"))


# Footer
st.markdown("---")
st.caption("Built with ❤️ by Thiruselvan M | Powered by Streamlit & Scikit-Learn")
