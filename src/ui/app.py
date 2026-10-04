import os
import requests
import streamlit as st

# read the API location from the environment, fall back to the local dev server
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Chest Cancer Classifier", page_icon="🫁")
st.title("Chest Cancer Classification")
st.write("Upload a chest CT image to classify it.")

uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    # 1. show the image
    st.image(uploaded, caption=uploaded.name, use_container_width=True)

    if st.button("Predict"):
        # 2. send it to the API as multipart form data, field name must match api.py ("file")
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        try:
            with st.spinner("Classifying..."):
                response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the API at {API_URL}: {e}")
            st.stop()

        # 3. display the result
        result = response.json()
        st.success(f"Prediction: **{result['prediction']}**")
        st.metric("Confidence", f"{result['confidence']:.2%}")

        st.subheader("Class breakdown")
        st.bar_chart(result["class_breakdown"])