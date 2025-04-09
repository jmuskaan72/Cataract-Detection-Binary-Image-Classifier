import streamlit as st
import requests
from PIL import Image
import os
import io

st.title("Cataract Detection")
st.write("Upload an eye image or select a sample image to predict if it's Cataract or Normal.")

col1, col2 = st.columns(2, gap='large')

with col1:
    # === Sample Images Section ===
    st.subheader("Try with Sample Images")

    sample_images_dir = "sample_images"
    sample_files = [f for f in os.listdir(sample_images_dir) if f.endswith((".jpg", ".jpeg", ".png"))][:5]

    images_per_row = 2
    rows = [sample_files[i:i + images_per_row] for i in range(0, len(sample_files), images_per_row)]

    for row in rows:
        cols = st.columns(images_per_row)
        for i, file in enumerate(row):
            with cols[i]:
                image_path = os.path.join(sample_images_dir, file)
                image = Image.open(image_path)
                st.image(image, caption=file, use_container_width=True)
                if st.button(f"Predict {file}", key=f"predict_{file}"):
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    buffered.seek(0)
                    files = {"file": (file, buffered.read(), "image/png")}
                    response = requests.post("http://127.0.0.1:8000/predict/", files=files)

                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Prediction: {result['prediction']} (Confidence: {result['confidence']}%)")
                    else:
                        st.error("Prediction failed. Try again.")

with col2:
    # === Upload Image Section ===
    st.subheader("Or Upload Your Own Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict Uploaded Image"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post("http://127.0.0.1:8000/predict/", files=files)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['prediction']} (Confidence: {result['confidence']}%)")
            else:
                st.error("Prediction failed. Try again.")
