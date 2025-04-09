# Cataract-Detection-Binary-Image-Classifier
This project is a deep learning-based solution for classifying eye images into two categories: **Cataract** and **Normal**. It includes data preprocessing, model training using transfer learning, evaluation, and deployment using FastAPI and Streamlit.

---

## 📂 Project Structure

```
├── data/preproceseed_images
│   ├── test/                        # Test images data
|   |      └─cataract/
|   |      └─normal/           
│   ├── train/                       # Train images data
|   |      └─cataract/
|   |      └─normal/     
├── notebooks/
│   ├── datapreprocess.ipynb        # EDA and data preprocessing
│   └── evaluation_test.ipynb       # Model evaluation notebook
├── src/
│   ├── preprocessing.py            # Data augmentation setup
│   └── model.py                    # Model training script
│   └── ml_requirements.txt         # Model training script
├── models/
│   └── best_model_v2_vgg16.h5      # Trained VGG16 model
├── api/
│   ├── main.py                     # FastAPI app
│   └── streamlit_app.py            # Streamlit frontend
│   └── requirements.txt           # Streamlit frontend
│   └── sample_images/             # Sample test images
```
---

## 📊 Dataset Overview

- **Training Set:** 491 images (`cataract`, `normal`)
- **Testing Set:** 121 images (`cataract`, `normal`)
- Balanced across both classes, but relatively small for typical deep learning.
---

## 🧪 Preprocessing

### EDA
- **Image Thresholding** to segment foreground and background
- **Edge detection** to highlight medical features

### Data Augmentation (`src/preprocessing.py`)
Applied with `ImageDataGenerator`:
- Rotation (±10°), shifts (10%), zoom (±10%)
- Brightness adjustment (±15%)
- Horizontal flip (valid in medical context)
- Rescale pixel values to `[0,1]`
---

## 🧠 Model Training (`src/model.py`)

- **Base Model:** VGG16 with transfer learning
- **Fine-tuning:** Last few layers, 10 additional epochs
- **Optimizer:** Adam, LR = 0.0001
- **Callbacks:** Early stopping, model checkpoint
- **Label smoothing** for handling noisy labels

---

## 📈 Evaluation (`notebooks/evaluation_test.ipynb`)

- **Test Accuracy:** 99.95%
- **Precision:** 93.75%
- **Recall:** 100.00%
- **AUC:** 96.69%
- **Optimal Threshold:** `0.60`
  - F1-score: `0.9836`
  - Precision: `0.9677`
  - Recall: `1.0000`
- **Confusion Matrix:** Only 2 false negatives, no false positives for `normal`
- **ROC Curve:** AUC = 1.00

---

## 🚀 Deployment

### FastAPI Backend (`api/main.py`)
- Loads model `best_model_v2_vgg16.h5`
- Preprocesses image (resize to 224x224, normalize)
- API Endpoint: `POST /predict/`
- Returns:
  ```json
  {
    "prediction": "Cataract" or "Normal",
    "confidence": 97.98
  }
  ```

### Example cURL
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@image_253.png;type=image/png'
```

### Streamlit UI (`api/streamlit_app.py`)
- Upload image or test with sample images
- Interacts with FastAPI backend
- Displays prediction label and confidence %

---

## ⚙️ How to Run

### 1. Setup
```bash
cd api/
pip install -r requirements.txt
```

### 2. Run FastAPI Backend
```bash
uvicorn main:app --reload
```
- Visit: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Click on the POST /predict/ endpoint.
- Click "Try it out".
- Use the "Choose File" button to upload your image (.jpg, .png, etc.).
- Click "Execute" to get the prediction with confidence.

### 3. Run Streamlit Frontend
On another terminal, 
```bash
streamlit run streamlit_app.py
```
- Visit: [http://localhost:8501](http://localhost:8501)
- Click on sample image ‘Predict’ button to see results.
- Or, upload an image of your own from app/samples/ folder, then click ‘Predict Uploaded image’ to see the results - Predicted Class with Confidence%.

---

## 📚 Reference

- [Medical Imaging - Thresholding Techniques](https://pmc.ncbi.nlm.nih.gov/articles/PMC5977656/)
- To know more in detail regarding this project, you can view the **Binary Cataract Classification Documentation.pdf** attached. 

---

## 🏁 Conclusion

This project demonstrates the power of transfer learning and careful preprocessing in tackling medical image classification tasks with limited data, backed by an interactive and robust API + UI pipeline.
