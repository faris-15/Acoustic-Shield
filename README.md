# 🛡️ Acoustic Shield: Privacy-Preserving Urban Acoustic AI

**Computer Science Department**  
**Course:** CS4507 – Pattern Recognition (Section 1)  
**Institution:** Umm Al-Qura University

---

## 📌 Project Overview

This repository contains the source code, pre-trained neural network weights, and an interactive Streamlit dashboard for the **Acoustic Shield** project.

The system leverages a robust Convolutional Neural Network (CNN) to classify environmental audio into a binary framework (**Emergency** vs. **Normal**) to assist Smart City infrastructure without relying on privacy-invasive visual surveillance.

### ✨ Key Features

- Binary Classification: Accurately isolates Emergency threats (Sirens & Gunshots) from regular urban background noise.
- Privacy-First Approach: Processes only acoustic spatial features.
- Batch Processing Dashboard built with Streamlit.
- High Accuracy: Validation accuracy of **97.02%**.

---

## 📂 Repository Structure

```text
SmartCity_AcousticProject/
│
├── app.py
├── robust_model.keras
├── smart_city_logo.jpg
├── README.md
└── Demo_Sounds/
```

## ⚙️ System Requirements

- Python 3.8+
- pip

## 📦 Installation

```bash
pip install streamlit tensorflow-cpu librosa numpy matplotlib resampy
```

## 🚀 Run the Application

```bash
streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

## 🎙️ Usage

1. Open the dashboard.
2. Navigate to **Live System Inference**.
3. Upload one or more `.wav` files.
4. Click **Run AI Analysis**.
5. View the prediction and Mel-Spectrogram visualization.

---

## 🔬 Model Architecture

### Input

- 128 × 128 × 1 Mel-Spectrogram

### Data Augmentation

- RandomTranslation
- RandomZoom

### Feature Extraction

- Conv2D
- BatchNormalization
- MaxPooling2D

(Repeated for 3 convolutional blocks)

### Classifier

```text
Flatten
 ↓
Dense(128)
 ↓
Dropout(0.25)
 ↓
Dense(1)
 ↓
Sigmoid
```

## 📊 Performance

| Metric | Value |
|----------|---------|
| Validation Accuracy | 97.02% |
| Classes | 2 |
| Framework | TensorFlow / Keras |

---

## 👨‍💻 Engineering Team

| Name |
|--------|
| Faris Mohammed AlSulami |
| Hussain Mash | 
| Abdulrahim Alharbi |



---

## 📜 License

This project was developed for academic purposes only.

---

**Developed for Academic Purposes – 2026**
