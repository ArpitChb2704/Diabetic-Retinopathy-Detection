# 🧠 Diabetic Retinopathy Classification using Deep Learning

This project presents a Deep Learning-based solution for automatic classification of **Diabetic Retinopathy (DR)** from retinal fundus images. The project includes a user-friendly **Streamlit web application** that allows image upload, prediction, Grad-CAM visualization, and **PDF report generation**.

---

## 🚀 Features

- ✅ DR classification into severity levels (No DR, Mild, Moderate, Severe, Proliferative)
- 🧾 Downloadable PDF report of predictions
- 📊 Accuracy, AUC, confusion matrix, and training history
- 🌐 Streamlit web app for interactive use
- 🧠 EfficientNetB0-based CNN model (ImageNet pre-trained)

---

## Model. Summary

- Model: EfficientNetB0 (with ImageNet weights)
- Input Size: 224x224x3
- Loss: Categorical Crossentropy
- Optimizer: Adam
- Training Data: APTOS / Kaggle DR dataset (preprocessed)
- Validation Accuracy: ~92%
- Evaluation Metrics: Accuracy, AUC, Confusion Matrix

## Dataset

The model was trained on the publicly available APTOS 2019 Blindness Detection dataset from Kaggle.

## Snapshots

![image](https://github.com/user-attachments/assets/8c6a4abd-de7a-4d19-84a7-251fcf30013c)
![image](https://github.com/user-attachments/assets/7a67703f-c535-47dc-bbfd-c1e1bfa24464)



