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
![alt text](https://file%2B.vscode-resource.vscode-cdn.net/Users/apsarg/Desktop/diabetic_retinopathy_app/DR/app.png?version%3D1751298788594)

![alt text](https://file%2B.vscode-resource.vscode-cdn.net/Users/apsarg/Desktop/diabetic_retinopathy_app/DR/report.png?version%3D1751298793873)