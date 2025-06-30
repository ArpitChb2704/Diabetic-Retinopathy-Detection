import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
from fpdf import FPDF
import tempfile
from datetime import datetime
import requests
from gen_pdf import generate_pdf



# Load model once at start
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("DR_model_1.h5")

model = load_model()
# Class names and descriptions
class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
class_descriptions = {
    "No DR": "No signs of diabetic retinopathy.",
    "Mild": "Early signs; microaneurysms may be present.",
    "Moderate": "More damage to blood vessels, possible hemorrhages.",
    "Severe": "Extensive retinal damage, risk of vision loss.",
    "Proliferative DR": "Growth of new blood vessels; high risk of blindness.",
}

# Example images
example_images = {
    "Example 1": "https://utils.vipas.ai/vipas-images/diabetic_retinopathy_test_images/0.jpeg",
    "Example 2": "https://utils.vipas.ai/vipas-images/diabetic_retinopathy_test_images/1.jpeg",
    "Example 3": "https://utils.vipas.ai/vipas-images/diabetic_retinopathy_test_images/2.jpeg",
    "Example 4": "https://utils.vipas.ai/vipas-images/diabetic_retinopathy_test_images/3.jpeg",
}

# Image preprocessing
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize
    if image.shape[-1] == 4:  # Convert RGBA to RGB
        image = image[:, :, :3]
    return np.expand_dims(image, axis=0)

# UI
st.title("🩺 Diabetic Retinopathy Classifier")
st.write("Upload or choose a sample image to detect diabetic retinopathy severity.")

# Get optional patient details
name = st.text_input("Patient Name (Optional)", "")
age = st.text_input("Age (Optional)", "")
gender = st.selectbox("Gender (Optional)", ["", "Male", "Female", "Other"])


option = st.selectbox("Select Image Source:", ["Upload an Image", "Use Example Image"])
selected_image = None

if option == "Use Example Image":
    image_choice = st.selectbox("Choose an Example Image:", list(example_images.keys()))
    image_url = example_images[image_choice]
    response = requests.get(image_url)
    selected_image = Image.open(BytesIO(response.content))
elif option == "Upload an Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        selected_image = Image.open(uploaded_file)

if selected_image:
    col1, col2 = st.columns(2)

    with col1:
        st.image(selected_image, caption="Selected Image", use_column_width=True)

    with col2:
        st.markdown("### Prediction Result:")
        predict_button = st.button("🧠 Predict", use_container_width=True)

        if predict_button:
            try:
                preprocessed = preprocess_image(selected_image)
                predictions = model.predict(preprocessed)
                predicted_index = np.argmax(predictions)
                confidence_score = float(predictions[0][predicted_index])
                predicted_label = class_names[predicted_index]
                description = class_descriptions[predicted_label]

                st.success(f"**Predicted Class:** {predicted_label}")
                st.info(f"**Confidence Score:** {confidence_score:.2f}")
                st.write(f"**Description:** {description}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    st.markdown("---")
    

    # Use defaults if blank
    name = name if name else "Anonymous"
    age = age if age else "-"
    gender = gender if gender else "-"
    st.markdown("### 📄 Download Your Report")
    

    if selected_image and predict_button:
        try:
            

            # Save and offer download
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            #Download Button
                pdf_bytes = generate_pdf(selected_image, str(predicted_label), confidence_score, str(description),
                                        name, age, gender)
                st.download_button(
                        label="📄 Download Report as PDF",
                        data=pdf_bytes,
                        file_name=f"DR_Report_{name.replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )

        except Exception as e:
            import traceback
            st.error(f"Prediction failed: {e}")
            st.text(traceback.format_exc())
    elif predict_button and not selected_image:
        st.warning("⚠️ Please upload or select an image before prediction.")




