import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras

# Basic page setup
st.set_page_config(page_title="AgriAI Crop Detector", layout="centered")

st.title("AgriAI: Crop Disease Detection")
st.write("### Save Your Crop From Diseases")
st.write("---")

# Function to load the model & labels
@st.cache_resource
def load_model_and_labels():
    # Model file load ho rahi hai
    model = keras.models.load_model('Cotton_Crop_Disease_Model.h5', compile=False)
    
    # Labels file read ho rahi hai
    labels = []
    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f.readlines()]
        
    return model, labels

# Load Model
model = None
labels = []

try:
    with st.spinner("Loading Model... Please wait"):
        model, labels = load_model_and_labels()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error("Error loading model! Make sure 'Cotton_Crop_Disease_Model.h5' and 'labels.txt' are in GitHub repository.")

# Updated Remedies Dictionary (Cotton & Okra diseases included)
remedies_dict = {
    "Cotton Bacterial Blight": "Ilaaj: Copper Oxychloride 3g/L ka spray karein aur infected patte jala dein.",
    "Cotton Healthy": "Status: Fasal bilkul theek hai. Standard schedule follow karein.",
    "Okra Leaf Curl": "Ilaaj: Imidacloprid 0.5ml/L spray karein taake Whitefly control ho sake.",
    "Okra Yellow Vein Mosaic": "Ilaaj: Infected podon ko nikalein aur Neem oil 5ml/L spray karein.",
    "Okra Healthy": "Status: Bhindi ki fasal bilkul sehatmand hai!"
}

st.write("### Select Input Method")
input_type = st.radio("Choose one:", ("Camera", "Upload File"))

img_file = None

if input_type == "Camera":
    img_file = st.camera_input("Take a picture of the crop")
else:
    img_file = st.file_uploader("Upload image here (jpg/png)", type=["jpg", "jpeg", "png"])

# Main logic for prediction
if img_file is not None:
    if model is None:
        st.error("Model load nahi ho saka. Pehle model file upload check karein.")
    else:
        img = Image.open(img_file)
        
        # Show uploaded image
        if input_type == "Upload File":
            st.image(img, caption="Your Uploaded Image", use_container_width=True)
            
        st.write("### Prediction Result:")
        
        with st.spinner("Analyzing..."):
            # Resize image to 224x224
            img_resized = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
            img_array = np.asarray(img_resized)
            
            # Normalize image data
            normalized_img = (img_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_img
            
            # Getting predictions
            prediction = model.predict(data)
            index = np.argmax(prediction)
            
            # Cleaning the class name
            raw_class = labels[index] if index < len(labels) else "Unknown"
            if raw_class[0].isdigit():
                class_name = " ".join(raw_class.split()[1:])
            else:
                class_name = raw_class
                
            confidence_score = prediction[0][index] * 100

        # Displaying the final output
        if "Healthy" in class_name or "healthy" in class_name.lower():
            st.success(f"Result: {class_name}")
            st.info(f"Accuracy: {confidence_score:.2f}%")
            st.balloons()
        else:
            st.error(f"Disease Detected: {class_name}")
            st.warning(f"Accuracy: {confidence_score:.2f}%")
            
            # Check and display remedy
            remedy = remedies_dict.get(class_name, "Is disease ki remedy system me added nahi hai.")
            st.write("### Recommended Remedy:")
            st.write(remedy)

st.write("---")
st.caption("Developed by Azad Ahmed Bhurgrri | Roll No: 2K23/CSME/9 | IMCS, University of Sindh")
