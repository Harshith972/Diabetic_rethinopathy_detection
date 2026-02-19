import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Page Setup
st.set_page_config(page_title="Diabetic Retinopathy Scanner", page_icon="👁️")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('final_messidor_model.keras')

def process_image(image_data):
    # Resize to 300x300 (Same as training)
    size = (300, 300)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.array(image)
    
    # Handle Grayscale or RGBA
    if img_array.ndim == 2: img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4: img_array = img_array[..., :3]
    
    # Preprocess
    img_batch = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(img_batch)

# UI Header
st.title("👁️ Diabetic Retinopathy Grader")
st.write("Upload a retinal fundus image to detect severity.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Scan', use_column_width=True)
    
    with st.spinner('Scanning retina...'):
        try:
            model = load_model()
            processed = process_image(image)
            pred = model.predict(processed)
            
            # Get Result
            idx = np.argmax(pred)
            conf = np.max(pred) * 100
            
            # --- UPDATED LABELS WITH GRADES ---
            labels = {
                0: "Grade 0: No DR (Healthy)",
                1: "Grade 1: Mild DR",
                2: "Grade 2: Moderate DR",
                3: "Grade 3: Severe DR"
            }
            result_text = labels.get(idx, "Unknown")
            
            # --- DISPLAY MAIN DIAGNOSIS ---
            st.divider()
            if idx == 0:
                st.success(f"✅ **Diagnosis:** {result_text}")
            else:
                st.error(f"⚠️ **Diagnosis:** {result_text}")
            
            st.write(f"**Top Confidence:** {conf:.2f}%")
            
            # --- DISPLAY ALL PROBABILITIES ---
            st.divider()
            st.subheader("📊 Full Confidence Report")
            probs = pred[0] # Get the array of 4 probabilities
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Grade 0 (Healthy):** {probs[0]*100:.2f}%")
                st.progress(int(probs[0]*100))
                
                st.warning(f"**Grade 1 (Mild):** {probs[1]*100:.2f}%")
                st.progress(int(probs[1]*100))

            with col2:
                st.warning(f"**Grade 2 (Moderate):** {probs[2]*100:.2f}%")
                st.progress(int(probs[2]*100))
                
                st.error(f"**Grade 3 (Severe):** {probs[3]*100:.2f}%")
                st.progress(int(probs[3]*100))
            
        except Exception as e:
            st.error(f"Error: {e}")