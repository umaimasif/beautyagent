import streamlit as st
import cv2
import numpy as np
import os
import urllib.request
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from langchain_groq import ChatGroq
from PIL import Image

# --- 1. MODEL DOWNLOAD FALLBACK ---
# This ensures the .task file exists before the app tries to load it
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading required AI models..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# --- 2. INITIALIZATION ---
# Use Streamlit Secrets for the API Key
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("GROQ_API_KEY not found in Streamlit Secrets!")
    st.stop()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=groq_api_key
)

st.set_page_config(page_title="Beauty Agent", layout="wide")

# Custom CSS for horizontal tab styling
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 50px; }
    .stTabs [data-baseweb="tab"] { font-size: 20px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("✨ Interactive Beauty Agent")

# --- 3. CORE LOGIC FUNCTION ---
def get_analysis(img_path):
    # Initialize with the verified absolute path
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    
    try:
        with vision.FaceLandmarker.create_from_options(options) as detector:
            mp_image = mp.Image.create_from_file(img_path)
            result = detector.detect(mp_image)
            
            if not result.face_landmarks: 
                return None, None

            landmarks = result.face_landmarks[0]
            h, w = mp_image.height, mp_image.width
            def pt(i): return np.array([landmarks[i].x * w, landmarks[i].y * h])

            # Analysis Logic
            f_h = np.linalg.norm(pt(10) - pt(152))
            c_w = np.linalg.norm(pt(234) - pt(454))
            j_w = np.linalg.norm(pt(58) - pt(288))
            
            ratio = f_h / c_w
            if ratio > 1.5: shape = "Oval"
            elif j_w > (c_w * 0.85): shape = "Square"
            elif c_w > (f_h * 0.9): shape = "Round"
            else: shape = "Heart"

            # Skin Tone Sampling
            raw_img = cv2.imread(img_path)
            cheek = pt(205)
            sample = raw_img[max(0, int(cheek[1])-5):min(h, int(cheek[1])+5), 
                             max(0, int(cheek[0])-5):min(w, int(cheek[0])+5)]
            avg = np.mean(sample, axis=(0, 1))
            hex_val = '#{:02x}{:02x}{:02x}'.format(int(avg[2]), int(avg[1]), int(avg[0]))
            
            return shape, hex_val
            
    except Exception as e:
        st.error(f"Analysis Error: {e}")
        return None, None

# --- 4. MAIN UI ---
uploaded_file = st.file_uploader("Upload your photo to begin", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Process uploaded image
    img_array = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)
    temp_path = "temp_img.jpg"
    cv2.imwrite(temp_path, img)
    
    # Display image at a fixed width
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=400)

    with st.spinner("Analyzing your features..."):
        shape, hex_code = get_analysis(temp_path)

    if shape:
        st.success(f"Analysis Complete: **{shape}** Face Shape | **{hex_code}** Skin Tone")
        
        # --- HORIZONTAL TABS ---
        tab1, tab2, tab3 = st.tabs(["🎨 Color Palette", "💇 Hair Recommendations", "💄 Makeup Tips"])

        with tab1:
            st.header("Recommended Colors")
            with st.spinner("Generating palette..."):
                prompt = f"Suggest a clothing color palette for {hex_code} skin tone. Use bullet points."
                st.write(llm.invoke(prompt).content)
            st.color_picker("Your Detected Tone", hex_code, disabled=True)

        with tab2:
            st.header("Best Hairstyles")
            with st.spinner("Finding styles..."):
                prompt = f"Suggest specific haircuts for a {shape} face shape. Focus on balance."
                st.write(llm.invoke(prompt).content)

        with tab3:
            st.header("Makeup Suggestions")
            with st.spinner("Consulting makeup artist..."):
                prompt = f"Suggest lipstick and blush shades for {hex_code} skin tone and {shape} face."
                st.write(llm.invoke(prompt).content)
    else:
        st.error("No face detected. Please ensure your face is clearly visible and try again.")
