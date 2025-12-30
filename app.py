import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from langchain_groq import ChatGroq
from PIL import Image
import urllib.request
import os

# Official Google Model URL
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

# This check ensures we have a REAL file, not a tiny 1KB pointer
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000000:
    with st.spinner("Repairing AI model file..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
# --- INITIALIZATION ---
groq_api_key = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=groq_api_key
)
st.set_page_config(page_title="Beauty Agent", layout="wide")
import os
from pathlib import Path

# Get the directory where app.py is located
base_path = Path(__file__).parent
model_path = str(base_path / "face_landmarker.task")

# Check if file exists before running to debug
if not os.path.exists(model_path):
    st.error(f"Model file NOT found at {model_path}. Please ensure it is in your GitHub repo root.")
else:
    base_options = python.BaseOptions(model_asset_path=model_path)
# Custom CSS for horizontal styling
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 50px; }
    .stTabs [data-baseweb="tab"] { font-size: 20px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("✨ Interactive Beauty Agent")

# --- CORE LOGIC FUNCTIONS ---
def get_analysis(img_path):
    # 1. Use absolute pathing for Streamlit Cloud stability
    # This finds the file in the same folder as your app.py
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'face_landmarker.task')
    
    # Safety check: Display an error if the file is missing in GitHub
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}. Please ensure 'face_landmarker.task' is in your GitHub root.")
        return None, None

    # 2. Initialize with the corrected path
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    
    # Use context manager (with) to ensure the detector is closed properly
    try:
        with vision.FaceLandmarker.create_from_options(options) as detector:
            mp_image = mp.Image.create_from_file(img_path)
            result = detector.detect(mp_image)
            
            if not result.face_landmarks: 
                return None, None

            landmarks = result.face_landmarks[0]
            h, w = mp_image.height, mp_image.width
            def pt(i): return np.array([landmarks[i].x * w, landmarks[i].y * h])

            # --- YOUR ORIGINAL ANALYSIS LOGIC ---
            f_h = np.linalg.norm(pt(10) - pt(152))
            c_w = np.linalg.norm(pt(234) - pt(454))
            j_w = np.linalg.norm(pt(58) - pt(288))
            
            ratio = f_h / c_w
            if ratio > 1.5: shape = "Oval"
            elif j_w > (c_w * 0.85): shape = "Square"
            elif c_w > (f_h * 0.9): shape = "Round"
            else: shape = "Heart"

            raw_img = cv2.imread(img_path)
            cheek = pt(205)
            sample = raw_img[max(0, int(cheek[1])-5):min(h, int(cheek[1])+5), 
                             max(0, int(cheek[0])-5):min(w, int(cheek[0])+5)]
            avg = np.mean(sample, axis=(0, 1))
            hex_val = '#{:02x}{:02x}{:02x}'.format(int(avg[2]), int(avg[1]), int(avg[0]))
            
            return shape, hex_val
            
    except Exception as e:
        st.error(f"MediaPipe Initialization Error: {e}")
        return None, None
# --- MAIN UI ---
uploaded_file = st.file_uploader("Upload your photo to begin", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save and display image
    img_array = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, img)
    
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=400)

    with st.spinner("Analyzing..."):
        shape, hex_code = get_analysis(temp_path)

    if shape:
        st.success(f"Analysis Complete: **{shape}** Face Shape | **{hex_code}** Skin Tone")
        
        # --- HORIZONTAL OPTIONS ---
        tab1, tab2, tab3 = st.tabs(["🎨 Color Palette", "💇 Hair Recommendations", "💄 Makeup Tips"])

        with tab1:
            st.header("Recommended Colors")
            prompt = f"Suggest a clothing color palette for {hex_code} skin tone. Use bullet points."
            st.write(llm.invoke(prompt).content)
            st.color_picker("Your Detected Tone", hex_code)

        with tab2:
            st.header("Best Hairstyles")
            prompt = f"Suggest haircuts for a {shape} face shape. Be specific about lengths and styles."
            st.write(llm.invoke(prompt).content)

        with tab3:
            st.header("Makeup Suggestions")
            prompt = f"Suggest lipstick and blush colors for {hex_code} skin tone and {shape} face."
            st.write(llm.invoke(prompt).content)
    else:
        st.error("No face detected. Please try a clearer photo.")