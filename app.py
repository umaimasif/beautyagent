import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from langchain_groq import ChatGroq
from PIL import Image

# --- INITIALIZATION ---
groq_api_key = st.secrets["GROQ_API_KEY"]
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=groq_api_key
)
st.set_page_config(page_title="Beauty Agent", layout="wide")

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
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    
    mp_image = mp.Image.create_from_file(img_path)
    result = detector.detect(mp_image)
    
    if not result.face_landmarks: return None, None

    landmarks = result.face_landmarks[0]
    h, w = mp_image.height, mp_image.width
    def pt(i): return np.array([landmarks[i].x * w, landmarks[i].y * h])

    # Analysis
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
    sample = raw_img[int(cheek[1])-5:int(cheek[1])+5, int(cheek[0])-5:int(cheek[0])+5]
    avg = np.mean(sample, axis=(0, 1))
    hex_val = '#{:02x}{:02x}{:02x}'.format(int(avg[2]), int(avg[1]), int(avg[0]))
    
    return shape, hex_val

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