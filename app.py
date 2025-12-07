import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

st.set_page_config(page_title="Pedestrian-Detection", page_icon="🚶", layout="centered")

# Background styling
page_bg = '''
<style>
.stApp {
    background: linear-gradient(135deg, #a7d8ff, #0047ab);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
div.stButton > button {
    background-color: #ff4b8e;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.6em 1.2em;
    font-size: 1.05em;
    font-weight: 600;
}
div.stButton > button:hover {
    background-color: #ff2a78;
}
.signal-box {
    width:120px;
    height:120px;
    border-radius:15px;
    margin:auto;
    margin-top:20px;
}
</style>
'''
st.markdown(page_bg, unsafe_allow_html=True)

st.title("🚶 Pedestrian Detection with Traffic Signal Control")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

def detect_pedestrians(img_np):
    results = model(img_np)
    result = results[0]
    img_with_boxes = img_np.copy()
    count = 0

    for box in result.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if cls == 0:
            count += 1
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img_with_boxes, f'Person {conf:.2f}', (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
    return img_with_boxes, count

mode = st.radio("Choose Input Mode:", ["📂 Upload Image", "📷 Use Webcam"])

image_np = None

if mode == "📂 Upload Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_np = np.array(image)

elif mode == "📷 Use Webcam":
    cam_img = st.camera_input("Take a picture")
    if cam_img:
        image = Image.open(cam_img).convert("RGB")
        image_np = np.array(image)

if image_np is not None:
    if st.button("🔍 Run Detection"):
        processed, count = detect_pedestrians(image_np)
        st.image(processed, caption="Processed Image", use_container_width=True)

        st.subheader("🚦 Traffic Signal Status")
        if count >= 3:
            st.markdown('<div class="signal-box" style="background:red;"></div>', unsafe_allow_html=True)
            st.error("🔴 RED LIGHT — Too many pedestrians!")
        elif count == 0:
            st.markdown('<div class="signal-box" style="background:green;"></div>', unsafe_allow_html=True)
            st.success("🟢 GREEN LIGHT — No pedestrians detected.")
        else:
            st.markdown('<div class="signal-box" style="background:yellow;"></div>', unsafe_allow_html=True)
            st.warning("🟡 YELLOW LIGHT — Pedestrians present.")

        st.write(f"### Total pedestrians detected: **{count}**")

    st.button("🔄 Upload Another Photo", on_click=lambda: st.experimental_rerun())
