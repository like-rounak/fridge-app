import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

# Load model
model = YOLO("best.pt")  # upload trained weights in the same folder

st.title("🧊 Fridge Item Detector")
st.write("Upload a fridge image and get the items detected!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Run prediction
    results = model.predict(img, conf=0.5)
    boxes = results[0].boxes

    st.subheader("Detected Items:")
    labels = []
    for box in boxes:
        cls_id = int(box.cls)
        label = results[0].names[cls_id]
        labels.append(label)
    st.write(list(set(labels)))

    # Show image with boxes
    results[0].plot()
    st.image(results[0].plot(), caption="Detections", use_container_width=True)
