import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image

st.set_page_config(page_title="MediScan", page_icon="🩺", layout="centered")

st.title("🩺 MediScan")
st.subheader("Medical Image Classification System")
st.write("Upload a chest X-ray image to classify it as Normal or Pneumonia and view a Grad-CAM explanation.")

st.info("Educational/research prototype only — not a medical diagnosis tool.")

MODEL_PATH = Path("mediscan_resnet50.pt")

uploaded = st.file_uploader("Upload chest X-ray image", type=["png","jpg","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if not MODEL_PATH.exists():
        st.warning("The trained model file is not present yet. Run the Colab notebook first, download mediscan_resnet50.pt, and place it beside app.py.")
        st.stop()

    try:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image

        device = torch.device("cpu")
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Dropout(0.30), nn.Linear(model.fc.in_features, 2))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        x = tf(image).unsqueeze(0)
        with torch.no_grad():
            prob = torch.softmax(model(x), dim=1)[0]
            pred = int(prob.argmax())
            conf = float(prob[pred])

        labels = ["Normal", "Pneumonia"]
        st.success(f"Prediction: {labels[pred]}")
        st.metric("Confidence", f"{conf*100:.2f}%")

        cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
        gray = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
        rgb = np.asarray(image.resize((224,224))).astype(np.float32) / 255.0
        overlay = show_cam_on_image(rgb, gray, use_rgb=True)
        st.image(overlay, caption="Grad-CAM visualization", use_container_width=True)
    except Exception as e:
        st.error("The model/app could not be loaded. Check the requirements and trained model file.")
        st.code(str(e))
