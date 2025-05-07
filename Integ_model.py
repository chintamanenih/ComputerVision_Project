import streamlit as st
import torch
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import joblib
import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from ultralytics import YOLO
from io import BytesIO

# ---------- Load Models ----------
reg_model = joblib.load("C:\\Users\\Harsha\\Downloads\\CV\\fog_regression_model.pkl")
scaler = joblib.load("C:\\Users\\Harsha\\Downloads\\CV\\fog_scaler.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet_model = smp.Unet(
    encoder_name="resnet34", encoder_weights=None,
    in_channels=3, classes=3, activation=None
).to(device)
resnet_model.load_state_dict(torch.load("C:\\Users\\Harsha\\Downloads\\CV\\best_resnet34_unet_colortry1.pth", map_location=device))
resnet_model.eval()

yolo_model = YOLO("yolov8n.pt")

transform = transforms.Compose([
    transforms.ToTensor()
])

# ---------- Fog Detection ----------
def glcm_features(image, levels=8):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = (gray / (256 / levels)).astype(np.uint8)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    return [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'dissimilarity').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean(),
        graycoprops(glcm, 'ASM').mean()
    ]

def predict_fog_percentage(image):
    features = np.array(glcm_features(image)).reshape(1, -1)
    scaled = scaler.transform(features)
    return float(np.clip(reg_model.predict(scaled)[0], 0, 100))

# ---------- DCP ----------
def dark_channel(image, size=15):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark = cv2.erode(min_channel, kernel)
    return dark

def atmospheric_light(image, dark_channel, top_percent=0.2):
    h, w = image.shape[:2]
    num_pixels = int(h * w * top_percent)
    dark_vec = dark_channel.ravel()
    img_vec = image.reshape(h * w, 3)
    indices = np.argsort(dark_vec)[-num_pixels:]
    A = np.mean(img_vec[indices], axis=0)
    return A

def transmission_estimate(image, A, omega=0.98, size=15):
    norm_image = image.astype(np.float32) / A
    dark = dark_channel(norm_image, size)
    transmission = 1 - omega * dark
    return transmission

def recover_image(image, A, transmission, t0=0.5):
    transmission = np.maximum(transmission, t0)
    J = (image - A) / transmission[:, :, np.newaxis] + A
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J

def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(32, 32))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb

def smooth_edges(image):
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def dehaze_image(image):
    dark = dark_channel(image)
    A = atmospheric_light(image, dark)
    transmission = transmission_estimate(image, A)
    dehazed = recover_image(image, A, transmission)
    enhanced = enhance_image(dehazed)
    smoothed = smooth_edges(enhanced)
    return smoothed

# ---------- ResNet Dehazing ----------
def resnet_dehaze(image_np):
    input_tensor = transform(image_np).unsqueeze(0).to(device)
    with torch.no_grad():
        output = resnet_model(input_tensor)[0].cpu().permute(1, 2, 0).numpy()
    output = (np.clip(output, 0, 1) * 255).astype(np.uint8)
    return smooth_edges(output)

# ---------- Combined Dehaze Pipelines ----------
def full_pipeline_variants(image_np):
    dcp = dehaze_image(image_np)
    resnet = resnet_dehaze(image_np)
    dcp_resnet = resnet_dehaze(dcp)
    return dcp, resnet, dcp_resnet

# ---------- YOLO Detection ----------
def detect_objects_boxes_only(image_pil):
    image_np = np.array(image_pil)
    results = yolo_model.predict(image_np, conf=0.05, iou=0.5, verbose=False)

    annotated_img = image_np.copy()
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            label = f"{yolo_model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    return Image.fromarray(annotated_img)

# ---------- Helpers ----------
def resize_for_display(pil_img, size=(256, 256)):
    return pil_img.resize(size)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Fog Detection ➜ Multi-Dehaze ➜ YOLO", layout="centered")
st.title("🌫️ Fog Detection ➜ Dehazing (DCP + ResNet) ➜ YOLO")
st.write("Upload an image. Fog level is detected, and 3 dehazing methods + YOLO are applied.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    img_np = np.array(pil_img)

    st.image(resize_for_display(pil_img), caption="Original Image (256x256)")
    fog_level = predict_fog_percentage(img_np)
    st.markdown(f"### Fog Level: {fog_level:.2f}%")

    with st.spinner("Generating all dehazed versions..."):
        dcp_img, resnet_img, dcp_resnet_img = full_pipeline_variants(img_np)

    # Show All Dehazed Variants
    st.subheader("🌀 Dehazed Versions")
    col1, col2 = st.columns(2)
    col1.image(resize_for_display(Image.fromarray(dcp_img)), caption="DCP Only")
    col2.image(resize_for_display(Image.fromarray(resnet_img)), caption="ResNet Only")

    st.image(resize_for_display(Image.fromarray(dcp_resnet_img)), caption="DCP ➜ ResNet")

    # YOLO on All
    st.subheader("🧠 YOLO on Dehazed Versions")
    with st.spinner("Running YOLO detections..."):
        yolo_orig = detect_objects_boxes_only(pil_img)
        yolo_dcp = detect_objects_boxes_only(Image.fromarray(dcp_img))
        yolo_resnet = detect_objects_boxes_only(Image.fromarray(resnet_img))
        yolo_dcp_resnet = detect_objects_boxes_only(Image.fromarray(dcp_resnet_img))

    st.image(resize_for_display(yolo_orig), caption="YOLO on Original Image")
    #st.image(resize_for_display(yolo_dcp), caption="YOLO on DCP Only")
    #st.image(resize_for_display(yolo_resnet), caption="YOLO on ResNet Only")
    st.image(resize_for_display(yolo_dcp_resnet), caption="YOLO on DCP ➜ ResNet")