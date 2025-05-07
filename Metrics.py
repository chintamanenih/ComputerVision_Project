import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import joblib
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image

# ---------- Configuration ----------
FOG_FOLDER = "C:/Users/Harsha/Downloads/CV/fog/"
GT_FOLDER = "C:/Users/Harsha/Downloads/CV/gt/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models
reg_model = joblib.load("C:/Users/Harsha/Downloads/CV/fog_regression_model.pkl")
scaler = joblib.load("C:/Users/Harsha/Downloads/CV/fog_scaler.pkl")

resnet_model = smp.Unet(
    encoder_name="resnet34", encoder_weights=None,
    in_channels=3, classes=3, activation=None
).to(DEVICE)
resnet_model.load_state_dict(torch.load("C:/Users/Harsha/Downloads/CV/best_resnet34_unet_colortry1.pth", map_location=DEVICE))
resnet_model.eval()

# Image Transformation
transform = torch.nn.Sequential(
    torch.nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
    torch.nn.Sigmoid()
)

# DCP Components

def dark_channel(image, size=15):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_channel, kernel)

def atmospheric_light(image, dark_channel, top_percent=0.2):
    h, w = image.shape[:2]
    num_pixels = int(h * w * top_percent)
    dark_vec = dark_channel.ravel()
    img_vec = image.reshape(h * w, 3)
    indices = np.argsort(dark_vec)[-num_pixels:]
    return np.mean(img_vec[indices], axis=0)

def transmission_estimate(image, A, omega=0.98, size=15):
    norm_image = image.astype(np.float32) / A
    dark = dark_channel(norm_image, size)
    return 1 - omega * dark

def recover_image(image, A, transmission, t0=0.1):
    transmission = np.maximum(transmission, t0)
    J = (image - A) / transmission[:, :, np.newaxis] + A
    return np.clip(J, 0, 255).astype(np.uint8)

def resnet_dehaze(image_np):
    input_tensor = torch.tensor(image_np / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        output = resnet_model(input_tensor)[0].cpu().permute(1, 2, 0).numpy()
    return (output * 255).astype(np.uint8)

def dehaze_image_dcp(image):
    dark = dark_channel(image)
    A = atmospheric_light(image, dark)
    transmission = transmission_estimate(image, A)
    return recover_image(image, A, transmission)

# Processing Images
fog_files = sorted(os.listdir(FOG_FOLDER))
gt_files = sorted(os.listdir(GT_FOLDER))

psnr_results = {"DCP": [], "ResNet": [], "DCP ➜ ResNet": []}
ssim_results = {"DCP": [], "ResNet": [], "DCP ➜ ResNet": []}

for fog_file, gt_file in zip(fog_files, gt_files):
    fog_path = os.path.join(FOG_FOLDER, fog_file)
    gt_path = os.path.join(GT_FOLDER, gt_file)
    fog_img = cv2.imread(fog_path)
    gt_img = cv2.imread(gt_path)
    gt_img = cv2.resize(gt_img, (fog_img.shape[1], fog_img.shape[0]))
    
    # Dehazing
    dcp_img = dehaze_image_dcp(fog_img)
    resnet_img = resnet_dehaze(fog_img)
    dcp_resnet_img = resnet_dehaze(dcp_img)
    
    # PSNR and SSIM Calculation
    psnr_results["DCP"].append(psnr(gt_img, dcp_img))
    psnr_results["ResNet"].append(psnr(gt_img, resnet_img))
    psnr_results["DCP ➜ ResNet"].append(psnr(gt_img, dcp_resnet_img))
    
    # Ensure SSIM window size is valid
    min_dim = min(gt_img.shape[0], gt_img.shape[1])
    win_size = min(7, min_dim)
    ssim_results["DCP"].append(ssim(gt_img, dcp_img, channel_axis=2, win_size=win_size))
    ssim_results["ResNet"].append(ssim(gt_img, resnet_img, channel_axis=2, win_size=win_size))
    ssim_results["DCP ➜ ResNet"].append(ssim(gt_img, dcp_resnet_img, channel_axis=2, win_size=win_size))

# Plot PSNR Results
plt.figure(figsize=(12, 6))
for method in psnr_results.keys():
    plt.plot(psnr_results[method], label=f"{method} (PSNR)")
plt.title("Dehazing Method Comparison - PSNR")
plt.xlabel("Image Index")
plt.ylabel("PSNR Value")
plt.legend()
plt.show()

# Plot SSIM Results
plt.figure(figsize=(12, 6))
for method in ssim_results.keys():
    plt.plot(ssim_results[method], label=f"{method} (SSIM)")
plt.title("Dehazing Method Comparison - SSIM")
plt.xlabel("Image Index")
plt.ylabel("SSIM Value")
plt.legend()
plt.show()

print("Processing Complete. Separate PSNR and SSIM metrics plotted.")
