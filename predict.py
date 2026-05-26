import torch
import cv2
import numpy as np
from model import UNet, AttentionUNet

device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")

# ── Load Models ─────────────────────────────────────
unet_model = UNet()
unet_model.load_state_dict(
    torch.load("best_unet.pth", map_location=device))
unet_model.to(device).eval()

attn_model = AttentionUNet()
attn_model.load_state_dict(
    torch.load("best_attention.pth", map_location=device))
attn_model.to(device).eval()


# ── Preprocess ───────────────────────────────────────
def preprocess(image):
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    return image


# ── Predict ──────────────────────────────────────────
def predict_image(image, model_type="attention"):
    if image is None:
        raise ValueError("Input image is None")

    if model_type == "unet":
        model = unet_model
    elif model_type == "attention":
        model = attn_model
    else:
        raise ValueError("model_type must be unet or attention")

    # Convert BGR to RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    original_size = (image.shape[1], image.shape[0])

    img_tensor = preprocess(image)

    with torch.no_grad():
        output = model(img_tensor)
        output = torch.sigmoid(output)

    pred = output.squeeze().cpu().numpy()
    pred = cv2.resize(pred, original_size,
                      interpolation=cv2.INTER_LINEAR)
    pred = np.clip(pred, 0, 1)

    return pred