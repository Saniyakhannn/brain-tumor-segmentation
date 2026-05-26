import torch
import torch.nn.functional as F
import cv2
import numpy as np
from model import UNet, AttentionUNet
from predict import unet_model, attn_model, device


class GradCAM:
    def __init__(self, model, target_layer, device):
        self.model       = model
        self.device      = device
        self.gradients   = None
        self.activations = None
        self._fwd_hook   = target_layer.register_forward_hook(
            self._save_activation)
        self._bwd_hook   = target_layer.register_full_backward_hook(
            self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, img_tensor, target_size=(256, 256)):
        self.model.eval()
        img_tensor = img_tensor.to(self.device)
        img_tensor.requires_grad_(True)
        output = self.model(img_tensor)
        score  = torch.sigmoid(output).mean()
        self.model.zero_grad()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(
            dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=target_size,
                  mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam    -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def apply_heatmap(image_rgb, cam, alpha=0.55):
    heatmap     = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_uint8   = (image_rgb * 255).astype(np.uint8) \
                  if image_rgb.max() <= 1 \
                  else image_rgb.astype(np.uint8)
    img_uint8   = cv2.resize(img_uint8, (256, 256))
    blended     = cv2.addWeighted(
        img_uint8, 1-alpha, heatmap_rgb, alpha, 0)
    return blended


def run_gradcam(image_bgr, show_gates=False):
    """
    Run Grad-CAM on both models
    Returns dict with heatmaps and overlays
    """
    # Preprocess
    img_rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img       = cv2.resize(img_rgb, (256, 256)).astype(
                np.float32) / 255.0
    tensor    = torch.from_numpy(
                img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    img_uint8 = (img * 255).astype(np.uint8)

    results = {}

    # UNet Grad-CAM
    gc_unet  = GradCAM(unet_model,
               unet_model.bottleneck, device)
    cam_unet = gc_unet.generate(tensor.clone())
    gc_unet.remove_hooks()
    results["unet_cam"]     = cam_unet
    results["unet_overlay"] = apply_heatmap(img, cam_unet)

    # Attention UNet Grad-CAM
    gc_attn  = GradCAM(attn_model,
               attn_model.bottleneck, device)
    cam_attn = gc_attn.generate(tensor.clone())
    gc_attn.remove_hooks()
    results["attn_cam"]     = cam_attn
    results["attn_overlay"] = apply_heatmap(img, cam_attn)

    # Difference map
    diff       = cam_attn.astype(float) - \
                 cam_unet.astype(float)
    diff_norm  = ((diff + 1) / 2 * 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(
                 diff_norm, cv2.COLORMAP_COOL)
    results["diff"] = cv2.cvtColor(
                      diff_color, cv2.COLOR_BGR2RGB)

    return results