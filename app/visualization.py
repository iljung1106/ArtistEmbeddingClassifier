"""
Visualization utilities for artist embedding model:
- Grad-CAM heatmaps
- View attention weights (whole/face/eyes)
- Branch attention weights (Gram/Cov/Spectrum/Stats)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class ViewAnalysis:
    """Analysis results for a single inference."""
    # View attention weights [3] for whole/face/eyes
    view_weights: Dict[str, float]
    # Branch attention weights per view {view_name: {branch_name: weight}}
    branch_weights: Dict[str, Dict[str, float]]
    # Grad-CAM heatmaps per view (PIL Images)
    gradcam_heatmaps: Dict[str, Optional[Image.Image]]
    # Original images for overlay
    original_images: Dict[str, Optional[Image.Image]]


def _get_branch_weights(encoder, x: torch.Tensor) -> Dict[str, float]:
    """
    Extract branch attention weights from a ViewEncoder forward pass.
    Returns dict with keys: gram, cov, spectrum, stats
    """
    # We need to do a partial forward to get the branch gate weights
    with torch.no_grad():
        x_lab = encoder._rgb_to_lab(x)
        f0 = encoder.stem(x_lab)
        f1 = encoder.b1(f0)
        f2 = encoder.b2(f1)
        f3 = encoder.b3(f2)
        f4 = encoder.b4(f3)

        g3 = encoder.h_gram3(f3)
        c3 = encoder.h_cov3(f3)
        sp3 = encoder.h_sp3(f3)
        st3 = encoder.h_st3(f3)

        g4 = encoder.h_gram4(f4)
        c4 = encoder.h_cov4(f4)
        sp4 = encoder.h_sp4(f4)
        st4 = encoder.h_st4(f4)

        b_gram = torch.cat([g3, g4], dim=1)
        b_cov = torch.cat([c3, c4], dim=1)
        b_sp = torch.cat([sp3, sp4], dim=1)
        b_st = torch.cat([st3, st4], dim=1)

        flat = torch.cat([b_gram, b_cov, b_sp, b_st], dim=1)
        gate_logits = encoder.branch_gate(flat)
        w = torch.softmax(gate_logits, dim=-1)

    # w is [1, 4] for single image
    w_np = w[0].cpu().numpy()
    return {
        "Gram": float(w_np[0]),
        "Cov": float(w_np[1]),
        "Spectrum": float(w_np[2]),
        "Stats": float(w_np[3]),
    }


def _compute_xgradcam(
    encoder,
    x: torch.Tensor,
    target_layer_name: str = "b3",
) -> np.ndarray:
    """
    Compute XGrad-CAM heatmap for a ViewEncoder.
    XGrad-CAM is an improved variant that uses element-wise gradient-activation
    products normalized by activation sums, providing better localization.

    Reference: Axiom-based Grad-CAM (Fu et al., BMVC 2020)
    Returns a heatmap as numpy array [H, W] normalized to [0, 1].
    """
    # Storage for activations and gradients
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output.detach().clone()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach().clone()

    # Get the target layer
    target_layer = getattr(encoder, target_layer_name, None)
    if target_layer is None:
        # Fallback to b2 or b1
        for fallback in ["b2", "b1", "stem"]:
            target_layer = getattr(encoder, fallback, None)
            if target_layer is not None:
                break

    if target_layer is None:
        return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)

    # Register hooks
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward pass
        x.requires_grad_(True)
        output = encoder(x)

        # Backward pass - use the L2 norm of output as target
        target = output.norm(dim=1).sum()
        encoder.zero_grad()
        target.backward(retain_graph=True)

        # Get activations and gradients
        acts = activations.get("value")
        grads = gradients.get("value")

        if acts is None or grads is None:
            return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)

        # XGrad-CAM: weights = sum(grads * acts, spatial) / (sum(acts, spatial) + eps)
        # This normalizes by the activation magnitude, improving localization
        grad_act_product = grads * acts  # [B, C, H, W]
        sum_grad_act = grad_act_product.sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        sum_acts = acts.sum(dim=(2, 3), keepdim=True) + 1e-7  # [B, C, 1, 1]
        weights = sum_grad_act / sum_acts  # [B, C, 1, 1]

        # Weighted combination of activations
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)  # Only positive contributions

        # Normalize
        cam = cam[0, 0].cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input size
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8))
        cam_pil = cam_pil.resize((x.shape[3], x.shape[2]), Image.BILINEAR)
        cam = np.array(cam_pil).astype(np.float32) / 255.0

        return cam

    finally:
        fwd_handle.remove()
        bwd_handle.remove()
        x.requires_grad_(False)


def _overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> Image.Image:
    """Overlay a heatmap on an image."""
    import matplotlib.pyplot as plt

    # Ensure heatmap is 2D and normalized
    heatmap = np.clip(heatmap, 0, 1)

    # Get colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]  # RGB only, no alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # Resize heatmap to match image
    heatmap_pil = Image.fromarray(heatmap_colored)
    heatmap_pil = heatmap_pil.resize(image.size, Image.BILINEAR)

    # Blend
    image_rgb = image.convert("RGB")
    blended = Image.blend(image_rgb, heatmap_pil, alpha)

    return blended


def analyze_views(
    model: torch.nn.Module,
    views: Dict[str, Optional[torch.Tensor]],
    original_images: Dict[str, Optional[Image.Image]],
    device: torch.device,
) -> ViewAnalysis:
    """
    Perform full analysis on a set of views.
    Returns view weights, branch weights per view, and Grad-CAM heatmaps.
    """
    model.eval()

    # Prepare masks
    masks = {}
    view_tensors = {}
    for k in ("whole", "face", "eyes"):
        if views.get(k) is not None:
            view_tensors[k] = views[k].unsqueeze(0).to(device)
            masks[k] = torch.ones(1, dtype=torch.bool, device=device)
        else:
            view_tensors[k] = None
            masks[k] = torch.zeros(1, dtype=torch.bool, device=device)

    # Get view attention weights from forward pass
    with torch.no_grad():
        _, _, W = model(view_tensors, masks)

    # W is [1, num_present_views]
    W_np = W[0].cpu().numpy()

    # Map W back to view names (only present views have weights)
    view_order = ["whole", "face", "eyes"]
    present_views = [k for k in view_order if view_tensors[k] is not None]

    view_weights = {}
    for i, k in enumerate(present_views):
        view_weights[k] = float(W_np[i])
    for k in view_order:
        if k not in view_weights:
            view_weights[k] = 0.0

    # Get branch weights and Grad-CAM for each view
    branch_weights = {}
    gradcam_heatmaps = {}

    # Get encoder (shared or separate)
    enc_whole = model.enc_whole
    enc_face = model.enc_face
    enc_eyes = model.enc_eyes

    encoders = {"whole": enc_whole, "face": enc_face, "eyes": enc_eyes}

    for k in view_order:
        if view_tensors[k] is not None:
            enc = encoders[k]
            x = view_tensors[k]

            # Branch weights
            try:
                branch_weights[k] = _get_branch_weights(enc, x)
            except Exception:
                branch_weights[k] = {"Gram": 0.25, "Cov": 0.25, "Spectrum": 0.25, "Stats": 0.25}

            # Grad-CAM
            try:
                heatmap = _compute_xgradcam(enc, x.clone(), target_layer_name="b3")
                if original_images.get(k) is not None:
                    gradcam_heatmaps[k] = _overlay_heatmap(original_images[k], heatmap, alpha=0.5)
                else:
                    gradcam_heatmaps[k] = None
            except Exception:
                gradcam_heatmaps[k] = None
        else:
            branch_weights[k] = {}
            gradcam_heatmaps[k] = None

    return ViewAnalysis(
        view_weights=view_weights,
        branch_weights=branch_weights,
        gradcam_heatmaps=gradcam_heatmaps,
        original_images={k: original_images.get(k) for k in view_order},
    )


def format_view_weights_html(analysis: ViewAnalysis) -> str:
    """Format view weights as clean HTML with styled progress bars."""
    # View labels with descriptions
    view_info = {
        "whole": ("Whole Image", "#4CAF50"),  # green
        "face": ("Face", "#2196F3"),  # blue
        "eyes": ("Eye", "#FF9800"),  # orange (single-eye crop)
    }

    html_parts = ['<div style="font-family: sans-serif; padding: 10px;">']
    html_parts.append('<h3 style="margin-bottom: 15px;">📊 View Contribution</h3>')

    for k in ("whole", "face", "eyes"):
        w = analysis.view_weights.get(k, 0.0)
        label, color = view_info[k]
        pct = int(w * 100)

        html_parts.append(f'''
        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-weight: 500;">{label}</span>
                <span style="font-weight: 600; color: {color};">{pct}%</span>
            </div>
            <div style="background: #e0e0e0; border-radius: 4px; height: 20px; overflow: hidden;">
                <div style="background: {color}; width: {pct}%; height: 100%; border-radius: 4px; transition: width 0.3s;"></div>
            </div>
        </div>
        ''')

    html_parts.append('</div>')
    return "".join(html_parts)

