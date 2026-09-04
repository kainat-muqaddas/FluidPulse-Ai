"""
POD-FCDNN Engine: Core machine learning and data processing logic.
Optimized for high-speed inference in Streamlit.
"""

import numpy as np
import torch
import torch.nn as nn
import streamlit as st


class FCDNN(nn.Module):
    """Fully Connected Deep Neural Network for POD coefficient prediction."""
    def __init__(self, out_dim: int, width: int = 128, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(1, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PODModel:
    """Wrapper container for POD variables."""
    def __init__(self, mean, Phi, xy, N, r):
        self.mean = mean
        self.Phi = Phi
        self.xy = xy
        self.N = N
        self.r = r


class PODFCDNNTrainer:
    """Inference class for POD-FCDNN predictions."""
    def __init__(self, pod, model, x_mean, x_std, y_mean, y_std, device="cpu"):
        self.pod = pod
        self.model = model
        self.device = torch.device(device)
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def predict_coefficients(self, Re: float) -> np.ndarray:
        x_in = np.log(np.array([[Re]], dtype=np.float32))
        x_norm = (x_in - self.x_mean) / self.x_std

        self.model.eval()
        with torch.no_grad():
            y_norm = self.model(torch.tensor(x_norm, device=self.device))
            y_norm = y_norm.cpu().numpy()

        coeffs = y_norm * self.y_std + self.y_mean
        return coeffs.ravel()


@st.cache_resource
def load_checkpoint(path: str, device: str = "cpu") -> PODFCDNNTrainer:
    """Fast cached loader supporting both class-based and dict-based checkpoints."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "pod" in checkpoint:
        pod = checkpoint["pod"]
        r = pod.r
    else:
        # Extract from dictionary checkpoint structure
        pod = PODModel(
            mean=checkpoint["pod_mean"],
            Phi=checkpoint["pod_phi"],
            xy=checkpoint["pod_xy"],
            N=checkpoint["pod_N"],
            r=checkpoint["pod_r"]
        )
        r = checkpoint["pod_r"]

    model = FCDNN(out_dim=r).to(device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])

    model.eval()

    trainer = PODFCDNNTrainer(
        pod=pod,
        model=model,
        x_mean=checkpoint["x_mean"],
        x_std=checkpoint["x_std"],
        y_mean=checkpoint["y_mean"],
        y_std=checkpoint["y_std"],
        device=device
    )

    return trainer


def predict_and_reconstruct(trainer: PODFCDNNTrainer, Re_query: float):
    """Predict flow fields (u, v, p) with downsampling for rendering speed."""
    coeffs = trainer.predict_coefficients(Re_query)

    pod = trainer.pod
    phi = pod.Phi.numpy() if hasattr(pod.Phi, "numpy") else pod.Phi
    mean = pod.mean.numpy() if hasattr(pod.mean, "numpy") else pod.mean
    xy = pod.xy.numpy() if hasattr(pod.xy, "numpy") else pod.xy

    X_recon = np.dot(phi, coeffs) + mean

    N = pod.N
    u = X_recon[:N]
    v = X_recon[N:2*N]
    p = X_recon[2*N:3*N]

    # Downsample large spatial meshes (>15,000 nodes) to maintain smooth Plotly rendering speed
    if N > 15000:
        step = int(np.ceil(N / 15000))
        u, v, p, xy = u[::step], v[::step], p[::step], xy[::step]

    return {
        "u": u,
        "v": v,
        "p": p,
        "xy": xy,
        "Re": Re_query
    }
