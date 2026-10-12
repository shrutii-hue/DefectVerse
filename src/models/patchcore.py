"""PatchCore: Towards Total Recall in Industrial Anomaly Detection.

Implementation of the PatchCore anomaly detection and localization algorithm
tailored for industrial visual inspection (MVTec AD).
Enhanced with ML-based Out-of-Distribution (OOD) Guardrail for enterprise robustness.

Reference:
    Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms


class PatchCore(nn.Module):
    """PatchCore Anomaly Detection and Localization Model with OOD Guardrail.

    Attributes:
        backbone_name: Name of torchvision backbone (e.g. 'resnet18', 'resnet50').
        layers: Intermediate layer names to extract features from.
        device: Torch device (cuda or cpu).
        patch_size: Kernel size for neighborhood average pooling.
        coreset_sampling_ratio: Fraction of patch embeddings to keep in memory bank.
        memory_bank: Subsampled normal patch features tensor of shape (M, D).
        category_centroid: Global structural feature centroid of shape (1, D).
        ood_threshold: Similarity bound to classify in-distribution vs out-of-distribution.
        threshold: Anomaly score decision threshold.
    """

    # Class-level defaults for backward compatibility

    ood_threshold: float = 0.65
    category_centroid: Optional[torch.Tensor] = None
    threshold: float = 0.5
    min_score: float = 0.0
    max_score: float = 1.0

    def __init__(

        self,
        backbone_name: str = "resnet18",
        layers: Tuple[str, ...] = ("layer2", "layer3"),
        device: Optional[Union[str, torch.device]] = None,
        patch_size: int = 3,
        coreset_sampling_ratio: float = 0.1,
        input_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.layers = list(layers)
        self.patch_size = patch_size
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.input_size = input_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Standard ImageNet preprocessing transform
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.input_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Initialize backbone feature extractor
        self.backbone = self._build_backbone(self.backbone_name)
        self.backbone.to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Hook storage
        self.features: Dict[str, torch.Tensor] = {}
        self._hooks: List[Any] = []
        self._register_hooks()

        # Local aggregation pooling
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.patch_size,
            stride=1,
            padding=self.patch_size // 2,
        )

        # Memory bank, calibration, and OOD guardrail state
        self.memory_bank: Optional[torch.Tensor] = None
        self.category_centroid: Optional[torch.Tensor] = None
        self.ood_threshold: float = 0.65
        self.threshold: float = 0.5
        self.min_score: float = 0.0
        self.max_score: float = 1.0

    def _build_backbone(self, backbone_name: str) -> nn.Module:
        """Load pretrained ResNet backbone."""
        backbone_name = backbone_name.lower()
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
        elif backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT
            model = models.resnet34(weights=weights)
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        elif backbone_name == "wide_resnet50_2":
            weights = models.Wide_ResNet50_2_Weights.DEFAULT
            model = models.wide_resnet50_2(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: '{backbone_name}'")
        return model

    def _register_hooks(self) -> None:
        """Register forward hooks on selected intermediate layers."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        def get_hook(layer_name: str):
            def hook(module, input, output):
                self.features[layer_name] = output
            return hook

        for layer_name in self.layers:
            if not hasattr(self.backbone, layer_name):
                raise AttributeError(f"Backbone does not have layer '{layer_name}'")
            layer = getattr(self.backbone, layer_name)
            self._hooks.append(layer.register_forward_hook(get_hook(layer_name)))

    def extract_patch_features(
        self, x: torch.Tensor, return_spatial_dims: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, int, int]]:
        """Extract multi-scale patch features with local neighborhood pooling.

        Args:
            x: Input image tensor of shape (B, 3, H, W) on self.device.
            return_spatial_dims: If True, also return (H_feat, W_feat).

        Returns:
            features: Tensor of shape (B, H_feat * W_feat, Feature_Dim)
            (optional) H_feat, W_feat
        """
        self.features.clear()
        _ = self.backbone(x)

        layer_features = []
        target_h, target_w = None, None

        for idx, layer_name in enumerate(self.layers):
            feat = self.features[layer_name]  # (B, C, H_i, W_i)
            # Local neighborhood pooling
            feat = self.avg_pool(feat)

            if idx == 0:
                target_h, target_w = feat.shape[2], feat.shape[3]
            else:
                # Bilinearly interpolate deeper layers to first layer's resolution
                feat = F.interpolate(
                    feat,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            layer_features.append(feat)

        # Concatenate across channels: (B, total_channels, H_feat, W_feat)
        concat_features = torch.cat(layer_features, dim=1)
        b, c, h, w = concat_features.shape

        # Reshape to (B, H*W, C)
        patch_tokens = concat_features.permute(0, 2, 3, 1).reshape(b, h * w, c)

        if return_spatial_dims:
            return patch_tokens, h, w
        return patch_tokens

    def _subsample_coreset(
        self, features: torch.Tensor, target_count: int
    ) -> torch.Tensor:
        """Perform greedy minimax k-center coreset selection.

        Args:
            features: Memory bank features of shape (N, D).
            target_count: Number of coreset points to retain.

        Returns:
            Subsampled features of shape (target_count, D).
        """
        n, d = features.shape
        if target_count >= n:
            return features

        print(f"[PatchCore] Subsampling {n} patches down to {target_count} ({self.coreset_sampling_ratio * 100:.1f}%)...")

        # Use random projection to speed up distance calculations
        proj_dim = min(128, d)
        random_proj = torch.randn(d, proj_dim, device=features.device)
        random_proj = F.normalize(random_proj, p=2, dim=0)
        projected = torch.matmul(features, random_proj)  # (N, proj_dim)

        # Greedy k-Center selection
        select_indices = torch.zeros(target_count, dtype=torch.long, device=features.device)
        first_idx = torch.randint(0, n, (1,), device=features.device).item()
        select_indices[0] = first_idx

        dists = torch.norm(projected - projected[first_idx : first_idx + 1], dim=1)

        for i in range(1, target_count):
            furthest_idx = torch.argmax(dists).item()
            select_indices[i] = furthest_idx
            new_dists = torch.norm(projected - projected[furthest_idx : furthest_idx + 1], dim=1)
            dists = torch.minimum(dists, new_dists)

        return features[select_indices]

    def fit(
        self,
        data_loader: Union[DataLoader, List[torch.Tensor], List[str]],
        calibrate_threshold: bool = True,
    ) -> None:
        """Build normal memory bank from training data and calibrate OOD guardrail.

        Args:
            data_loader: DataLoader yielding image tensors (or list of tensors/image paths).
            calibrate_threshold: If True, computes empirical threshold from normal training scores.
        """
        self.backbone.eval()
        patch_list = []
        global_repr_list = []

        print(f"[PatchCore] Building memory bank using backbone '{self.backbone_name}' on {self.device}...")

        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch

                if isinstance(images, str):
                    pil_img = Image.open(images).convert("RGB")
                    images = self.transform(pil_img).unsqueeze(0)
                elif isinstance(images, list) and len(images) > 0 and isinstance(images[0], str):
                    tensors = [self.transform(Image.open(p).convert("RGB")) for p in images]
                    images = torch.stack(tensors, dim=0)

                images = images.to(self.device)
                patches = self.extract_patch_features(images)  # (B, H*W, D)
                
                # Global representation per image for OOD Guardrail
                img_global = F.normalize(patches.mean(dim=1), p=2, dim=-1)  # (B, D)
                global_repr_list.append(img_global.cpu())

                patches_flat = patches.reshape(-1, patches.shape[-1])  # (B*H*W, D)
                patch_list.append(patches_flat.cpu())

        all_patches = torch.cat(patch_list, dim=0).to(self.device)
        total_patches = all_patches.shape[0]

        target_count = max(1, int(total_patches * self.coreset_sampling_ratio))
        if target_count < total_patches:
            self.memory_bank = self._subsample_coreset(all_patches, target_count)
        else:
            self.memory_bank = all_patches

        print(f"[PatchCore] Memory bank successfully built with {self.memory_bank.shape[0]} patches (dim={self.memory_bank.shape[1]}).")

        # Calibrate Category Centroid for Out-of-Distribution (OOD) Guardrail
        if global_repr_list:
            all_globals = torch.cat(global_repr_list, dim=0).to(self.device)  # (N_imgs, D)
            self.category_centroid = F.normalize(all_globals.mean(dim=0, keepdim=True), p=2, dim=-1)  # (1, D)
            train_sims = torch.mm(all_globals, self.category_centroid.T).squeeze(-1)
            min_train_sim = float(train_sims.min().item())
            # Guardrail cutoff set with margin below min normal training similarity
            self.ood_threshold = max(0.55, min_train_sim - 0.20)
            print(f"[Guardrail] Category structural profile calibrated: centroid_dim={self.category_centroid.shape[1]}, min_train_sim={min_train_sim:.4f}, ood_cutoff={self.ood_threshold:.4f}")
        else:
            self.category_centroid = F.normalize(self.memory_bank.mean(dim=0, keepdim=True), p=2, dim=-1)
            self.ood_threshold = 0.65

        # Calibrate baseline normal scores to initialize decision threshold
        if calibrate_threshold:
            print("[PatchCore] Calibrating score baseline on normal training data...")
            sample_count = min(total_patches, 2000)
            sub_sample = all_patches[:sample_count]
            with torch.no_grad():
                dists = self._compute_nn_distances(sub_sample)
                self.min_score = float(dists.min().item())
                self.threshold = float(torch.quantile(dists, 0.995).item())
                self.max_score = max(self.threshold * 2.0, float(dists.max().item()))
            print(f"[PatchCore] Calibration complete: baseline threshold={self.threshold:.4f}, min_score={self.min_score:.4f}")

    def _compute_nn_distances(
        self, query_patches: torch.Tensor, chunk_size: int = 1024
    ) -> torch.Tensor:
        """Compute nearest neighbor Euclidean distance to memory bank in chunks."""
        assert self.memory_bank is not None, "Memory bank is not fitted yet."
        num_patches = query_patches.shape[0]
        min_dists = []

        for i in range(0, num_patches, chunk_size):
            chunk = query_patches[i : i + chunk_size]
            dist_matrix = torch.cdist(chunk, self.memory_bank, p=2)
            chunk_min_dist, _ = torch.min(dist_matrix, dim=1)
            min_dists.append(chunk_min_dist)

        return torch.cat(min_dists, dim=0)

    def validate_ood(
        self, image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]
    ) -> Dict[str, Any]:
        """Perform ML-based Out-of-Distribution (OOD) Guardrail validation.

        Computes global feature cosine similarity against the category's structural profile.
        If an image is completely unrelated (e.g. human face, random photo, screenshot),
        the guardrail flags it before downstream anomaly scoring.

        Returns:
            Dict containing:
                'is_ood': bool
                'similarity': float [0, 1]
                'threshold': float
                'guardrail_message': str or None
        """
        if self.category_centroid is None:
            if self.memory_bank is not None:
                self.category_centroid = F.normalize(self.memory_bank.mean(dim=0, keepdim=True), p=2, dim=-1)
            else:
                return {"is_ood": False, "similarity": 1.0, "threshold": self.ood_threshold, "guardrail_message": None}

        orig_np_rgb = self._load_as_rgb_numpy(image_input)
        pil_img = Image.fromarray(orig_np_rgb)
        tensor_x = self.transform(pil_img).unsqueeze(0).to(self.device)

        self.backbone.eval()
        with torch.no_grad():
            patches = self.extract_patch_features(tensor_x)  # (1, H*W, D)
            query_global = F.normalize(patches.mean(dim=1), p=2, dim=-1)  # (1, D)
            similarity = float(torch.mm(query_global, self.category_centroid.T).item())

        is_ood = bool(similarity < self.ood_threshold)
        guardrail_msg = (
            "[GUARDRAIL TRIGGERED]: Out-of-Distribution input. "
            "This image does not match the structural profile of the selected industrial category."
        ) if is_ood else None

        return {
            "is_ood": is_ood,
            "similarity": similarity,
            "threshold": self.ood_threshold,
            "guardrail_message": guardrail_msg,
        }

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        blur_sigma: float = 4.0,
    ) -> Dict[str, Any]:
        """Predict anomaly score, binary decision, localization heatmap, and OOD status.

        Args:
            image_input: File path, PIL Image, HWC uint8 numpy array, or (1, 3, H, W) tensor.
            blur_sigma: Sigma for Gaussian smoothing of the localization heatmap.

        Returns:
            Dict containing:
                'anomaly_score': float, global image-level anomaly score
                'is_anomaly': bool, True if score >= threshold
                'threshold': float, current decision threshold
                'is_ood': bool, True if input fails OOD structural guardrail
                'ood_similarity': float, structural similarity to category profile
                'ood_threshold': float, OOD cutoff bound
                'guardrail_message': str or None
                'anomaly_map': np.ndarray of shape (H_orig, W_orig), float [0, 1]
                'heatmap_colored': np.ndarray of shape (H_orig, W_orig, 3), RGB uint8 heatmap
                'heatmap_overlay': np.ndarray of shape (H_orig, W_orig, 3), RGB uint8 blend
                'original_image': np.ndarray of shape (H_orig, W_orig, 3), RGB uint8
        """
        if self.memory_bank is None:
            raise RuntimeError("PatchCore must be fitted or loaded before making predictions.")

        # Load and preserve original image for visualization
        orig_np_rgb = self._load_as_rgb_numpy(image_input)
        h_orig, w_orig = orig_np_rgb.shape[:2]

        # Prepare input tensor
        pil_img = Image.fromarray(orig_np_rgb)
        tensor_x = self.transform(pil_img).unsqueeze(0).to(self.device)

        self.backbone.eval()
        with torch.no_grad():
            patch_tokens, feat_h, feat_w = self.extract_patch_features(
                tensor_x, return_spatial_dims=True
            )
            patches = patch_tokens[0]  # (feat_h * feat_w, D)

            # OOD Structural Guardrail Check
            if self.category_centroid is None:
                self.category_centroid = F.normalize(self.memory_bank.mean(dim=0, keepdim=True), p=2, dim=-1)
            query_global = F.normalize(patches.mean(dim=0, keepdim=True), p=2, dim=-1)  # (1, D)
            similarity = float(torch.mm(query_global, self.category_centroid.T).item())

            # Compute nearest neighbor patch distances
            patch_scores = self._compute_nn_distances(patches)  # (feat_h * feat_w,)

        # Check OOD status
        is_ood = bool(similarity < self.ood_threshold)
        guardrail_msg = (
            "[GUARDRAIL TRIGGERED]: Out-of-Distribution input. "
            "This image does not match the structural profile of the selected industrial category."
        ) if is_ood else None

        # Spatial 2D anomaly map: (feat_h, feat_w)
        score_map = patch_scores.reshape(feat_h, feat_w).cpu().numpy()

        # Image-level anomaly score
        image_score = float(score_map.max())
        is_anomaly = bool(image_score >= self.threshold)

        # Bilinear resize to original image dimensions
        anomaly_map = cv2.resize(
            score_map, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC
        )

        # Gaussian smoothing
        if blur_sigma > 0:
            anomaly_map = gaussian_filter(anomaly_map, sigma=blur_sigma)

        # Normalize heatmap to [0, 1] for visualization
        denom = max(1e-6, (self.max_score - self.min_score))
        norm_map = np.clip((anomaly_map - self.min_score) / denom, 0.0, 1.0)

        # Colorize heatmap (COLORMAP_JET)
        heatmap_uint8 = (norm_map * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Overlay blend onto original image
        alpha = 0.45
        overlay = (alpha * heatmap_rgb + (1 - alpha) * orig_np_rgb).astype(np.uint8)

        return {
            "anomaly_score": image_score,
            "is_anomaly": is_anomaly,
            "threshold": self.threshold,
            "is_ood": is_ood,
            "ood_similarity": similarity,
            "ood_threshold": self.ood_threshold,
            "guardrail_message": guardrail_msg,
            "anomaly_map": norm_map,
            "heatmap_colored": heatmap_rgb,
            "heatmap_overlay": overlay,
            "original_image": orig_np_rgb,
        }

    def _load_as_rgb_numpy(
        self, image_input: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """Helper to convert any input format to standard RGB HWC uint8 numpy array."""
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
            return np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return np.stack([image_input] * 3, axis=-1)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            elif image_input.shape[2] == 3:
                return image_input.copy()
            raise ValueError(f"Unexpected numpy image shape: {image_input.shape}")
        elif isinstance(image_input, torch.Tensor):
            t = image_input.detach().cpu()
            if t.ndim == 4:
                t = t[0]
            if t.dtype in (torch.float32, torch.float64):
                t = (t.clamp(0, 1) * 255).to(torch.uint8)
            return t.permute(1, 2, 0).numpy()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def save(self, filepath: Union[str, Path]) -> str:
        """Persist model configuration, weights, memory bank, and OOD profile."""
        assert self.memory_bank is not None, "Cannot save an unfitted model. Call fit() first."
        save_path = Path(filepath).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "backbone_name": self.backbone_name,
            "layers": self.layers,
            "patch_size": self.patch_size,
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "input_size": self.input_size,
            "threshold": self.threshold,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "memory_bank": self.memory_bank.cpu(),
            "category_centroid": self.category_centroid.cpu() if self.category_centroid is not None else None,
            "ood_threshold": self.ood_threshold,
        }

        torch.save(payload, str(save_path))
        print(f"[PatchCore] Model successfully saved to {save_path}")
        return str(save_path)

    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> "PatchCore":
        """Load a persisted PatchCore checkpoint from disk."""
        checkpoint_path = Path(filepath).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        print(f"[PatchCore] Loading model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

        model = cls(
            backbone_name=checkpoint["backbone_name"],
            layers=tuple(checkpoint["layers"]),
            device=device,
            patch_size=checkpoint.get("patch_size", 3),
            coreset_sampling_ratio=checkpoint.get("coreset_sampling_ratio", 0.1),
            input_size=tuple(checkpoint.get("input_size", (224, 224))),
        )

        model.threshold = checkpoint.get("threshold", 0.5)
        model.min_score = checkpoint.get("min_score", 0.0)
        model.max_score = checkpoint.get("max_score", 1.0)
        model.memory_bank = checkpoint["memory_bank"].to(model.device)

        # Restore OOD Guardrail state
        cat_cent = checkpoint.get("category_centroid")
        if cat_cent is not None:
            model.category_centroid = cat_cent.to(model.device)
        else:
            model.category_centroid = F.normalize(model.memory_bank.mean(dim=0, keepdim=True), p=2, dim=-1)

        model.ood_threshold = checkpoint.get("ood_threshold", 0.65)

        print(
            f"[PatchCore] Restored successfully: backbone={model.backbone_name}, "
            f"patches={model.memory_bank.shape[0]}, threshold={model.threshold:.4f}, ood_cutoff={model.ood_threshold:.4f}"
        )
        return model
