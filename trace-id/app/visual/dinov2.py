"""
TRACE-ID — DINOv2 Visual Verifier
Independent visual similarity verification using Facebook Research DINOv2.

Purpose:
  Confirm that a candidate image is visually related to the input image.
  This is a secondary verification step — NOT a reverse image search engine.

Model:
  dinov2_vits14 (ViT-S/14, 21M params, 384-dim embeddings)
  Loaded via torch.hub on first use (~80 MB download).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app import config

logger = logging.getLogger(__name__)


@dataclass
class VisualVerificationResult:
    """Result of DINOv2 visual similarity comparison."""

    similarity: float
    """Cosine similarity in [0, 1] between DINOv2 embeddings."""

    passed: bool
    """True if similarity >= VISUAL_SIMILARITY_THRESHOLD."""

    threshold_used: float
    """The threshold applied."""

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"


class DINOv2Verifier:
    """
    Singleton DINOv2 visual similarity verifier.

    Usage:
        verifier = DINOv2Verifier()
        result = verifier.compare(image_path_a, image_path_b)
    """

    _instance: Optional["DINOv2Verifier"] = None
    _model = None
    _transform = None
    _device = None

    def __new__(cls) -> "DINOv2Verifier":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self) -> None:
        """Lazy-load DINOv2 model on first use."""
        if self._model is not None:
            return

        try:
            import torch
            import torchvision.transforms as T
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch/torchvision not installed. "
                "Run: pip install torch torchvision"
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading DINOv2 model '%s' on %s — first run downloads ~80 MB",
            config.DINOV2_MODEL_NAME,
            device,
        )

        try:
            model = torch.hub.load(
                "facebookresearch/dinov2",
                config.DINOV2_MODEL_NAME,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load DINOv2 '{config.DINOV2_MODEL_NAME}': {exc}\n"
                "Check internet connection or set MOCK_MODE=true for offline testing."
            ) from exc

        model.eval()
        model = model.to(device)

        size = config.DINOV2_IMAGE_SIZE
        transform = T.Compose(
            [
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self._model = model
        self._transform = transform
        self._device = device
        logger.info("DINOv2 model loaded successfully.")

    def embed(self, image_path: Path | str | np.ndarray) -> np.ndarray:
        """
        Generate a DINOv2 embedding for an image.

        Args:
            image_path: Path to image file, or a BGR numpy array.

        Returns:
            L2-normalized embedding vector (384-dim for ViT-S/14).
        """
        self._ensure_loaded()

        import torch
        from PIL import Image as PILImage

        if isinstance(image_path, np.ndarray):
            # Convert BGR numpy → PIL RGB
            import cv2
            rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
            pil_image = PILImage.fromarray(rgb)
        else:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found for DINOv2: {path}")
            pil_image = PILImage.open(path).convert("RGB")

        tensor = self._transform(pil_image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self._model(tensor)  # (1, embed_dim)

        embedding = features.squeeze(0).cpu().numpy().astype(np.float32)

        # L2-normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        return embedding

    def compare(
        self,
        image_a: Path | str | np.ndarray,
        image_b: Path | str | np.ndarray,
        threshold: float | None = None,
    ) -> VisualVerificationResult:
        """
        Compare two images using DINOv2 cosine similarity.

        Args:
            image_a: Input image (path or BGR numpy array).
            image_b: Candidate image (path or BGR numpy array).
            threshold: Override for VISUAL_SIMILARITY_THRESHOLD.

        Returns:
            VisualVerificationResult with similarity score and PASS/FAIL verdict.
        """
        if threshold is None:
            threshold = config.VISUAL_SIMILARITY_THRESHOLD

        if isinstance(image_a, np.ndarray) and image_a.ndim == 1 and image_a.shape[0] == 384:
            emb_a = image_a
        else:
            emb_a = self.embed(image_a)
        emb_b = self.embed(image_b)

        similarity = float(np.dot(emb_a, emb_b))  # both L2-normalized → dot = cosine

        passed = similarity >= threshold

        logger.debug(
            "DINOv2 visual similarity: %.4f (threshold=%.4f) → %s",
            similarity,
            threshold,
            "PASS" if passed else "FAIL",
        )

        return VisualVerificationResult(
            similarity=round(similarity, 6),
            passed=passed,
            threshold_used=threshold,
        )


class MockDINOv2Verifier:
    """
    Mock DINOv2 verifier for MOCK_MODE.
    Always returns a fixed passing score. Clearly labeled as mock.
    Never used in production.
    """

    def compare(
        self,
        image_a,
        image_b,
        threshold: float | None = None,
    ) -> VisualVerificationResult:
        logger.warning("[MOCK MODE] DINOv2 returning mock visual similarity score")
        if threshold is None:
            threshold = config.VISUAL_SIMILARITY_THRESHOLD
        mock_sim = 0.72  # Fixed mock value — clearly not a real measurement
        return VisualVerificationResult(
            similarity=mock_sim,
            passed=mock_sim >= threshold,
            threshold_used=threshold,
        )

    def embed(self, image_path) -> np.ndarray:
        logger.warning("[MOCK MODE] DINOv2 returning mock embedding")
        return np.ones(384, dtype=np.float32) / np.sqrt(384)


def get_verifier() -> DINOv2Verifier | MockDINOv2Verifier:
    """Return the appropriate verifier based on MOCK_MODE."""
    if config.MOCK_MODE:
        return MockDINOv2Verifier()
    return DINOv2Verifier()
