from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from safetensors.numpy import save_file
from transformers import SiglipModel, SiglipProcessor

from src.schemas.embedding.vision import VisionEmbeddingConfig

logger = logging.getLogger(__name__)


class VisionEmbeddingService:
    """Extract SigLIP vision embeddings from a CSV's image column.

    Images listed in a CSV column are loaded from ``image_dir``, processed
    by the SigLIP vision tower, and saved as a single .safetensors file
    ordered by CSV row.

    Usage:
        service = VisionEmbeddingService("configs/embedding/vision.json")
        embeddings = service.process_all()
    """

    def __init__(self, config_path: str) -> None:
        """Initialize the service and validate the config file.

        Args:
            config_path: Path to the JSON config file.
        """
        with open(config_path) as f:
            self.config = VisionEmbeddingConfig.model_validate_json(f.read())

        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.processor: SiglipProcessor | None = None
        self.model: SiglipModel | None = None

    def _load_model(self) -> None:
        """Lazily load the SigLIP model and processor onto the target device."""
        if self.model is not None:
            return
        logger.info("Loading SigLIP model from %s", self.config.model_path)
        self.processor = SiglipProcessor.from_pretrained(
            pretrained_model_name_or_path=str(self.config.model_path),
            local_files_only=True,
        )
        self.model = SiglipModel.from_pretrained(
            pretrained_model_name_or_path=str(self.config.model_path),
            local_files_only=True,
        )
        self.model = self.model.to(self.device)
        self.model.compile()
        self.model.eval()

    def _load_images(self, filenames: list[str]) -> list[Image.Image]:
        """Load a list of filenames from the image directory as RGB PIL images.

        Args:
            filenames: List of image filenames relative to ``image_dir``.

        Returns:
            List of loaded RGB PIL images.
        """
        images = []
        for fname in filenames:
            path = self.config.image_dir / fname
            images.append(Image.open(fp=path).convert("RGB"))
        return images

    def _get_image_filenames(self) -> list[str]:
        """Return ordered list of image filenames from the CSV.

        Returns:
            Ordered list of image filenames.
        """
        df = pd.read_csv(filepath_or_buffer=self.config.csv_path)
        return df[self.config.column].astype(str).tolist()

    def process_all(self) -> np.ndarray:
        """Process all images and save embeddings to disk.

        Returns:
            Array of shape (num_images, hidden_size) with the vision embeddings.
        """
        self._load_model()
        filenames = self._get_image_filenames()
        logger.info("Processing %d images", len(filenames))

        all_embeddings = []
        for i in tqdm(
            range(0, len(filenames), self.config.batch_size),
            desc="Encoding images",
            unit="batch",
        ):
            batch = filenames[i : i + self.config.batch_size]
            images = self._load_images(batch)
            inputs = self.processor(images=images, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model.vision_model(**inputs)
                emb = outputs.pooler_output.cpu().numpy()
            all_embeddings.append(emb)

        embeddings = np.concatenate(all_embeddings, axis=0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        save_file(
            tensor_dict={"embeddings": embeddings},
            filename=str(self.config.output_dir / "embedding.safetensors"),
        )
        logger.info("Saved %d image embeddings", len(embeddings))

        df = pd.read_csv(filepath_or_buffer=self.config.csv_path)
        index = {
            str(idx): row[self.config.id_column]
            for idx, row in df.iterrows()
        }
        with open(
            file=self.config.output_dir / "index.json", mode="w"
        ) as f:
            json.dump(obj=index, fp=f, indent=2)

        return embeddings


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    VisionEmbeddingService(config_path=args.config).process_all()
