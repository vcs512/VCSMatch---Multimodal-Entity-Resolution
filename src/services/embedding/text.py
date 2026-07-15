from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from safetensors.numpy import save_file
from transformers import SiglipModel, SiglipProcessor

from src.schemas.embedding.text import TextEmbeddingConfig

logger = logging.getLogger(__name__)


class TextEmbeddingService:
    """Extract SigLIP text embeddings from a CSV column.

    Usage:
        service = TextEmbeddingService("configs/embedding/text_titles.json")
        service.process()
    """

    def __init__(self, config_path: str) -> None:
        """Initialize the service and validate the config file.

        Args:
            config_path: Path to the JSON config file.
        """
        with open(config_path) as f:
            self.config = TextEmbeddingConfig.model_validate_json(f.read())

        self.model_path = self.config.model_path
        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor: SiglipProcessor | None = None
        self.model: SiglipModel | None = None

    def _load_model(self) -> None:
        """Lazily load the SigLIP model and processor onto the target device."""
        if self.model is not None:
            return
        logger.info("Loading SigLIP model from %s", self.model_path)
        self.processor = SiglipProcessor.from_pretrained(
            pretrained_model_name_or_path=str(self.model_path),
            local_files_only=True,
        )
        self.model = SiglipModel.from_pretrained(
            pretrained_model_name_or_path=str(self.model_path),
            local_files_only=True,
        )
        self.model = self.model.to(self.device)
        self.model.compile()
        self.model.eval()

    def _encode(
        self, texts: list[str], batch_size: int | None = None
    ) -> np.ndarray:
        """Encode texts through the SigLIP text model.

        Args:
            texts: List of strings to embed.
            batch_size: Max texts per forward pass. If None, all texts
                        are processed in a single batch.

        Returns:
            Array of shape (len(texts), hidden_size).
        """
        self._load_model()
        if batch_size is None:
            batch_size = len(texts)

        all_embeddings = []
        for i in tqdm(
            range(0, len(texts), batch_size),
            desc="Encoding texts",
            unit="batch",
        ):
            batch_texts = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.text_model(**inputs)
                all_embeddings.append(outputs.pooler_output.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def process(self) -> None:
        """Read texts from a CSV column, encode, and save as .safetensors.

        The embeddings are L2-normalized before being persisted.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(filepath_or_buffer=self.config.csv_path)
        texts = df[self.config.column].astype(str).tolist()
        logger.info(
            "Encoding %d texts from column '%s'", len(texts), self.config.column
        )

        embeddings = self._encode(texts=texts, batch_size=self.config.batch_size)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        save_file(
            tensor_dict={"embeddings": embeddings},
            filename=str(
                self.config.output_dir / "embedding.safetensors"
            ),
        )
        logger.info("Saved %d title embeddings", len(embeddings))

        index = {
            str(idx): row[self.config.id_column]
            for idx, row in df.iterrows()
        }
        with open(file=self.config.output_dir / "index.json", mode="w") as f:
            json.dump(obj=index, fp=f, indent=2)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    TextEmbeddingService(config_path=args.config).process()
