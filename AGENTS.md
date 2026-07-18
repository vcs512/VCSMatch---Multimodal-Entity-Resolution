# Multimodal Entity Resolution

Entity resolution using image and text.

Applied in the "Shopee - Price Match Garantee" Kaggle competition/dataset.

## Development

- Python language
- uv dependency manager
- docker compose for running services
- SOLID principles
- Concise comments (never inline)
- Google style docstrings (always state args and returns when not 'None')
- Ruff linter
- Explicit arguments names in functions (avoid positionals)
- Layered architecture (folders: `services/`, `core/`, `schemas/`)
- DTO for more than one object returned in a method (pydantic)
- Services configurations in JSON (`configs/` directory)
- Every service must have the inputs, usage and outputs expected described in
    the project README.md
- Every service should run as a docker compose service
- `data/` directory contains the dataset to be used
- `checkpoints/` directory contains the AI models
- `tests/` directory contains unity tests
- Working with reduced VRAM (4 GB)

## Pipelines

- Normalized embeddings from modalities (vision and text) are computed offline
    by the `embedding` services (numpy safetensors)
- Embeddings (safetensors) accompany a `*_index.json` file which maps each line
    of the object loaded (keys) to what it represents (values)
