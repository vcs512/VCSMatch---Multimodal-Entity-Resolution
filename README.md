# VCSMatch: Multimodal Entity Resolution

Framework/Pipelines to work with vision-language entity resolution.

Based in the `Shopee - Price Match Garantee` Kaggle competition/dataset
([reference](https://www.kaggle.com/competitions/shopee-product-matching)).

## Roadmap

Experiments include:

- EDA (exploratory data analysis): information to split holdout set
- Isolated analysis: vision and language isolated experiments
- Naive fusion analysis: simple multimodal composition (concatenation, sum)
- Specialized head tuning: usage of metric learning heads

## Docker

All services are exposed as Docker Compose services in `docker-compose.yml`.

## Services

### 1. Vision Embedding Service

Extract SigLIP vision embeddings from product images.

#### Config schema (`configs/embedding/vision.json`)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `model_path` | string | yes | — | Path to SigLIP checkpoint directory |
| `image_dir` | string | yes | — | Directory containing product images |
| `output_dir` | string | yes | — | Directory to write embeddings |
| `csv_path` | string | no | null | CSV with image filenames; if null, scans `image_dir` |
| `image_column` | string | no | `"image"` | Column name in CSV with filenames |
| `batch_size` | int | no | 64 | Images per forward pass |
| `device` | string | no | `"cuda"` if available else `"cpu"` | Torch device |
| `num_workers` | int | no | 0 | DataLoader workers |

#### Outputs - Vision Embedding

- `{output_dir}/image_embeddings.safetensors` — tensor key `"embeddings"`,
    shape `(N, embedding_dim)`, ordered by CSV row (or filename sort)
- `{output_dir}/image_index.json` — maps integer index → filename

#### Usage - Vision Embedding

```bash
docker compose up embedding-vision
```

### 2. Text Embedding Service

Extract SigLIP text embeddings from a CSV column.

#### Config schema (`configs/embedding/text_titles.json`)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `model_path` | string | yes | — | Path to SigLIP checkpoint directory |
| `csv_path` | string | yes | — | CSV file to read texts from |
| `column` | string | yes | — | Column with text data |
| `output_dir` | string | yes | — | Directory to write embeddings |
| `id_column` | string | no | null | Column to use for index mapping |
| `batch_size` | int | no | 128 | Texts per forward pass |
| `device` | string | no | `"cuda"` if available else `"cpu"` | Torch device |

#### Outputs

- `{output_dir}/title_embeddings.safetensors` — tensor key `"embeddings"`, shape `(N, 768)`
- `{output_dir}/title_index.json` — maps integer index → id (only if `id_column` set)

#### Usage

For product titles:

```bash
docker compose up embedding-text-titles
```

For zero-shot prompts:

```bash
docker compose up embedding-text-prompts
```
