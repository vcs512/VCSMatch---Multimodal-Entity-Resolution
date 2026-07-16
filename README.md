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
| `csv_path` | string | yes | — | CSV file with image filenames |
| `column` | string | yes | — | Column name in CSV with filenames |
| `id_column` | string | yes | — | Column to use for index mapping |
| `batch_size` | int | no | 64 | Images per forward pass |
| `device` | string | no | `"cuda"` if available else `"cpu"` | Torch device |

#### Outputs - Vision Embedding

- `{output_dir}/embedding.safetensors` — tensor key `"embeddings"`,
    shape `(N, embedding_dim)`, ordered by CSV row
- `{output_dir}/index.json` — maps integer index → id from `id_column`

#### Usage - Vision Embedding

```bash
docker compose run --rm embedding-vision
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
| `id_column` | string | yes | — | Column to use for index mapping |
| `batch_size` | int | no | 128 | Texts per forward pass |
| `device` | string | no | `"cuda"` if available else `"cpu"` | Torch device |

#### Outputs - Text Embedding

- `{output_dir}/embedding.safetensors` — tensor key `"embeddings"`, shape `(N, 768)`
- `{output_dir}/index.json` — maps integer index → id from `id_column`

#### Usage - Text Embedding

##### Dataset product titles

```bash
docker compose run --rm embedding-text-titles
```

##### Zero-shot prompts

Example of `prompts.csv`:

```csv
posting_id,title
a product isolated,a product isolated
many products together,many products together
```

Run service:

```bash
docker compose run --rm embedding-text-prompts
```

### 3. EDA Service

Runs zero-shot classification (vision vs. prompt embeddings) and language
detection on product titles, then merges both scores into the source CSV.

#### Config schema (`configs/eda/eda.json`)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `train_csv_path` | string | yes | — | Path to the training CSV |
| `text_column` | string | no | `"title"` | Column with product titles for language detection |
| `vision_embeddings_path` | string | yes | — | Path to pre-computed image embeddings (`.safetensors`) |
| `text_prompt_embeddings_path` | string | yes | — | Path to zero-shot prompt embeddings (`.safetensors`) |
| `output_dir` | string | yes | — | Root directory for all EDA outputs |

#### Outputs - EDA

- `{output_dir}/` — root directory containing the following:

  - `train_augmented.csv` — CSV with two extra columns:
    - `multiples_score` — probability the image shows multiple products together (cosine-similarity zero-shot classification)
    - `indonesian_score` — confidence that the product title is Indonesian (lingua language detection)

  - `row/` — per-row statistics:
    - `train_augmented_rows_statistics.csv` — `describe()` on `multiples_score` and `indonesian_score`
    - `multiples_score_boxplot.jpg`
    - `indonesian_score_boxplot.jpg`

  - `group/` — per-group statistics (grouped by `label_group`):
    - `train_augmented_grouped_statistics.csv` — aggregates: `product_count`, `multiples_score_mean`, `multiples_score_std`, `indonesian_score_mean`, `indonesian_score_std`
    - `train_augmented_grouped_statistics_describe.csv` — `describe()` on `product_count`, `multiples_score_std`, `indonesian_score_std`
    - `product_count_boxplot.jpg`
    - `multiples_score_std_boxplot.jpg`
    - `indonesian_score_std_boxplot.jpg`

#### Usage - EDA

```bash
docker compose up eda
```

### 4. Dataset Split Service

Stratifies product groups by variance and splits into train/val/test preserving
stratum proportions.

#### Config schema (`configs/datasets/dataset_split.json`)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `eda_dir` | string | yes | — | Path to EDA output directory (must contain `train_augmented.csv` and `group/train_augmented_grouped_statistics.csv`) |
| `output_dir` | string | yes | — | Directory to write split outputs |

#### Strategy - Dataset Split

A group is labelled `"higher_variance"` if any of `multiples_score_std`,
`indonesian_score_std`, or `product_count` exceeds its column's 75th percentile
(Q3) across any group.  Otherwise it is `"lower_variance"`.
Groups are then split 70/15/15 (train/val/test) while preserving stratum
proportions.

#### Inputs - Dataset Split

- `train_augmented.csv` — per-row augmented data with `label_group`, `multiples_score`, `indonesian_score`
- `group/train_augmented_grouped_statistics.csv` — pre-computed group aggregates (from EDA service)

#### Outputs - Dataset Split

- `group_summary.csv` — group-level data with `stratum` and `split` columns
- `assignments.csv` — per-row data with `stratum` and `split` columns added
- `stratum_distribution.csv` — group count per stratum
- `group_split_distribution.csv` — group count per split

#### Usage - Dataset Split

```bash
docker compose up dataset-split
```

### 5. Retrieval Service

Runs k-nearest-neighbors search using FAISS (GPU) on pre-computed embeddings,
filtered to a specific dataset split. Returns retrieved posting IDs as JSON
lists for each query product.

#### Config schema (`configs/retrieval/retrieval.json`)

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `embeddings_dir` | string | yes | — | Directory containing `embedding.safetensors` and `index.json` |
| `assignments_path` | string | yes | — | Path to `assignments.csv` with `split` column |
| `output_dir` | string | yes | — | Directory to write results CSV |
| `split` | string | no | `"test"` | Which split to query and search within |
| `k` | int | no | 50 | Number of nearest neighbors |
| `threshold` | float | no | 0.5 | Minimum cosine similarity to retain a neighbor |

#### Strategy - Retrieval

Embeddings are L2-normalized, so FAISS inner product equals cosine similarity.
Only embeddings belonging to the configured split are loaded into the index.
For each query product, the top-k nearest neighbors are retrieved (self-match
is included) and any neighbor below the cosine-similarity threshold
is discarded. Retrieved posting IDs are serialized as a JSON string list.

#### Inputs - Retrieval

- `{embeddings_dir}/embedding.safetensors` — pre-computed embeddings
- `{embeddings_dir}/index.json` — maps integer index → posting_id
- `assignments.csv` — per-row data with `split` and `stratum` columns (from Dataset Split service)

#### Outputs - Retrieval

- `retrieval_results.csv` — one row per query product with columns:
  - `posting_id` — query product ID
  - `label_group` — ground-truth group label
  - `stratum` — stratum assignment
  - `retrieved_products` — JSON string list of retrieved posting IDs

#### Usage - Retrieval

```bash
docker compose up retrieval
```
