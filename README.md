#  Crypto Keyword Clustering System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)


An BERT-based keyword clustering system specialized for cryptocurrency SEO, capable of processing 130,000+ keywords with hierarchical topic modeling.

## Features

- **Hierarchical Clustering**: 4-level taxonomy (Pillar → Primary → Secondary → Subtopic)
- **BERT Embeddings**: State-of-the-art sentence transformers for semantic understanding
- **Crypto-Specific**: 200+ validated crypto terms and patterns
- **Ultra-Accurate**: HDBSCAN clustering with validation and refinement
- **Scalable**: Processes 130k+ keywords in ~26 minutes
- **SEO-Optimized**: Built for search engine keyword taxonomy

##  Installation

### Prerequisites
- Python 3.8 or higher
- 8GB+ RAM recommended
- CUDA-capable GPU (optional, for acceleration)

### Setup

1. Clone the repository:
\`\`\`bash
git clone  https://github.com/abushanisro/bert-Lama
cd crypto-keyword-clustering
\`\`\`

2. Create virtual environment:
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Quick Start

### Basic Usage

\`\`\`python
from src.crypto_clustering_v2 import run_crypto_clustering_pipeline

# Run the complete pipeline
result = run_crypto_clustering_pipeline()
\`\`\`

### Custom Configuration

\`\`\`python
from src.crypto_clustering_v2 import CryptoConfig, CryptoKeywordProcessor, CryptoClusteringEngine

# Configure settings
CryptoConfig.INPUT_FILE = 'your_keywords.csv'
CryptoConfig.PILLAR_CLUSTERS = 15
CryptoConfig.MIN_CLUSTER_SIZE = 10

# Process keywords
processor = CryptoKeywordProcessor()
df = processor.load_and_process_keywords(CryptoConfig.INPUT_FILE)

# Create clusters
engine = CryptoClusteringEngine()
result = engine.create_hierarchical_clusters(df)
\`\`\`

## Output Structure

The system generates an Excel file with multiple sheets:

1. **Crypto_Keywords_Clustered**: Main results with hierarchical clustering
2. **Hierarchy_Tree**: Complete taxonomy structure
3. **Summary_Statistics**: Clustering metrics and statistics

### Output Fields

- `original_keyword`: Original keyword
- `cleaned_keyword`: Processed keyword
- `pillar_id`, `pillar_name`: Top-level category
- `primary_id`, `primary_name`: Primary topic
- `secondary_id`, `secondary_name`: Secondary topic
- `subtopic_id`, `subtopic_name`: Granular subtopic
- `search_volume`, `competition`, `cpc`: SEO metrics
- Various crypto-specific features

## Architecture

- **BERT Embeddings**: `sentence-transformers/all-mpnet-base-v2`
- **Dimension Reduction**: PCA (776d → 100d) → UMAP (100d → 75d)
- **Clustering**: HDBSCAN with K-Means fallback
- **Validation**: Multi-stage cluster refinement

## Performance

- **Processing Time**: ~26 minutes for 130k keywords
- **Memory Usage**: 2.5GB peak
- **Accuracy**: >90% relevant keyword retention

## Documentation

- [Technical Documentation v1](docs/technical_docs_v1.md)
- [Technical Documentation v2](docs/technical_docs_v2.md)
- [API Reference](docs/api_reference.md)

## Testing

Run tests with pytest:

\`\`\`bash
pytest tests/
\`\`\`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## Acknowledgments

- Sentence Transformers for BERT models
- HDBSCAN for clustering algorithms
- UMAP for dimension reduction

## Contact

Mohammed Abushan email@giottusabu@gmail.com
