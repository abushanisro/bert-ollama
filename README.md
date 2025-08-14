#  Crypto Keyword Clustering System

[Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)


An BERT-based keyword clustering system specialized for cryptocurrency SEO, capable of processing 130,000+ keywords with hierarchical topic modeling.

## Features
Hybrid BERT + Local Llama Clustering System
- **Hierarchical Clustering**: 2-level taxonomy (Pillar →topic)
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
cd 
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


https://docs.google.com/document/d/1gtgU58ZzfXGYsbtq7UwmaSOnXETX6wFqnaBuarlZfi0/edit?usp=sharing


# Acknowledgments

- Sentence Transformers for BERT models
- HDBSCAN for clustering algorithms
- UMAP for dimension reduction

## Contact

Mohammed Abushan email@giottusabu@gmail.com
