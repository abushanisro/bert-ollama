"""
Enhanced BERT-Based Keyword Clustering System - All Keywords Version
Processes ALL keywords without removal - 200k+ keywords with 2-level hierarchy
"""

import os
import gc
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Set
from collections import defaultdict, Counter
import re
import pickle
from pathlib import Path
import json

# Core ML and NLP libraries
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import umap
import torch

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============= OPTIMIZED CONFIGURATION =============
class Config:
    """Configuration for processing 200k keywords"""
    
    # File paths - UPDATE THESE TO YOUR PATHS
    INPUT_FILE = '/home/admin1/Downloads/demo_crypto/FInal list of crypto terms.xlsx'
    OUTPUT_DIR = '/home/admin1/Downloads/demo_crypto/output'
    OUTPUT_FILE = f'{OUTPUT_DIR}/crypto_clusters_all_keywords.xlsx'
    
    # BERT Model - Best quality
    EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
    
    # Processing Configuration for 200k keywords
    BATCH_SIZE = 1000  # Larger batch for 200k keywords
    MAX_SEQUENCE_LENGTH = 128
    MAX_KEYWORDS = 200000  # Process up to 2 lakh keywords
    
    # Simple 2-Level Clustering
    PILLAR_CLUSTERS = 15      # Main pillar categories
    TOPICS_PER_PILLAR = 10    # Average topics per pillar
    TOTAL_TOPICS = 150        # Total topic clusters
    
    # Clustering Parameters
    UMAP_N_NEIGHBORS = 15
    UMAP_N_COMPONENTS = 50
    UMAP_MIN_DIST = 0.0
    UMAP_METRIC = 'cosine'
    
    # Quality Control
    MIN_CLUSTER_SIZE = 10
    MIN_KEYWORDS_FOR_TOPIC = 5
    
    # Memory Management
    ENABLE_CACHING = True
    CACHE_DIR = 'cache_2lakh'

# ============= KEYWORD PROCESSOR (SIMPLIFIED - NO REMOVAL) =============
class KeywordProcessor:
    """Process and clean keywords without removing any"""
    
    def __init__(self):
        self.stats = {
            'total_processed': 0,
            'duplicates_found': 0
        }
        
    def load_and_process(self, file_path: str, max_keywords: int = 200000) -> pd.DataFrame:
        """Load and process ALL keywords without removal"""
        print(f"\n Loading keywords from: {file_path}")
        
        # Load data
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, nrows=max_keywords)
        else:
            df = pd.read_csv(file_path, nrows=max_keywords)
        
        print(f"✓ Loaded {len(df)} keywords")
        
        # Find keyword column
        keyword_col = self._find_keyword_column(df)
        print(f"✓ Using column: '{keyword_col}'")
        
        # Process keywords (cleaning only, no removal)
        processed_df = self._process_keywords(df, keyword_col)
        
        return processed_df
    
    def _find_keyword_column(self, df: pd.DataFrame) -> str:
        """Find the keyword column"""
        possible_names = ['keyword', 'keywords', 'query', 'term', 'search_term']
        
        for col in df.columns:
            if col.lower() in possible_names:
                return col
        
        # Return first text column
        for col in df.columns:
            if df[col].dtype == 'object':
                return col
        
        return df.columns[0]
    
    def _process_keywords(self, df: pd.DataFrame, keyword_col: str) -> pd.DataFrame:
        """Process keywords - clean but keep ALL"""
        print("\n🔧 Processing keywords (keeping all)...")
        
        # Create working dataframe
        all_keywords = []
        
        for idx, row in df.iterrows():
            keyword = str(row[keyword_col])
            
            # Light cleaning - preserve original meaning
            cleaned = keyword.strip()
            # Only remove excessive whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = cleaned.strip()
            
            # If keyword becomes empty after cleaning, use original
            if not cleaned:
                cleaned = keyword
            
            all_keywords.append({
                'original_keyword': keyword,
                'cleaned_keyword': cleaned,
                'search_volume': row.get('search_volume', 0) if 'search_volume' in row else 0,
                'competition': row.get('competition', 0) if 'competition' in row else 0,
                'cpc': row.get('cpc', 0.0) if 'cpc' in row else 0.0
            })
        
        # Create dataframe
        result_df = pd.DataFrame(all_keywords)
        
        # Track duplicates but don't remove them (optional: you can uncomment to remove duplicates)
        before_dedup = len(result_df)
        # Uncomment the next line if you want to remove exact duplicates
        # result_df = result_df.drop_duplicates(subset=['cleaned_keyword'], keep='first')
        duplicates = before_dedup - len(result_df)
        
        if duplicates > 0:
            print(f"   • Found {duplicates} duplicate keywords (kept all)")
            self.stats['duplicates_found'] = duplicates
        
        # Add features
        result_df = self._add_features(result_df)
        result_df = result_df.reset_index(drop=True)
        
        self.stats['total_processed'] = len(result_df)
        
        print(f"\n Processing Results:")
        print(f"   • Total Keywords Processed: {len(result_df):,}")
        print(f"   • All keywords retained for clustering")
        
        return result_df
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for clustering"""
        df['word_count'] = df['cleaned_keyword'].str.split().str.len()
        df['char_count'] = df['cleaned_keyword'].str.len()
        df['has_numbers'] = df['cleaned_keyword'].str.contains(r'\d', na=False)
        df['has_question'] = df['cleaned_keyword'].str.contains(
            r'how|what|why|when|where|which|who', na=False, case=False
        )
        
        # Handle any NaN values
        df['word_count'] = df['word_count'].fillna(0).astype(int)
        df['char_count'] = df['char_count'].fillna(0).astype(int)
        df['has_numbers'] = df['has_numbers'].fillna(False)
        df['has_question'] = df['has_question'].fillna(False)
        
        return df

# ============= TOPIC NAME GENERATOR =============
class TopicNameGenerator:
    """Generate clean, non-duplicate topic names"""
    
    def __init__(self):
        self.used_words = set()  # Track used words to avoid duplicates
        self.topic_cache = {}
        
    def generate_topic_name(self, keywords: List[str], level: str = 'topic') -> str:
        """Generate unique topic name without duplicate words"""
        if not keywords:
            return f"Empty_{level.title()}"
        
        # Extract important terms using TF-IDF
        important_terms = self._extract_important_terms(keywords)
        
        # Filter out already used words
        available_terms = [term for term in important_terms if term not in self.used_words]
        
        if not available_terms:
            # If all terms are used, use with numbering
            available_terms = important_terms[:3] if important_terms else ['Topic']
            topic_name = self._create_topic_name(available_terms, level)
            topic_name = f"{topic_name}_{len(self.topic_cache) + 1}"
        else:
            topic_name = self._create_topic_name(available_terms[:3], level)
        
        # Add used words to set
        for word in topic_name.split():
            self.used_words.add(word.lower())
        
        return topic_name
    
    def _extract_important_terms(self, keywords: List[str]) -> List[str]:
        """Extract most important terms from keywords"""
        if len(keywords) < 2:
            return keywords[0].split() if keywords else []
        
        try:
            # Use TF-IDF to find important terms
            tfidf = TfidfVectorizer(
                max_features=10,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=1,
                lowercase=True
            )
            
            # Filter out empty keywords
            valid_keywords = [k for k in keywords if k and len(k) > 0]
            if not valid_keywords:
                return ['General']
            
            tfidf_matrix = tfidf.fit_transform(valid_keywords)
            feature_names = tfidf.get_feature_names_out()
            
            # Get importance scores
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-10:][::-1]
            
            # Extract top terms
            top_terms = []
            for idx in top_indices:
                term = feature_names[idx]
                # Clean term
                term = re.sub(r'[^\w\s]', '', term).strip()
                if term and len(term) > 2:
                    top_terms.append(term.title())
            
            return top_terms if top_terms else ['General']
            
        except Exception as e:
            # Fallback to frequency-based extraction
            return self._extract_by_frequency(keywords)
    
    def _extract_by_frequency(self, keywords: List[str]) -> List[str]:
        """Extract terms by frequency"""
        word_freq = Counter()
        
        for keyword in keywords:
            if keyword:  # Check if keyword is not empty
                words = keyword.lower().split()
                for word in words:
                    if len(word) > 2:
                        word_freq[word] += 1
        
        # Get top words
        top_words = [word.title() for word, _ in word_freq.most_common(10)]
        return top_words if top_words else ['General']
    
    def _create_topic_name(self, terms: List[str], level: str) -> str:
        """Create topic name from terms"""
        if not terms:
            return f"General_{level.title()}"
        
        # Remove duplicates within the name
        seen = set()
        unique_terms = []
        for term in terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                unique_terms.append(term)
        
        if level == 'pillar':
            # Pillar: 1-2 words
            return " ".join(unique_terms[:2])
        else:
            # Topic: 2-3 words
            return " ".join(unique_terms[:3])
    
    def reset_used_words(self):
        """Reset used words tracker"""
        self.used_words = set()

# ============= CLUSTERING ENGINE =============
class HierarchicalClusteringEngine:
    """Create 2-level hierarchical clusters"""
    
    def __init__(self):
        self.embedding_model = None
        self.embeddings = None
        self.topic_generator = TopicNameGenerator()
        
    def create_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create 2-level clusters: Pillars and Topics"""
        if len(df) == 0:
            return df
        
        print("\n Starting BERT-based clustering...")
        print(f"   Processing {len(df):,} keywords")
        
        keywords = df['cleaned_keyword'].tolist()
        
        # Step 1: Create embeddings
        print("   Creating BERT embeddings...")
        self.embeddings = self._create_embeddings(keywords)
        
        # Step 2: Reduce dimensions
        print("   Reducing dimensions with UMAP...")
        reduced_embeddings = self._reduce_dimensions(self.embeddings)
        
        # Step 3: Create pillar clusters
        print(f"   Creating {Config.PILLAR_CLUSTERS} pillar clusters...")
        pillar_labels = self._create_pillar_clusters(reduced_embeddings)
        
        # Step 4: Create topic clusters within pillars
        print(f"   Creating topic clusters within pillars...")
        topic_labels = self._create_topic_clusters(reduced_embeddings, pillar_labels)
        
        # Step 5: Generate names for clusters
        print("   Generating unique topic names...")
        result_df = df.copy()
        result_df['pillar_id'] = pillar_labels
        result_df['topic_id'] = topic_labels
        
        # Generate pillar names
        self.topic_generator.reset_used_words()
        pillar_names = self._generate_cluster_names(keywords, pillar_labels, 'pillar')
        result_df['pillar_name'] = [pillar_names[label] for label in pillar_labels]
        
        # Generate topic names
        topic_names = self._generate_cluster_names(keywords, topic_labels, 'topic')
        result_df['topic_name'] = [topic_names[label] for label in topic_labels]
        
        print("✓ Clustering completed successfully")
        return result_df
    
    def _create_embeddings(self, keywords: List[str]) -> np.ndarray:
        """Create BERT embeddings"""
        # Initialize model
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.embedding_model.max_seq_length = Config.MAX_SEQUENCE_LENGTH
        
        # Handle empty keywords
        processed_keywords = []
        for k in keywords:
            if not k or len(k) == 0:
                processed_keywords.append("empty")
            else:
                processed_keywords.append(k)
        
        # Create embeddings in batches
        embeddings = []
        batch_size = Config.BATCH_SIZE
        
        for i in range(0, len(processed_keywords), batch_size):
            batch = processed_keywords[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
            
            # Progress update
            if i % (batch_size * 10) == 0 and i > 0:
                progress = (i / len(processed_keywords)) * 100
                print(f"      Embedding progress: {progress:.1f}%")
        
        embeddings = np.vstack(embeddings)
        return embeddings
    
    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensions using PCA + UMAP"""
        # First PCA
        if embeddings.shape[1] > 100:
            pca = PCA(n_components=100, random_state=42)
            embeddings = pca.fit_transform(embeddings)
        
        # Then UMAP
        umap_model = umap.UMAP(
            n_neighbors=min(Config.UMAP_N_NEIGHBORS, len(embeddings) - 1),
            n_components=min(Config.UMAP_N_COMPONENTS, embeddings.shape[0] - 1),
            min_dist=Config.UMAP_MIN_DIST,
            metric=Config.UMAP_METRIC,
            random_state=42,
            low_memory=True
        )
        
        reduced_embeddings = umap_model.fit_transform(embeddings)
        return reduced_embeddings
    
    def _create_pillar_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        """Create main pillar clusters"""
        n_clusters = min(Config.PILLAR_CLUSTERS, len(embeddings) // 100, len(embeddings))
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        labels = kmeans.fit_predict(embeddings)
        return labels
    
    def _create_topic_clusters(self, embeddings: np.ndarray, pillar_labels: np.ndarray) -> np.ndarray:
        """Create topic clusters within each pillar"""
        topic_labels = np.zeros_like(pillar_labels)
        global_topic_id = 0
        
        # Group by pillar
        unique_pillars = np.unique(pillar_labels)
        
        for pillar_id in unique_pillars:
            # Get indices for this pillar
            pillar_mask = pillar_labels == pillar_id
            pillar_indices = np.where(pillar_mask)[0]
            
            if len(pillar_indices) < Config.MIN_KEYWORDS_FOR_TOPIC:
                # Too small for sub-clustering
                topic_labels[pillar_indices] = global_topic_id
                global_topic_id += 1
                continue
            
            # Determine number of topics for this pillar
            n_topics = min(
                Config.TOPICS_PER_PILLAR,
                max(2, len(pillar_indices) // Config.MIN_CLUSTER_SIZE),
                len(pillar_indices)
            )
            
            # Cluster within pillar
            pillar_embeddings = embeddings[pillar_indices]
            
            kmeans = KMeans(
                n_clusters=n_topics,
                random_state=42,
                n_init=5
            )
            
            sub_labels = kmeans.fit_predict(pillar_embeddings)
            
            # Assign global topic IDs
            for i, idx in enumerate(pillar_indices):
                topic_labels[idx] = global_topic_id + sub_labels[i]
            
            global_topic_id += n_topics
        
        return topic_labels
    
    def _generate_cluster_names(self, keywords: List[str], labels: np.ndarray, 
                               level: str) -> Dict[int, str]:
        """Generate unique names for each cluster"""
        # Group keywords by cluster
        cluster_keywords = defaultdict(list)
        for keyword, label in zip(keywords, labels):
            if keyword:  # Only add non-empty keywords
                cluster_keywords[label].append(keyword)
        
        # Generate name for each cluster
        cluster_names = {}
        for cluster_id, cluster_kws in cluster_keywords.items():
            if cluster_kws:
                cluster_names[cluster_id] = self.topic_generator.generate_topic_name(
                    cluster_kws, level
                )
            else:
                cluster_names[cluster_id] = f"Cluster_{cluster_id}"
        
        return cluster_names

# ============= EXCEL OUTPUT GENERATOR =============
class ExcelOutputGenerator:
    """Generate comprehensive Excel output"""
    
    def create_output(self, clustered_df: pd.DataFrame, output_path: str):
        """Create Excel with clustered keywords and statistics"""
        print(f"\n Creating Excel output: {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Clustered Keywords
            self._create_clustered_sheet(clustered_df, writer)
            
            # Sheet 2: Pillar Summary
            self._create_pillar_summary(clustered_df, writer)
            
            # Sheet 3: Topic Summary
            self._create_topic_summary(clustered_df, writer)
            
            # Sheet 4: Statistics
            self._create_statistics_sheet(clustered_df, writer)
        
        print("✓ Excel file created successfully")
    
    def _create_clustered_sheet(self, df: pd.DataFrame, writer):
        """Create main clustered keywords sheet"""
        output_df = df[[
            'original_keyword', 'cleaned_keyword', 
            'pillar_id', 'pillar_name',
            'topic_id', 'topic_name',
            'word_count', 'search_volume', 'competition', 'cpc'
        ]].copy()
        
        # Add combined hierarchy
        output_df['full_path'] = output_df['pillar_name'] + ' > ' + output_df['topic_name']
        
        # Sort by pillar and topic
        output_df = output_df.sort_values(['pillar_id', 'topic_id'])
        
        output_df.to_excel(writer, sheet_name='Clustered_Keywords', index=False)
    
    def _create_pillar_summary(self, df: pd.DataFrame, writer):
        """Create pillar-level summary"""
        if len(df) == 0:
            return
        
        pillar_summary = df.groupby(['pillar_id', 'pillar_name']).agg({
            'cleaned_keyword': 'count',
            'topic_id': 'nunique',
            'search_volume': 'sum',
            'competition': 'mean',
            'cpc': 'mean'
        }).round(2)
        
        pillar_summary.columns = [
            'Total_Keywords', 'Total_Topics', 
            'Total_Search_Volume', 'Avg_Competition', 'Avg_CPC'
        ]
        
        pillar_summary = pillar_summary.reset_index()
        pillar_summary = pillar_summary.sort_values('Total_Keywords', ascending=False)
        
        pillar_summary.to_excel(writer, sheet_name='Pillar_Summary', index=False)
    
    def _create_topic_summary(self, df: pd.DataFrame, writer):
        """Create topic-level summary"""
        if len(df) == 0:
            return
        
        topic_summary = df.groupby(['pillar_name', 'topic_id', 'topic_name']).agg({
            'cleaned_keyword': 'count',
            'search_volume': 'sum',
            'competition': 'mean',
            'cpc': 'mean'
        }).round(2)
        
        topic_summary.columns = [
            'Keyword_Count', 'Total_Search_Volume', 
            'Avg_Competition', 'Avg_CPC'
        ]
        
        topic_summary = topic_summary.reset_index()
        topic_summary = topic_summary.sort_values('Keyword_Count', ascending=False)
        
        topic_summary.to_excel(writer, sheet_name='Topic_Summary', index=False)
    
    def _create_statistics_sheet(self, clustered_df: pd.DataFrame, writer):
        """Create overall statistics sheet"""
        stats = []
        
        # Overall stats
        stats.append({
            'Metric': 'Total Keywords Processed',
            'Value': len(clustered_df)
        })
        stats.append({
            'Metric': 'Total Pillars',
            'Value': clustered_df['pillar_name'].nunique()
        })
        stats.append({
            'Metric': 'Total Topics',
            'Value': clustered_df['topic_name'].nunique()
        })
        stats.append({
            'Metric': 'Avg Keywords per Pillar',
            'Value': round(len(clustered_df) / clustered_df['pillar_name'].nunique())
        })
        stats.append({
            'Metric': 'Avg Keywords per Topic',
            'Value': round(len(clustered_df) / clustered_df['topic_name'].nunique())
        })
        stats.append({
            'Metric': 'Keywords with Search Volume > 0',
            'Value': (clustered_df['search_volume'] > 0).sum()
        })
        stats.append({
            'Metric': 'Total Search Volume',
            'Value': clustered_df['search_volume'].sum()
        })
        stats.append({
            'Metric': 'Average CPC',
            'Value': f"${clustered_df['cpc'].mean():.2f}"
        })
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)

# ============= MAIN PIPELINE =============
def run_clustering_pipeline():
    """Main execution pipeline"""
    start_time = datetime.now()
    
    print("="*80)
    print(" ENHANCED BERT CLUSTERING SYSTEM - ALL KEYWORDS VERSION")
    print("   Processing ALL keywords (no removal)")
    print("   2-Level Hierarchy: Pillars → Topics")
    print("="*80)
    
    try:
        # Create output directory
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        
        # Step 1: Load and process keywords (no removal)
        processor = KeywordProcessor()
        processed_df = processor.load_and_process(
            Config.INPUT_FILE, 
            max_keywords=Config.MAX_KEYWORDS
        )
        
        # Step 2: Create clusters
        if len(processed_df) > 0:
            clustering_engine = HierarchicalClusteringEngine()
            clustered_df = clustering_engine.create_clusters(processed_df)
            
            # Step 3: Generate output
            excel_generator = ExcelOutputGenerator()
            excel_generator.create_output(clustered_df, Config.OUTPUT_FILE)
            
            # Print summary
            elapsed_time = datetime.now() - start_time
            
            print("\n" + "="*80)
            print("CLUSTERING COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"\n FINAL RESULTS:")
            print(f"   • Total Keywords Processed: {len(clustered_df):,}")
            print(f"   • No Keywords Removed (All Retained)")
            print(f"   • Pillars Created: {clustered_df['pillar_name'].nunique()}")
            print(f"   • Topics Created: {clustered_df['topic_name'].nunique()}")
            print(f"\n Processing Time: {elapsed_time}")
            print(f"\n Output File: {Config.OUTPUT_FILE}")
            
            # Show sample topics
            print(f"\n Sample Pillar Structure:")
            for pillar in clustered_df['pillar_name'].unique()[:5]:
                pillar_data = clustered_df[clustered_df['pillar_name'] == pillar]
                topics = pillar_data['topic_name'].unique()[:3]
                keyword_count = len(pillar_data)
                print(f"\n  {pillar} ({keyword_count:,} keywords)")
                for topic in topics:
                    topic_count = len(pillar_data[pillar_data['topic_name'] == topic])
                    print(f"      └─ {topic} ({topic_count} keywords)")
            
            # Show distribution
            print(f"\n Cluster Size Distribution:")
            pillar_sizes = clustered_df.groupby('pillar_name').size().sort_values(ascending=False)
            print(f"   • Largest Pillar: {pillar_sizes.index[0]} ({pillar_sizes.values[0]:,} keywords)")
            print(f"   • Smallest Pillar: {pillar_sizes.index[-1]} ({pillar_sizes.values[-1]:,} keywords)")
            print(f"   • Median Pillar Size: {pillar_sizes.median():.0f} keywords")
            
        else:
            print("\n No keywords found to cluster!")
        
        return clustered_df if len(processed_df) > 0 else pd.DataFrame()
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Run the clustering pipeline
    clustered_data = run_clustering_pipeline()