"""
Enhanced BERT-Based Keyword Clustering System for Crypto Ecosystem
Processes 200k+ keywords with 2-level hierarchy (Pillar & Topics only)
Shows removed keywords and prevents duplicate words in topics
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
    
    # File paths
    INPUT_FILE = '/home/admin1/Downloads/demo_crypto/csv list keywords.csv'
    OUTPUT_DIR = '/home/admin1/Downloads/demo_crypto/output'
    OUTPUT_FILE = f'{OUTPUT_DIR}/crypto_clusters_2lakh_clean.xlsx'
    REMOVED_KEYWORDS_FILE = f'{OUTPUT_DIR}/removed_keywords.csv'
    
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

# ============= CRYPTO RELEVANCE CHECKER =============
class CryptoRelevanceChecker:
    """Enhanced relevance checking for crypto keywords"""
    
    def __init__(self):
        # Comprehensive crypto terms list
        self.crypto_terms = self._load_crypto_terms()
        self.proxy_terms = self._load_proxy_terms()
        self.gibberish_patterns = self._compile_gibberish_patterns()
        
    def _load_crypto_terms(self) -> Set[str]:
        """Load comprehensive crypto terms"""
        return set([
            # Major Cryptocurrencies
            'bitcoin', 'btc', 'ethereum', 'eth', 'ether', 'binance', 'bnb', 
            'cardano', 'ada', 'solana', 'sol', 'ripple', 'xrp', 'polkadot', 'dot',
            'dogecoin', 'doge', 'shiba', 'shib', 'avalanche', 'avax', 'chainlink', 'link',
            'polygon', 'matic', 'cosmos', 'atom', 'algorand', 'algo', 'vechain', 'vet',
            'stellar', 'xlm', 'filecoin', 'fil', 'tron', 'trx', 'monero', 'xmr',
            'litecoin', 'ltc', 'uniswap', 'uni', 'bitcoin cash', 'bch', 'near', 'icp',
            'aptos', 'apt', 'arbitrum', 'arb', 'optimism', 'op', 'hedera', 'hbar',
            
            # DeFi & Protocols
            'defi', 'aave', 'compound', 'maker', 'mkr', 'curve', 'crv', 'sushi',
            'pancakeswap', 'cake', 'yearn', 'yfi', 'synthetix', 'snx', 'balancer',
            '1inch', 'dydx', 'lido', 'ldo', 'frax', 'gmx', 'venus', 'cream',
            
            # Technical Terms
            'blockchain', 'cryptocurrency', 'crypto', 'altcoin', 'token', 'coin',
            'wallet', 'exchange', 'dex', 'cex', 'amm', 'liquidity', 'pool',
            'yield', 'farming', 'staking', 'mining', 'validator', 'node',
            'smart contract', 'dapp', 'web3', 'dao', 'nft', 'metaverse',
            'gas', 'gwei', 'hash', 'hashrate', 'block', 'transaction', 'txn',
            
            # Trading Terms
            'trading', 'trade', 'buy', 'sell', 'swap', 'convert', 'bridge',
            'price', 'chart', 'analysis', 'technical', 'fundamental', 'ta',
            'bullish', 'bearish', 'pump', 'dump', 'hodl', 'fomo', 'fud',
            'whale', 'portfolio', 'invest', 'investment', 'profit', 'loss',
            
            # Stablecoins
            'stablecoin', 'usdt', 'tether', 'usdc', 'circle', 'dai', 'busd',
            'tusd', 'pax', 'paxos', 'gusd', 'gemini', 'usdd', 'frax',
            
            # Exchanges
            'binance', 'coinbase', 'kraken', 'bitfinex', 'huobi', 'okx', 'okex',
            'kucoin', 'gate', 'bittrex', 'bitstamp', 'bybit', 'bitget', 'mexc',
            
            # Layer 2 & Scaling
            'layer2', 'l2', 'rollup', 'zk', 'zero knowledge', 'optimistic',
            'zksync', 'starknet', 'starkware', 'lightning', 'network',
            
            # Security & Regulation
            'security', 'audit', 'hack', 'exploit', 'vulnerability', 'kyc', 'aml',
            'regulation', 'sec', 'cftc', 'compliance', 'tax', 'legal',
            
            # Other
            'airdrop', 'whitepaper', 'roadmap', 'mainnet', 'testnet', 'fork',
            'halving', 'burn', 'mint', 'supply', 'market cap', 'volume',
            'consensus', 'proof of work', 'pow', 'proof of stake', 'pos'
        ])
    
    def _load_proxy_terms(self) -> Set[str]:
        """Terms that are crypto-related when combined with other words"""
        return set([
            'proxy', 'node', 'rpc', 'api', 'endpoint', 'gateway', 'bridge',
            'oracle', 'feed', 'index', 'tracker', 'explorer', 'scan', 'scanner',
            'monitor', 'alert', 'bot', 'automation', 'tool', 'platform',
            'protocol', 'network', 'chain', 'cross-chain', 'multi-chain',
            'layer', 'sidechain', 'parachain', 'subnet', 'shard', 'rollup'
        ])
    
    def _compile_gibberish_patterns(self) -> List:
        """Compile regex patterns for gibberish detection"""
        return [
            re.compile(r'^[a-z]{25,}$'),  # Very long single words
            re.compile(r'^[0-9]+$'),  # Pure numbers
            re.compile(r'^[a-z0-9]{1,2}$'),  # Too short
            re.compile(r'(.)\1{5,}'),  # Repeated characters
            re.compile(r'^test\d+'),  # Test keywords
            re.compile(r'^demo\d+'),  # Demo keywords
            re.compile(r'^example\d+'),  # Example keywords
            re.compile(r'xxx|porn|sex|adult'),  # Adult content
            re.compile(r'^asdf|^qwer|^zxcv'),  # Keyboard mashing
            re.compile(r'^[^a-zA-Z0-9\s\-\_]+$'),  # Only special chars
        ]
    
    def is_relevant(self, keyword: str) -> bool:
        """Check if keyword is crypto-relevant"""
        keyword_lower = keyword.lower()
        
        # Check for gibberish
        for pattern in self.gibberish_patterns:
            if pattern.search(keyword_lower):
                return False
        
        # Check for crypto terms
        words = keyword_lower.split()
        
        # Direct crypto term match
        for word in words:
            if word in self.crypto_terms:
                return True
        
        # Check for proxy terms with context
        has_proxy = any(term in keyword_lower for term in self.proxy_terms)
        has_crypto = any(term in keyword_lower for term in self.crypto_terms)
        
        if has_proxy and has_crypto:
            return True
        
        # Check for crypto-related patterns
        crypto_patterns = [
            'crypto', 'blockchain', 'bitcoin', 'ethereum', 'defi', 'nft',
            'token', 'coin', 'mining', 'trading', 'wallet', 'exchange'
        ]
        
        return any(pattern in keyword_lower for pattern in crypto_patterns)
    
    def classify_removal_reason(self, keyword: str) -> str:
        """Classify why a keyword was removed"""
        keyword_lower = keyword.lower()
        
        # Check length
        if len(keyword_lower) < 3:
            return "Too Short"
        if len(keyword_lower) > 100:
            return "Too Long"
        
        # Check for gibberish patterns
        if re.match(r'^[a-z]{25,}$', keyword_lower):
            return "Gibberish - Long Word"
        if re.match(r'^[0-9]+$', keyword_lower):
            return "Pure Numbers"
        if re.search(r'(.)\1{5,}', keyword_lower):
            return "Repeated Characters"
        if re.match(r'^test\d*|^demo\d*|^example\d*', keyword_lower):
            return "Test/Demo Keyword"
        if re.search(r'xxx|porn|sex|adult', keyword_lower):
            return "Adult Content"
        if re.match(r'^asdf|^qwer|^zxcv', keyword_lower):
            return "Keyboard Mashing"
        
        # Check for non-crypto
        has_any_crypto = any(term in keyword_lower for term in self.crypto_terms)
        has_any_proxy = any(term in keyword_lower for term in self.proxy_terms)
        
        if not has_any_crypto and not has_any_proxy:
            return "Not Crypto Related"
        
        return "Other"

# ============= KEYWORD PROCESSOR =============
class KeywordProcessor:
    """Process and clean keywords efficiently"""
    
    def __init__(self):
        self.relevance_checker = CryptoRelevanceChecker()
        self.removed_keywords = []
        
    def load_and_process(self, file_path: str, max_keywords: int = 200000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process keywords, return both kept and removed"""
        print(f"\n📂 Loading keywords from: {file_path}")
        
        # Load data
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, nrows=max_keywords)
        else:
            df = pd.read_csv(file_path, nrows=max_keywords)
        
        print(f"✓ Loaded {len(df)} keywords")
        
        # Find keyword column
        keyword_col = self._find_keyword_column(df)
        print(f"✓ Using column: '{keyword_col}'")
        
        # Process keywords
        processed_df, removed_df = self._process_keywords(df, keyword_col)
        
        return processed_df, removed_df
    
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
    
    def _process_keywords(self, df: pd.DataFrame, keyword_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process keywords and separate kept vs removed"""
        print("\n🔧 Processing keywords...")
        
        # Create working dataframe
        all_keywords = []
        
        for idx, row in df.iterrows():
            keyword = str(row[keyword_col])
            
            # Basic cleaning
            cleaned = keyword.lower().strip()
            cleaned = re.sub(r'http\S+|www\S+', '', cleaned)  # Remove URLs
            cleaned = re.sub(r'\S+@\S+', '', cleaned)  # Remove emails
            cleaned = re.sub(r'[^\w\s\-\$\#\.]', ' ', cleaned)  # Keep crypto chars
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
            cleaned = cleaned.strip()
            
            all_keywords.append({
                'original_keyword': keyword,
                'cleaned_keyword': cleaned,
                'search_volume': row.get('search_volume', 0) if 'search_volume' in row else 0,
                'competition': row.get('competition', 0) if 'competition' in row else 0,
                'cpc': row.get('cpc', 0.0) if 'cpc' in row else 0.0
            })
        
        # Create dataframe
        all_df = pd.DataFrame(all_keywords)
        
        # Separate valid and removed keywords
        kept_keywords = []
        removed_keywords = []
        
        for idx, row in all_df.iterrows():
            keyword = row['cleaned_keyword']
            
            # Check if valid
            if len(keyword) < 3 or len(keyword) > 100:
                row['removal_reason'] = 'Invalid Length'
                removed_keywords.append(row)
            elif not self.relevance_checker.is_relevant(keyword):
                row['removal_reason'] = self.relevance_checker.classify_removal_reason(keyword)
                removed_keywords.append(row)
            else:
                kept_keywords.append(row)
        
        kept_df = pd.DataFrame(kept_keywords)
        removed_df = pd.DataFrame(removed_keywords)
        
        # Remove exact duplicates from kept
        if len(kept_df) > 0:
            before_dedup = len(kept_df)
            kept_df = kept_df.drop_duplicates(subset=['cleaned_keyword'], keep='first')
            duplicates_removed = before_dedup - len(kept_df)
            
            if duplicates_removed > 0:
                print(f"✓ Removed {duplicates_removed} duplicate keywords")
        
        # Add features to kept keywords
        if len(kept_df) > 0:
            kept_df = self._add_features(kept_df)
            kept_df = kept_df.reset_index(drop=True)
        
        print(f"\n📊 Processing Results:")
        print(f"   • Keywords Kept: {len(kept_df):,}")
        print(f"   • Keywords Removed: {len(removed_df):,}")
        
        if len(removed_df) > 0:
            print(f"\n📋 Removal Reasons:")
            removal_stats = removed_df['removal_reason'].value_counts()
            for reason, count in removal_stats.items():
                print(f"   • {reason}: {count:,}")
        
        return kept_df, removed_df
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for clustering"""
        df['word_count'] = df['cleaned_keyword'].str.split().str.len()
        df['char_count'] = df['cleaned_keyword'].str.len()
        df['has_numbers'] = df['cleaned_keyword'].str.contains(r'\d', na=False)
        df['has_question'] = df['cleaned_keyword'].str.contains(
            r'how|what|why|when|where|which|who', na=False
        )
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
            available_terms = important_terms[:3]
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
                min_df=1
            )
            
            tfidf_matrix = tfidf.fit_transform(keywords)
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
            
            return top_terms
            
        except Exception as e:
            # Fallback to frequency-based extraction
            return self._extract_by_frequency(keywords)
    
    def _extract_by_frequency(self, keywords: List[str]) -> List[str]:
        """Extract terms by frequency"""
        word_freq = Counter()
        
        for keyword in keywords:
            words = keyword.lower().split()
            for word in words:
                if len(word) > 2:
                    word_freq[word] += 1
        
        # Get top words
        top_words = [word.title() for word, _ in word_freq.most_common(10)]
        return top_words
    
    def _create_topic_name(self, terms: List[str], level: str) -> str:
        """Create topic name from terms"""
        if not terms:
            return f"Crypto_{level.title()}"
        
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
        
        print("\n🤖 Starting BERT-based clustering...")
        
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
        
        # Create embeddings in batches
        embeddings = []
        batch_size = Config.BATCH_SIZE
        
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
            
            # Progress update
            if i % (batch_size * 10) == 0 and i > 0:
                progress = (i / len(keywords)) * 100
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
            n_neighbors=Config.UMAP_N_NEIGHBORS,
            n_components=Config.UMAP_N_COMPONENTS,
            min_dist=Config.UMAP_MIN_DIST,
            metric=Config.UMAP_METRIC,
            random_state=42,
            low_memory=True
        )
        
        reduced_embeddings = umap_model.fit_transform(embeddings)
        return reduced_embeddings
    
    def _create_pillar_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        """Create main pillar clusters"""
        n_clusters = min(Config.PILLAR_CLUSTERS, len(embeddings) // 100)
        
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
                max(2, len(pillar_indices) // Config.MIN_CLUSTER_SIZE)
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
            cluster_keywords[label].append(keyword)
        
        # Generate name for each cluster
        cluster_names = {}
        for cluster_id, cluster_kws in cluster_keywords.items():
            cluster_names[cluster_id] = self.topic_generator.generate_topic_name(
                cluster_kws, level
            )
        
        return cluster_names

# ============= EXCEL OUTPUT GENERATOR =============
class ExcelOutputGenerator:
    """Generate comprehensive Excel output"""
    
    def create_output(self, clustered_df: pd.DataFrame, removed_df: pd.DataFrame, output_path: str):
        """Create Excel with clustered keywords and statistics"""
        print(f"\n📝 Creating Excel output: {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Clustered Keywords
            self._create_clustered_sheet(clustered_df, writer)
            
            # Sheet 2: Removed Keywords
            if len(removed_df) > 0:
                self._create_removed_sheet(removed_df, writer)
            
            # Sheet 3: Pillar Summary
            self._create_pillar_summary(clustered_df, writer)
            
            # Sheet 4: Topic Summary
            self._create_topic_summary(clustered_df, writer)
            
            # Sheet 5: Statistics
            self._create_statistics_sheet(clustered_df, removed_df, writer)
        
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
    
    def _create_removed_sheet(self, df: pd.DataFrame, writer):
        """Create removed keywords sheet"""
        if len(df) > 0:
            output_df = df[[
                'original_keyword', 'cleaned_keyword', 'removal_reason'
            ]].copy()
            
            output_df = output_df.sort_values('removal_reason')
            output_df.to_excel(writer, sheet_name='Removed_Keywords', index=False)
    
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
    
    def _create_statistics_sheet(self, clustered_df: pd.DataFrame, removed_df: pd.DataFrame, writer):
        """Create overall statistics sheet"""
        stats = []
        
        # Overall stats
        total_processed = len(clustered_df) + len(removed_df)
        stats.append({
            'Metric': 'Total Keywords Processed',
            'Value': total_processed
        })
        stats.append({
            'Metric': 'Keywords Kept',
            'Value': len(clustered_df)
        })
        stats.append({
            'Metric': 'Keywords Removed',
            'Value': len(removed_df)
        })
        stats.append({
            'Metric': 'Retention Rate',
            'Value': f"{(len(clustered_df) / total_processed * 100):.1f}%" if total_processed > 0 else "0%"
        })
        
        if len(clustered_df) > 0:
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
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)

# ============= MAIN PIPELINE =============
def run_clustering_pipeline():
    """Main execution pipeline"""
    start_time = datetime.now()
    
    print("="*80)
    print("🚀 ENHANCED BERT CLUSTERING SYSTEM")
    print("   Processing up to 200,000 keywords")
    print("   2-Level Hierarchy: Pillars → Topics")
    print("="*80)
    
    try:
        # Create output directory
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        
        # Step 1: Load and process keywords
        processor = KeywordProcessor()
        processed_df, removed_df = processor.load_and_process(
            Config.INPUT_FILE, 
            max_keywords=Config.MAX_KEYWORDS
        )
        
        # Save removed keywords
        if len(removed_df) > 0:
            removed_df.to_csv(Config.REMOVED_KEYWORDS_FILE, index=False)
            print(f"\n💾 Saved removed keywords to: {Config.REMOVED_KEYWORDS_FILE}")
        
        # Step 2: Create clusters (only if we have keywords)
        if len(processed_df) > 0:
            clustering_engine = HierarchicalClusteringEngine()
            clustered_df = clustering_engine.create_clusters(processed_df)
            
            # Step 3: Generate output
            excel_generator = ExcelOutputGenerator()
            excel_generator.create_output(clustered_df, removed_df, Config.OUTPUT_FILE)
            
            # Print summary
            elapsed_time = datetime.now() - start_time
            
            print("\n" + "="*80)
            print("✅ CLUSTERING COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"\n📊 FINAL RESULTS:")
            print(f"   • Total Keywords Processed: {len(processed_df) + len(removed_df):,}")
            print(f"   • Keywords Clustered: {len(clustered_df):,}")
            print(f"   • Keywords Removed: {len(removed_df):,}")
            print(f"   • Pillars Created: {clustered_df['pillar_name'].nunique()}")
            print(f"   • Topics Created: {clustered_df['topic_name'].nunique()}")
            print(f"\n⏱️ Processing Time: {elapsed_time}")
            print(f"\n📁 Output Files:")
            print(f"   • Clustered Keywords: {Config.OUTPUT_FILE}")
            print(f"   • Removed Keywords: {Config.REMOVED_KEYWORDS_FILE}")
            
            # Show sample topics
            print(f"\n🎯 Sample Pillar Structure:")
            for pillar in clustered_df['pillar_name'].unique()[:5]:
                pillar_data = clustered_df[clustered_df['pillar_name'] == pillar]
                topics = pillar_data['topic_name'].unique()[:3]
                keyword_count = len(pillar_data)
                print(f"\n   📌 {pillar} ({keyword_count:,} keywords)")
                for topic in topics:
                    topic_count = len(pillar_data[pillar_data['topic_name'] == topic])
                    print(f"      └─ {topic} ({topic_count} keywords)")
        else:
            print("\n⚠️ No valid keywords found to cluster!")
        
        return clustered_df if len(processed_df) > 0 else pd.DataFrame()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    clustered_data = run_clustering_pipeline()

    """
    Enhanced BERT-Based Keyword Clustering System for Crypto Ecosystem
    Processes 200k+ keywords with 2-level hierarchy (Pillar & Topics only)
    Shows removed keywords and prevents duplicate words in topics
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
    
    # File paths
    INPUT_FILE = '/home/admin1/Downloads/demo_crypto/Book6.csv'
    OUTPUT_DIR = '/home/admin1/Downloads/demo_crypto/output'
    OUTPUT_FILE = f'{OUTPUT_DIR}/crypto_clusters_2lakh_clean.xlsx'
    REMOVED_KEYWORDS_FILE = f'{OUTPUT_DIR}/removed_keywords.csv'
    
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

# ============= CRYPTO RELEVANCE CHECKER =============
class CryptoRelevanceChecker:
    """Enhanced relevance checking for crypto keywords"""
    
    def __init__(self):
        # Comprehensive crypto terms list
        self.crypto_terms = self._load_crypto_terms()
        self.proxy_terms = self._load_proxy_terms()
        self.gibberish_patterns = self._compile_gibberish_patterns()
        
    def _load_crypto_terms(self) -> Set[str]:
        """Load comprehensive crypto terms"""
        return set([
            # Major Cryptocurrencies
            'bitcoin', 'btc', 'ethereum', 'eth', 'ether', 'binance', 'bnb', 
            'cardano', 'ada', 'solana', 'sol', 'ripple', 'xrp', 'polkadot', 'dot',
            'dogecoin', 'doge', 'shiba', 'shib', 'avalanche', 'avax', 'chainlink', 'link',
            'polygon', 'matic', 'cosmos', 'atom', 'algorand', 'algo', 'vechain', 'vet',
            'stellar', 'xlm', 'filecoin', 'fil', 'tron', 'trx', 'monero', 'xmr',
            'litecoin', 'ltc', 'uniswap', 'uni', 'bitcoin cash', 'bch', 'near', 'icp',
            'aptos', 'apt', 'arbitrum', 'arb', 'optimism', 'op', 'hedera', 'hbar',
            
            # DeFi & Protocols
            'defi', 'aave', 'compound', 'maker', 'mkr', 'curve', 'crv', 'sushi',
            'pancakeswap', 'cake', 'yearn', 'yfi', 'synthetix', 'snx', 'balancer',
            '1inch', 'dydx', 'lido', 'ldo', 'frax', 'gmx', 'venus', 'cream',
            
            # Technical Terms
            'blockchain', 'cryptocurrency', 'crypto', 'altcoin', 'token', 'coin',
            'wallet', 'exchange', 'dex', 'cex', 'amm', 'liquidity', 'pool',
            'yield', 'farming', 'staking', 'mining', 'validator', 'node',
            'smart contract', 'dapp', 'web3', 'dao', 'nft', 'metaverse',
            'gas', 'gwei', 'hash', 'hashrate', 'block', 'transaction', 'txn',
            
            # Trading Terms
            'trading', 'trade', 'buy', 'sell', 'swap', 'convert', 'bridge',
            'price', 'chart', 'analysis', 'technical', 'fundamental', 'ta',
            'bullish', 'bearish', 'pump', 'dump', 'hodl', 'fomo', 'fud',
            'whale', 'portfolio', 'invest', 'investment', 'profit', 'loss',
            
            # Stablecoins
            'stablecoin', 'usdt', 'tether', 'usdc', 'circle', 'dai', 'busd',
            'tusd', 'pax', 'paxos', 'gusd', 'gemini', 'usdd', 'frax',
            
            # Exchanges
            'binance', 'coinbase', 'kraken', 'bitfinex', 'huobi', 'okx', 'okex',
            'kucoin', 'gate', 'bittrex', 'bitstamp', 'bybit', 'bitget', 'mexc',
            
            # Layer 2 & Scaling
            'layer2', 'l2', 'rollup', 'zk', 'zero knowledge', 'optimistic',
            'zksync', 'starknet', 'starkware', 'lightning', 'network',
            
            # Security & Regulation
            'security', 'audit', 'hack', 'exploit', 'vulnerability', 'kyc', 'aml',
            'regulation', 'sec', 'cftc', 'compliance', 'tax', 'legal',
            
            # Other
            'airdrop', 'whitepaper', 'roadmap', 'mainnet', 'testnet', 'fork',
            'halving', 'burn', 'mint', 'supply', 'market cap', 'volume',
            'consensus', 'proof of work', 'pow', 'proof of stake', 'pos'
        ])
    
    def _load_proxy_terms(self) -> Set[str]:
        """Terms that are crypto-related when combined with other words"""
        return set([
            'proxy', 'node', 'rpc', 'api', 'endpoint', 'gateway', 'bridge',
            'oracle', 'feed', 'index', 'tracker', 'explorer', 'scan', 'scanner',
            'monitor', 'alert', 'bot', 'automation', 'tool', 'platform',
            'protocol', 'network', 'chain', 'cross-chain', 'multi-chain',
            'layer', 'sidechain', 'parachain', 'subnet', 'shard', 'rollup'
        ])
    
    def _compile_gibberish_patterns(self) -> List:
        """Compile regex patterns for gibberish detection"""
        return [
            re.compile(r'^[a-z]{25,}$'),  # Very long single words
            re.compile(r'^[0-9]+$'),  # Pure numbers
            re.compile(r'^[a-z0-9]{1,2}$'),  # Too short
            re.compile(r'(.)\1{5,}'),  # Repeated characters
            re.compile(r'^test\d+'),  # Test keywords
            re.compile(r'^demo\d+'),  # Demo keywords
            re.compile(r'^example\d+'),  # Example keywords
            re.compile(r'xxx|porn|sex|adult'),  # Adult content
            re.compile(r'^asdf|^qwer|^zxcv'),  # Keyboard mashing
            re.compile(r'^[^a-zA-Z0-9\s\-\_]+$'),  # Only special chars
        ]
    
    def is_relevant(self, keyword: str) -> bool:
        """Check if keyword is crypto-relevant"""
        keyword_lower = keyword.lower()
        
        # Check for gibberish
        for pattern in self.gibberish_patterns:
            if pattern.search(keyword_lower):
                return False
        
        # Check for crypto terms
        words = keyword_lower.split()
        
        # Direct crypto term match
        for word in words:
            if word in self.crypto_terms:
                return True
        
        # Check for proxy terms with context
        has_proxy = any(term in keyword_lower for term in self.proxy_terms)
        has_crypto = any(term in keyword_lower for term in self.crypto_terms)
        
        if has_proxy and has_crypto:
            return True
        
        # Check for crypto-related patterns
        crypto_patterns = [
            'crypto', 'blockchain', 'bitcoin', 'ethereum', 'defi', 'nft',
            'token', 'coin', 'mining', 'trading', 'wallet', 'exchange'
        ]
        
        return any(pattern in keyword_lower for pattern in crypto_patterns)
    
    def classify_removal_reason(self, keyword: str) -> str:
        """Classify why a keyword was removed"""
        keyword_lower = keyword.lower()
        
        # Check length
        if len(keyword_lower) < 3:
            return "Too Short"
        if len(keyword_lower) > 100:
            return "Too Long"
        
        # Check for gibberish patterns
        if re.match(r'^[a-z]{25,}$', keyword_lower):
            return "Gibberish - Long Word"
        if re.match(r'^[0-9]+$', keyword_lower):
            return "Pure Numbers"
        if re.search(r'(.)\1{5,}', keyword_lower):
            return "Repeated Characters"
        if re.match(r'^test\d*|^demo\d*|^example\d*', keyword_lower):
            return "Test/Demo Keyword"
        if re.search(r'xxx|porn|sex|adult', keyword_lower):
            return "Adult Content"
        if re.match(r'^asdf|^qwer|^zxcv', keyword_lower):
            return "Keyboard Mashing"
        
        # Check for non-crypto
        has_any_crypto = any(term in keyword_lower for term in self.crypto_terms)
        has_any_proxy = any(term in keyword_lower for term in self.proxy_terms)
        
        if not has_any_crypto and not has_any_proxy:
            return "Not Crypto Related"
        
        return "Other"

# ============= KEYWORD PROCESSOR =============
class KeywordProcessor:
    """Process and clean keywords efficiently"""
    
    def __init__(self):
        self.relevance_checker = CryptoRelevanceChecker()
        self.removed_keywords = []
        
    def load_and_process(self, file_path: str, max_keywords: int = 200000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process keywords, return both kept and removed"""
        print(f"\n📂 Loading keywords from: {file_path}")
        
        # Load data
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, nrows=max_keywords)
        else:
            df = pd.read_csv(file_path, nrows=max_keywords)
        
        print(f"✓ Loaded {len(df)} keywords")
        
        # Find keyword column
        keyword_col = self._find_keyword_column(df)
        print(f"✓ Using column: '{keyword_col}'")
        
        # Process keywords
        processed_df, removed_df = self._process_keywords(df, keyword_col)
        
        return processed_df, removed_df
    
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
    
    def _process_keywords(self, df: pd.DataFrame, keyword_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process keywords and separate kept vs removed"""
        print("\n🔧 Processing keywords...")
        
        # Create working dataframe
        all_keywords = []
        
        for idx, row in df.iterrows():
            keyword = str(row[keyword_col])
            
            # Basic cleaning
            cleaned = keyword.lower().strip()
            cleaned = re.sub(r'http\S+|www\S+', '', cleaned)  # Remove URLs
            cleaned = re.sub(r'\S+@\S+', '', cleaned)  # Remove emails
            cleaned = re.sub(r'[^\w\s\-\$\#\.]', ' ', cleaned)  # Keep crypto chars
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
            cleaned = cleaned.strip()
            
            all_keywords.append({
                'original_keyword': keyword,
                'cleaned_keyword': cleaned,
                'search_volume': row.get('search_volume', 0) if 'search_volume' in row else 0,
                'competition': row.get('competition', 0) if 'competition' in row else 0,
                'cpc': row.get('cpc', 0.0) if 'cpc' in row else 0.0
            })
        
        # Create dataframe
        all_df = pd.DataFrame(all_keywords)
        
        # Separate valid and removed keywords
        kept_keywords = []
        removed_keywords = []
        
        for idx, row in all_df.iterrows():
            keyword = row['cleaned_keyword']
            
            # Check if valid
            if len(keyword) < 3 or len(keyword) > 100:
                row['removal_reason'] = 'Invalid Length'
                removed_keywords.append(row)
            elif not self.relevance_checker.is_relevant(keyword):
                row['removal_reason'] = self.relevance_checker.classify_removal_reason(keyword)
                removed_keywords.append(row)
            else:
                kept_keywords.append(row)
        
        kept_df = pd.DataFrame(kept_keywords)
        removed_df = pd.DataFrame(removed_keywords)
        
        # Remove exact duplicates from kept
        if len(kept_df) > 0:
            before_dedup = len(kept_df)
            kept_df = kept_df.drop_duplicates(subset=['cleaned_keyword'], keep='first')
            duplicates_removed = before_dedup - len(kept_df)
            
            if duplicates_removed > 0:
                print(f"✓ Removed {duplicates_removed} duplicate keywords")
        
        # Add features to kept keywords
        if len(kept_df) > 0:
            kept_df = self._add_features(kept_df)
            kept_df = kept_df.reset_index(drop=True)
        
        print(f"\n📊 Processing Results:")
        print(f"   • Keywords Kept: {len(kept_df):,}")
        print(f"   • Keywords Removed: {len(removed_df):,}")
        
        if len(removed_df) > 0:
            print(f"\n📋 Removal Reasons:")
            removal_stats = removed_df['removal_reason'].value_counts()
            for reason, count in removal_stats.items():
                print(f"   • {reason}: {count:,}")
        
        return kept_df, removed_df
    
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features for clustering"""
        df['word_count'] = df['cleaned_keyword'].str.split().str.len()
        df['char_count'] = df['cleaned_keyword'].str.len()
        df['has_numbers'] = df['cleaned_keyword'].str.contains(r'\d', na=False)
        df['has_question'] = df['cleaned_keyword'].str.contains(
            r'how|what|why|when|where|which|who', na=False
        )
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
            available_terms = important_terms[:3]
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
                min_df=1
            )
            
            tfidf_matrix = tfidf.fit_transform(keywords)
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
            
            return top_terms
            
        except Exception as e:
            # Fallback to frequency-based extraction
            return self._extract_by_frequency(keywords)
    
    def _extract_by_frequency(self, keywords: List[str]) -> List[str]:
        """Extract terms by frequency"""
        word_freq = Counter()
        
        for keyword in keywords:
            words = keyword.lower().split()
            for word in words:
                if len(word) > 2:
                    word_freq[word] += 1
        
        # Get top words
        top_words = [word.title() for word, _ in word_freq.most_common(10)]
        return top_words
    
    def _create_topic_name(self, terms: List[str], level: str) -> str:
        """Create topic name from terms"""
        if not terms:
            return f"Crypto_{level.title()}"
        
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
        
        print("\n🤖 Starting BERT-based clustering...")
        
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
        
        # Create embeddings in batches
        embeddings = []
        batch_size = Config.BATCH_SIZE
        
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
            
            # Progress update
            if i % (batch_size * 10) == 0 and i > 0:
                progress = (i / len(keywords)) * 100
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
            n_neighbors=Config.UMAP_N_NEIGHBORS,
            n_components=Config.UMAP_N_COMPONENTS,
            min_dist=Config.UMAP_MIN_DIST,
            metric=Config.UMAP_METRIC,
            random_state=42,
            low_memory=True
        )
        
        reduced_embeddings = umap_model.fit_transform(embeddings)
        return reduced_embeddings
    
    def _create_pillar_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        """Create main pillar clusters"""
        n_clusters = min(Config.PILLAR_CLUSTERS, len(embeddings) // 100)
        
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
                max(2, len(pillar_indices) // Config.MIN_CLUSTER_SIZE)
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
            cluster_keywords[label].append(keyword)
        
        # Generate name for each cluster
        cluster_names = {}
        for cluster_id, cluster_kws in cluster_keywords.items():
            cluster_names[cluster_id] = self.topic_generator.generate_topic_name(
                cluster_kws, level
            )
        
        return cluster_names

# ============= EXCEL OUTPUT GENERATOR =============
class ExcelOutputGenerator:
    """Generate comprehensive Excel output"""
    
    def create_output(self, clustered_df: pd.DataFrame, removed_df: pd.DataFrame, output_path: str):
        """Create Excel with clustered keywords and statistics"""
        print(f"\n📝 Creating Excel output: {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Clustered Keywords
            self._create_clustered_sheet(clustered_df, writer)
            
            # Sheet 2: Removed Keywords
            if len(removed_df) > 0:
                self._create_removed_sheet(removed_df, writer)
            
            # Sheet 3: Pillar Summary
            self._create_pillar_summary(clustered_df, writer)
            
            # Sheet 4: Topic Summary
            self._create_topic_summary(clustered_df, writer)
            
            # Sheet 5: Statistics
            self._create_statistics_sheet(clustered_df, removed_df, writer)
        
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
    
    def _create_removed_sheet(self, df: pd.DataFrame, writer):
        """Create removed keywords sheet"""
        if len(df) > 0:
            output_df = df[[
                'original_keyword', 'cleaned_keyword', 'removal_reason'
            ]].copy()
            
            output_df = output_df.sort_values('removal_reason')
            output_df.to_excel(writer, sheet_name='Removed_Keywords', index=False)
    
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
    
    def _create_statistics_sheet(self, clustered_df: pd.DataFrame, removed_df: pd.DataFrame, writer):
        """Create overall statistics sheet"""
        stats = []
        
        # Overall stats
        total_processed = len(clustered_df) + len(removed_df)
        stats.append({
            'Metric': 'Total Keywords Processed',
            'Value': total_processed
        })
        stats.append({
            'Metric': 'Keywords Kept',
            'Value': len(clustered_df)
        })
        stats.append({
            'Metric': 'Keywords Removed',
            'Value': len(removed_df)
        })
        stats.append({
            'Metric': 'Retention Rate',
            'Value': f"{(len(clustered_df) / total_processed * 100):.1f}%" if total_processed > 0 else "0%"
        })
        
        if len(clustered_df) > 0:
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
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)

# ============= MAIN PIPELINE =============
def run_clustering_pipeline():
    """Main execution pipeline"""
    start_time = datetime.now()
    
    print("="*80)
    print("🚀 ENHANCED BERT CLUSTERING SYSTEM")
    print("   Processing up to 200,000 keywords")
    print("   2-Level Hierarchy: Pillars → Topics")
    print("="*80)
    
    try:
        # Create output directory
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        
        # Step 1: Load and process keywords
        processor = KeywordProcessor()
        processed_df, removed_df = processor.load_and_process(
            Config.INPUT_FILE, 
            max_keywords=Config.MAX_KEYWORDS
        )
        
        # Save removed keywords
        if len(removed_df) > 0:
            removed_df.to_csv(Config.REMOVED_KEYWORDS_FILE, index=False)
            print(f"\n💾 Saved removed keywords to: {Config.REMOVED_KEYWORDS_FILE}")
        
        # Step 2: Create clusters (only if we have keywords)
        if len(processed_df) > 0:
            clustering_engine = HierarchicalClusteringEngine()
            clustered_df = clustering_engine.create_clusters(processed_df)
            
            # Step 3: Generate output
            excel_generator = ExcelOutputGenerator()
            excel_generator.create_output(clustered_df, removed_df, Config.OUTPUT_FILE)
            
            # Print summary
            elapsed_time = datetime.now() - start_time
            
            print("\n" + "="*80)
            print("✅ CLUSTERING COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"\n📊 FINAL RESULTS:")
            print(f"   • Total Keywords Processed: {len(processed_df) + len(removed_df):,}")
            print(f"   • Keywords Clustered: {len(clustered_df):,}")
            print(f"   • Keywords Removed: {len(removed_df):,}")
            print(f"   • Pillars Created: {clustered_df['pillar_name'].nunique()}")
            print(f"   • Topics Created: {clustered_df['topic_name'].nunique()}")
            print(f"\n⏱️ Processing Time: {elapsed_time}")
            print(f"\n📁 Output Files:")
            print(f"   • Clustered Keywords: {Config.OUTPUT_FILE}")
            print(f"   • Removed Keywords: {Config.REMOVED_KEYWORDS_FILE}")
            
            # Show sample topics
            print(f"\n🎯 Sample Pillar Structure:")
            for pillar in clustered_df['pillar_name'].unique()[:5]:
                pillar_data = clustered_df[clustered_df['pillar_name'] == pillar]
                topics = pillar_data['topic_name'].unique()[:3]
                keyword_count = len(pillar_data)
                print(f"\n   📌 {pillar} ({keyword_count:,} keywords)")
                for topic in topics:
                    topic_count = len(pillar_data[pillar_data['topic_name'] == topic])
                    print(f"      └─ {topic} ({topic_count} keywords)")
        else:
            print("\n⚠️ No valid keywords found to cluster!")
        
        return clustered_df if len(processed_df) > 0 else pd.DataFrame()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Run the clustering pipeline
    clustered_data = run_clustering_pipeline()
