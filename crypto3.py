"""
Advanced Cryptocurrency BERT-Based SEO Keyword Clustering System
Specialized for 130k+ crypto keywords with intelligent topic modeling
Ultra-accurate clustering with crypto-specific cleaning and validation
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

# Core ML and NLP libraries
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import umap
import hdbscan
from transformers import pipeline
import torch

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ============= CRYPTO-OPTIMIZED CONFIGURATION =============
class CryptoConfig:
    """Configuration optimized for cryptocurrency keyword clustering"""
    
    # File paths
    INPUT_FILE = '/home/admin1/Downloads/demo_crypto/Book6.csv'
    OUTPUT_FILE = '/home/admin1/Downloads/demo_crypto/output/crypto_clusters_ultra_accurate.xlsx'
    
    # BERT Model - Use finance/crypto optimized model if available
    EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'  # Better quality for domain-specific
    
    # Processing Configuration for 130k keywords
    BATCH_SIZE = 256  # Smaller batch for better accuracy
    MAX_SEQUENCE_LENGTH = 128
    
    # Hierarchical Clustering - Crypto-specific structure
    PILLAR_CLUSTERS = 12      # Main crypto pillars (Bitcoin, Ethereum, DeFi, etc.)
    PRIMARY_CLUSTERS = 50     # Major topic areas
    SECONDARY_CLUSTERS = 250  # Specific topics
    SUBTOPIC_CLUSTERS = 800   # Granular subtopics
    
    # Clustering Parameters - Tuned for crypto
    UMAP_N_NEIGHBORS = 20
    UMAP_N_COMPONENTS = 75
    UMAP_MIN_DIST = 0.0
    UMAP_METRIC = 'cosine'
    
    # Quality Control
    MIN_CLUSTER_SIZE = 5
    MIN_TOPIC_SIZE = 8
    SIMILARITY_THRESHOLD = 0.75
    
    # Memory Management
    ENABLE_CACHING = True
    CACHE_DIR = 'crypto_cache'

# ============= CRYPTO DOMAIN KNOWLEDGE =============
class CryptoDomainKnowledge:
    """Comprehensive crypto domain knowledge for accurate clustering"""
    
    # Core crypto concepts for validation
    CRYPTO_PILLARS = {
        'bitcoin': ['btc', 'bitcoin', 'satoshi', 'lightning', 'segwit', 'taproot', 'halving'],
        'ethereum': ['eth', 'ethereum', 'ether', 'gas', 'gwei', 'eip', 'merge', 'sharding'],
        'defi': ['defi', 'decentralized finance', 'yield', 'farming', 'liquidity', 'amm', 'dex'],
        'nft': ['nft', 'non-fungible', 'opensea', 'erc721', 'erc1155', 'metadata'],
        'exchanges': ['exchange', 'binance', 'coinbase', 'kraken', 'trading', 'spot', 'futures'],
        'stablecoins': ['stablecoin', 'usdt', 'usdc', 'dai', 'peg', 'collateral'],
        'layer2': ['layer 2', 'l2', 'polygon', 'arbitrum', 'optimism', 'zk', 'rollup'],
        'web3': ['web3', 'dapp', 'decentralized', 'blockchain', 'smart contract'],
        'mining': ['mining', 'mining pool', 'hash', 'asic', 'gpu', 'proof of work'],
        'staking': ['staking', 'validator', 'delegation', 'rewards', 'apr', 'apy'],
        'privacy': ['privacy', 'monero', 'zcash', 'mixer', 'tornado', 'anonymous'],
        'regulation': ['regulation', 'sec', 'compliance', 'kyc', 'aml', 'tax', 'legal']
    }
    
    # Valid crypto terms (extended list)
    VALID_CRYPTO_TERMS = {
        # Cryptocurrencies
        'bitcoin', 'btc', 'ethereum', 'eth', 'binance', 'bnb', 'cardano', 'ada',
        'solana', 'sol', 'ripple', 'xrp', 'polkadot', 'dot', 'dogecoin', 'doge',
        'avalanche', 'avax', 'chainlink', 'link', 'polygon', 'matic', 'cosmos', 'atom',
        'algorand', 'algo', 'vechain', 'vet', 'stellar', 'xlm', 'filecoin', 'fil',
        'tron', 'trx', 'monero', 'xmr', 'litecoin', 'ltc', 'uniswap', 'uni',
        
        # DeFi Protocols
        'aave', 'compound', 'maker', 'mkr', 'curve', 'crv', 'sushi', 'sushiswap',
        'pancakeswap', 'cake', 'yearn', 'yfi', 'synthetix', 'snx', 'balancer', 'bal',
        '1inch', 'dydx', 'venus', 'cream', 'bancor', 'bnt', 'kyber', 'knc',
        
        # Technical Terms
        'blockchain', 'cryptocurrency', 'crypto', 'defi', 'nft', 'dao', 'dex', 'cex',
        'amm', 'liquidity', 'pool', 'yield', 'farming', 'staking', 'validator',
        'mining', 'miner', 'hash', 'hashrate', 'difficulty', 'block', 'transaction',
        'wallet', 'address', 'private', 'key', 'public', 'seed', 'phrase', 'ledger',
        'metamask', 'trustwallet', 'phantom', 'gas', 'gwei', 'fee', 'slippage',
        
        # Concepts
        'smart', 'contract', 'token', 'tokenomics', 'whitepaper', 'roadmap', 'mainnet',
        'testnet', 'fork', 'hardfork', 'softfork', 'consensus', 'proof', 'work', 'stake',
        'burn', 'mint', 'supply', 'market', 'cap', 'volume', 'price', 'chart',
        'bullish', 'bearish', 'pump', 'dump', 'whale', 'hodl', 'fomo', 'fud',
        'altcoin', 'memecoin', 'shitcoin', 'rugpull', 'scam', 'hack', 'exploit',
        
        # Layer 2 & Scaling
        'layer', 'l1', 'l2', 'rollup', 'optimistic', 'zk', 'zero', 'knowledge',
        'arbitrum', 'optimism', 'zksync', 'starknet', 'lightning', 'network',
        'sidechain', 'bridge', 'cross', 'chain', 'interoperability', 'cosmos', 'ibc',
        
        # Web3 & Metaverse
        'web3', 'metaverse', 'gamefi', 'play', 'earn', 'p2e', 'sandbox', 'decentraland',
        'axie', 'infinity', 'ens', 'domain', 'ipfs', 'arweave', 'storage',
        
        # Stablecoins
        'stablecoin', 'usdt', 'tether', 'usdc', 'circle', 'dai', 'busd', 'tusd',
        'pax', 'paxos', 'gusd', 'gemini', 'frax', 'ust', 'luna', 'algorithmic',
        
        # Exchanges & Trading
        'exchange', 'binance', 'coinbase', 'kraken', 'bitfinex', 'huobi', 'okex',
        'kucoin', 'gate', 'bittrex', 'bitstamp', 'trading', 'spot', 'futures',
        'margin', 'leverage', 'perpetual', 'options', 'derivatives', 'order', 'book',
        'limit', 'market', 'stop', 'loss', 'take', 'profit', 'liquidation',
        
        # Security & Privacy
        'security', 'privacy', 'anonymous', 'kyc', 'aml', 'audit', 'hack', 'exploit',
        'vulnerability', 'multisig', '2fa', 'cold', 'hot', 'hardware', 'custody',
        
        # Regulation & Compliance
        'regulation', 'sec', 'cftc', 'regulatory', 'compliance', 'tax', 'legal',
        'securities', 'commodity', 'etf', 'bitcoin', 'spot', 'futures', 'grayscale'
    }
    
    # Gibberish patterns to filter out
    GIBBERISH_PATTERNS = [
        r'^[a-z]{20,}$',  # Very long single words
        r'^[0-9]+$',  # Pure numbers
        r'^[a-z0-9]{1,2}$',  # Too short
        r'[^a-zA-Z0-9\s\-\_\.]',  # Special characters
        r'(.)\1{4,}',  # Repeated characters (e.g., "aaaaaaa")
        r'^test',  # Test keywords
        r'^demo',  # Demo keywords
        r'^example',  # Example keywords
        r'xxx|porn|sex',  # Adult content
        r'^click|^download|^free\s+download',  # Spam patterns
    ]
    
    # Topic name templates for crypto
    TOPIC_TEMPLATES = {
        'trading': ['Trading', 'Analysis', 'Strategies', 'Signals', 'Charts'],
        'investment': ['Investment', 'Portfolio', 'Holdings', 'Allocation'],
        'technical': ['Development', 'Protocol', 'Implementation', 'Architecture'],
        'guides': ['Guide', 'Tutorial', 'How-to', 'Explained', 'Basics'],
        'news': ['News', 'Updates', 'Announcements', 'Events'],
        'comparison': ['Comparison', 'vs', 'Alternatives', 'Differences'],
        'tools': ['Tools', 'Platforms', 'Software', 'Applications'],
        'security': ['Security', 'Safety', 'Protection', 'Best Practices']
    }

# ============= CRYPTO KEYWORD PROCESSOR =============
class CryptoKeywordProcessor:
    """Advanced crypto-specific keyword processing and cleaning"""
    
    def __init__(self):
        self.cache_dir = Path(CryptoConfig.CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)
        self.domain = CryptoDomainKnowledge()
        self.crypto_terms = self.domain.VALID_CRYPTO_TERMS
        
    def load_and_process_keywords(self, file_path: str) -> pd.DataFrame:
        """Load and process crypto keywords with ultra-accurate cleaning"""
        logging.info(f"Loading 130k crypto keywords from: {file_path}")
        
        try:
            # Load raw data
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path, low_memory=False)
            
            logging.info(f"Loaded {len(df)} keywords")
            
            # Find keyword column
            keyword_col = self._find_keyword_column(df)
            
            # Process with crypto-specific cleaning
            processed_df = self._process_crypto_keywords(df, keyword_col)
            
            return processed_df
            
        except Exception as e:
            logging.error(f"Error processing keywords: {e}")
            raise
    
    def _find_keyword_column(self, df: pd.DataFrame) -> str:
        """Find keyword column"""
        possible_names = ['keyword', 'keywords', 'query', 'term', 'search_term']
        
        for col in df.columns:
            if col.lower() in possible_names:
                return col
        
        # Return first text column
        for col in df.columns:
            if df[col].dtype == 'object':
                return col
        
        return df.columns[0]
    
    def _process_crypto_keywords(self, df: pd.DataFrame, keyword_col: str) -> pd.DataFrame:
        """Ultra-accurate crypto keyword processing"""
        logging.info("Processing crypto keywords with advanced cleaning...")
        
        # Create working dataframe
        processed_df = pd.DataFrame({
            'original_keyword': df[keyword_col].astype(str),
            'search_volume': df.get('search_volume', 0) if 'search_volume' in df.columns else 0,
            'competition': df.get('competition', 0) if 'competition' in df.columns else 0,
            'cpc': df.get('cpc', 0.0) if 'cpc' in df.columns else 0.0
        })
        
        initial_count = len(processed_df)
        
        # Clean keywords
        processed_df['cleaned_keyword'] = processed_df['original_keyword'].str.lower().str.strip()
        
        # Remove URLs and emails
        processed_df['cleaned_keyword'] = processed_df['cleaned_keyword'].str.replace(r'http\S+|www\S+', '', regex=True)
        processed_df['cleaned_keyword'] = processed_df['cleaned_keyword'].str.replace(r'\S+@\S+', '', regex=True)
        
        # Clean special characters but keep crypto-relevant ones
        processed_df['cleaned_keyword'] = processed_df['cleaned_keyword'].str.replace(r'[^\w\s\-\$\#]', ' ', regex=True)
        processed_df['cleaned_keyword'] = processed_df['cleaned_keyword'].str.replace(r'\s+', ' ', regex=True)
        processed_df['cleaned_keyword'] = processed_df['cleaned_keyword'].str.strip()
        
        # Filter by length
        processed_df = processed_df[
            (processed_df['cleaned_keyword'].str.len() >= 3) & 
            (processed_df['cleaned_keyword'].str.len() <= 100)
        ]
        
        # Remove gibberish using patterns
        for pattern in self.domain.GIBBERISH_PATTERNS:
            processed_df = processed_df[~processed_df['cleaned_keyword'].str.contains(pattern, regex=True, na=False)]
        
        # Filter for crypto relevance
        processed_df = self._filter_crypto_relevant(processed_df)
        
        # Remove duplicates
        processed_df = processed_df.drop_duplicates(subset=['cleaned_keyword'], keep='first')
        
        # Add crypto-specific features
        processed_df = self._add_crypto_features(processed_df)
        
        processed_df = processed_df.reset_index(drop=True)
        
        logging.info(f"✓ Cleaned: {len(processed_df)} valid crypto keywords ({initial_count - len(processed_df)} removed)")
        return processed_df
    
    def _filter_crypto_relevant(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter only crypto-relevant keywords"""
        # Create relevance score
        df['crypto_relevance'] = 0
        
        # Check for crypto terms
        for term in self.crypto_terms:
            df.loc[df['cleaned_keyword'].str.contains(term, na=False), 'crypto_relevance'] += 1
        
        # Check for pillar terms
        for pillar, terms in self.domain.CRYPTO_PILLARS.items():
            for term in terms:
                df.loc[df['cleaned_keyword'].str.contains(term, na=False), 'crypto_relevance'] += 2
        
        # Keep only relevant keywords
        df = df[df['crypto_relevance'] > 0]
        
        return df.drop('crypto_relevance', axis=1)
    
    def _add_crypto_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add crypto-specific features"""
        df['word_count'] = df['cleaned_keyword'].str.split().str.len()
        df['char_count'] = df['cleaned_keyword'].str.len()
        
        # Crypto-specific features
        df['has_ticker'] = df['cleaned_keyword'].str.contains(r'\b[A-Z]{3,5}\b', na=False)
        df['has_price_terms'] = df['cleaned_keyword'].str.contains('price|prediction|forecast|analysis', na=False)
        df['has_technical_terms'] = df['cleaned_keyword'].str.contains('blockchain|smart contract|defi|nft|dao', na=False)
        df['has_trading_terms'] = df['cleaned_keyword'].str.contains('buy|sell|trade|exchange|swap', na=False)
        df['has_question_words'] = df['cleaned_keyword'].str.contains('how|what|why|when|where|which|who', na=False)
        df['is_long_tail'] = df['word_count'] > 3
        
        # Identify keyword type
        df['keyword_type'] = df.apply(self._classify_keyword_type, axis=1)
        
        return df
    
    def _classify_keyword_type(self, row) -> str:
        """Classify keyword type"""
        keyword = row['cleaned_keyword']
        
        if row['has_price_terms']:
            return 'price_analysis'
        elif row['has_trading_terms']:
            return 'trading'
        elif row['has_technical_terms']:
            return 'technical'
        elif row['has_question_words']:
            return 'informational'
        elif row['word_count'] == 1:
            return 'branded'
        elif row['is_long_tail']:
            return 'long_tail'
        else:
            return 'general'

# ============= CRYPTO TOPIC GENERATOR =============
class CryptoTopicGenerator:
    """Generate meaningful crypto-specific topic names"""
    
    def __init__(self):
        self.domain = CryptoDomainKnowledge()
        self.tfidf_vectorizer = None
        
    def generate_topic_from_cluster(self, keywords: List[str], level: str = 'secondary') -> str:
        """Generate topic name AFTER clustering based on keyword patterns"""
        if not keywords or len(keywords) == 0:
            return f"Empty_{level.title()}_Topic"
        
        # Analyze keyword patterns
        crypto_entities = self._extract_crypto_entities(keywords)
        common_patterns = self._extract_common_patterns(keywords)
        
        # Generate topic based on patterns
        if crypto_entities:
            topic = self._generate_entity_based_topic(crypto_entities, common_patterns, level)
        else:
            topic = self._generate_pattern_based_topic(keywords, common_patterns, level)
        
        return self._clean_topic_name(topic, level)
    
    def _extract_crypto_entities(self, keywords: List[str]) -> Dict[str, int]:
        """Extract crypto entities from keywords"""
        entities = defaultdict(int)
        
        for keyword in keywords:
            words = keyword.lower().split()
            for word in words:
                # Check major cryptos
                if word in ['bitcoin', 'btc']:
                    entities['Bitcoin'] += 1
                elif word in ['ethereum', 'eth']:
                    entities['Ethereum'] += 1
                elif word in ['binance', 'bnb']:
                    entities['Binance'] += 1
                elif word in ['cardano', 'ada']:
                    entities['Cardano'] += 1
                elif word in ['solana', 'sol']:
                    entities['Solana'] += 1
                elif word in ['polygon', 'matic']:
                    entities['Polygon'] += 1
                elif word in ['defi', 'decentralized']:
                    entities['DeFi'] += 1
                elif word in ['nft', 'nfts']:
                    entities['NFT'] += 1
                elif word in ['staking', 'stake']:
                    entities['Staking'] += 1
                elif word in ['mining', 'miner']:
                    entities['Mining'] += 1
                elif word in ['trading', 'trade']:
                    entities['Trading'] += 1
                elif word in ['wallet', 'wallets']:
                    entities['Wallets'] += 1
        
        return dict(sorted(entities.items(), key=lambda x: x[1], reverse=True))
    
    def _extract_common_patterns(self, keywords: List[str]) -> Dict[str, int]:
        """Extract common patterns from keywords"""
        patterns = defaultdict(int)
        
        # Pattern categories
        pattern_types = {
            'price': ['price', 'prediction', 'forecast', 'analysis', 'chart'],
            'guide': ['how', 'what', 'guide', 'tutorial', 'learn'],
            'comparison': ['vs', 'versus', 'compare', 'difference', 'alternative'],
            'investment': ['buy', 'invest', 'portfolio', 'best', 'top'],
            'technical': ['blockchain', 'protocol', 'smart', 'contract', 'gas'],
            'exchange': ['exchange', 'swap', 'convert', 'trade', 'platform'],
            'security': ['secure', 'safe', 'hack', 'scam', 'protect'],
            'news': ['news', 'update', 'latest', 'announcement', 'launch']
        }
        
        for keyword in keywords:
            for pattern_type, terms in pattern_types.items():
                if any(term in keyword.lower() for term in terms):
                    patterns[pattern_type] += 1
        
        return dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True))
    
    def _generate_entity_based_topic(self, entities: Dict[str, int], patterns: Dict[str, int], level: str) -> str:
        """Generate topic based on crypto entities"""
        top_entities = list(entities.keys())[:2]
        top_patterns = list(patterns.keys())[:1] if patterns else []
        
        if level == 'pillar':
            # Broad pillar topics
            if top_entities:
                return top_entities[0]
        elif level == 'primary':
            # Primary topics
            if top_entities and top_patterns:
                pattern_map = {
                    'price': 'Price Analysis',
                    'guide': 'Guides',
                    'comparison': 'Comparisons',
                    'investment': 'Investment',
                    'technical': 'Technical',
                    'exchange': 'Exchanges',
                    'security': 'Security',
                    'news': 'News'
                }
                pattern_name = pattern_map.get(top_patterns[0], top_patterns[0].title())
                return f"{top_entities[0]} {pattern_name}"
            elif top_entities:
                return f"{top_entities[0]} Overview"
        elif level == 'secondary':
            # Secondary topics
            if len(top_entities) >= 2:
                return f"{top_entities[0]} {top_entities[1]}"
            elif top_entities and top_patterns:
                return f"{top_entities[0]} {top_patterns[0].title()}"
            elif top_entities:
                return f"{top_entities[0]} Topics"
        else:  # subtopic
            # Most specific
            if top_entities and top_patterns:
                pattern_suffix = {
                    'price': 'Price Predictions',
                    'guide': 'How-To Guides',
                    'comparison': 'Detailed Comparison',
                    'investment': 'Investment Strategy',
                    'technical': 'Technical Details',
                    'exchange': 'Exchange Guide',
                    'security': 'Security Guide',
                    'news': 'Latest Updates'
                }
                suffix = pattern_suffix.get(top_patterns[0], 'Analysis')
                return f"{top_entities[0]} {suffix}"
            elif top_entities:
                return f"{top_entities[0]} Specific"
        
        return self._generate_pattern_based_topic([], patterns, level)
    
    def _generate_pattern_based_topic(self, keywords: List[str], patterns: Dict[str, int], level: str) -> str:
        """Generate topic based on patterns when no clear entity"""
        if not keywords:
            return f"Crypto_{level.title()}"
        
        # Use TF-IDF to find most important terms
        try:
            tfidf = TfidfVectorizer(
                max_features=5,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            tfidf_matrix = tfidf.fit_transform(keywords)
            feature_names = tfidf.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-3:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            
            # Clean and format terms
            clean_terms = []
            for term in top_terms:
                if any(crypto_term in term.lower() for crypto_term in self.domain.VALID_CRYPTO_TERMS):
                    clean_terms.append(term.title())
            
            if clean_terms:
                return " ".join(clean_terms[:2])
            else:
                return f"Crypto {level.title()}"
                
        except:
            return f"Crypto {level.title()}"
    
    def _clean_topic_name(self, topic: str, level: str) -> str:
        """Clean and format topic name"""
        # Remove duplicates
        words = topic.split()
        seen = set()
        clean_words = []
        for word in words:
            if word.lower() not in seen:
                seen.add(word.lower())
                clean_words.append(word)
        
        topic = " ".join(clean_words)
        
        # Ensure proper length
        if level == 'pillar' and len(clean_words) > 2:
            topic = " ".join(clean_words[:2])
        elif level == 'primary' and len(clean_words) > 3:
            topic = " ".join(clean_words[:3])
        elif level == 'secondary' and len(clean_words) > 4:
            topic = " ".join(clean_words[:4])
        
        # Capitalize properly
        return topic.strip()

# ============= ULTRA-ACCURATE CLUSTERING ENGINE =============
class CryptoClusteringEngine:
    """Ultra-accurate hierarchical clustering for crypto keywords"""
    
    def __init__(self):
        self.embedding_model = None
        self.embeddings = None
        self.topic_generator = CryptoTopicGenerator()
        self.hierarchy_data = {}
        
    def create_hierarchical_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create multi-level hierarchical clusters with ultra accuracy"""
        logging.info("Starting ultra-accurate crypto clustering...")
        
        keywords = df['cleaned_keyword'].tolist()
        
        # Step 1: Create domain-specific embeddings
        self.embeddings = self._create_crypto_embeddings(keywords, df)
        
        # Step 2: Advanced dimensionality reduction
        reduced_embeddings = self._reduce_dimensions_advanced(self.embeddings)
        
        # Step 3: Create hierarchical clusters with validation
        result_df = df.copy()
        
        # Level 1: Pillar topics (major crypto categories)
        logging.info("Creating crypto pillar clusters...")
        pillar_labels, pillar_names = self._create_validated_clusters(
            reduced_embeddings, keywords, CryptoConfig.PILLAR_CLUSTERS, 'pillar'
        )
        result_df['pillar_id'] = pillar_labels
        result_df['pillar_name'] = pillar_names
        
        # Level 2: Primary topics
        logging.info("Creating primary topic clusters...")
        primary_labels, primary_names = self._create_hierarchical_clusters(
            reduced_embeddings, keywords, pillar_labels, CryptoConfig.PRIMARY_CLUSTERS, 'primary'
        )
        result_df['primary_id'] = primary_labels
        result_df['primary_name'] = primary_names
        
        # Level 3: Secondary topics
        logging.info("Creating secondary topic clusters...")
        secondary_labels, secondary_names = self._create_hierarchical_clusters(
            reduced_embeddings, keywords, primary_labels, CryptoConfig.SECONDARY_CLUSTERS, 'secondary'
        )
        result_df['secondary_id'] = secondary_labels
        result_df['secondary_name'] = secondary_names
        
        # Level 4: Subtopics
        logging.info("Creating granular subtopic clusters...")
        subtopic_labels, subtopic_names = self._create_hierarchical_clusters(
            reduced_embeddings, keywords, secondary_labels, CryptoConfig.SUBTOPIC_CLUSTERS, 'subtopic'
        )
        result_df['subtopic_id'] = subtopic_labels
        result_df['subtopic_name'] = subtopic_names
        
        # Post-process for quality
        result_df = self._post_process_clusters(result_df)
        
        # Build hierarchy tree
        self._build_hierarchy_tree(result_df)
        
        logging.info("✓ Ultra-accurate clustering completed")
        return result_df
    
    def _create_crypto_embeddings(self, keywords: List[str], df: pd.DataFrame) -> np.ndarray:
        """Create embeddings with crypto domain knowledge"""
        logging.info(f"Creating crypto-optimized embeddings for {len(keywords)} keywords...")
        
        # Use better model for crypto
        self.embedding_model = SentenceTransformer(CryptoConfig.EMBEDDING_MODEL)
        self.embedding_model.max_seq_length = CryptoConfig.MAX_SEQUENCE_LENGTH
        
        # Create base embeddings
        embeddings = []
        batch_size = CryptoConfig.BATCH_SIZE
        
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
            
            if i % (batch_size * 10) == 0:
                progress = (i / len(keywords)) * 100
                logging.info(f"Embedding progress: {progress:.1f}%")
                gc.collect()
        
        embeddings = np.vstack(embeddings)
        
        # Enhance embeddings with crypto features
        crypto_features = self._extract_crypto_features(df)
        
        # Combine embeddings with features
        enhanced_embeddings = np.hstack([embeddings, crypto_features])
        
        logging.info(f"✓ Created enhanced embeddings with shape: {enhanced_embeddings.shape}")
        return enhanced_embeddings
    
    def _extract_crypto_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract crypto-specific features for better clustering"""
        features = []
        
        # Binary features
        features.append(df['has_ticker'].values.reshape(-1, 1))
        features.append(df['has_price_terms'].values.reshape(-1, 1))
        features.append(df['has_technical_terms'].values.reshape(-1, 1))
        features.append(df['has_trading_terms'].values.reshape(-1, 1))
        features.append(df['has_question_words'].values.reshape(-1, 1))
        
        # Normalized numeric features
        features.append((df['word_count'] / df['word_count'].max()).values.reshape(-1, 1))
        
        # One-hot encode keyword type
        keyword_types = pd.get_dummies(df['keyword_type'], prefix='type')
        features.append(keyword_types.values)
        
        return np.hstack(features).astype(np.float32)
    
    def _reduce_dimensions_advanced(self, embeddings: np.ndarray) -> np.ndarray:
        """Advanced dimensionality reduction for better clustering"""
        logging.info("Advanced dimensionality reduction...")
        
        # First PCA for noise reduction
        if embeddings.shape[1] > 100:
            pca = PCA(n_components=100, random_state=42)
            embeddings = pca.fit_transform(embeddings)
            logging.info(f"PCA reduced to {embeddings.shape[1]} dimensions")
        
        # UMAP for better cluster separation
        umap_model = umap.UMAP(
            n_neighbors=CryptoConfig.UMAP_N_NEIGHBORS,
            n_components=CryptoConfig.UMAP_N_COMPONENTS,
            min_dist=CryptoConfig.UMAP_MIN_DIST,
            metric=CryptoConfig.UMAP_METRIC,
            random_state=42,
            low_memory=True,
            angular_rp_forest=True  # Better for high-dimensional data
        )
        
        reduced_embeddings = umap_model.fit_transform(embeddings)
        logging.info(f"✓ UMAP reduced to {reduced_embeddings.shape[1]} dimensions")
        
        return reduced_embeddings
    
    def _create_validated_clusters(self, embeddings: np.ndarray, keywords: List[str], 
                                  n_clusters: int, level: str) -> Tuple[List[int], List[str]]:
        """Create clusters with validation for quality"""
        logging.info(f"Creating validated {level} clusters...")
        
        # Try HDBSCAN first for better quality
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=CryptoConfig.MIN_CLUSTER_SIZE,
                min_samples=5,
                cluster_selection_epsilon=0.5,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            
            cluster_labels = clusterer.fit_predict(embeddings)
            n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            # If too few clusters, use KMeans
            if n_clusters_found < n_clusters * 0.5:
                raise ValueError("Too few clusters from HDBSCAN")
                
        except:
            # Fallback to KMeans for controlled number of clusters
            kmeans = KMeans(
                n_clusters=min(n_clusters, len(keywords) // 10),
                random_state=42,
                n_init=20,
                max_iter=500
            )
            cluster_labels = kmeans.fit_predict(embeddings)
        
        # Validate and refine clusters
        cluster_labels = self._validate_clusters(cluster_labels, embeddings, keywords)
        
        # Generate topic names from clusters
        cluster_names = self._generate_cluster_names(keywords, cluster_labels, level)
        
        logging.info(f"✓ Created {len(set(cluster_labels))} validated {level} clusters")
        return cluster_labels.tolist(), cluster_names
    
    def _validate_clusters(self, labels: np.ndarray, embeddings: np.ndarray, 
                          keywords: List[str]) -> np.ndarray:
        """Validate and refine clusters for quality"""
        # Check for outliers (-1 labels from HDBSCAN)
        if -1 in labels:
            # Assign outliers to nearest cluster
            outlier_indices = np.where(labels == -1)[0]
            for idx in outlier_indices:
                # Find nearest non-outlier point
                non_outlier_mask = labels != -1
                if non_outlier_mask.any():
                    distances = np.linalg.norm(
                        embeddings[non_outlier_mask] - embeddings[idx], axis=1
                    )
                    nearest_idx = np.where(non_outlier_mask)[0][np.argmin(distances)]
                    labels[idx] = labels[nearest_idx]
        
        # Merge very small clusters
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if count < CryptoConfig.MIN_CLUSTER_SIZE:
                # Merge with nearest larger cluster
                small_cluster_mask = labels == label
                other_mask = ~small_cluster_mask
                
                if other_mask.any():
                    # Calculate centroid of small cluster
                    small_centroid = embeddings[small_cluster_mask].mean(axis=0)
                    
                    # Find nearest large cluster
                    other_labels = labels[other_mask]
                    other_embeddings = embeddings[other_mask]
                    
                    unique_other = np.unique(other_labels)
                    min_dist = float('inf')
                    nearest_label = unique_other[0]
                    
                    for other_label in unique_other:
                        other_centroid = embeddings[labels == other_label].mean(axis=0)
                        dist = np.linalg.norm(small_centroid - other_centroid)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_label = other_label
                    
                    labels[small_cluster_mask] = nearest_label
        
        return labels
    
    def _create_hierarchical_clusters(self, embeddings: np.ndarray, keywords: List[str],
                                     parent_labels: List[int], n_clusters: int, 
                                     level: str) -> Tuple[List[int], List[str]]:
        """Create hierarchical sub-clusters within parent clusters"""
        logging.info(f"Creating hierarchical {level} clusters...")
        
        final_labels = [-1] * len(keywords)
        final_names = ["Uncategorized"] * len(keywords)
        global_cluster_id = 0
        
        # Group by parent clusters
        parent_groups = defaultdict(list)
        for idx, parent_id in enumerate(parent_labels):
            parent_groups[parent_id].append(idx)
        
        for parent_id, indices in parent_groups.items():
            if len(indices) < CryptoConfig.MIN_CLUSTER_SIZE:
                # Too small to cluster further
                for idx in indices:
                    final_labels[idx] = global_cluster_id
                    final_names[idx] = f"Small_{level.title()}_Group"
                global_cluster_id += 1
                continue
            
            # Determine optimal number of subclusters
            parent_size = len(indices)
            target_clusters = max(2, min(
                n_clusters // len(parent_groups),
                parent_size // CryptoConfig.MIN_CLUSTER_SIZE
            ))
            
            # Extract parent embeddings and keywords
            parent_embeddings = embeddings[indices]
            parent_keywords = [keywords[i] for i in indices]
            
            # Cluster within parent
            if target_clusters >= parent_size // 2:
                # Too many clusters requested, reduce
                target_clusters = max(2, parent_size // 10)
            
            # Use HDBSCAN for quality within parent
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=max(3, parent_size // target_clusters),
                    min_samples=2,
                    metric='euclidean'
                )
                sub_labels = clusterer.fit_predict(parent_embeddings)
                
                # Handle outliers
                if -1 in sub_labels:
                    # Assign outliers to nearest cluster
                    for i, label in enumerate(sub_labels):
                        if label == -1:
                            distances = np.linalg.norm(
                                parent_embeddings - parent_embeddings[i], axis=1
                            )
                            distances[i] = float('inf')
                            nearest = np.argmin(distances)
                            sub_labels[i] = sub_labels[nearest] if sub_labels[nearest] != -1 else 0
                
            except:
                # Fallback to KMeans
                kmeans = KMeans(
                    n_clusters=target_clusters,
                    random_state=42,
                    n_init=10
                )
                sub_labels = kmeans.fit_predict(parent_embeddings)
            
            # Generate names for subclusters
            sub_names = self._generate_cluster_names(parent_keywords, sub_labels, level)
            
            # Assign global labels and names
            unique_sub_labels = np.unique(sub_labels)
            label_mapping = {old: new for new, old in enumerate(unique_sub_labels)}
            
            for i, original_idx in enumerate(indices):
                mapped_label = label_mapping[sub_labels[i]]
                global_sub_id = global_cluster_id + mapped_label
                final_labels[original_idx] = global_sub_id
                final_names[original_idx] = sub_names[i]
            
            global_cluster_id += len(unique_sub_labels)
        
        logging.info(f"✓ Created {len(set(final_labels))} hierarchical {level} clusters")
        return final_labels, final_names
    
    def _generate_cluster_names(self, keywords: List[str], labels: List[int], level: str) -> List[str]:
        """Generate meaningful names for clusters using topic generator"""
        # Group keywords by cluster
        cluster_groups = defaultdict(list)
        for keyword, label in zip(keywords, labels):
            cluster_groups[label].append(keyword)
        
        # Generate names for each cluster
        cluster_name_map = {}
        for cluster_id, cluster_keywords in cluster_groups.items():
            # Use topic generator to create name from cluster keywords
            cluster_name_map[cluster_id] = self.topic_generator.generate_topic_from_cluster(
                cluster_keywords, level
            )
        
        # Map back to original order
        return [cluster_name_map[label] for label in labels]
    
    def _post_process_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process clusters for quality assurance"""
        logging.info("Post-processing clusters for quality...")
        
        # Fix any inconsistencies in hierarchy
        for idx, row in df.iterrows():
            # Ensure hierarchy makes sense
            if 'uncategorized' in row['pillar_name'].lower():
                # Try to assign based on keywords
                keyword = row['cleaned_keyword']
                if 'bitcoin' in keyword or 'btc' in keyword:
                    df.at[idx, 'pillar_name'] = 'Bitcoin'
                elif 'ethereum' in keyword or 'eth' in keyword:
                    df.at[idx, 'pillar_name'] = 'Ethereum'
                elif 'defi' in keyword:
                    df.at[idx, 'pillar_name'] = 'DeFi'
                else:
                    df.at[idx, 'pillar_name'] = 'General Crypto'
        
        # Remove duplicate topic names at same level
        df = self._deduplicate_topic_names(df)
        
        return df
    
    def _deduplicate_topic_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure unique topic names at each level"""
        levels = ['pillar_name', 'primary_name', 'secondary_name', 'subtopic_name']
        
        for level in levels:
            # Count occurrences
            name_counts = df[level].value_counts()
            duplicates = name_counts[name_counts > 1].index
            
            for dup_name in duplicates:
                # Add numeric suffix to duplicates
                dup_mask = df[level] == dup_name
                dup_indices = df[dup_mask].index
                
                for i, idx in enumerate(dup_indices[1:], 1):
                    df.at[idx, level] = f"{dup_name} {i+1}"
        
        return df
    
    def _build_hierarchy_tree(self, df: pd.DataFrame):
        """Build comprehensive hierarchy tree structure"""
        logging.info("Building crypto hierarchy tree...")
        
        hierarchy_data = []
        
        # Build tree at each level
        for level_config in [
            ('Pillar', 'pillar_id', 'pillar_name', None),
            ('Primary', 'primary_id', 'primary_name', 'pillar_id'),
            ('Secondary', 'secondary_id', 'secondary_name', 'primary_id'),
            ('Subtopic', 'subtopic_id', 'subtopic_name', 'secondary_id')
        ]:
            level_name, id_col, name_col, parent_col = level_config
            
            if parent_col:
                group_cols = [parent_col, id_col, name_col]
            else:
                group_cols = [id_col, name_col]
            
            level_stats = df.groupby(group_cols).agg({
                'cleaned_keyword': 'count',
                'search_volume': 'sum',
                'competition': 'mean',
                'cpc': 'mean'
            }).reset_index()
            
            for _, row in level_stats.iterrows():
                hierarchy_data.append({
                    'level': level_name,
                    'id': f"{level_name[0]}_{row[id_col]}",
                    'name': row[name_col],
                    'parent_id': f"{level_config[0][0]}_{row[parent_col]}" if parent_col else None,
                    'keyword_count': row['cleaned_keyword'],
                    'total_search_volume': row['search_volume'],
                    'avg_competition': round(row['competition'], 3),
                    'avg_cpc': round(row['cpc'], 2)
                })
        
        self.hierarchy_data = pd.DataFrame(hierarchy_data)
        logging.info(f"✓ Built hierarchy tree with {len(hierarchy_data)} nodes")

# ============= MAIN PIPELINE =============
def run_crypto_clustering_pipeline():
    """Execute the complete crypto keyword clustering pipeline"""
    start_time = datetime.now()
    
    print("="*80)
    print("🚀 ULTRA-ACCURATE CRYPTO KEYWORD CLUSTERING SYSTEM")
    print("="*80)
    
    try:
        # Initialize components
        processor = CryptoKeywordProcessor()
        clustering_engine = CryptoClusteringEngine()
        
        # Step 1: Load and clean data
        print("\n[STEP 1/3] Loading and cleaning 130k crypto keywords...")
        df = processor.load_and_process_keywords(CryptoConfig.INPUT_FILE)
        print(f"✓ Processed {len(df)} valid crypto keywords")
        
        # Step 2: Create ultra-accurate clusters
        print("\n[STEP 2/3] Creating ultra-accurate BERT clusters...")
        clustered_df = clustering_engine.create_hierarchical_clusters(df)
        
        # Step 3: Generate output
        print("\n[STEP 3/3] Generating comprehensive Excel output...")
        
        # Create Excel with all sheets
        with pd.ExcelWriter(CryptoConfig.OUTPUT_FILE, engine='openpyxl') as writer:
            # Main clustered keywords
            clustered_df.to_excel(writer, sheet_name='Crypto_Keywords_Clustered', index=False)
            
            # Hierarchy tree
            clustering_engine.hierarchy_data.to_excel(writer, sheet_name='Hierarchy_Tree', index=False)
            
            # Summary statistics
            summary_df = create_summary_statistics(clustered_df)
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        elapsed_time = datetime.now() - start_time
        
        print("\n" + "="*80)
        print("✅ CRYPTO CLUSTERING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📊 RESULTS:")
        print(f"   • Total Keywords Processed: {len(clustered_df):,}")
        print(f"   • Pillar Topics: {clustered_df['pillar_name'].nunique()}")
        print(f"   • Primary Topics: {clustered_df['primary_name'].nunique()}")
        print(f"   • Secondary Topics: {clustered_df['secondary_name'].nunique()}")
        print(f"   • Subtopics: {clustered_df['subtopic_name'].nunique()}")
        print(f"\n⏱️ Processing Time: {elapsed_time}")
        print(f"📁 Output File: {CryptoConfig.OUTPUT_FILE}")
        
        return clustered_df
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics for the clustering"""
    summary = []
    
    # Top pillars by keyword count
    pillar_stats = df.groupby('pillar_name').agg({
        'cleaned_keyword': 'count',
        'search_volume': 'sum'
    }).sort_values('cleaned_keyword', ascending=False).head(20)
    
    for pillar, stats in pillar_stats.iterrows():
        summary.append({
            'Category': 'Top Pillar',
            'Name': pillar,
            'Keyword_Count': stats['cleaned_keyword'],
            'Total_Search_Volume': stats['search_volume']
        })
    
    return pd.DataFrame(summary)

if __name__ == "__main__":
    os.makedirs(os.path.dirname(CryptoConfig.OUTPUT_FILE), exist_ok=True)
    os.makedirs(CryptoConfig.CACHE_DIR, exist_ok=True)
    run_crypto_clustering_pipeline()