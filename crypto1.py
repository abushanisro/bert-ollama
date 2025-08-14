"""
Advanced BERT-Based Crypto SEO Keyword Clustering System
Specialized for cryptocurrency and blockchain keywords with intelligent filtering
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

# Topic Modeling
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ============= CRYPTO-SPECIFIC CONFIGURATION =============
class CryptoConfig:
    """Configuration optimized for cryptocurrency keyword clustering"""
    
    # File paths
    INPUT_FILE = '/home/admin1/Downloads/demo_crypto/Book6.csv'
    OUTPUT_FILE = '/home/admin1/Downloads/demo_crypto/output/crypto_seo_clusters_clean.xlsx'
    
    # BERT Model - Using finance/crypto optimized model if available
    EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'  # High quality embeddings
    
    # Processing Configuration
    BATCH_SIZE = 512
    MAX_SEQUENCE_LENGTH = 128
    MIN_KEYWORD_LENGTH = 2  # Minimum keyword length
    MAX_KEYWORD_LENGTH = 100  # Maximum keyword length
    
    # Crypto-Specific Clustering Configuration
    PILLAR_CLUSTERS = 15      # Main crypto categories (DeFi, NFT, Trading, etc.)
    PRIMARY_CLUSTERS = 50     # Primary crypto topics
    SECONDARY_CLUSTERS = 150  # Secondary crypto topics
    SUBTOPIC_CLUSTERS = 400   # Detailed subtopics
    
    # Clustering Parameters
    UMAP_N_NEIGHBORS = 15
    UMAP_N_COMPONENTS = 50
    UMAP_MIN_DIST = 0.0
    UMAP_METRIC = 'cosine'
    
    # BERTopic Configuration
    MIN_TOPIC_SIZE = 5
    TOP_N_WORDS = 15
    NGRAM_RANGE = (1, 3)
    
    # Memory Management
    ENABLE_CACHING = True
    CACHE_DIR = 'crypto_cache'
    
    # Crypto Filtering Threshold
    CRYPTO_RELEVANCE_THRESHOLD = 0.3  # Minimum relevance score to keep keyword

# ============= CRYPTO DOMAIN KNOWLEDGE =============
class CryptoDomainKnowledge:
    """Comprehensive cryptocurrency domain knowledge for filtering and classification"""
    
    # Core crypto terms that must be present or related
    CRYPTO_CORE_TERMS = {
        # Cryptocurrencies
        'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency', 'altcoin',
        'satoshi', 'wei', 'gwei', 'binance', 'bnb', 'cardano', 'ada', 'solana', 'sol',
        'polkadot', 'dot', 'dogecoin', 'doge', 'shiba', 'shib', 'ripple', 'xrp',
        'litecoin', 'ltc', 'chainlink', 'link', 'polygon', 'matic', 'avalanche', 'avax',
        'cosmos', 'atom', 'algorand', 'algo', 'stellar', 'xlm', 'tron', 'trx',
        'monero', 'xmr', 'tether', 'usdt', 'usdc', 'dai', 'busd', 'stablecoin',
        
        # Blockchain Technology
        'blockchain', 'distributed', 'ledger', 'consensus', 'node', 'validator',
        'mining', 'miner', 'hash', 'hashrate', 'difficulty', 'block', 'chain',
        'merkle', 'tree', 'genesis', 'fork', 'hardfork', 'softfork', 'mainnet',
        'testnet', 'devnet', 'sidechain', 'layer1', 'layer2', 'l1', 'l2',
        'rollup', 'optimistic', 'zk', 'zkrollup', 'plasma', 'sharding', 'beacon',
        
        # DeFi Terms
        'defi', 'decentralized', 'finance', 'yield', 'farming', 'liquidity',
        'pool', 'lp', 'amm', 'dex', 'swap', 'uniswap', 'sushiswap', 'pancakeswap',
        'curve', 'balancer', 'aave', 'compound', 'maker', 'makerdao', 'lending',
        'borrowing', 'collateral', 'liquidation', 'impermanent', 'loss', 'slippage',
        'tvl', 'apy', 'apr', 'vault', 'staking', 'stake', 'unstake', 'delegation',
        
        # NFT & Gaming
        'nft', 'nfts', 'non-fungible', 'token', 'erc721', 'erc1155', 'opensea',
        'rarible', 'mintable', 'minting', 'metadata', 'ipfs', 'collectible',
        'gamefi', 'play2earn', 'p2e', 'metaverse', 'sandbox', 'decentraland',
        'axie', 'infinity', 'gaming', 'virtual', 'land', 'avatar',
        
        # Trading Terms
        'trading', 'trader', 'exchange', 'cex', 'centralized', 'spot', 'futures',
        'margin', 'leverage', 'long', 'short', 'hodl', 'hodling', 'whale',
        'pump', 'dump', 'fomo', 'fud', 'dyor', 'bullish', 'bearish', 'bull',
        'bear', 'market', 'candle', 'candlestick', 'chart', 'technical', 'analysis',
        'ta', 'fundamental', 'fa', 'rsi', 'macd', 'bollinger', 'fibonacci',
        'support', 'resistance', 'breakout', 'volume', 'liquidity', 'orderbook',
        
        # Wallet & Security
        'wallet', 'metamask', 'trustwallet', 'ledger', 'trezor', 'hardware',
        'software', 'hot', 'cold', 'storage', 'private', 'key', 'public', 'address',
        'seed', 'phrase', 'mnemonic', 'recovery', 'backup', '2fa', 'security',
        'hack', 'scam', 'phishing', 'rugpull', 'honeypot', 'audit', 'kyc', 'aml',
        
        # Smart Contracts & Development
        'smart', 'contract', 'solidity', 'vyper', 'rust', 'move', 'cairo',
        'evm', 'ethereum', 'virtual', 'machine', 'gas', 'gasprice', 'gaslimit',
        'wei', 'gwei', 'transaction', 'tx', 'txhash', 'receipt', 'event', 'log',
        'abi', 'bytecode', 'opcode', 'deploy', 'deployment', 'dapp', 'web3',
        'ethers', 'web3js', 'web3py', 'truffle', 'hardhat', 'foundry', 'remix',
        
        # Protocols & Standards
        'erc20', 'erc721', 'erc1155', 'bep20', 'trc20', 'spl', 'token', 'standard',
        'protocol', 'dao', 'governance', 'proposal', 'voting', 'quorum', 'treasury',
        'multisig', 'gnosis', 'safe', 'timelock', 'vesting', 'cliff', 'unlock',
        
        # Investment & Finance
        'ico', 'ido', 'ieo', 'launchpad', 'presale', 'tokensale', 'tokenomics',
        'whitepaper', 'roadmap', 'team', 'advisor', 'investor', 'vc', 'venture',
        'capital', 'portfolio', 'diversify', 'risk', 'reward', 'roi', 'profit',
        'loss', 'pnl', 'unrealized', 'realized', 'tax', 'capital', 'gains',
        
        # Emerging Tech
        'ai', 'artificial', 'intelligence', 'machine', 'learning', 'oracle',
        'chainlink', 'band', 'api3', 'bridge', 'cross-chain', 'interoperability',
        'cosmos', 'polkadot', 'wormhole', 'layerzero', 'multichain', 'omnichain'
    }
    
    # Expanded crypto-related terms (secondary relevance)
    CRYPTO_RELATED_TERMS = {
        'invest', 'investment', 'price', 'value', 'worth', 'buy', 'sell', 'trade',
        'platform', 'app', 'application', 'mobile', 'desktop', 'browser', 'extension',
        'community', 'telegram', 'discord', 'twitter', 'reddit', 'forum', 'group',
        'news', 'update', 'announcement', 'launch', 'release', 'upgrade', 'improvement',
        'guide', 'tutorial', 'learn', 'education', 'course', 'academy', 'university',
        'review', 'comparison', 'versus', 'alternative', 'best', 'top', 'list',
        'regulation', 'legal', 'compliance', 'law', 'sec', 'cftc', 'regulatory',
        'bank', 'banking', 'financial', 'institution', 'payment', 'transfer', 'remittance',
        'technology', 'innovation', 'future', 'adoption', 'mainstream', 'institutional',
        'retail', 'user', 'holder', 'owner', 'participant', 'member', 'contributor'
    }
    
    # Terms to explicitly exclude (not crypto-related)
    EXCLUDE_TERMS = {
        'recipe', 'cooking', 'food', 'restaurant', 'fashion', 'clothing', 'weather',
        'sports', 'football', 'basketball', 'baseball', 'soccer', 'tennis', 'golf',
        'movie', 'film', 'actor', 'actress', 'celebrity', 'entertainment', 'music',
        'song', 'album', 'artist', 'band', 'concert', 'tour', 'ticket', 'show',
        'travel', 'vacation', 'hotel', 'flight', 'airline', 'destination', 'tourism',
        'real estate', 'property', 'mortgage', 'rent', 'lease', 'apartment', 'house',
        'health', 'medical', 'doctor', 'hospital', 'medicine', 'treatment', 'disease',
        'education', 'school', 'college', 'university', 'degree', 'student', 'teacher',
        'job', 'career', 'employment', 'resume', 'interview', 'salary', 'benefits'
    }
    
    # Crypto project categories for better clustering
    CRYPTO_CATEGORIES = {
        'Layer 1 Blockchains': ['bitcoin', 'ethereum', 'solana', 'cardano', 'avalanche', 'polkadot', 'cosmos'],
        'Layer 2 Solutions': ['polygon', 'arbitrum', 'optimism', 'zksync', 'starknet', 'lightning'],
        'DeFi Protocols': ['uniswap', 'aave', 'compound', 'curve', 'makerdao', 'yearn', 'sushiswap'],
        'CEX Platforms': ['binance', 'coinbase', 'kraken', 'ftx', 'okx', 'huobi', 'kucoin'],
        'NFT Marketplaces': ['opensea', 'rarible', 'foundation', 'superrare', 'nifty', 'mintable'],
        'Gaming & Metaverse': ['axie', 'sandbox', 'decentraland', 'gala', 'enjin', 'immutable'],
        'Privacy Coins': ['monero', 'zcash', 'dash', 'secret', 'oasis', 'tornado'],
        'Stablecoins': ['usdt', 'usdc', 'dai', 'busd', 'frax', 'ust', 'tusd'],
        'Oracle Networks': ['chainlink', 'band', 'api3', 'dia', 'tellor', 'uma'],
        'Storage Networks': ['filecoin', 'arweave', 'storj', 'sia', 'ipfs', 'swarm']
    }

# ============= CRYPTO KEYWORD FILTER =============
class CryptoKeywordFilter:
    """Advanced filtering system for crypto-related keywords"""
    
    def __init__(self):
        self.domain = CryptoDomainKnowledge()
        self.crypto_terms = self.domain.CRYPTO_CORE_TERMS
        self.related_terms = self.domain.CRYPTO_RELATED_TERMS
        self.exclude_terms = self.domain.EXCLUDE_TERMS
        
        # Create regex patterns for efficient matching
        self.crypto_pattern = self._create_pattern(self.crypto_terms)
        self.related_pattern = self._create_pattern(self.related_terms)
        self.exclude_pattern = self._create_pattern(self.exclude_terms)
        
    def _create_pattern(self, terms: Set[str]) -> re.Pattern:
        """Create regex pattern from terms"""
        # Sort by length (longest first) to match longer terms first
        sorted_terms = sorted(terms, key=len, reverse=True)
        # Escape special regex characters and create pattern
        escaped_terms = [re.escape(term) for term in sorted_terms]
        pattern = r'\b(' + '|'.join(escaped_terms) + r')\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def calculate_crypto_relevance(self, keyword: str) -> float:
        """Calculate how relevant a keyword is to cryptocurrency"""
        keyword_lower = keyword.lower()
        
        # Check for excluded terms first
        if self.exclude_pattern.search(keyword_lower):
            return 0.0
        
        # Count crypto term matches
        crypto_matches = len(self.crypto_pattern.findall(keyword_lower))
        related_matches = len(self.related_pattern.findall(keyword_lower))
        
        # Calculate relevance score
        word_count = len(keyword_lower.split())
        
        # Strong crypto relevance
        if crypto_matches > 0:
            base_score = 0.7 + (0.3 * min(crypto_matches / word_count, 1.0))
            return min(base_score + (0.1 * related_matches / max(word_count, 1)), 1.0)
        
        # Moderate relevance (only related terms)
        if related_matches >= 2:
            return 0.3 + (0.2 * min(related_matches / word_count, 1.0))
        
        # Check for crypto symbols or common patterns
        if self._has_crypto_patterns(keyword_lower):
            return 0.5
        
        return 0.0
    
    def _has_crypto_patterns(self, keyword: str) -> bool:
        """Check for crypto-specific patterns"""
        patterns = [
            r'\b[a-z]{2,5}\/usdt?\b',  # Trading pairs
            r'\b[a-z]{2,5}\/btc\b',
            r'\b[a-z]{2,5}\/eth\b',
            r'\$[a-z]{2,5}\b',  # Ticker symbols
            r'\b0x[a-f0-9]{40}\b',  # Ethereum addresses
            r'\bblock\s*#?\d+\b',  # Block numbers
            r'\btx[a-f0-9]{64}\b',  # Transaction hashes
            r'\b\d+\s*gwei\b',  # Gas prices
            r'\b\d+x\s*(leverage|long|short)\b',  # Trading terms
        ]
        
        for pattern in patterns:
            if re.search(pattern, keyword, re.IGNORECASE):
                return True
        return False
    
    def filter_crypto_keywords(self, keywords: List[str], threshold: float = 0.3) -> Tuple[List[str], List[float]]:
        """Filter keywords to keep only crypto-related ones"""
        filtered_keywords = []
        relevance_scores = []
        
        for keyword in keywords:
            score = self.calculate_crypto_relevance(keyword)
            if score >= threshold:
                filtered_keywords.append(keyword)
                relevance_scores.append(score)
        
        return filtered_keywords, relevance_scores

# ============= IMPROVED TOPIC GENERATOR =============
class CryptoTopicGenerator:
    """Generate meaningful crypto-specific topic titles"""
    
    def __init__(self):
        self.domain = CryptoDomainKnowledge()
        self.category_map = self.domain.CRYPTO_CATEGORIES
        
    def generate_topic_title(self, keywords: List[str], level: str = 'secondary') -> str:
        """Generate clean, meaningful topic title for crypto keywords"""
        if not keywords:
            return f"Empty_{level.title()}_Topic"
        
        # Extract key crypto terms from keywords
        crypto_terms = self._extract_crypto_terms(keywords)
        
        # Identify category if possible
        category = self._identify_category(crypto_terms)
        
        # Generate title based on level and content
        if level == 'pillar':
            return self._generate_pillar_title(crypto_terms, category)
        elif level == 'primary':
            return self._generate_primary_title(crypto_terms, category)
        elif level == 'secondary':
            return self._generate_secondary_title(crypto_terms, category)
        else:  # subtopic
            return self._generate_subtopic_title(crypto_terms, keywords)
    
    def _extract_crypto_terms(self, keywords: List[str]) -> Counter:
        """Extract and count crypto-specific terms"""
        all_terms = []
        
        for keyword in keywords[:100]:  # Sample for efficiency
            # Extract meaningful crypto terms
            words = re.findall(r'\b[a-zA-Z]{2,}\b', keyword.lower())
            for word in words:
                if word in self.domain.CRYPTO_CORE_TERMS:
                    all_terms.append(word)
                elif len(word) <= 5 and word.isupper():  # Potential ticker
                    all_terms.append(word.lower())
        
        return Counter(all_terms)
    
    def _identify_category(self, crypto_terms: Counter) -> Optional[str]:
        """Identify the primary category based on terms"""
        category_scores = {}
        
        for category, terms in self.category_map.items():
            score = sum(crypto_terms.get(term, 0) for term in terms)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return None
    
    def _generate_pillar_title(self, crypto_terms: Counter, category: Optional[str]) -> str:
        """Generate broad pillar topic title"""
        if category:
            # Use identified category as base
            base = category.replace('_', ' ')
        elif crypto_terms:
            # Use most common terms
            top_terms = [term.title() for term, _ in crypto_terms.most_common(2)]
            base = ' '.join(top_terms)
        else:
            base = "General Crypto"
        
        return self._clean_title(base)
    
    def _generate_primary_title(self, crypto_terms: Counter, category: Optional[str]) -> str:
        """Generate primary topic title"""
        if crypto_terms:
            top_terms = [term.title() for term, _ in crypto_terms.most_common(3)]
            if category and category not in ' '.join(top_terms):
                return self._clean_title(f"{' '.join(top_terms[:2])} {category.split()[0]}")
            return self._clean_title(' '.join(top_terms))
        elif category:
            return self._clean_title(f"{category} Topics")
        else:
            return "Crypto Topics"
    
    def _generate_secondary_title(self, crypto_terms: Counter, category: Optional[str]) -> str:
        """Generate secondary topic title"""
        if crypto_terms:
            top_terms = [term.title() for term, _ in crypto_terms.most_common(4)]
            return self._clean_title(' '.join(top_terms[:3]))
        else:
            return self._generate_fallback_title([], 'secondary')
    
    def _generate_subtopic_title(self, crypto_terms: Counter, keywords: List[str]) -> str:
        """Generate specific subtopic title"""
        if crypto_terms:
            # Use specific combination of terms
            top_terms = [term.title() for term, _ in crypto_terms.most_common(4)]
            return self._clean_title(' '.join(top_terms))
        else:
            # Fallback to keyword-based generation
            return self._generate_fallback_title(keywords[:3], 'subtopic')
    
    def _generate_fallback_title(self, keywords: List[str], level: str) -> str:
        """Generate fallback title when no clear crypto terms found"""
        if not keywords:
            return f"Crypto_{level.title()}"
        
        # Extract key words from first few keywords
        words = []
        for keyword in keywords[:3]:
            keyword_words = re.findall(r'\b[a-zA-Z]{3,}\b', keyword.lower())
            words.extend([w.title() for w in keyword_words[:2] if w not in {'the', 'and', 'for', 'with'}])
        
        if words:
            return self._clean_title(' '.join(words[:3]))
        else:
            return f"Crypto_{level.title()}"
    
    def _clean_title(self, title: str) -> str:
        """Clean and format title"""
        # Remove duplicates while preserving order
        words = title.split()
        seen = set()
        unique_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower not in seen:
                seen.add(word_lower)
                unique_words.append(word)
        
        # Join and clean
        clean_title = ' '.join(unique_words[:5])  # Limit length
        clean_title = re.sub(r'\s+', ' ', clean_title).strip()
        
        # Ensure title is not empty
        if not clean_title:
            return "Crypto Topic"
        
        return clean_title

# ============= ENHANCED DATA PROCESSOR =============
class CryptoDataProcessor:
    """Process and filter crypto keyword data"""
    
    def __init__(self):
        self.cache_dir = Path(CryptoConfig.CACHE_DIR)
        self.cache_dir.mkdir(exist_ok=True)
        self.filter = CryptoKeywordFilter()
        
    def load_and_process_keywords(self, file_path: str) -> pd.DataFrame:
        """Load, process, and filter crypto keywords"""
        logger.info(f"Loading keywords from: {file_path}")
        
        # Try cache first
        cache_file = self.cache_dir / f"crypto_processed_{Path(file_path).stem}.pkl"
        if CryptoConfig.ENABLE_CACHING and cache_file.exists():
            try:
                logger.info("Loading cached crypto data...")
                return pd.read_pickle(cache_file)
            except Exception as e:
                logger.warning(f"Cache loading failed: {e}")
        
        try:
            # Load raw data
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_csv(file_path, low_memory=False)
            
            logger.info(f"Loaded {len(df)} rows")
            
            # Find keyword column
            keyword_col = self._find_keyword_column(df)
            
            # Process keywords
            processed_df = self._process_crypto_keywords(df, keyword_col)
            
            # Cache if enabled
            if CryptoConfig.ENABLE_CACHING:
                processed_df.to_pickle(cache_file)
                logger.info(f"Cached to {cache_file}")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing keywords: {e}")
            raise
    
    def _find_keyword_column(self, df: pd.DataFrame) -> str:
        """Find keyword column"""
        possible_names = [
            'keyword', 'keywords', 'query', 'term', 'search_term',
            'search', 'phrase', 'keyphrase', 'search_query'
        ]
        
        for col in df.columns:
            if col.lower() in possible_names:
                return col
        
        for col in df.columns:
            if any(name in col.lower() for name in possible_names):
                return col
        
        # Return first text column
        for col in df.columns:
            if df[col].dtype == 'object':
                return col
        
        return df.columns[0]
    
    def _process_crypto_keywords(self, df: pd.DataFrame, keyword_col: str) -> pd.DataFrame:
        """Process and filter keywords for crypto relevance"""
        logger.info("Processing crypto keywords...")
        
        # Create working dataframe
        processed_df = pd.DataFrame({
            'original_keyword': df[keyword_col].astype(str),
            'search_volume': df.get('search_volume', 0),
            'competition': df.get('competition', 0),
            'cpc': df.get('cpc', 0.0)
        })
        
        # Clean keywords
        processed_df['cleaned_keyword'] = processed_df['original_keyword'].str.lower().str.strip()
        processed_df['cleaned_keyword'] = processed_df['cleaned_keyword'].str.replace(r'[^\w\s-]', ' ', regex=True)
        processed_df['cleaned_keyword'] = processed_df['cleaned_keyword'].str.replace(r'\s+', ' ', regex=True)
        
        # Filter by length
        processed_df = processed_df[
            (processed_df['cleaned_keyword'].str.len() >= CryptoConfig.MIN_KEYWORD_LENGTH) & 
            (processed_df['cleaned_keyword'].str.len() <= CryptoConfig.MAX_KEYWORD_LENGTH)
        ]
        
        initial_count = len(processed_df)
        
        # Calculate crypto relevance scores
        logger.info("Filtering for crypto relevance...")
        keywords = processed_df['cleaned_keyword'].tolist()
        filtered_keywords, relevance_scores = self.filter.filter_crypto_keywords(
            keywords, 
            threshold=CryptoConfig.CRYPTO_RELEVANCE_THRESHOLD
        )
        
        # Keep only crypto-relevant keywords
        crypto_mask = processed_df['cleaned_keyword'].isin(filtered_keywords)
        processed_df = processed_df[crypto_mask].copy()
        
        # Add relevance scores
        score_dict = dict(zip(filtered_keywords, relevance_scores))
        processed_df['crypto_relevance'] = processed_df['cleaned_keyword'].map(score_dict)
        
        # Remove duplicates
        processed_df = processed_df.drop_duplicates(subset=['cleaned_keyword'], keep='first')
        
        # Add features
        processed_df['word_count'] = processed_df['cleaned_keyword'].str.split().str.len()
        processed_df['char_count'] = processed_df['cleaned_keyword'].str.len()
        processed_df['has_trading_terms'] = processed_df['cleaned_keyword'].str.contains(
            '|'.join(['buy', 'sell', 'trade', 'trading', 'price', 'chart', 'analysis']), na=False
        )
        processed_df['has_defi_terms'] = processed_df['cleaned_keyword'].str.contains(
            '|'.join(['defi', 'yield', 'farm', 'stake', 'liquidity', 'pool', 'swap']), na=False
        )
        processed_df['has_nft_terms'] = processed_df['cleaned_keyword'].str.contains(
            '|'.join(['nft', 'collectible', 'opensea', 'mint', 'metadata']), na=False
        )
        
        processed_df = processed_df.reset_index(drop=True)
        
        filtered_count = initial_count - len(processed_df)
        logger.info(f"✓ Filtered {filtered_count} non-crypto keywords")
        logger.info(f"✓ Retained {len(processed_df)} crypto-relevant keywords")
        
        return processed_df

# ============= CRYPTO CLUSTERING ENGINE =============
class CryptoClusteringEngine:
    """Clustering engine optimized for cryptocurrency keywords"""
    
    def __init__(self):
        self.embedding_model = None
        self.embeddings = None
        self.topic_generator = CryptoTopicGenerator()
        self.hierarchy_data = {}
        
    def create_hierarchical_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create crypto-optimized hierarchical clusters"""
        logger.info("Starting crypto clustering...")
        
        keywords = df['cleaned_keyword'].tolist()
        
        # Create embeddings with crypto context
        self.embeddings = self._create_crypto_embeddings(keywords, df['crypto_relevance'].tolist())
        
        # Reduce dimensions
        reduced_embeddings = self._reduce_dimensions(self.embeddings)
        
        # Create hierarchical clusters
        result_df = df.copy()
        
        # Level 1: Pillar topics (main crypto categories)
        logger.info("Creating crypto pillar topics...")
        pillar_labels, pillar_names = self._create_crypto_clusters(
            reduced_embeddings, keywords, CryptoConfig.PILLAR_CLUSTERS, 'pillar'
        )
        result_df['pillar_id'] = pillar_labels
        result_df['pillar_name'] = pillar_names
        
        # Level 2: Primary topics
        logger.info("Creating primary crypto topics...")
        primary_labels, primary_names = self._create_hierarchical_clusters(
            reduced_embeddings, keywords, pillar_labels, CryptoConfig.PRIMARY_CLUSTERS, 'primary'
        )
        result_df['primary_id'] = primary_labels
        result_df['primary_name'] = primary_names
        
        # Level 3: Secondary topics
        logger.info("Creating secondary crypto topics...")
        secondary_labels, secondary_names = self._create_hierarchical_clusters(
            reduced_embeddings, keywords, primary_labels, CryptoConfig.SECONDARY_CLUSTERS, 'secondary'
        )
        result_df['secondary_id'] = secondary_labels
        result_df['secondary_name'] = secondary_names
        
        # Level 4: Subtopics
        logger.info("Creating crypto subtopics...")
        subtopic_labels, subtopic_names = self._create_hierarchical_clusters(
            reduced_embeddings, keywords, secondary_labels, CryptoConfig.SUBTOPIC_CLUSTERS, 'subtopic'
        )
        result_df['subtopic_id'] = subtopic_labels
        result_df['subtopic_name'] = subtopic_names
        
        # Build hierarchy
        self._build_hierarchy_tree(result_df)
        
        logger.info("✓ Crypto clustering completed")
        return result_df
    
    def _create_crypto_embeddings(self, keywords: List[str], relevance_scores: List[float]) -> np.ndarray:
        """Create embeddings with crypto context weighting"""
        logger.info(f"Creating crypto-weighted embeddings for {len(keywords)} keywords...")
        
        # Initialize model
        self.embedding_model = SentenceTransformer(CryptoConfig.EMBEDDING_MODEL)
        self.embedding_model.max_seq_length = CryptoConfig.MAX_SEQUENCE_LENGTH
        
        # Create embeddings
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
            
            # Weight by crypto relevance
            batch_scores = relevance_scores[i:i+batch_size]
            for j, score in enumerate(batch_scores):
                batch_embeddings[j] = batch_embeddings[j] * (0.7 + 0.3 * score)
            
            embeddings.append(batch_embeddings)
            
            if i % (batch_size * 10) == 0:
                logger.info(f"Progress: {(i / len(keywords)) * 100:.1f}%")
        
        embeddings = np.vstack(embeddings)
        logger.info(f"✓ Created embeddings shape: {embeddings.shape}")
        
        return embeddings
    
    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensions using UMAP"""
        logger.info("Reducing dimensions...")
        
        # PCA first if needed
        if embeddings.shape[1] > 100:
            pca = PCA(n_components=100, random_state=42)
            embeddings = pca.fit_transform(embeddings)
        
        # UMAP reduction
        umap_model = umap.UMAP(
            n_neighbors=CryptoConfig.UMAP_N_NEIGHBORS,
            n_components=CryptoConfig.UMAP_N_COMPONENTS,
            min_dist=CryptoConfig.UMAP_MIN_DIST,
            metric=CryptoConfig.UMAP_METRIC,
            random_state=42,
            low_memory=True
        )
        
        reduced = umap_model.fit_transform(embeddings)
        logger.info(f"✓ Reduced to {reduced.shape[1]} dimensions")
        
        return reduced
    
    def _create_crypto_clusters(self, embeddings: np.ndarray, keywords: List[str],
                               n_clusters: int, level: str) -> Tuple[List[int], List[str]]:
        """Create crypto-specific clusters"""
        logger.info(f"Creating {n_clusters} {level} clusters...")
        
        # Use K-means for consistent sizing
        kmeans = KMeans(
            n_clusters=min(n_clusters, len(keywords)),
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        labels = kmeans.fit_predict(embeddings)
        
        # Generate crypto-specific names
        names = self._generate_cluster_names(keywords, labels, level)
        
        logger.info(f"✓ Created {len(set(labels))} {level} clusters")
        return labels.tolist(), names
    
    def _create_hierarchical_clusters(self, embeddings: np.ndarray, keywords: List[str],
                                     parent_labels: List[int], n_clusters: int, 
                                     level: str) -> Tuple[List[int], List[str]]:
        """Create hierarchical sub-clusters"""
        logger.info(f"Creating {level} hierarchical clusters...")
        
        final_labels = [-1] * len(keywords)
        final_names = ["Uncategorized"] * len(keywords)
        global_id = 0
        
        # Group by parent
        parent_groups = defaultdict(list)
        for idx, parent_id in enumerate(parent_labels):
            parent_groups[parent_id].append(idx)
        
        for parent_id, indices in parent_groups.items():
            if len(indices) < 3:
                for idx in indices:
                    final_labels[idx] = global_id
                    final_names[idx] = f"Small_{level.title()}_Group"
                global_id += 1
                continue
            
            # Sub-cluster size
            parent_size = len(indices)
            target_clusters = max(2, min(n_clusters // 15, parent_size // 5))
            
            # Extract subset
            parent_embeddings = embeddings[indices]
            parent_keywords = [keywords[i] for i in indices]
            
            # Cluster
            if target_clusters >= parent_size:
                sub_labels = list(range(parent_size))
            else:
                kmeans = KMeans(n_clusters=target_clusters, random_state=42, n_init=5)
                sub_labels = kmeans.fit_predict(parent_embeddings)
            
            # Generate names
            sub_names = self._generate_cluster_names(parent_keywords, sub_labels, level)
            
            # Assign
            for i, sub_label in enumerate(sub_labels):
                orig_idx = indices[i]
                final_labels[orig_idx] = global_id + sub_label
                final_names[orig_idx] = sub_names[i]
            
            global_id += max(sub_labels) + 1
        
        logger.info(f"✓ Created {len(set(final_labels))} {level} clusters")
        return final_labels, final_names
    
    def _generate_cluster_names(self, keywords: List[str], labels: List[int], level: str) -> List[str]:
        """Generate clean cluster names"""
        cluster_groups = defaultdict(list)
        for keyword, label in zip(keywords, labels):
            cluster_groups[label].append(keyword)
        
        cluster_names = {}
        for cluster_id, cluster_keywords in cluster_groups.items():
            cluster_names[cluster_id] = self.topic_generator.generate_topic_title(
                cluster_keywords, level
            )
        
        return [cluster_names[label] for label in labels]
    
    def _build_hierarchy_tree(self, df: pd.DataFrame):
        """Build hierarchy tree structure"""
        logger.info("Building hierarchy tree...")
        
        hierarchy_data = []
        
        # Process each level
        for level_name, level_prefix, parent_level in [
            ('Pillar', 'P', None),
            ('Primary', 'PR', 'pillar'),
            ('Secondary', 'S', 'primary'),
            ('Subtopic', 'ST', 'secondary')
        ]:
            level_col_id = f"{level_name.lower()}_id"
            level_col_name = f"{level_name.lower()}_name"
            
            if parent_level:
                parent_col_id = f"{parent_level}_id"
                group_cols = [parent_col_id, level_col_id, level_col_name]
            else:
                group_cols = [level_col_id, level_col_name]
            
            level_stats = df.groupby(group_cols).agg({
                'cleaned_keyword': 'count',
                'search_volume': 'sum',
                'competition': 'mean',
                'cpc': 'mean',
                'crypto_relevance': 'mean'
            }).reset_index()
            
            for _, row in level_stats.iterrows():
                node = {
                    'level': level_name,
                    'id': f"{level_prefix}_{row[level_col_id]}",
                    'name': row[level_col_name],
                    'parent_id': f"{parent_level[0].upper()}_{row[parent_col_id]}" if parent_level else None,
                    'keyword_count': row['cleaned_keyword'],
                    'total_search_volume': row['search_volume'],
                    'avg_competition': round(row['competition'], 3),
                    'avg_cpc': round(row['cpc'], 2),
                    'avg_crypto_relevance': round(row['crypto_relevance'], 3)
                }
                hierarchy_data.append(node)
        
        self.hierarchy_data = pd.DataFrame(hierarchy_data)
        logger.info(f"✓ Built hierarchy with {len(hierarchy_data)} nodes")

# ============= LOGGING SETUP =============
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('crypto_clustering.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============= EXCEL OUTPUT GENERATOR =============
class CryptoExcelOutputGenerator:
    """Generate crypto-focused Excel output"""
    
    def create_crypto_excel(self, df: pd.DataFrame, hierarchy_data: pd.DataFrame, output_path: str):
        """Create comprehensive crypto analysis Excel file"""
        logger.info(f"Creating Excel output: {output_path}")
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main sheets
                self._create_main_sheet(df, writer)
                self._create_pillar_analysis(df, writer)
                self._create_defi_analysis(df, writer)
                self._create_trading_analysis(df, writer)
                self._create_nft_analysis(df, writer)
                hierarchy_data.to_excel(writer, sheet_name='Hierarchy_Tree', index=False)
                
                self._auto_adjust_columns(writer)
            
            logger.info("✓ Excel file created successfully")
            self._print_summary(df, hierarchy_data)
            
        except Exception as e:
            logger.error(f"Error creating Excel: {e}")
            raise
    
    def _create_main_sheet(self, df: pd.DataFrame, writer):
        """Create main keywords sheet"""
        main_df = df[[
            'original_keyword', 'cleaned_keyword', 'crypto_relevance',
            'pillar_name', 'primary_name', 'secondary_name', 'subtopic_name',
            'search_volume', 'competition', 'cpc',
            'has_trading_terms', 'has_defi_terms', 'has_nft_terms'
        ]].copy()
        
        main_df['full_path'] = (
            main_df['pillar_name'] + ' > ' +
            main_df['primary_name'] + ' > ' +
            main_df['secondary_name'] + ' > ' +
            main_df['subtopic_name']
        )
        
        main_df.to_excel(writer, sheet_name='Crypto_Keywords', index=False)
    
    def _create_pillar_analysis(self, df: pd.DataFrame, writer):
        """Create pillar analysis sheet"""
        pillar_df = df.groupby(['pillar_id', 'pillar_name']).agg({
            'cleaned_keyword': 'count',
            'search_volume': ['sum', 'mean'],
            'competition': 'mean',
            'cpc': 'mean',
            'crypto_relevance': 'mean',
            'has_trading_terms': 'sum',
            'has_defi_terms': 'sum',
            'has_nft_terms': 'sum'
        }).round(2)
        
        pillar_df.columns = [
            'Total_Keywords', 'Total_Volume', 'Avg_Volume',
            'Avg_Competition', 'Avg_CPC', 'Avg_Relevance',
            'Trading_Keywords', 'DeFi_Keywords', 'NFT_Keywords'
        ]
        
        pillar_df = pillar_df.reset_index()
        pillar_df = pillar_df.sort_values('Total_Volume', ascending=False)
        pillar_df.to_excel(writer, sheet_name='Pillar_Analysis', index=False)
    
    def _create_defi_analysis(self, df: pd.DataFrame, writer):
        """Create DeFi-specific analysis"""
        defi_df = df[df['has_defi_terms']].copy()
        if len(defi_df) > 0:
            defi_analysis = defi_df.groupby(['primary_name', 'secondary_name']).agg({
                'cleaned_keyword': 'count',
                'search_volume': 'sum',
                'competition': 'mean'
            }).round(2)
            defi_analysis = defi_analysis.reset_index()
            defi_analysis = defi_analysis.sort_values('search_volume', ascending=False).head(50)
            defi_analysis.to_excel(writer, sheet_name='DeFi_Topics', index=False)
    
    def _create_trading_analysis(self, df: pd.DataFrame, writer):
        """Create trading-specific analysis"""
        trading_df = df[df['has_trading_terms']].copy()
        if len(trading_df) > 0:
            trading_analysis = trading_df.groupby(['primary_name', 'secondary_name']).agg({
                'cleaned_keyword': 'count',
                'search_volume': 'sum',
                'competition': 'mean'
            }).round(2)
            trading_analysis = trading_analysis.reset_index()
            trading_analysis = trading_analysis.sort_values('search_volume', ascending=False).head(50)
            trading_analysis.to_excel(writer, sheet_name='Trading_Topics', index=False)
    
    def _create_nft_analysis(self, df: pd.DataFrame, writer):
        """Create NFT-specific analysis"""
        nft_df = df[df['has_nft_terms']].copy()
        if len(nft_df) > 0:
            nft_analysis = nft_df.groupby(['primary_name', 'secondary_name']).agg({
                'cleaned_keyword': 'count',
                'search_volume': 'sum',
                'competition': 'mean'
            }).round(2)
            nft_analysis = nft_analysis.reset_index()
            nft_analysis = nft_analysis.sort_values('search_volume', ascending=False).head(50)
            nft_analysis.to_excel(writer, sheet_name='NFT_Topics', index=False)
    
    def _auto_adjust_columns(self, writer):
        """Auto-adjust column widths"""
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    def _print_summary(self, df: pd.DataFrame, hierarchy_data: pd.DataFrame):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("🚀 CRYPTO SEO CLUSTERING COMPLETE")
        print("="*80)
        print(f"\n📊 Results:")
        print(f"   • Total Crypto Keywords: {len(df):,}")
        print(f"   • Average Crypto Relevance: {df['crypto_relevance'].mean():.2%}")
        print(f"   • Pillar Topics: {df['pillar_name'].nunique()}")
        print(f"   • Primary Topics: {df['primary_name'].nunique()}")
        print(f"   • Secondary Topics: {df['secondary_name'].nunique()}")
        print(f"   • Subtopics: {df['subtopic_name'].nunique()}")
        
        if df['search_volume'].sum() > 0:
            print(f"\n📈 Metrics:")
            print(f"   • Total Search Volume: {df['search_volume'].sum():,}")
            print(f"   • DeFi Keywords: {df['has_defi_terms'].sum():,}")
            print(f"   • Trading Keywords: {df['has_trading_terms'].sum():,}")
            print(f"   • NFT Keywords: {df['has_nft_terms'].sum():,}")
        
        print(f"\n✅ Output saved to: {CryptoConfig.OUTPUT_FILE}")
        print("="*80)

# ============= MAIN PIPELINE =============
class CryptoSEOPipeline:
    """Main pipeline for crypto SEO clustering"""
    
    def __init__(self):
        self.processor = CryptoDataProcessor()
        self.clustering_engine = CryptoClusteringEngine()
        self.excel_generator = CryptoExcelOutputGenerator()
    
    def run_pipeline(self):
        """Execute the complete pipeline"""
        start_time = datetime.now()
        
        print("="*80)
        print("🚀 CRYPTO SEO KEYWORD CLUSTERING SYSTEM")
        print("="*80)
        
        try:
            # Step 1: Load and filter
            logger.info("[STEP 1/3] Loading and filtering crypto keywords...")
            df = self.processor.load_and_process_keywords(CryptoConfig.INPUT_FILE)
            
            if len(df) == 0:
                logger.error("No crypto-relevant keywords found!")
                return None
            
            # Step 2: Cluster
            logger.info("[STEP 2/3] Creating crypto clusters...")
            clustered_df = self.clustering_engine.create_hierarchical_clusters(df)
            
            # Step 3: Output
            logger.info("[STEP 3/3] Generating Excel output...")
            self.excel_generator.create_crypto_excel(
                clustered_df,
                self.clustering_engine.hierarchy_data,
                CryptoConfig.OUTPUT_FILE
            )
            
            elapsed = datetime.now() - start_time
            logger.info(f"\n✅ Pipeline completed in {elapsed}")
            
            # Cleanup
            gc.collect()
            
            return clustered_df
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

if __name__ == "__main__":
    # Create output directory
    os.makedirs(os.path.dirname(CryptoConfig.OUTPUT_FILE), exist_ok=True)
    
    # Run pipeline
    pipeline = CryptoSEOPipeline()
    pipeline.run_pipeline()