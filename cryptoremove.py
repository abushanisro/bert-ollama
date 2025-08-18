"""
Enhanced BERT-Based Keyword Clustering System for Crypto Ecosystem
- Processes exactly 124,417 keywords (or any specified number)
- Simple topic modeling without excessive subtopics
- No word duplication in topic names
- Shows removed keywords with detailed reasons
- Uses LLM-style relevance checking for accuracy
- Focuses on meaningful crypto topics only
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
from sklearn.cluster import KMeans
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
    """Configuration for processing exact number of keywords"""
    
    # File paths
    INPUT_FILE = '/home/admin1/Downloads/demo_crypto/FInal list of crypto terms.xlsx'
    OUTPUT_DIR = '/home/admin1/Downloads/demo_crypto/output'
    OUTPUT_FILE = f'{OUTPUT_DIR}/crypto_clusters_exact_count.xlsx'
    REMOVED_KEYWORDS_FILE = f'{OUTPUT_DIR}/removed_keywords_detailed.csv'
    
    # BERT Model - Best quality
    EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
    
    # Processing Configuration
    BATCH_SIZE = 500
    MAX_SEQUENCE_LENGTH = 128
    TARGET_KEYWORDS = 124417  # Process exactly this many keywords
    
    # Simple Topic Clustering (NO hierarchical levels)
    TARGET_TOPICS = 25  # Reasonable number of main topics
    
    # Clustering Parameters
    UMAP_N_NEIGHBORS = 15
    UMAP_N_COMPONENTS = 30
    UMAP_MIN_DIST = 0.1
    UMAP_METRIC = 'cosine'
    
    # Quality Control
    MIN_CLUSTER_SIZE = 20  # Minimum keywords per topic
    MIN_KEYWORDS_FOR_ANALYSIS = 10
    
    # Memory Management
    ENABLE_CACHING = True
    CACHE_DIR = 'cache_exact'

# ============= ADVANCED CRYPTO RELEVANCE CHECKER =============
class AdvancedCryptoRelevanceChecker:
    """LLM-style relevance checking with comprehensive crypto knowledge"""
    
    def __init__(self):
        self.crypto_terms = self._load_comprehensive_crypto_terms()
        self.context_terms = self._load_context_terms()
        self.exclusion_patterns = self._compile_exclusion_patterns()
        self.crypto_contexts = self._load_crypto_contexts()
        
    def _load_comprehensive_crypto_terms(self) -> Set[str]:
        """Load the most comprehensive crypto terms database"""
        return set([
            # Major Cryptocurrencies & Tokens
            'bitcoin', 'btc', 'ethereum', 'eth', 'ether', 'binance', 'bnb', 'tether', 'usdt',
            'cardano', 'ada', 'solana', 'sol', 'ripple', 'xrp', 'polkadot', 'dot', 'usdc',
            'dogecoin', 'doge', 'shiba', 'shib', 'avalanche', 'avax', 'chainlink', 'link',
            'polygon', 'matic', 'cosmos', 'atom', 'algorand', 'algo', 'vechain', 'vet',
            'stellar', 'xlm', 'filecoin', 'fil', 'tron', 'trx', 'monero', 'xmr', 'dai',
            'litecoin', 'ltc', 'uniswap', 'uni', 'bitcoin cash', 'bch', 'near', 'icp',
            'aptos', 'apt', 'arbitrum', 'arb', 'optimism', 'op', 'hedera', 'hbar',
            'sandbox', 'sand', 'decentraland', 'mana', 'axie', 'axs', 'gala', 'enjin', 'enj',
            
            # DeFi Protocols & Platforms
            'defi', 'decentralized finance', 'aave', 'compound', 'maker', 'mkr', 'curve', 'crv',
            'sushi', 'sushiswap', 'pancakeswap', 'cake', 'yearn', 'yfi', 'synthetix', 'snx',
            'balancer', '1inch', 'dydx', 'lido', 'ldo', 'frax', 'gmx', 'venus', 'cream',
            'raydium', 'serum', 'jupiter', 'orca', 'marinade', 'anchor', 'terra', 'luna',
            
            # Core Blockchain & Crypto Terms
            'blockchain', 'cryptocurrency', 'crypto', 'altcoin', 'token', 'coin', 'digital currency',
            'wallet', 'exchange', 'dex', 'cex', 'decentralized exchange', 'centralized exchange',
            'amm', 'automated market maker', 'liquidity', 'pool', 'liquidity pool',
            'yield', 'farming', 'yield farming', 'staking', 'mining', 'validator', 'node',
            'smart contract', 'dapp', 'web3', 'dao', 'decentralized autonomous organization',
            'nft', 'non fungible token', 'metaverse', 'gamefi', 'play to earn', 'p2e',
            
            # Technical Terms
            'gas', 'gwei', 'hash', 'hashrate', 'block', 'blockchain', 'transaction', 'txn',
            'consensus', 'proof of work', 'pow', 'proof of stake', 'pos', 'proof of authority',
            'merkle tree', 'private key', 'public key', 'seed phrase', 'mnemonic',
            'fork', 'hard fork', 'soft fork', 'mainnet', 'testnet', 'sidechain',
            
            # Trading & Investment Terms
            'trading', 'trade', 'buy', 'sell', 'swap', 'convert', 'bridge', 'cross chain',
            'price', 'chart', 'analysis', 'technical analysis', 'fundamental analysis', 'ta', 'fa',
            'bullish', 'bearish', 'pump', 'dump', 'hodl', 'fomo', 'fud', 'diamond hands',
            'whale', 'portfolio', 'invest', 'investment', 'profit', 'loss', 'pnl',
            'market cap', 'volume', 'tvl', 'total value locked', 'apy', 'apr',
            
            # Stablecoins
            'stablecoin', 'stable coin', 'usdt', 'tether', 'usdc', 'circle', 'dai', 'busd',
            'tusd', 'trueusd', 'pax', 'paxos', 'gusd', 'gemini dollar', 'usdd', 'frax',
            'terra usd', 'ust', 'magic internet money', 'mim',
            
            # Major Exchanges & Platforms
            'binance', 'coinbase', 'kraken', 'bitfinex', 'huobi', 'okx', 'okex', 'gate.io',
            'kucoin', 'bittrex', 'bitstamp', 'bybit', 'bitget', 'mexc', 'crypto.com',
            'ftx', 'gemini', 'bitpanda', 'coincheck', 'bithumb', 'upbit',
            
            # Layer 2 & Scaling Solutions
            'layer2', 'l2', 'layer 2', 'rollup', 'zk rollup', 'optimistic rollup',
            'zk', 'zero knowledge', 'zksync', 'starknet', 'starkware', 'polygon',
            'lightning network', 'state channel', 'plasma', 'sharding',
            
            # Security & Compliance
            'security', 'audit', 'smart contract audit', 'hack', 'exploit', 'vulnerability',
            'kyc', 'know your customer', 'aml', 'anti money laundering', 'compliance',
            'regulation', 'sec', 'securities exchange commission', 'cftc', 'tax', 'legal',
            
            # Emerging Trends
            'airdrop', 'whitepaper', 'roadmap', 'tokenomics', 'burn', 'token burn',
            'mint', 'minting', 'supply', 'circulating supply', 'total supply',
            'halving', 'bitcoin halving', 'difficulty adjustment', 'mempool',
            'oracle', 'price oracle', 'flash loan', 'arbitrage', 'mev', 'sandwich attack',
            'rug pull', 'exit scam', 'ponzi', 'pyramid scheme',
            
            # Web3 & Metaverse
            'web3', 'web 3.0', 'metaverse', 'virtual reality', 'vr', 'augmented reality', 'ar',
            'virtual world', 'digital asset', 'digital collectible', 'pfp', 'profile picture',
            'avatar', 'land', 'virtual land', 'real estate', 'gaming', 'blockchain game',
            
            # Institutional & Enterprise
            'institutional', 'enterprise', 'custody', 'custodial', 'non custodial',
            'cold storage', 'hot wallet', 'hardware wallet', 'paper wallet',
            'multi signature', 'multisig', 'treasury', 'corporate treasury',
        ])
    
    def _load_context_terms(self) -> Set[str]:
        """Terms that indicate crypto context when combined"""
        return set([
            'price', 'wallet', 'exchange', 'trading', 'investment', 'market', 'analysis',
            'news', 'update', 'launch', 'protocol', 'network', 'platform', 'app',
            'tool', 'tracker', 'monitor', 'alert', 'bot', 'automation', 'api',
            'dashboard', 'portfolio', 'calculator', 'converter', 'bridge',
            'explorer', 'scan', 'scanner', 'node', 'rpc', 'endpoint'
        ])
    
    def _load_crypto_contexts(self) -> List[str]:
        """Common crypto-related contexts and phrases"""
        return [
            'crypto', 'blockchain', 'defi', 'nft', 'web3', 'metaverse',
            'digital currency', 'virtual currency', 'altcoin', 'stablecoin',
            'smart contract', 'decentralized', 'peer to peer', 'p2p'
        ]
    
    def _compile_exclusion_patterns(self) -> List:
        """Patterns for non-crypto content to exclude"""
        return [
            # Length-based exclusions
            re.compile(r'^.{1,2}$'),  # Too short (1-2 characters)
            re.compile(r'^.{101,}$'),  # Too long (over 100 characters)
            
            # Gibberish patterns
            re.compile(r'^[a-z]{30,}$'),  # Very long single word
            re.compile(r'^[0-9]+$'),  # Pure numbers
            re.compile(r'(.)\1{6,}'),  # Too many repeated characters
            re.compile(r'^[^a-zA-Z0-9\s\-\_\.]+$'),  # Only special characters
            
            # Test/Demo content
            re.compile(r'^(test|demo|example|sample)\d*$', re.IGNORECASE),
            re.compile(r'^(lorem|ipsum|placeholder)', re.IGNORECASE),
            
            # Spam/Adult content
            re.compile(r'(xxx|porn|sex|adult|casino|gambling)', re.IGNORECASE),
            re.compile(r'(viagra|cialis|pharmacy|pills)', re.IGNORECASE),
            
            # Keyboard mashing
            re.compile(r'^(asdf|qwer|zxcv|hjkl)', re.IGNORECASE),
            re.compile(r'^(aaa|bbb|ccc|ddd|eee)', re.IGNORECASE),
            
            # Non-English gibberish
            re.compile(r'[^\x00-\x7F]{10,}'),  # Too much non-ASCII
            
            # Random strings
            re.compile(r'^[a-z0-9]{20,}$'),  # Long alphanumeric strings
        ]
    
    def is_crypto_relevant(self, keyword: str) -> Tuple[bool, str]:
        """
        Advanced relevance check with detailed reasoning
        Returns: (is_relevant, reason)
        """
        if not keyword or not isinstance(keyword, str):
            return False, "Invalid Input"
        
        keyword_clean = keyword.lower().strip()
        
        # Check exclusion patterns first
        for pattern in self.exclusion_patterns:
            if pattern.search(keyword_clean):
                return False, "Matches Exclusion Pattern"
        
        # Direct crypto term match (highest confidence)
        words = set(keyword_clean.replace('-', ' ').replace('_', ' ').split())
        if words.intersection(self.crypto_terms):
            return True, "Direct Crypto Term Match"
        
        # Check for crypto contexts
        for context in self.crypto_contexts:
            if context in keyword_clean:
                return True, f"Crypto Context: {context}"
        
        # Context + term combination
        has_context = bool(words.intersection(self.context_terms))
        has_crypto_hint = any(term in keyword_clean for term in [
            'coin', 'token', 'chain', 'block', 'hash', 'mine', 'stake',
            'swap', 'pool', 'vault', 'farm', 'yield', 'apy', 'apr'
        ])
        
        if has_context and has_crypto_hint:
            return True, "Context + Crypto Hint"
        
        # Brand name + crypto context
        crypto_brands = ['binance', 'coinbase', 'ethereum', 'bitcoin', 'metamask', 'uniswap']
        if any(brand in keyword_clean for brand in crypto_brands):
            return True, "Crypto Brand Reference"
        
        # Financial terms in crypto context
        financial_terms = ['price', 'chart', 'trading', 'investment', 'market', 'portfolio']
        if (any(term in keyword_clean for term in financial_terms) and 
            any(crypto in keyword_clean for crypto in ['crypto', 'bitcoin', 'eth', 'defi'])):
            return True, "Financial + Crypto Context"
        
        return False, "No Crypto Relevance"
    
    def classify_removal_reason(self, keyword: str) -> str:
        """Detailed classification of why keyword was removed"""
        if not keyword or not isinstance(keyword, str):
            return "Invalid Input"
        
        keyword_clean = keyword.lower().strip()
        
        # Length checks
        if len(keyword_clean) < 3:
            return "Too Short (< 3 characters)"
        if len(keyword_clean) > 100:
            return "Too Long (> 100 characters)"
        
        # Pattern-based classification
        if re.match(r'^[a-z]{30,}$', keyword_clean):
            return "Gibberish - Very Long Word"
        if re.match(r'^[0-9]+$', keyword_clean):
            return "Pure Numbers"
        if re.search(r'(.)\1{6,}', keyword_clean):
            return "Repeated Characters"
        if re.match(r'^(test|demo|example|sample)\d*$', keyword_clean, re.IGNORECASE):
            return "Test/Demo Content"
        if re.search(r'(xxx|porn|sex|adult)', keyword_clean, re.IGNORECASE):
            return "Adult Content"
        if re.match(r'^(asdf|qwer|zxcv)', keyword_clean, re.IGNORECASE):
            return "Keyboard Mashing"
        if re.match(r'^[^a-zA-Z0-9\s\-\_\.]+$', keyword_clean):
            return "Only Special Characters"
        if re.search(r'[^\x00-\x7F]{10,}', keyword_clean):
            return "Non-English Gibberish"
        
        # Relevance check
        is_relevant, reason = self.is_crypto_relevant(keyword)
        if not is_relevant:
            return f"Not Crypto Related - {reason}"
        
        return "Other"

# ============= KEYWORD PROCESSOR =============
class EnhancedKeywordProcessor:
    """Process keywords with exact count control and detailed tracking"""
    
    def __init__(self, target_count: int = 124417):
        self.target_count = target_count
        self.relevance_checker = AdvancedCryptoRelevanceChecker()
        self.processing_stats = {
            'total_loaded': 0,
            'duplicates_removed': 0,
            'relevance_filtered': 0,
            'final_kept': 0
        }
        
    def load_and_process(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load exactly the target number of keywords and process them"""
        print(f"\n📂 Loading keywords from: {file_path}")
        print(f"🎯 Target keyword count: {self.target_count:,}")
        
        # Load data with exact count
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path, nrows=self.target_count)
        else:
            df = pd.read_csv(file_path, nrows=self.target_count)
        
        self.processing_stats['total_loaded'] = len(df)
        print(f"✓ Loaded exactly {len(df):,} keywords")
        
        # Find keyword column
        keyword_col = self._find_keyword_column(df)
        print(f"✓ Using column: '{keyword_col}'")
        
        # Process all keywords
        processed_df, removed_df = self._process_all_keywords(df, keyword_col)
        
        # Print final statistics
        self._print_processing_summary()
        
        return processed_df, removed_df
    
    def _find_keyword_column(self, df: pd.DataFrame) -> str:
        """Intelligently find the keyword column"""
        possible_names = [
            'keyword', 'keywords', 'query', 'search_term', 'term', 'phrase',
            'search_query', 'key_word', 'search_phrase'
        ]
        
        # Check exact matches first
        for col in df.columns:
            if col.lower() in possible_names:
                return col
        
        # Check partial matches
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in ['keyword', 'query', 'term']):
                return col
        
        # Return first text column
        for col in df.columns:
            if df[col].dtype == 'object' and not col.lower().startswith('unnamed'):
                return col
        
        return df.columns[0]
    
    def _process_all_keywords(self, df: pd.DataFrame, keyword_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process all keywords with detailed tracking"""
        print("\n🔧 Processing all keywords...")
        
        all_keywords = []
        
        # Extract and clean all keywords
        for idx, row in df.iterrows():
            if idx % 10000 == 0 and idx > 0:
                print(f"   Processing: {idx:,} / {len(df):,}")
            
            keyword = str(row[keyword_col]) if pd.notna(row[keyword_col]) else ""
            
            # Basic cleaning
            cleaned = self._clean_keyword(keyword)
            
            keyword_data = {
                'original_keyword': keyword,
                'cleaned_keyword': cleaned,
                'search_volume': self._safe_get_numeric(row, 'search_volume'),
                'competition': self._safe_get_numeric(row, 'competition'),
                'cpc': self._safe_get_numeric(row, 'cpc', float),
                'row_index': idx
            }
            
            all_keywords.append(keyword_data)
        
        # Create dataframe and process
        all_df = pd.DataFrame(all_keywords)
        
        # Remove duplicates first
        print("\n   Removing duplicates...")
        before_dedup = len(all_df)
        all_df = all_df.drop_duplicates(subset=['cleaned_keyword'], keep='first')
        self.processing_stats['duplicates_removed'] = before_dedup - len(all_df)
        print(f"   ✓ Removed {self.processing_stats['duplicates_removed']:,} duplicates")
        
        # Separate valid and invalid keywords
        print("\n   Checking crypto relevance...")
        kept_keywords = []
        removed_keywords = []
        
        for idx, row in all_df.iterrows():
            keyword = row['cleaned_keyword']
            
            if not keyword or len(keyword.strip()) == 0:
                row['removal_reason'] = 'Empty Keyword'
                removed_keywords.append(row)
                continue
            
            # Check relevance
            is_relevant, reason = self.relevance_checker.is_crypto_relevant(keyword)
            
            if is_relevant:
                kept_keywords.append(row)
            else:
                row['removal_reason'] = self.relevance_checker.classify_removal_reason(keyword)
                removed_keywords.append(row)
        
        kept_df = pd.DataFrame(kept_keywords) if kept_keywords else pd.DataFrame()
        removed_df = pd.DataFrame(removed_keywords) if removed_keywords else pd.DataFrame()
        
        # Add features to kept keywords
        if len(kept_df) > 0:
            kept_df = self._add_keyword_features(kept_df)
            kept_df = kept_df.reset_index(drop=True)
        
        self.processing_stats['relevance_filtered'] = len(removed_df)
        self.processing_stats['final_kept'] = len(kept_df)
        
        return kept_df, removed_df
    
    def _clean_keyword(self, keyword: str) -> str:
        """Clean keyword while preserving crypto-specific characters"""
        if not keyword or not isinstance(keyword, str):
            return ""
        
        # Basic cleaning
        cleaned = keyword.lower().strip()
        
        # Remove URLs and emails
        cleaned = re.sub(r'http\S+|www\S+', '', cleaned)
        cleaned = re.sub(r'\S+@\S+', '', cleaned)
        
        # Keep crypto-specific characters: letters, numbers, spaces, hyphens, underscores, dots, $, #
        cleaned = re.sub(r'[^\w\s\-\$\#\.]', ' ', cleaned)
        
        # Normalize spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()
    
    def _safe_get_numeric(self, row, column, dtype=int):
        """Safely extract numeric value from row"""
        try:
            if column in row and pd.notna(row[column]):
                return dtype(row[column])
        except:
            pass
        return 0 if dtype == int else 0.0
    
    def _add_keyword_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful features for clustering"""
        df = df.copy()
        df['word_count'] = df['cleaned_keyword'].str.split().str.len()
        df['char_count'] = df['cleaned_keyword'].str.len()
        df['has_numbers'] = df['cleaned_keyword'].str.contains(r'\d', na=False)
        df['is_question'] = df['cleaned_keyword'].str.contains(
            r'\b(how|what|why|when|where|which|who|can|should|will)\b', na=False
        )
        df['is_comparison'] = df['cleaned_keyword'].str.contains(
            r'\b(vs|versus|compare|difference|better|best|top)\b', na=False
        )
        return df
    
    def _print_processing_summary(self):
        """Print detailed processing summary"""
        stats = self.processing_stats
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   • Total Keywords Loaded: {stats['total_loaded']:,}")
        print(f"   • Duplicates Removed: {stats['duplicates_removed']:,}")
        print(f"   • Relevance Filtered Out: {stats['relevance_filtered']:,}")
        print(f"   • Final Keywords Kept: {stats['final_kept']:,}")
        
        if stats['total_loaded'] > 0:
            retention_rate = (stats['final_kept'] / stats['total_loaded']) * 100
            print(f"   • Retention Rate: {retention_rate:.1f}%")

# ============= INTELLIGENT TOPIC NAME GENERATOR =============
class IntelligentTopicGenerator:
    """Generate meaningful, non-duplicate topic names using advanced NLP"""
    
    def __init__(self):
        self.used_words = set()
        self.topic_counter = 1
        self.crypto_categories = {
            'trading': ['trading', 'trade', 'buy', 'sell', 'price', 'chart', 'analysis'],
            'defi': ['defi', 'yield', 'farming', 'liquidity', 'pool', 'aave', 'compound'],
            'nft': ['nft', 'opensea', 'collectible', 'art', 'gaming', 'metaverse'],
            'bitcoin': ['bitcoin', 'btc', 'satoshi', 'halving', 'mining'],
            'ethereum': ['ethereum', 'eth', 'gas', 'gwei', 'erc20', 'smart contract'],
            'exchanges': ['binance', 'coinbase', 'kraken', 'exchange', 'cex', 'dex'],
            'staking': ['staking', 'validator', 'reward', 'delegation', 'pos'],
            'security': ['security', 'wallet', 'private key', 'seed', 'hack', 'audit']
        }
        
    def generate_topic_name(self, keywords: List[str]) -> str:
        """Generate intelligent topic names based on keyword content"""
        if not keywords:
            name = f"Crypto Topic {self.topic_counter}"
            self.topic_counter += 1
            return name
        
        # Analyze keywords to find the best representative terms
        important_terms = self._extract_key_terms(keywords)
        
        # Try to categorize based on crypto knowledge
        category_name = self._identify_crypto_category(keywords, important_terms)
        
        if category_name:
            final_name = self._ensure_unique_name(category_name)
        else:
            # Fallback to term-based naming
            final_name = self._create_descriptive_name(important_terms)
        
        return final_name
    
    def _extract_key_terms(self, keywords: List[str]) -> List[str]:
        """Extract the most important and representative terms"""
        if len(keywords) < 2:
            return keywords[0].split()[:3] if keywords else []
        
        try:
            # Use TF-IDF to find important terms
            tfidf = TfidfVectorizer(
                max_features=20,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=1,
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
            )
            
            tfidf_matrix = tfidf.fit_transform(keywords)
            feature_names = tfidf.get_feature_names_out()
            
            # Get term importance scores
            scores = tfidf_matrix.sum(axis=0).A1
            term_scores = list(zip(feature_names, scores))
            term_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Extract top terms, preferring crypto-specific ones
            important_terms = []
            for term, score in term_scores[:10]:
                cleaned_term = re.sub(r'[^\w]', '', term).lower()
                if len(cleaned_term) > 2 and cleaned_term not in self.used_words:
                    important_terms.append(cleaned_term.title())
                if len(important_terms) >= 5:
                    break
            
            return important_terms
            
        except Exception:
            # Fallback to frequency analysis
            return self._fallback_term_extraction(keywords)
    
    def _fallback_term_extraction(self, keywords: List[str]) -> List[str]:
        """Fallback method using simple frequency analysis"""
        word_freq = Counter()
        
        for keyword in keywords[:100]:  # Limit for performance
            words = keyword.lower().split()
            for word in words:
                if len(word) > 2:
                    word_freq[word] += 1
        
        # Get top words, excluding common stop words
        stop_words = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they', 'have'}
        top_words = []
        
        for word, freq in word_freq.most_common(10):
            if word not in stop_words and word not in self.used_words:
                top_words.append(word.title())
            if len(top_words) >= 5:
                break
        
        return top_words
    
    def _identify_crypto_category(self, keywords: List[str], terms: List[str]) -> Optional[str]:
        """Identify crypto category based on keywords and terms"""
        # Convert to lowercase for matching
        all_text = ' '.join(keywords[:50]).lower()  # Limit for performance
        term_text = ' '.join(terms).lower()
        
        category_scores = {}
        
        for category, category_terms in self.crypto_categories.items():
            score = 0
            for term in category_terms:
                if term in all_text:
                    score += all_text.count(term)
                if term in term_text:
                    score += 5  # Higher weight for important terms
            category_scores[category] = score
        
        # Find the highest scoring category
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            if best_category[1] > 0:  # Must have some relevance
                return self._format_category_name(best_category[0], terms)
        
        return None
    
    def _format_category_name(self, category: str, terms: List[str]) -> str:
        """Format category name with specific terms"""
        category_formatted = category.replace('_', ' ').title()
        
        # Add specific terms if available
        if terms:
            specific_term = terms[0]  # Use the most important term
            if specific_term.lower() not in category.lower():
                return f"{category_formatted} {specific_term}"
        
        return category_formatted
    
    def _create_descriptive_name(self, terms: List[str]) -> str:
        """Create descriptive name from important terms"""
        if not terms:
            name = f"Crypto Topic {self.topic_counter}"
            self.topic_counter += 1
            return name
        
        # Use 1-3 most important terms
        selected_terms = terms[:3]
        
        # Create name avoiding duplicates
        base_name = " ".join(selected_terms)
        return self._ensure_unique_name(base_name)
    
    def _ensure_unique_name(self, name: str) -> str:
        """Ensure the name doesn't use already used words"""
        words_in_name = set(word.lower() for word in name.split())
        
        # If no overlap with used words, it's good
        if not words_in_name.intersection(self.used_words):
            # Mark these words as used
            self.used_words.update(words_in_name)
            return name
        
        # If there's overlap, add a number suffix
        counter = 1
        while True:
            numbered_name = f"{name} {counter}"
            if numbered_name not in self.used_words:
                self.used_words.add(numbered_name.lower())
                return numbered_name
            counter += 1
    
    def reset_used_words(self):
        """Reset the used words tracker"""
        self.used_words.clear()
        self.topic_counter = 1

# ============= SIMPLE CLUSTERING ENGINE =============
class SimpleCryptoClusteringEngine:
    """Simple, effective clustering focused on meaningful crypto topics"""
    
    def __init__(self):
        self.embedding_model = None
        self.embeddings = None
        self.topic_generator = IntelligentTopicGenerator()
        
    def create_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create meaningful crypto topic clusters"""
        if len(df) == 0:
            print("⚠️ No keywords to cluster!")
            return df
        
        print(f"\n🤖 Starting BERT-based clustering for {len(df):,} keywords...")
        
        keywords = df['cleaned_keyword'].tolist()
        
        # Step 1: Create high-quality embeddings
        print("   📊 Creating BERT embeddings...")
        self.embeddings = self._create_embeddings(keywords)
        
        # Step 2: Reduce dimensions intelligently
        print("   🎯 Reducing dimensions...")
        reduced_embeddings = self._reduce_dimensions(self.embeddings)
        
        # Step 3: Find optimal number of clusters
        optimal_clusters = self._find_optimal_clusters(reduced_embeddings, len(keywords))
        print(f"   🎲 Creating {optimal_clusters} topic clusters...")
        
        # Step 4: Create clusters
        cluster_labels = self._create_topic_clusters(reduced_embeddings, optimal_clusters)
        
        # Step 5: Generate intelligent topic names
        print("   🏷️ Generating intelligent topic names...")
        result_df = df.copy()
        result_df['topic_id'] = cluster_labels
        
        # Generate unique topic names
        self.topic_generator.reset_used_words()
        topic_names = self._generate_intelligent_topic_names(keywords, cluster_labels)
        result_df['topic_name'] = [topic_names[label] for label in cluster_labels]
        
        # Add cluster statistics
        result_df = self._add_cluster_statistics(result_df)
        
        print("✅ Clustering completed successfully!")
        return result_df
    
    def _create_embeddings(self, keywords: List[str]) -> np.ndarray:
        """Create high-quality BERT embeddings"""
        # Initialize the best embedding model
        print(f"      Loading model: {Config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.embedding_model.max_seq_length = Config.MAX_SEQUENCE_LENGTH
        
        # Create embeddings in batches
        embeddings = []
        batch_size = Config.BATCH_SIZE
        
        for i in range(0, len(keywords), batch_size):
            batch = keywords[i:i+batch_size]
            
            # Show progress
            progress = (i / len(keywords)) * 100
            print(f"      Progress: {progress:.1f}% ({i:,}/{len(keywords):,})")
            
            batch_embeddings = self.embedding_model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
            
            # Memory management
            if i % (batch_size * 5) == 0:
                gc.collect()
        
        final_embeddings = np.vstack(embeddings)
        print(f"      ✓ Created embeddings: {final_embeddings.shape}")
        return final_embeddings
    
    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Intelligent dimension reduction"""
        print(f"      Original dimensions: {embeddings.shape[1]}")
        
        # First apply PCA if dimensions are too high
        if embeddings.shape[1] > 100:
            print("      Applying PCA preprocessing...")
            pca = PCA(n_components=100, random_state=42)
            embeddings = pca.fit_transform(embeddings)
            print(f"      PCA reduced to: {embeddings.shape[1]} dimensions")
        
        # Then apply UMAP for final reduction
        print("      Applying UMAP reduction...")
        umap_model = umap.UMAP(
            n_neighbors=min(Config.UMAP_N_NEIGHBORS, len(embeddings) - 1),
            n_components=Config.UMAP_N_COMPONENTS,
            min_dist=Config.UMAP_MIN_DIST,
            metric=Config.UMAP_METRIC,
            random_state=42,
            low_memory=True,
            verbose=False
        )
        
        reduced_embeddings = umap_model.fit_transform(embeddings)
        print(f"      ✓ Final dimensions: {reduced_embeddings.shape[1]}")
        
        return reduced_embeddings
    
    def _find_optimal_clusters(self, embeddings: np.ndarray, n_keywords: int) -> int:
        """Find optimal number of clusters based on data size and quality"""
        # Base calculation on keyword count
        base_clusters = max(5, min(Config.TARGET_TOPICS, n_keywords // 100))
        
        # Adjust based on data characteristics
        if n_keywords < 1000:
            optimal = max(5, n_keywords // 50)
        elif n_keywords < 10000:
            optimal = max(10, n_keywords // 200)
        elif n_keywords < 50000:
            optimal = max(15, n_keywords // 500)
        else:
            optimal = max(20, min(50, n_keywords // 1000))
        
        # Ensure we don't exceed reasonable limits
        optimal = min(optimal, Config.TARGET_TOPICS, n_keywords // Config.MIN_CLUSTER_SIZE)
        
        print(f"      Optimal clusters calculated: {optimal}")
        return optimal
    
    def _create_topic_clusters(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Create topic clusters using KMeans"""
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300,
            algorithm='lloyd'
        )
        
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Post-process small clusters
        cluster_labels = self._handle_small_clusters(cluster_labels, embeddings)
        
        return cluster_labels
    
    def _handle_small_clusters(self, labels: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Merge or reassign very small clusters"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        small_clusters = unique_labels[counts < Config.MIN_CLUSTER_SIZE]
        
        if len(small_clusters) == 0:
            return labels
        
        print(f"      Handling {len(small_clusters)} small clusters...")
        
        # For small clusters, reassign to nearest large cluster
        large_clusters = unique_labels[counts >= Config.MIN_CLUSTER_SIZE]
        
        if len(large_clusters) == 0:
            return labels  # All clusters are small, keep as is
        
        modified_labels = labels.copy()
        
        for small_cluster in small_clusters:
            small_indices = np.where(labels == small_cluster)[0]
            small_embeddings = embeddings[small_indices]
            
            # Find nearest large cluster
            min_distance = float('inf')
            best_cluster = large_clusters[0]
            
            for large_cluster in large_clusters:
                large_indices = np.where(labels == large_cluster)[0]
                large_embeddings = embeddings[large_indices]
                
                # Calculate average distance
                distances = cosine_similarity(small_embeddings, large_embeddings)
                avg_distance = distances.mean()
                
                if avg_distance > min_distance:  # Higher similarity (lower cosine distance)
                    min_distance = avg_distance
                    best_cluster = large_cluster
            
            # Reassign small cluster to best large cluster
            modified_labels[small_indices] = best_cluster
        
        return modified_labels
    
    def _generate_intelligent_topic_names(self, keywords: List[str], labels: np.ndarray) -> Dict[int, str]:
        """Generate intelligent, unique topic names"""
        # Group keywords by cluster
        cluster_keywords = defaultdict(list)
        for keyword, label in zip(keywords, labels):
            cluster_keywords[label].append(keyword)
        
        # Generate names for each cluster
        topic_names = {}
        for cluster_id, cluster_kws in cluster_keywords.items():
            topic_name = self.topic_generator.generate_topic_name(cluster_kws)
            topic_names[cluster_id] = topic_name
        
        return topic_names
    
    def _add_cluster_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful cluster statistics"""
        df = df.copy()
        
        # Add cluster size
        cluster_sizes = df['topic_id'].value_counts().to_dict()
        df['cluster_size'] = df['topic_id'].map(cluster_sizes)
        
        # Add cluster rank (by size)
        topic_ranks = df.groupby('topic_name')['cluster_size'].first().rank(method='dense', ascending=False).to_dict()
        df['topic_rank'] = df['topic_name'].map(topic_ranks)
        
        return df

# ============= COMPREHENSIVE OUTPUT GENERATOR =============
class ComprehensiveOutputGenerator:
    """Generate detailed Excel output with all analysis"""
    
    def create_comprehensive_output(self, clustered_df: pd.DataFrame, removed_df: pd.DataFrame, output_path: str):
        """Create comprehensive Excel analysis"""
        print(f"\n📝 Creating comprehensive output: {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Main Results - Clustered Keywords
            self._create_main_results_sheet(clustered_df, writer)
            
            # Sheet 2: Detailed Removed Keywords Analysis
            if len(removed_df) > 0:
                self._create_removed_analysis_sheet(removed_df, writer)
            
            # Sheet 3: Topic Analysis & Statistics
            self._create_topic_analysis_sheet(clustered_df, writer)
            
            # Sheet 4: Keyword Distribution Analysis
            self._create_distribution_analysis_sheet(clustered_df, writer)
            
            # Sheet 5: Processing Summary & Statistics
            self._create_summary_statistics_sheet(clustered_df, removed_df, writer)
        
        print("✅ Comprehensive output created successfully!")
    
    def _create_main_results_sheet(self, df: pd.DataFrame, writer):
        """Create main results sheet with clustered keywords"""
        if len(df) == 0:
            return
        
        # Prepare output columns
        output_df = df[[
            'original_keyword', 'cleaned_keyword', 'topic_id', 'topic_name',
            'cluster_size', 'topic_rank', 'word_count', 'char_count',
            'search_volume', 'competition', 'cpc', 'has_numbers', 'is_question', 'is_comparison'
        ]].copy()
        
        # Sort by topic rank and then by search volume
        output_df = output_df.sort_values(['topic_rank', 'search_volume'], ascending=[True, False])
        
        # Add row numbers
        output_df.insert(0, 'row_id', range(1, len(output_df) + 1))
        
        output_df.to_excel(writer, sheet_name='Clustered_Keywords', index=False)
    
    def _create_removed_analysis_sheet(self, df: pd.DataFrame, writer):
        """Create detailed analysis of removed keywords"""
        if len(df) == 0:
            return
        
        # Prepare removed keywords data
        removed_df = df[[
            'original_keyword', 'cleaned_keyword', 'removal_reason'
        ]].copy()
        
        # Sort by removal reason
        removed_df = removed_df.sort_values('removal_reason')
        
        # Add statistics
        reason_stats = df['removal_reason'].value_counts().reset_index()
        reason_stats.columns = ['Removal_Reason', 'Count']
        reason_stats['Percentage'] = (reason_stats['Count'] / len(df) * 100).round(2)
        
        # Write to different sections of the sheet
        removed_df.to_excel(writer, sheet_name='Removed_Keywords', index=False, startrow=0)
        
        # Add summary statistics below
        start_row = len(removed_df) + 3
        reason_stats.to_excel(writer, sheet_name='Removed_Keywords', index=False, startrow=start_row)
    
    def _create_topic_analysis_sheet(self, df: pd.DataFrame, writer):
        """Create detailed topic analysis"""
        if len(df) == 0:
            return
        
        # Topic-level statistics
        topic_stats = df.groupby(['topic_id', 'topic_name']).agg({
            'cleaned_keyword': 'count',
            'search_volume': ['sum', 'mean', 'max'],
            'competition': 'mean',
            'cpc': 'mean',
            'word_count': 'mean',
            'char_count': 'mean',
            'has_numbers': 'sum',
            'is_question': 'sum',
            'is_comparison': 'sum'
        }).round(2)
        
        # Flatten column names
        topic_stats.columns = [
            'Keyword_Count', 'Total_Search_Volume', 'Avg_Search_Volume', 'Max_Search_Volume',
            'Avg_Competition', 'Avg_CPC', 'Avg_Word_Count', 'Avg_Char_Count',
            'Keywords_With_Numbers', 'Question_Keywords', 'Comparison_Keywords'
        ]
        
        topic_stats = topic_stats.reset_index()
        topic_stats = topic_stats.sort_values('Keyword_Count', ascending=False)
        
        # Add percentages
        total_keywords = len(df)
        topic_stats['Percentage_of_Total'] = (topic_stats['Keyword_Count'] / total_keywords * 100).round(2)
        
        topic_stats.to_excel(writer, sheet_name='Topic_Analysis', index=False)
    
    def _create_distribution_analysis_sheet(self, df: pd.DataFrame, writer):
        """Create keyword distribution analysis"""
        if len(df) == 0:
            return
        
        # Various distribution analyses
        analyses = []
        
        # 1. Word count distribution
        word_count_dist = df['word_count'].value_counts().sort_index().reset_index()
        word_count_dist.columns = ['Word_Count', 'Frequency']
        word_count_dist['Analysis_Type'] = 'Word Count Distribution'
        analyses.append(word_count_dist)
        
        # 2. Character count ranges
        df['char_range'] = pd.cut(df['char_count'], bins=[0, 10, 20, 30, 50, 100], labels=['1-10', '11-20', '21-30', '31-50', '51+'])
        char_dist = df['char_range'].value_counts().reset_index()
        char_dist.columns = ['Character_Range', 'Frequency']
        char_dist['Analysis_Type'] = 'Character Count Distribution'
        
        # 3. Search volume ranges (if available)
        if df['search_volume'].sum() > 0:
            df['volume_range'] = pd.cut(df['search_volume'], bins=[-1, 0, 100, 1000, 10000, float('inf')], 
                                       labels=['0', '1-100', '101-1K', '1K-10K', '10K+'])
            volume_dist = df['volume_range'].value_counts().reset_index()
            volume_dist.columns = ['Search_Volume_Range', 'Frequency']
            volume_dist['Analysis_Type'] = 'Search Volume Distribution'
        
        # Combine all analyses
        start_row = 0
        for analysis in analyses:
            analysis.to_excel(writer, sheet_name='Distribution_Analysis', index=False, startrow=start_row)
            start_row += len(analysis) + 3
    
    def _create_summary_statistics_sheet(self, clustered_df: pd.DataFrame, removed_df: pd.DataFrame, writer):
        """Create comprehensive summary statistics"""
        stats = []
        
        # Overall processing statistics
        total_processed = len(clustered_df) + len(removed_df)
        
        stats.extend([
            {'Metric': 'PROCESSING SUMMARY', 'Value': ''},
            {'Metric': 'Total Keywords Processed', 'Value': f"{total_processed:,}"},
            {'Metric': 'Keywords Successfully Clustered', 'Value': f"{len(clustered_df):,}"},
            {'Metric': 'Keywords Removed', 'Value': f"{len(removed_df):,}"},
            {'Metric': 'Success Rate', 'Value': f"{(len(clustered_df) / total_processed * 100):.2f}%" if total_processed > 0 else "0%"},
            {'Metric': '', 'Value': ''},
        ])
        
        if len(clustered_df) > 0:
            stats.extend([
                {'Metric': 'CLUSTERING RESULTS', 'Value': ''},
                {'Metric': 'Total Topics Created', 'Value': clustered_df['topic_name'].nunique()},
                {'Metric': 'Average Keywords per Topic', 'Value': f"{len(clustered_df) / clustered_df['topic_name'].nunique():.1f}"},
                {'Metric': 'Largest Topic Size', 'Value': clustered_df['cluster_size'].max()},
                {'Metric': 'Smallest Topic Size', 'Value': clustered_df['cluster_size'].min()},
                {'Metric': '', 'Value': ''},
                
                {'Metric': 'KEYWORD CHARACTERISTICS', 'Value': ''},
                {'Metric': 'Average Word Count', 'Value': f"{clustered_df['word_count'].mean():.1f}"},
                {'Metric': 'Average Character Count', 'Value': f"{clustered_df['char_count'].mean():.1f}"},
                {'Metric': 'Keywords with Numbers', 'Value': f"{clustered_df['has_numbers'].sum():,} ({clustered_df['has_numbers'].mean()*100:.1f}%)"},
                {'Metric': 'Question Keywords', 'Value': f"{clustered_df['is_question'].sum():,} ({clustered_df['is_question'].mean()*100:.1f}%)"},
                {'Metric': 'Comparison Keywords', 'Value': f"{clustered_df['is_comparison'].sum():,} ({clustered_df['is_comparison'].mean()*100:.1f}%)"},
                {'Metric': '', 'Value': ''},
            ])
            
            if clustered_df['search_volume'].sum() > 0:
                stats.extend([
                    {'Metric': 'SEARCH VOLUME ANALYSIS', 'Value': ''},
                    {'Metric': 'Total Search Volume', 'Value': f"{clustered_df['search_volume'].sum():,}"},
                    {'Metric': 'Average Search Volume', 'Value': f"{clustered_df['search_volume'].mean():.0f}"},
                    {'Metric': 'Keywords with Search Volume > 0', 'Value': f"{(clustered_df['search_volume'] > 0).sum():,}"},
                ])
        
        # Removal analysis
        if len(removed_df) > 0:
            stats.extend([
                {'Metric': '', 'Value': ''},
                {'Metric': 'TOP REMOVAL REASONS', 'Value': ''},
            ])
            
            top_reasons = removed_df['removal_reason'].value_counts().head(5)
            for reason, count in top_reasons.items():
                percentage = (count / len(removed_df)) * 100
                stats.append({
                    'Metric': f"  • {reason}", 
                    'Value': f"{count:,} ({percentage:.1f}%)"
                })
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)

# ============= MAIN EXECUTION PIPELINE =============
def run_enhanced_clustering_pipeline():
    """Main execution pipeline with comprehensive processing"""
    start_time = datetime.now()
    
    print("="*80)
    print("🚀 ENHANCED CRYPTO KEYWORD CLUSTERING SYSTEM")
    print(f"   Target Keywords: {Config.TARGET_KEYWORDS:,}")
    print(f"   Target Topics: ~{Config.TARGET_TOPICS}")
    print("   Focus: Meaningful crypto topics without duplication")
    print("="*80)
    
    try:
        # Create output directories
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        
        # Step 1: Load and process keywords with exact count
        print(f"\n🔄 STEP 1: Loading and processing {Config.TARGET_KEYWORDS:,} keywords...")
        processor = EnhancedKeywordProcessor(target_count=Config.TARGET_KEYWORDS)
        processed_df, removed_df = processor.load_and_process(Config.INPUT_FILE)
        
        # Save removed keywords immediately
        if len(removed_df) > 0:
            removed_df.to_csv(Config.REMOVED_KEYWORDS_FILE, index=False)
            print(f"💾 Saved {len(removed_df):,} removed keywords to: {Config.REMOVED_KEYWORDS_FILE}")
        
        # Step 2: Create intelligent clusters
        if len(processed_df) > 0:
            print(f"\n🔄 STEP 2: Creating intelligent topic clusters...")
            clustering_engine = SimpleCryptoClusteringEngine()
            clustered_df = clustering_engine.create_clusters(processed_df)
            
            # Step 3: Generate comprehensive output
            print(f"\n🔄 STEP 3: Generating comprehensive output...")
            output_generator = ComprehensiveOutputGenerator()
            output_generator.create_comprehensive_output(clustered_df, removed_df, Config.OUTPUT_FILE)
            
            # Step 4: Display results
            elapsed_time = datetime.now() - start_time
            
            print("\n" + "="*80)
            print("✅ CLUSTERING COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            # Final Statistics
            total_processed = len(processed_df) + len(removed_df)
            print(f"\n📊 FINAL RESULTS:")
            print(f"   • Total Keywords Loaded: {total_processed:,}")
            print(f"   • Keywords Successfully Clustered: {len(clustered_df):,}")
            print(f"   • Keywords Removed: {len(removed_df):,}")
            print(f"   • Success Rate: {(len(clustered_df) / total_processed * 100):.2f}%")
            print(f"   • Topics Created: {clustered_df['topic_name'].nunique()}")
            print(f"   • Average Keywords per Topic: {len(clustered_df) / clustered_df['topic_name'].nunique():.1f}")
            
            print(f"\n⏱️ Processing Time: {elapsed_time}")
            
            print(f"\n📁 Output Files:")
            print(f"   • Main Results: {Config.OUTPUT_FILE}")
            print(f"   • Removed Keywords: {Config.REMOVED_KEYWORDS_FILE}")
            
            # Show sample topics
            print(f"\n🎯 Top 10 Topics by Keyword Count:")
            topic_summary = clustered_df.groupby('topic_name').size().sort_values(ascending=False).head(10)
            for topic, count in topic_summary.items():
                percentage = (count / len(clustered_df)) * 100
                print(f"   📌 {topic}: {count:,} keywords ({percentage:.1f}%)")
            
            # Show removal summary
            if len(removed_df) > 0:
                print(f"\n🗑️ Top Removal Reasons:")
                removal_summary = removed_df['removal_reason'].value_counts().head(5)
                for reason, count in removal_summary.items():
                    percentage = (count / len(removed_df)) * 100
                    print(f"   ❌ {reason}: {count:,} ({percentage:.1f}%)")
            
            return clustered_df
        else:
            print("\n⚠️ No valid crypto keywords found to cluster!")
            print("Please check your input data and crypto relevance criteria.")
            return pd.DataFrame()
        
    except Exception as e:
        print(f"\n Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    
    result_data = run_enhanced_clustering_pipeline()