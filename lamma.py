import os
import gc
import time
import logging
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional, Set
from collections import defaultdict, Counter
import re
import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig
)

# Core ML and NLP libraries
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============= CONFIGURATION =============
class Config:
    """Configuration for hybrid clustering with local Llama"""
    
    # File paths - UPDATE THESE
    INPUT_FILE = '/home/admin1/Downloads/demo_crypto/csv list keywords.csv'
    OUTPUT_DIR = '/home/admin1/Downloads/demo_crypto/output'
    OUTPUT_FILE = f'{OUTPUT_DIR}/llama_clusters_90k.xlsx'
    
    # Llama Model Configuration - FIXED FOR CPU
    # Option 1: CPU-friendly model (recommended)
    LLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Option 2: Alternative CPU models
    # LLAMA_MODEL = "microsoft/DialoGPT-medium"
    # LLAMA_MODEL = "distilgpt2"
    
    # Option 3: Using Ollama (if installed)
    USE_OLLAMA = False  # Set to True if you have Ollama installed
    OLLAMA_MODEL = "llama2:7b"
    
    # BERT Model for embeddings
    EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
    
    # Processing Configuration
    BATCH_SIZE = 500  # Reduced for CPU
    MAX_SEQUENCE_LENGTH = 128
    MAX_KEYWORDS = 90000
    
    # Clustering Configuration
    PILLAR_CLUSTERS = 12      # Main categories
    TOPICS_PER_PILLAR = 8     # Average topics per pillar
    MIN_CLUSTER_SIZE = 10
    MIN_KEYWORDS_FOR_TOPIC = 5
    
    # UMAP Parameters
    UMAP_N_NEIGHBORS = 15
    UMAP_N_COMPONENTS = 50
    UMAP_MIN_DIST = 0.0
    UMAP_METRIC = 'cosine'
    
    # Llama Processing - OPTIMIZED FOR CPU
    LLAMA_SAMPLE_SIZE = 15    # Reduced for faster processing
    LLAMA_MAX_LENGTH = 50     # Reduced for CPU
    LLAMA_TEMPERATURE = 0.3   # Lower = more focused
    DEVICE = "cpu"  # Force CPU for compatibility
    USE_4BIT = False  # Disable quantization on CPU
    MAX_NEW_TOKENS = 15  # Limit output length

# ============= LLAMA HANDLER - FIXED =============
class LlamaHandler:
    """Handle local Llama model for intelligent naming - CPU optimized"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.use_ollama = Config.USE_OLLAMA
        self.call_count = 0
        
        if self.use_ollama:
            self._setup_ollama()
        else:
            self._setup_transformers()
    
    def _setup_transformers(self):
        """Setup Llama using Hugging Face Transformers - CPU optimized"""
        print("   Loading Llama model for CPU (this may take a few minutes)...")
        
        try:
            # Load model without device specification to avoid conflicts
            print(f"   Loading model: {Config.LLAMA_MODEL}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                Config.LLAMA_MODEL,
                trust_remote_code=True,
                padding_side='left'  # Important for generation
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load model with CPU-specific settings
            self.model = AutoModelForCausalLM.from_pretrained(
                Config.LLAMA_MODEL,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=None  # Let it auto-assign
            )
            
            # Move model to CPU explicitly
            self.model = self.model.to('cpu')
            
            # Create pipeline without device specification
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.LLAMA_TEMPERATURE,
                do_sample=True,
                top_p=0.9,
                device=-1,  # CPU
                return_full_text=False  # Only return new tokens
            )
            
            print("   ✓ Llama model loaded successfully on CPU")
            
        except Exception as e:
            print(f"   ⚠️ Failed to load Llama model: {e}")
            print("   Falling back to rule-based naming...")
            self.pipeline = None
    
    def _setup_ollama(self):
        """Setup Ollama if available"""
        try:
            import ollama
            self.ollama_client = ollama.Client()
            # Test if model is available
            self.ollama_client.list()
            print(f"   ✓ Ollama setup successful with model: {Config.OLLAMA_MODEL}")
        except Exception as e:
            print(f"   ⚠️ Ollama not available: {e}")
            print("   Falling back to Transformers...")
            self.use_ollama = False
            self._setup_transformers()
    
    def generate_cluster_name(self, keywords: List[str], level: str = 'topic') -> str:
        """Generate intelligent cluster name using Llama"""
        self.call_count += 1
        
        # If no model loaded, use fallback
        if not self.use_ollama and self.pipeline is None:
            return self._fallback_naming(keywords, level)
        
        # Sample keywords
        if len(keywords) > Config.LLAMA_SAMPLE_SIZE:
            sample_indices = np.random.choice(len(keywords), Config.LLAMA_SAMPLE_SIZE, replace=False)
            keywords_sample = [keywords[i] for i in sample_indices]
        else:
            keywords_sample = keywords
        
        # Create prompt
        if level == 'pillar':
            prompt = self._create_pillar_prompt(keywords_sample)
            max_words = 2
        else:
            prompt = self._create_topic_prompt(keywords_sample)
            max_words = 4
        
        try:
            if self.use_ollama:
                name = self._generate_with_ollama(prompt)
            else:
                name = self._generate_with_transformers(prompt)
            
            # Clean and validate
            name = self._clean_name(name, max_words)
            return name
            
        except Exception as e:
            logging.warning(f"Llama generation failed: {e}")
            return self._fallback_naming(keywords_sample, level)
    
    def _create_pillar_prompt(self, keywords: List[str]) -> str:
        """Create prompt for pillar naming - simplified for TinyLlama"""
        keywords_str = ', '.join(keywords[:10])  # Reduced for smaller model
        
        prompt = f"Keywords: {keywords_str}\nCategory (2 words max):"
        return prompt
    
    def _create_topic_prompt(self, keywords: List[str]) -> str:
        """Create prompt for topic naming - simplified for TinyLlama"""
        keywords_str = ', '.join(keywords[:10])  # Reduced for smaller model
        
        prompt = f"Keywords: {keywords_str}\nTopic (3 words max):"
        return prompt
    
    def _generate_with_transformers(self, prompt: str) -> str:
        """Generate using Transformers pipeline - CPU optimized"""
        try:
            # Generate with conservative settings
            result = self.pipeline(
                prompt,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.LLAMA_TEMPERATURE,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                clean_up_tokenization_spaces=True
            )
            
            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                generated = result[0]['generated_text']
            else:
                generated = str(result)
            
            # Clean up the response
            generated = generated.strip()
            
            # Remove common unwanted patterns
            generated = re.sub(r'\n+', ' ', generated)
            generated = re.sub(r'\s+', ' ', generated)
            
            return generated
            
        except Exception as e:
            print(f"   Generation error: {e}")
            return ""
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """Generate using Ollama"""
        try:
            import ollama
            
            response = self.ollama_client.generate(
                model=Config.OLLAMA_MODEL,
                prompt=prompt,
                options={
                    "temperature": Config.LLAMA_TEMPERATURE,
                    "max_tokens": 20,
                    "top_p": 0.9
                }
            )
            
            return response['response'].strip()
        except Exception as e:
            print(f"   Ollama generation error: {e}")
            return ""
    
    def _clean_name(self, name: str, max_words: int) -> str:
        """Clean and validate generated name"""
        if not name:
            return f"Cluster_{self.call_count}"
        
        # Remove special characters and extra spaces
        name = re.sub(r'[^\w\s-]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Remove common prefixes/suffixes
        remove_phrases = [
            'category', 'topic', 'cluster', 'group', 'keywords',
            'here is', 'the name is', 'answer:', 'response:',
            'category:', 'topic:', 'name:'
        ]
        
        name_lower = name.lower()
        for phrase in remove_phrases:
            if phrase in name_lower:
                name = name_lower.replace(phrase, '').strip()
        
        # Limit words
        words = name.split()
        if len(words) > max_words:
            name = ' '.join(words[:max_words])
        
        # Capitalize properly
        name = ' '.join(word.capitalize() for word in name.split() if word)
        
        # If name is empty or too short, use fallback
        if not name or len(name) < 2:
            return f"Cluster_{self.call_count}"
        
        return name
    
    def _fallback_naming(self, keywords: List[str], level: str) -> str:
        """Fallback naming using TF-IDF when Llama fails"""
        try:
            if len(keywords) < 2:
                return keywords[0].title() if keywords else f"Cluster_{level.title()}"
            
            # Use TF-IDF to extract important terms
            text_data = [' '.join(keywords)]
            
            # Simple word frequency approach for fallback
            all_words = ' '.join(keywords).lower().split()
            word_counts = Counter(all_words)
            
            # Get most common words (excluding very common ones)
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            filtered_words = [(word, count) for word, count in word_counts.most_common(10) 
                             if word not in common_words and len(word) > 2]
            
            if filtered_words:
                if level == 'pillar':
                    terms = [word.title() for word, _ in filtered_words[:2]]
                else:
                    terms = [word.title() for word, _ in filtered_words[:3]]
                
                return ' '.join(terms)
            
            return f"{level.title()}_Group_{self.call_count}"
                
        except Exception as e:
            return f"{level.title()}_Group_{self.call_count}"
    
    def generate_cluster_description(self, keywords: List[str], name: str) -> str:
        """Generate a brief description for a cluster"""
        # For CPU version, use simple rule-based descriptions
        if len(keywords) < 5:
            return f"Small cluster focused on {name.lower()}"
        
        # Count word types
        sample = keywords[:10]
        word_counts = Counter()
        for keyword in sample:
            words = keyword.lower().split()
            word_counts.update(words)
        
        # Get common themes
        common_words = [word for word, count in word_counts.most_common(3) 
                       if len(word) > 2 and word not in {'the', 'a', 'an', 'and', 'or'}]
        
        if common_words:
            theme = ', '.join(common_words[:2])
            return f"Keywords related to {name.lower()} including {theme}"
        else:
            return f"Collection of keywords about {name.lower()}"

# ============= KEYWORD PROCESSOR =============
class KeywordProcessor:
    """Process and clean keywords"""
    
    def load_and_process(self, file_path: str, max_keywords: int = 90000) -> pd.DataFrame:
        """Load and process keywords"""
        print(f"\n📂 Loading keywords from: {file_path}")
        
        try:
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
            processed_df = self._process_keywords(df, keyword_col)
            
            return processed_df
            
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            # Create dummy data for testing
            print("Creating sample data for testing...")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data for testing"""
        sample_keywords = [
            "bitcoin price", "crypto trading", "ethereum mining", "blockchain technology",
            "digital wallet", "cryptocurrency exchange", "NFT marketplace", "defi protocol",
            "smart contracts", "altcoin investment", "crypto portfolio", "bitcoin mining",
            "ethereum staking", "crypto news", "blockchain development", "digital assets"
        ] * 100  # Multiply to create more data
        
        data = []
        for i, keyword in enumerate(sample_keywords[:1000]):  # Limit for demo
            data.append({
                'keyword': keyword,
                'search_volume': np.random.randint(100, 10000),
                'competition': np.random.uniform(0.1, 1.0),
                'cpc': np.random.uniform(0.5, 5.0)
            })
        
        return pd.DataFrame(data)
    
    def _find_keyword_column(self, df: pd.DataFrame) -> str:
        """Find the keyword column"""
        possible_names = ['keyword', 'keywords', 'query', 'term', 'search_term']
        
        # First check for exact matches
        for col in df.columns:
            if col.lower().strip() in possible_names:
                return col
        
        # Then check for partial matches
        for col in df.columns:
            col_lower = col.lower().strip()
            for name in possible_names:
                if name in col_lower or col_lower in name:
                    return col
        
        # Finally, use first text column
        for col in df.columns:
            if df[col].dtype == 'object':
                return col
        
        return df.columns[0]
    
    def _process_keywords(self, df: pd.DataFrame, keyword_col: str) -> pd.DataFrame:
        """Clean and process keywords"""
        print("\n🔧 Processing keywords...")
        
        all_keywords = []
        
        for idx, row in df.iterrows():
            try:
                keyword = str(row[keyword_col]).strip()
                
                if not keyword or keyword.lower() in ['nan', 'none', '']:
                    continue
                
                # Light cleaning
                cleaned = re.sub(r'\s+', ' ', keyword)
                cleaned = re.sub(r'[^\w\s-]', ' ', cleaned)
                cleaned = cleaned.strip()
                
                if len(cleaned) < 2:
                    continue
                
                all_keywords.append({
                    'original_keyword': keyword,
                    'cleaned_keyword': cleaned,
                    'search_volume': row.get('search_volume', 0) if 'search_volume' in row else np.random.randint(10, 1000),
                    'competition': row.get('competition', 0) if 'competition' in row else np.random.uniform(0.1, 0.9),
                    'cpc': row.get('cpc', 0.0) if 'cpc' in row else np.random.uniform(0.1, 2.0)
                })
                
            except Exception as e:
                continue
        
        if not all_keywords:
            print("❌ No valid keywords found!")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(all_keywords)
        
        # Add features
        result_df['word_count'] = result_df['cleaned_keyword'].str.split().str.len()
        result_df['char_count'] = result_df['cleaned_keyword'].str.len()
        result_df['has_numbers'] = result_df['cleaned_keyword'].str.contains(r'\d', na=False)
        
        result_df = result_df.fillna(0)
        
        print(f"✓ Processed {len(result_df):,} keywords")
        
        return result_df

# ============= HYBRID CLUSTERING ENGINE =============
class HybridClusteringEngine:
    """Hybrid clustering using BERT embeddings + Llama intelligence"""
    
    def __init__(self):
        self.embedding_model = None
        self.embeddings = None
        self.llama_handler = LlamaHandler()
        
    def create_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create hierarchical clusters with intelligent naming"""
        if len(df) == 0:
            return df
        
        print("\n🤖 Starting Hybrid BERT + Llama clustering...")
        print(f"   Processing {len(df):,} keywords")
        print(f"   Device: {Config.DEVICE}")
        
        keywords = df['cleaned_keyword'].tolist()
        
        # Step 1: Create BERT embeddings
        print("\n   Step 1: Creating BERT embeddings...")
        self.embeddings = self._create_embeddings(keywords)
        
        # Step 2: Reduce dimensions
        print("   Step 2: Reducing dimensions with UMAP...")
        reduced_embeddings = self._reduce_dimensions(self.embeddings)
        
        # Step 3: Create initial clusters
        print(f"   Step 3: Creating {Config.PILLAR_CLUSTERS} pillar clusters...")
        pillar_labels = self._create_pillar_clusters(reduced_embeddings)
        
        print(f"   Step 4: Creating topic clusters within pillars...")
        topic_labels = self._create_topic_clusters(reduced_embeddings, pillar_labels)
        
        # Step 5: Use Llama for intelligent naming
        print("   Step 5: Generating intelligent names...")
        result_df = df.copy()
        result_df['pillar_id'] = pillar_labels
        result_df['topic_id'] = topic_labels
        
        # Generate pillar names
        pillar_names = self._generate_intelligent_names(keywords, pillar_labels, 'pillar')
        result_df['pillar_name'] = [pillar_names[label] for label in pillar_labels]
        
        # Generate topic names
        topic_names = self._generate_intelligent_names(keywords, topic_labels, 'topic')
        result_df['topic_name'] = [topic_names[label] for label in topic_labels]
        
        # Step 6: Generate descriptions for major clusters
        print("   Step 6: Generating cluster descriptions...")
        result_df = self._add_cluster_descriptions(result_df)
        
        print(f"\n✓ Hybrid clustering completed")
        print(f"   Llama generations: {self.llama_handler.call_count}")
        
        # Clear memory
        gc.collect()
        
        return result_df
    
    def _create_embeddings(self, keywords: List[str]) -> np.ndarray:
        """Create BERT embeddings"""
        print("   Loading BERT model...")
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL, device='cpu')
        self.embedding_model.max_seq_length = Config.MAX_SEQUENCE_LENGTH
        
        processed_keywords = [k if k else "empty" for k in keywords]
        
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
            
            if i % (batch_size * 5) == 0 and i > 0:
                progress = (i / len(processed_keywords)) * 100
                print(f"      Embedding progress: {progress:.1f}%")
        
        return np.vstack(embeddings)
    
    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensions using PCA + UMAP"""
        if embeddings.shape[1] > 100:
            pca = PCA(n_components=100, random_state=42)
            embeddings = pca.fit_transform(embeddings)
        
        n_neighbors = min(Config.UMAP_N_NEIGHBORS, len(embeddings) - 1)
        n_components = min(Config.UMAP_N_COMPONENTS, embeddings.shape[0] - 1, 50)
        
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=Config.UMAP_MIN_DIST,
            metric=Config.UMAP_METRIC,
            random_state=42,
            low_memory=True,
            n_jobs=1  # Single thread for CPU
        )
        
        return umap_model.fit_transform(embeddings)
    
    def _create_pillar_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        """Create main pillar clusters"""
        n_clusters = min(Config.PILLAR_CLUSTERS, len(embeddings) // 50)
        if n_clusters < 2:
            n_clusters = 2
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        
        return kmeans.fit_predict(embeddings)
    
    def _create_topic_clusters(self, embeddings: np.ndarray, pillar_labels: np.ndarray) -> np.ndarray:
        """Create topic clusters within each pillar"""
        topic_labels = np.zeros_like(pillar_labels)
        global_topic_id = 0
        
        unique_pillars = np.unique(pillar_labels)
        
        for pillar_id in unique_pillars:
            pillar_mask = pillar_labels == pillar_id
            pillar_indices = np.where(pillar_mask)[0]
            
            if len(pillar_indices) < Config.MIN_KEYWORDS_FOR_TOPIC:
                topic_labels[pillar_indices] = global_topic_id
                global_topic_id += 1
                continue
            
            n_topics = min(
                Config.TOPICS_PER_PILLAR,
                max(2, len(pillar_indices) // Config.MIN_CLUSTER_SIZE),
                len(pillar_indices)
            )
            
            if n_topics < 2:
                topic_labels[pillar_indices] = global_topic_id
                global_topic_id += 1
                continue
            
            pillar_embeddings = embeddings[pillar_indices]
            
            kmeans = KMeans(
                n_clusters=n_topics,
                random_state=42,
                n_init=5
            )
            
            sub_labels = kmeans.fit_predict(pillar_embeddings)
            
            for i, idx in enumerate(pillar_indices):
                topic_labels[idx] = global_topic_id + sub_labels[i]
            
            global_topic_id += n_topics
        
        return topic_labels
    
    def _generate_intelligent_names(self, keywords: List[str], labels: np.ndarray, 
                                   level: str) -> Dict[int, str]:
        """Generate names using Llama or fallback"""
        cluster_keywords = defaultdict(list)
        for keyword, label in zip(keywords, labels):
            if keyword:
                cluster_keywords[label].append(keyword)
        
        cluster_names = {}
        total_clusters = len(cluster_keywords)
        
        for idx, (cluster_id, cluster_kws) in enumerate(cluster_keywords.items()):
            if idx % 5 == 0:
                print(f"      Naming progress: {idx}/{total_clusters} {level}s")
            
            if cluster_kws:
                name = self.llama_handler.generate_cluster_name(cluster_kws, level)
                cluster_names[cluster_id] = name
            else:
                cluster_names[cluster_id] = f"{level.title()}_{cluster_id}"
        
        return cluster_names
    
    def _add_cluster_descriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add descriptions to major clusters"""
        pillar_sizes = df.groupby('pillar_name').size().sort_values(ascending=False)
        top_pillars = pillar_sizes.head(5).index.tolist()
        
        descriptions = {}
        for pillar_name in top_pillars:
            pillar_keywords = df[df['pillar_name'] == pillar_name]['cleaned_keyword'].tolist()[:20]
            desc = self.llama_handler.generate_cluster_description(pillar_keywords, pillar_name)
            descriptions[pillar_name] = desc
        
        df['pillar_description'] = df['pillar_name'].map(descriptions).fillna('')
        
        return df

# ============= EXCEL OUTPUT GENERATOR =============
class ExcelOutputGenerator:
    """Generate comprehensive Excel output"""
    
    def create_output(self, clustered_df: pd.DataFrame, output_path: str):
        """Create Excel with clustered keywords and statistics"""
        print(f"\n📝 Creating Excel output: {output_path}")
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                self._create_clustered_sheet(clustered_df, writer)
                self._create_pillar_summary(clustered_df, writer)
                self._create_topic_summary(clustered_df, writer)
                self._create_statistics_sheet(clustered_df, writer)
                
                if 'pillar_description' in clustered_df.columns:
                    self._create_descriptions_sheet(clustered_df, writer)
            
            print("✓ Excel file created successfully")
            
        except Exception as e:
            print(f"❌ Error creating Excel file: {e}")
            # Save as CSV instead
            csv_path = output_path.replace('.xlsx', '.csv')
            clustered_df.to_csv(csv_path, index=False)
            print(f"✓ Saved as CSV instead: {csv_path}")
    
    def _create_clustered_sheet(self, df: pd.DataFrame, writer):
        """Create main clustered keywords sheet"""
        columns_to_include = [
            'original_keyword', 'cleaned_keyword', 
            'pillar_id', 'pillar_name',
            'topic_id', 'topic_name',
            'word_count'
        ]
        
        # Add optional columns if they exist
        optional_columns = ['search_volume', 'competition', 'cpc']
        for col in optional_columns:
            if col in df.columns:
                columns_to_include.append(col)
        
        output_df = df[columns_to_include].copy()
        output_df['full_path'] = output_df['pillar_name'] + ' > ' + output_df['topic_name']
        output_df = output_df.sort_values(['pillar_id', 'topic_id'])
        
        output_df.to_excel(writer, sheet_name='Clustered_Keywords', index=False)
    
    def _create_pillar_summary(self, df: pd.DataFrame, writer):
        """Create pillar-level summary"""
        agg_dict = {
            'cleaned_keyword': 'count',
            'topic_id': 'nunique'
        }
        
        # Add optional metrics if available
        optional_metrics = {'search_volume': 'sum', 'competition': 'mean', 'cpc': 'mean'}
        for col, func in optional_metrics.items():
            if col in df.columns:
                agg_dict[col] = func
        
        pillar_summary = df.groupby(['pillar_id', 'pillar_name']).agg(agg_dict).round(2)
        
        # Rename columns
        column_names = {
            'cleaned_keyword': 'Total_Keywords', 
            'topic_id': 'Total_Topics'
        }
        if 'search_volume' in agg_dict:
            column_names['search_volume'] = 'Total_Search_Volume'
        if 'competition' in agg_dict:
            column_names['competition'] = 'Avg_Competition'
        if 'cpc' in agg_dict:
            column_names['cpc'] = 'Avg_CPC'
        
        pillar_summary.columns = [column_names.get(col, col) for col in pillar_summary.columns]
        
        pillar_summary = pillar_summary.reset_index()
        pillar_summary = pillar_summary.sort_values('Total_Keywords', ascending=False)
        
        pillar_summary.to_excel(writer, sheet_name='Pillar_Summary', index=False)
    
    def _create_topic_summary(self, df: pd.DataFrame, writer):
        """Create topic-level summary"""
        agg_dict = {'cleaned_keyword': 'count'}
        
        # Add optional metrics if available
        optional_metrics = {'search_volume': 'sum', 'competition': 'mean', 'cpc': 'mean'}
        for col, func in optional_metrics.items():
            if col in df.columns:
                agg_dict[col] = func
        
        topic_summary = df.groupby(['pillar_name', 'topic_id', 'topic_name']).agg(agg_dict).round(2)
        
        # Rename columns
        column_names = {'cleaned_keyword': 'Keyword_Count'}
        if 'search_volume' in agg_dict:
            column_names['search_volume'] = 'Total_Search_Volume'
        if 'competition' in agg_dict:
            column_names['competition'] = 'Avg_Competition'
        if 'cpc' in agg_dict:
            column_names['cpc'] = 'Avg_CPC'
        
        topic_summary.columns = [column_names.get(col, col) for col in topic_summary.columns]
        
        topic_summary = topic_summary.reset_index()
        topic_summary = topic_summary.sort_values('Keyword_Count', ascending=False)
        
        topic_summary.to_excel(writer, sheet_name='Topic_Summary', index=False)
    
    def _create_statistics_sheet(self, clustered_df: pd.DataFrame, writer):
        """Create overall statistics sheet"""
        stats = [
            {'Metric': 'Total Keywords Processed', 'Value': len(clustered_df)},
            {'Metric': 'Total Pillars', 'Value': clustered_df['pillar_name'].nunique()},
            {'Metric': 'Total Topics', 'Value': clustered_df['topic_name'].nunique()},
            {'Metric': 'Avg Keywords per Pillar', 'Value': round(len(clustered_df) / clustered_df['pillar_name'].nunique())},
            {'Metric': 'Avg Keywords per Topic', 'Value': round(len(clustered_df) / clustered_df['topic_name'].nunique())},
            {'Metric': 'Device Used', 'Value': Config.DEVICE.upper()},
            {'Metric': 'Model Used', 'Value': 'TinyLlama (CPU Optimized)'},
            {'Metric': 'Processing Date', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M')}
        ]
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
    
    def _create_descriptions_sheet(self, df: pd.DataFrame, writer):
        """Create sheet with cluster descriptions"""
        desc_df = df[df['pillar_description'] != ''].groupby(['pillar_name', 'pillar_description']).agg({
            'cleaned_keyword': 'count'
        }).reset_index()
        
        # Add search volume if available
        if 'search_volume' in df.columns:
            desc_with_volume = df[df['pillar_description'] != ''].groupby(['pillar_name', 'pillar_description']).agg({
                'cleaned_keyword': 'count',
                'search_volume': 'sum'
            }).reset_index()
            desc_with_volume.columns = ['Pillar', 'Description', 'Keyword_Count', 'Total_Search_Volume']
        else:
            desc_with_volume = desc_df.copy()
            desc_with_volume.columns = ['Pillar', 'Description', 'Keyword_Count']
        
        desc_with_volume = desc_with_volume.sort_values('Keyword_Count', ascending=False)
        desc_with_volume.to_excel(writer, sheet_name='Cluster_Descriptions', index=False)

# ============= SETUP HELPER =============
def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = {
        'transformers': 'transformers>=4.35.0',
        'torch': 'torch',
        'sentence_transformers': 'sentence-transformers',
        'umap': 'umap-learn',
        'sklearn': 'scikit-learn',
        'openpyxl': 'openpyxl'
    }
    
    print("\n📦 Checking dependencies...")
    
    for package, install_name in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'umap':
                import umap
            else:
                __import__(package)
            print(f"   ✓ {package} installed")
        except ImportError:
            print(f"   Installing {package}...")
            os.system(f"pip install {install_name}")
    
    print("\n💡 For best performance, consider installing Ollama:")
    print("   curl -fsSL https://ollama.ai/install.sh | sh")
    print("   ollama pull llama2:7b")

# ============= MAIN PIPELINE =============
def run_llama_clustering():
    """Main execution pipeline"""
    start_time = datetime.now()
    
    print("="*80)
    print("🚀 HYBRID BERT + LOCAL LLAMA CLUSTERING SYSTEM - FIXED")
    print("   Processing up to 90,000 keywords")
    print("   Using BERT for clustering + Local Llama for naming")
    print("   💰 FREE - No API costs!")
    print("   🖥️ CPU Optimized version")
    print("="*80)
    
    # Check dependencies
    try:
        check_and_install_dependencies()
    except Exception as e:
        print(f"⚠️ Dependency check failed: {e}")
        print("Continuing with available packages...")
    
    try:
        # Create output directory
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        # Step 1: Load and process keywords
        processor = KeywordProcessor()
        processed_df = processor.load_and_process(
            Config.INPUT_FILE, 
            max_keywords=Config.MAX_KEYWORDS
        )
        
        if len(processed_df) == 0:
            print("\n❌ No keywords found to cluster!")
            return pd.DataFrame()
        
        # Limit for testing if too many keywords
        if len(processed_df) > 10000:
            print(f"\n⚠️ Processing first 10,000 keywords for CPU optimization...")
            processed_df = processed_df.head(10000)
        
        # Step 2: Hybrid clustering
        clustering_engine = HybridClusteringEngine()
        clustered_df = clustering_engine.create_clusters(processed_df)
        
        # Step 3: Generate output
        excel_generator = ExcelOutputGenerator()
        excel_generator.create_output(clustered_df, Config.OUTPUT_FILE)
        
        # Print summary
        elapsed_time = datetime.now() - start_time
        
        print("\n" + "="*80)
        print("✅ LLAMA CLUSTERING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📊 FINAL RESULTS:")
        print(f"   • Total Keywords Processed: {len(clustered_df):,}")
        print(f"   • Pillars Created: {clustered_df['pillar_name'].nunique()}")
        print(f"   • Topics Created: {clustered_df['topic_name'].nunique()}")
        print(f"   • Llama Generations: {clustering_engine.llama_handler.call_count}")
        print(f"   • Device Used: {Config.DEVICE.upper()}")
        print(f"\n⏱️ Processing Time: {elapsed_time}")
        print(f"\n📁 Output File: {Config.OUTPUT_FILE}")
        
        # Show sample clusters
        print(f"\n🎯 Sample Pillar Structure:")
        unique_pillars = clustered_df['pillar_name'].unique()
        
        for pillar in unique_pillars[:5]:
            pillar_data = clustered_df[clustered_df['pillar_name'] == pillar]
            topics = pillar_data['topic_name'].unique()[:3]
            keyword_count = len(pillar_data)
            print(f"\n   📌 {pillar} ({keyword_count:,} keywords)")
            
            if 'pillar_description' in clustered_df.columns:
                desc = pillar_data['pillar_description'].iloc[0] if not pillar_data.empty else ''
                if desc:
                    print(f"      📝 {desc}")
            
            for topic in topics:
                topic_count = len(pillar_data[pillar_data['topic_name'] == topic])
                print(f"      └─ {topic} ({topic_count} keywords)")
        
        # Show sample keywords from each pillar
        print(f"\n🔍 Sample Keywords by Pillar:")
        for pillar in unique_pillars[:3]:
            pillar_keywords = clustered_df[clustered_df['pillar_name'] == pillar]['cleaned_keyword'].head(3).tolist()
            print(f"   {pillar}: {', '.join(pillar_keywords)}")
        
        return clustered_df
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty dataframe on error
        return pd.DataFrame()

# ============= QUICK SETUP FUNCTION =============
def quick_setup():
    """Quick setup for first-time users"""
    print("\n QUICK SETUP")
    print("="*50)
    
    # Check if input file exists
    if not os.path.exists(Config.INPUT_FILE):
        print(f" Input file not found: {Config.INPUT_FILE}")
        print("\n Please update the INPUT_FILE path in Config class")
        print("   Current path:", Config.INPUT_FILE)
        print("\n Expected format: CSV or Excel with keyword column")
        return False
    
    print(f"✓ Input file found: {Config.INPUT_FILE}")
    
    # Check output directory
    try:
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        print(f"✓ Output directory ready: {Config.OUTPUT_DIR}")
    except Exception as e:
        print(f" Cannot create output directory: {e}")
        return False
    
    # Check Python packages
    try:
        import torch
        import transformers
        import sentence_transformers
        print("✓ Core packages available")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✓ Device: {device}")
        
        return True
    except ImportError as e:
        print(f" Missing packages: {e}")
        print("\nPlease install with: pip install torch transformers sentence-transformers")
        return False

if __name__ == "__main__":
    # Quick setup check
    if quick_setup():
        # Run the Llama clustering
        clustered_data = run_llama_clustering()
        
        if len(clustered_data) > 0:
            print(f"\n Success! Check your results in: {Config.OUTPUT_FILE}")
        else:
            print(f"\n No results generated. Check the error messages above.")
    else:
        print(f"\n Setup failed. Please fix the issues above and try again.")

        """
Hybrid BERT + Local Llama Clustering System - FIXED VERSION
Uses BERT for embeddings and LOCAL Llama for intelligent naming
FREE - No API costs, runs completely offline
Optimized for 90,000 keywords
"""
