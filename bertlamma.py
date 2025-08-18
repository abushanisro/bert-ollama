"""
Hybrid BERT + Local Llama Clustering System
Uses BERT for embeddings and LOCAL Llama for intelligent naming
FREE - No API costs, runs completely offline
Optimized for 90,000 keywords
"""

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
    INPUT_FILE = '/home/admin1/Downloads/demo_crypto/FInal list of crypto terms.xlsx'
    OUTPUT_DIR = '/home/admin1/Downloads/demo_crypto/output'
    OUTPUT_FILE = f'{OUTPUT_DIR}/llama_clusters_90k.xlsx'
    
    # Llama Model Configuration (Choose one)
    # Option 1: Smaller, faster model (7B parameters)
    LLAMA_MODEL = "TheBloke/Llama-2-7B-Chat-GPTQ"  # Quantized for efficiency
    
    # Option 2: Medium model (13B parameters) - Better quality
    # LLAMA_MODEL = "TheBloke/Llama-2-13B-Chat-GPTQ"
    
    # Option 3: Using Ollama (if installed)
    USE_OLLAMA = False  # Set to True if you have Ollama installed
    OLLAMA_MODEL = "llama2:7b"
    
    # BERT Model for embeddings
    EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
    
    # Processing Configuration
    BATCH_SIZE = 1000
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
    
    # Llama Processing
    LLAMA_SAMPLE_SIZE = 20    # Keywords to sample for naming
    LLAMA_MAX_LENGTH = 100     # Max tokens for response
    LLAMA_TEMPERATURE = 0.3   # Lower = more focused
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_4BIT = True  # Use 4-bit quantization for memory efficiency

# ============= LLAMA HANDLER =============
class LlamaHandler:
    """Handle local Llama model for intelligent naming"""
    
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
        """Setup Llama using Hugging Face Transformers"""
        print("   Loading Llama model (this may take a few minutes)...")
        
        try:
            # Configure quantization for memory efficiency
            if Config.USE_4BIT and Config.DEVICE == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    Config.LLAMA_MODEL,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # Load without quantization (needs more memory)
                self.model = AutoModelForCausalLM.from_pretrained(
                    Config.LLAMA_MODEL,
                    torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                Config.LLAMA_MODEL,
                trust_remote_code=True
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                temperature=Config.LLAMA_TEMPERATURE,
                do_sample=True,
                top_p=0.95,
                device=Config.DEVICE if Config.DEVICE == "cuda" else -1
            )
            
            print("   ✓ Llama model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load full model: {e}")
            print("   Trying alternative lightweight model...")
            self._setup_lightweight_model()
    
    def _setup_lightweight_model(self):
        """Fallback to a lighter model if main model fails"""
        try:
            # Use a smaller model like Phi-2 or TinyLlama
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            print(f"   Loading lightweight model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=256,
                temperature=0.3,
                device=Config.DEVICE if Config.DEVICE == "cuda" else -1
            )
            
            print("   ✓ Lightweight model loaded successfully")
            
        except Exception as e:
            print(f" Could not load any model: {e}")
            print("   Falling back to TF-IDF naming only")
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
            print(f" Ollama not available: {e}")
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
        """Create prompt for pillar naming"""
        keywords_str = ', '.join(keywords[:20])
        
        prompt = f"""<s>[INST] You are a keyword categorization expert. Analyze these keywords and create a concise category name.

Keywords: {keywords_str}

Rules:
- Maximum 2 words
- Professional and clear
- Capture the main theme
- Be specific, not generic

Reply with ONLY the category name, nothing else. [/INST]

Category name:"""
        
        return prompt
    
    def _create_topic_prompt(self, keywords: List[str]) -> str:
        """Create prompt for topic naming"""
        keywords_str = ', '.join(keywords[:20])
        
        prompt = f"""<s>[INST] You are a keyword clustering expert. Create a specific topic name for these keywords.

Keywords: {keywords_str}

Rules:
- 2-4 words maximum
- Specific and descriptive
- Professional terminology
- Actionable and searchable

Reply with ONLY the topic name, nothing else. [/INST]

Topic name:"""
        
        return prompt
    
    def _generate_with_transformers(self, prompt: str) -> str:
        """Generate using Transformers pipeline"""
        result = self.pipeline(
            prompt,
            max_new_tokens=20,
            temperature=Config.LLAMA_TEMPERATURE,
            do_sample=True,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract generated text
        generated = result[0]['generated_text']
        
        # Remove the prompt from response
        if prompt in generated:
            generated = generated.replace(prompt, '').strip()
        
        # Extract the actual name (after "name:")
        if "name:" in generated.lower():
            parts = generated.lower().split("name:")
            if len(parts) > 1:
                generated = parts[-1].strip()
        
        return generated.strip()
    
    def _generate_with_ollama(self, prompt: str) -> str:
        """Generate using Ollama"""
        import ollama
        
        response = self.ollama_client.generate(
            model=Config.OLLAMA_MODEL,
            prompt=prompt,
            options={
                "temperature": Config.LLAMA_TEMPERATURE,
                "max_tokens": 20,
                "top_p": 0.95
            }
        )
        
        return response['response'].strip()
    
    def _clean_name(self, name: str, max_words: int) -> str:
        """Clean and validate generated name"""
        # Remove special characters and extra spaces
        name = re.sub(r'[^\w\s-]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Remove common prefixes/suffixes
        remove_phrases = [
            'category name', 'topic name', 'cluster', 'group',
            'here is', 'the name is', 'answer:', 'response:'
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
        name = ' '.join(word.capitalize() for word in name.split())
        
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
            tfidf = TfidfVectorizer(max_features=5, stop_words='english')
            tfidf_matrix = tfidf.fit_transform(keywords)
            feature_names = tfidf.get_feature_names_out()
            
            # Get top terms
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-3:][::-1]
            
            terms = [feature_names[idx].title() for idx in top_indices]
            
            if level == 'pillar':
                return ' '.join(terms[:2])
            else:
                return ' '.join(terms[:3])
                
        except:
            return f"{level.title()}_Group_{self.call_count}"
    
    def generate_cluster_description(self, keywords: List[str], name: str) -> str:
        """Generate a brief description for a cluster"""
        if not self.use_ollama and self.pipeline is None:
            return f"Keywords related to {name.lower()}"
        
        sample = keywords[:15] if len(keywords) > 15 else keywords
        keywords_str = ', '.join(sample)
        
        prompt = f"""<s>[INST] Create a one-sentence description for the topic "{name}" based on these keywords:

Keywords: {keywords_str}

The description should be under 100 characters and explain what this cluster represents.

Reply with ONLY the description. [/INST]

Description:"""
        
        try:
            if self.use_ollama:
                desc = self._generate_with_ollama(prompt)
            else:
                desc = self._generate_with_transformers(prompt)
            
            # Clean description
            desc = desc.strip()
            desc = re.sub(r'^description:?\s*', '', desc, flags=re.IGNORECASE)
            
            if len(desc) > 100:
                desc = desc[:97] + "..."
            
            return desc if desc else f"Keywords related to {name.lower()}"
            
        except:
            return f"Keywords related to {name.lower()}"

# ============= KEYWORD PROCESSOR =============
class KeywordProcessor:
    """Process and clean keywords"""
    
    def load_and_process(self, file_path: str, max_keywords: int = 90000) -> pd.DataFrame:
        """Load and process keywords"""
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
        processed_df = self._process_keywords(df, keyword_col)
        
        return processed_df
    
    def _find_keyword_column(self, df: pd.DataFrame) -> str:
        """Find the keyword column"""
        possible_names = ['keyword', 'keywords', 'query', 'term', 'search_term']
        
        for col in df.columns:
            if col.lower() in possible_names:
                return col
        
        for col in df.columns:
            if df[col].dtype == 'object':
                return col
        
        return df.columns[0]
    
    def _process_keywords(self, df: pd.DataFrame, keyword_col: str) -> pd.DataFrame:
        """Clean and process keywords"""
        print("\n🔧 Processing keywords...")
        
        all_keywords = []
        
        for idx, row in df.iterrows():
            keyword = str(row[keyword_col])
            
            # Light cleaning
            cleaned = keyword.strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            if not cleaned:
                cleaned = keyword
            
            all_keywords.append({
                'original_keyword': keyword,
                'cleaned_keyword': cleaned,
                'search_volume': row.get('search_volume', 0) if 'search_volume' in row else 0,
                'competition': row.get('competition', 0) if 'competition' in row else 0,
                'cpc': row.get('cpc', 0.0) if 'cpc' in row else 0.0
            })
        
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
        
        print("\n Starting Hybrid BERT + Llama clustering...")
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
        print("   Step 5: Generating intelligent names with Llama...")
        result_df = df.copy()
        result_df['pillar_id'] = pillar_labels
        result_df['topic_id'] = topic_labels
        
        # Generate pillar names with Llama
        pillar_names = self._generate_intelligent_names(keywords, pillar_labels, 'pillar')
        result_df['pillar_name'] = [pillar_names[label] for label in pillar_labels]
        
        # Generate topic names with Llama
        topic_names = self._generate_intelligent_names(keywords, topic_labels, 'topic')
        result_df['topic_name'] = [topic_names[label] for label in topic_labels]
        
        # Step 6: Generate descriptions for major clusters
        print("   Step 6: Generating cluster descriptions...")
        result_df = self._add_cluster_descriptions(result_df)
        
        print(f"\n✓ Hybrid clustering completed")
        print(f"   Llama generations: {self.llama_handler.call_count}")
        
        # Clear GPU memory if used
        if Config.DEVICE == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        return result_df
    
    def _create_embeddings(self, keywords: List[str]) -> np.ndarray:
        """Create BERT embeddings"""
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
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
            
            if i % (batch_size * 10) == 0 and i > 0:
                progress = (i / len(processed_keywords)) * 100
                print(f"      Embedding progress: {progress:.1f}%")
        
        return np.vstack(embeddings)
    
    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensions using PCA + UMAP"""
        if embeddings.shape[1] > 100:
            pca = PCA(n_components=100, random_state=42)
            embeddings = pca.fit_transform(embeddings)
        
        umap_model = umap.UMAP(
            n_neighbors=min(Config.UMAP_N_NEIGHBORS, len(embeddings) - 1),
            n_components=min(Config.UMAP_N_COMPONENTS, embeddings.shape[0] - 1),
            min_dist=Config.UMAP_MIN_DIST,
            metric=Config.UMAP_METRIC,
            random_state=42,
            low_memory=True
        )
        
        return umap_model.fit_transform(embeddings)
    
    def _create_pillar_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        """Create main pillar clusters"""
        n_clusters = min(Config.PILLAR_CLUSTERS, len(embeddings) // 100)
        
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
        """Generate names using Llama"""
        cluster_keywords = defaultdict(list)
        for keyword, label in zip(keywords, labels):
            if keyword:
                cluster_keywords[label].append(keyword)
        
        cluster_names = {}
        total_clusters = len(cluster_keywords)
        
        for idx, (cluster_id, cluster_kws) in enumerate(cluster_keywords.items()):
            if idx % 10 == 0:
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
            pillar_keywords = df[df['pillar_name'] == pillar_name]['cleaned_keyword'].tolist()[:30]
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
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            self._create_clustered_sheet(clustered_df, writer)
            self._create_pillar_summary(clustered_df, writer)
            self._create_topic_summary(clustered_df, writer)
            self._create_statistics_sheet(clustered_df, writer)
            
            if 'pillar_description' in clustered_df.columns:
                self._create_descriptions_sheet(clustered_df, writer)
        
        print("✓ Excel file created successfully")
    
    def _create_clustered_sheet(self, df: pd.DataFrame, writer):
        """Create main clustered keywords sheet"""
        output_df = df[[
            'original_keyword', 'cleaned_keyword', 
            'pillar_id', 'pillar_name',
            'topic_id', 'topic_name',
            'word_count', 'search_volume', 'competition', 'cpc'
        ]].copy()
        
        output_df['full_path'] = output_df['pillar_name'] + ' > ' + output_df['topic_name']
        output_df = output_df.sort_values(['pillar_id', 'topic_id'])
        
        output_df.to_excel(writer, sheet_name='Clustered_Keywords', index=False)
    
    def _create_pillar_summary(self, df: pd.DataFrame, writer):
        """Create pillar-level summary"""
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
        stats = [
            {'Metric': 'Total Keywords Processed', 'Value': len(clustered_df)},
            {'Metric': 'Total Pillars', 'Value': clustered_df['pillar_name'].nunique()},
            {'Metric': 'Total Topics', 'Value': clustered_df['topic_name'].nunique()},
            {'Metric': 'Avg Keywords per Pillar', 'Value': round(len(clustered_df) / clustered_df['pillar_name'].nunique())},
            {'Metric': 'Avg Keywords per Topic', 'Value': round(len(clustered_df) / clustered_df['topic_name'].nunique())},
            {'Metric': 'Device Used', 'Value': Config.DEVICE.upper()},
            {'Metric': 'Model Used', 'Value': 'Llama (Local)'}
        ]
        
        stats_df = pd.DataFrame(stats)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
    
    def _create_descriptions_sheet(self, df: pd.DataFrame, writer):
        """Create sheet with cluster descriptions"""
        desc_df = df[df['pillar_description'] != ''].groupby(['pillar_name', 'pillar_description']).agg({
            'cleaned_keyword': 'count',
            'search_volume': 'sum'
        }).reset_index()
        
        desc_df.columns = ['Pillar', 'Description', 'Keyword_Count', 'Total_Search_Volume']
        desc_df = desc_df.sort_values('Keyword_Count', ascending=False)
        
        desc_df.to_excel(writer, sheet_name='Cluster_Descriptions', index=False)

# ============= SETUP HELPER =============
def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = {
        'transformers': 'transformers>=4.35.0',
        'accelerate': 'accelerate',
        'bitsandbytes': 'bitsandbytes',  # For quantization
        'sentencepiece': 'sentencepiece',  # For tokenizer
        'protobuf': 'protobuf',
        'scipy': 'scipy'  # For UMAP
    }
    
    print("\nChecking dependencies...")
    
    for package, install_name in required_packages.items():
        try:
            __import__(package)
            print(f"   ✓ {package} installed")
        except ImportError:
            print(f"   Installing {package}...")
            os.system(f"pip install {install_name}")
    
    # Optional: Install Ollama
    print("\nFor best performance, consider installing Ollama:")
    print("   curl -fsSL https://ollama.ai/install.sh | sh")
    print("   ollama pull llama2:7b")

# ============= MAIN PIPELINE =============
def run_llama_clustering():
    """Main execution pipeline"""
    start_time = datetime.now()
    
    print("="*80)
    print("HYBRID BERT + LOCAL LLAMA CLUSTERING SYSTEM")
    print("   Processing 90,000 keywords")
    print("   Using BERT for clustering + Local Llama for naming")
    print(" FREE - No API costs!")
    print("="*80)
    
    # Check dependencies
    check_and_install_dependencies()
    
    try:
        # Create output directory
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        # Step 1: Load and process keywords
        processor = KeywordProcessor()
        processed_df = processor.load_and_process(
            Config.INPUT_FILE, 
            max_keywords=Config.MAX_KEYWORDS
        )
        
        # Step 2: Hybrid clustering
        if len(processed_df) > 0:
            clustering_engine = HybridClusteringEngine()
            clustered_df = clustering_engine.create_clusters(processed_df)
            
            # Step 3: Generate output
            excel_generator = ExcelOutputGenerator()
            excel_generator.create_output(clustered_df, Config.OUTPUT_FILE)
            
            # Print summary
            elapsed_time = datetime.now() - start_time
            
            print("\n" + "="*80)
            print("LLAMA CLUSTERING COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"\n FINAL RESULTS:")
            print(f"   • Total Keywords Processed: {len(clustered_df):,}")
            print(f"   • Pillars Created: {clustered_df['pillar_name'].nunique()}")
            print(f"   • Topics Created: {clustered_df['topic_name'].nunique()}")
            print(f"   • Llama Generations: {clustering_engine.llama_handler.call_count}")
            print(f"   • Device Used: {Config.DEVICE.upper()}")
            print(f"\nProcessing Time: {elapsed_time}")
            print(f"\n Output File: {Config.OUTPUT_FILE}")
            
            # Show sample clusters
            print(f"\n Sample Pillar Structure:")
            for pillar in clustered_df['pillar_name'].unique()[:5]:
                pillar_data = clustered_df[clustered_df['pillar_name'] == pillar]
                topics = pillar_data['topic_name'].unique()[:3]
                keyword_count = len(pillar_data)
                print(f"\n {pillar} ({keyword_count:,} keywords)")
                
                if 'pillar_description' in clustered_df.columns:
                    desc = pillar_data['pillar_description'].iloc[0] if not pillar_data.empty else ''
                    if desc:
                        print(f" {desc}")
                
                for topic in topics:
                    topic_count = len(pillar_data[pillar_data['topic_name'] == topic])
                    print(f"      └─ {topic} ({topic_count} keywords)")
            
        else:
            print("\n No keywords found to cluster!")
        
        return clustered_df if len(processed_df) > 0 else pd.DataFrame()
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Run the Llama clustering
    clustered_data = run_llama_clustering()

    """
