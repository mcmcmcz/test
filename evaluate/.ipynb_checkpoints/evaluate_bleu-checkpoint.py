import nltk
from bert_score import score
import torch
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import AutoModel, AutoTokenizer
import json
import re
from typing import List, Dict, Tuple
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_load import *

def load_data(data_dir):
    folders = [folder for folder in os.listdir(data_dir) if 'pdf' in folder]
    all_data = {}
    for folder in folders:
        path = data_dir + folder + '/' + 'introduction_results.json'
        if os.path.exists(path): 
            with open(path, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            if 'sections' in data.keys() and 'introduction' in data.keys():
                all_data[folder] = data
    return all_data

class TextSimilarityEvaluator:
    """Text Similarity Evaluator"""
    
    def __init__(self, local_model_path: str = "./bert-large-uncased", nltk_data_path: str = "./nltk_data"):
        """
        Initialize text similarity evaluator
        
        Args:
            local_model_path: Path to BERT model and tokenizer
            nltk_data_path: Path to NLTK data
        """
        self.local_model_path = local_model_path
        
        # Set NLTK data path
        if nltk_data_path:
            nltk.data.path.append(nltk_data_path)
        
        # Initialize model and tokenizer
        self._load_models()
    
    def _load_models(self):
        """Load necessary models and tokenizer"""
        # Load NLTK tokenizer
        try:
            self.nltk_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        except:
            print("Warning: Could not load NLTK punkt tokenizer")
        
        # Check and load local BERT model
        if not os.path.exists(self.local_model_path):
            raise FileNotFoundError(f"Model path not found: {self.local_model_path}")
            
        # Check GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
            
        self.model = AutoModel.from_pretrained(self.local_model_path)
        self.model.to(self.device)  # Move model to GPU
        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)

    def calculate_single_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Calculate similarity scores between two texts
        
        Args:
            text1: First text (reference text)
            text2: Second text (generated text)
            
        Returns:
            Dictionary containing BLEU, METEOR, BERT similarity, and Rouge-L scores
        """
        # Text preprocessing
        text1_processed = text1.strip().lower()
        text2_processed = text2.strip().lower()
        
        tokens1 = word_tokenize(text1_processed)
        tokens2 = word_tokenize(text2_processed)
        
        
        # Calculate BERT similarity
        def get_bert_embedding(text):
            inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            # Move inputs to GPU
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)
        
        embedding1 = get_bert_embedding(text1_processed)
        embedding2 = get_bert_embedding(text2_processed)
        # Calculate similarity and move back to CPU
        bert_similarity = float(torch.nn.functional.cosine_similarity(embedding1, embedding2).cpu())
        # bert_similarity = 0
        scores = {
            'BERT_Similarity': bert_similarity,
        }
        
        return scores

def load_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data

def get_results(truth, generated, item):
    truth_introduction = truth[item].strip()
    generated_introduction = generated[item]['introduction']
    return truth_introduction, generated_introduction
    
# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Text similarity evaluation with CSV export')
    parser.add_argument('--generated_path', type = str, default = '../writing_agents_results_cvpr/gpt_elaborate/', help = 'path of generated introduction')
    args = parser.parse_args()

    generated_path = args.generated_path
    print(generated_path)
    # Load data
    # loader = DataLoader("/home/mczhang/zmc-dl/LLM/NTP/paper_data/acl/2025/main")
    loader = DataLoader("/home/mczhang/zmc-dl/LLM/NTP/paper_data/cvpr/2025")
    all_data = loader.load_all()
    all_items = []
    for item in all_data.keys():
        if all_data[item].abstract:
            all_items.append(item)
    print(len(all_items))
    generated = load_data(f'{args.generated_path}')
    # outlines = load_json('../evaluate/outline_acl_2025_main.json')
    print(len(generated))
    truth = {}
    for item in all_items:
        truth[item] = all_data[item].introduction

    local_model_path: str = "/home/mczhang/zmc-dl/LLM/NTP/bert-large-uncased"
    nltk_data_path: str = "/home/mczhang/zmc-dl/LLM/NTP/nltk_data"
    evaluator = TextSimilarityEvaluator(local_model_path, nltk_data_path)
    
    intro_bert_score = []    
    for item in tqdm(all_items):
        truth_introduction, generated_introduction = get_results(truth, generated, item)
        intro_score = evaluator.calculate_single_similarity(truth_introduction, generated_introduction)
        intro_bert_score.append(intro_score['BERT_Similarity'])

    print(f"BERT Similarity: {np.round(np.mean(intro_bert_score),3)}")
    
