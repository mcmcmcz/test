import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
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
from rouge_score import rouge_scorer
from tqdm import tqdm
from data_load import *

def load_data(data_dir):
    folders = [folder for folder in os.listdir(data_dir) if folder.startswith('20')]
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
        
        # Initialize Rouge evaluator
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
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
        
        # Calculate BLEU score
        bleu_score = sentence_bleu([tokens1], tokens2, weights=(0.25,0.25,0.25,0.25))
        # bleu_score =
        
        # Calculate METEOR score
        meteor = meteor_score([tokens1], tokens2)
        
        # Calculate Rouge-L score
        rouge_scores = self.rouge_scorer.score(text1, text2)
        rouge_l_f1 = rouge_scores['rougeL'].fmeasure
        
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
            'BLEU': float(bleu_score),
            'METEOR': float(meteor),
            'BERT_Similarity': bert_similarity,
            'Rouge_L': float(rouge_l_f1)
        }
        
        return scores

def load_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data

def get_results(truth, generated, item):
    truth_introduction = truth[item]['introduction'].strip()
    truth_sections = truth[item]['sections']
    generated_introduction = generated[item]['introduction']
    generated_sections = generated[item]['sections']
    return truth_introduction, truth_sections, generated_introduction, generated_sections
    
# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Text similarity evaluation with CSV export')
    parser.add_argument('--generated_path', type = str, default = '../writing_agents_results/ft/', help = 'path of generated introduction')
    args = parser.parse_args()
    
    sections_name = ['background', 'problem_statement', 'method_overview', 'contributions']
    generated_path = args.generated_path
    print(generated_path)
    outlines = load_json('../evaluate/outline_acl_2025_main.json')
    # Load data
    loader = DataLoader("/home/mczhang/zmc-dl/LLM/NTP/paper_data/acl/2025/main")
    all_data = loader.load_all()
    all_items = []
    for item in all_data.keys():
        if all_data[item].abstract:
            all_items.append(item)
    print(len(all_items))
    generated = load_data(f'{args.generated_path}')
    print(len(generated))
    truth = {}
    for item in all_items:
        truth[item] = outlines[item]

    local_model_path: str = "/home/mczhang/zmc-dl/LLM/NTP/bert-large-uncased"
    nltk_data_path: str = "/home/mczhang/zmc-dl/LLM/NTP/nltk_data"
    evaluator = TextSimilarityEvaluator(local_model_path, nltk_data_path)
    
    # Lists for collecting results
    detailed_results = []
    
    # Introduction metrics
    intro_bleu = []
    intro_meteor = []
    intro_bert_score = []
    intro_rouge_l = []
    
    # Sections metrics
    sections_bleu = []
    sections_meteor = []
    sections_bert_score = []
    sections_rouge_l = []
    
    for item in tqdm(all_items):
        truth_introduction, truth_sections, generated_introduction, generated_sections = get_results(truth, generated, item)
        
        # Evaluate Introduction
        intro_score = evaluator.calculate_single_similarity(truth_introduction, generated_introduction)
        intro_bleu.append(intro_score['BLEU'])
        intro_meteor.append(intro_score['METEOR'])
        intro_bert_score.append(intro_score['BERT_Similarity'])
        intro_rouge_l.append(intro_score['Rouge_L'])
        
        # Evaluate Sections
        section_scores = []
        for section_name in sections_name:
            # Check if section exists in both and is not empty
            truth_section = truth_sections.get(section_name, "").strip()
            generated_section = generated_sections.get(section_name, "").strip()
            
            if truth_section and generated_section:  # Only evaluate if both sections have content
                score = evaluator.calculate_single_similarity(truth_section, generated_section)
                section_scores.append(score)
                print(f"Evaluated section: {section_name}")
            elif truth_section and not generated_section:
                        scores = {
                            'BLEU': float(0),
                            'METEOR': float(0),
                            'BERT_Similarity': 0,
                            'Rouge_L': float(0)
                        }
            else:
                print(f"Skipped empty section: {section_name}")
        
        if section_scores:
            avg_section_score = {
                'BLEU': np.mean([s['BLEU'] for s in section_scores]),
                'METEOR': np.mean([s['METEOR'] for s in section_scores]),
                'BERT_Similarity': np.mean([s['BERT_Similarity'] for s in section_scores]),
                'Rouge_L': np.mean([s['Rouge_L'] for s in section_scores])
            }
            sections_bleu.append(avg_section_score['BLEU'])
            sections_meteor.append(avg_section_score['METEOR'])
            sections_bert_score.append(avg_section_score['BERT_Similarity'])
            sections_rouge_l.append(avg_section_score['Rouge_L'])
        
        # Collect results
        result_row = {
            'generated_path': generated_path,
            # Introduction results
            'intro_BLEU': intro_score['BLEU'],
            'intro_METEOR': intro_score['METEOR'],
            'intro_BERT_Similarity': intro_score['BERT_Similarity'],
            'intro_Rouge_L': intro_score['Rouge_L'],
            # Sections results
            'sections_BLEU': avg_section_score['BLEU'] if section_scores else 0,
            'sections_METEOR': avg_section_score['METEOR'] if section_scores else 0,
            'sections_BERT_Similarity': avg_section_score['BERT_Similarity'] if section_scores else 0,
            'sections_Rouge_L': avg_section_score['Rouge_L'] if section_scores else 0
        }
        detailed_results.append(result_row)
    
    # Print Introduction results
    print("="*60)
    print("📝 Introduction Evaluation Results")
    print("="*60)
    print("Similarity Metrics:")
    print(f"BLEU: {np.round(np.mean(intro_bleu),3)}")
    print(f"METEOR: {np.round(np.mean(intro_meteor),3)}")
    print(f"BERT Similarity: {np.round(np.mean(intro_bert_score),3)}")
    print(f"Rouge-L: {np.round(np.mean(intro_rouge_l),3)}")
    
    # Print Sections results
    print("\n" + "="*60)
    print("📊 Sections Evaluation Results")
    print("="*60)
    if sections_bleu:  # Only print if we have valid section scores
        print("Similarity Metrics:")
        print(f"BLEU: {np.round(np.mean(sections_bleu),3)}")
        print(f"METEOR: {np.round(np.mean(sections_meteor),3)}")
        print(f"BERT Similarity: {np.round(np.mean(sections_bert_score),3)}")
        print(f"Rouge-L: {np.round(np.mean(sections_rouge_l),3)}")
        print(f"Number of evaluated sections: {len(sections_bleu)}")
    else:
        print("No valid sections were evaluated")
    
    # Create DataFrame with average results
    avg_row = {
        'generated_path': generated_path,
        # Introduction averages
        'intro_BLEU': np.mean(intro_bleu),
        'intro_METEOR': np.mean(intro_meteor),
        'intro_BERT_Similarity': np.mean(intro_bert_score),
        'intro_Rouge_L': np.mean(intro_rouge_l),
        # Sections averages
        # Sections averages (only if we have valid sections)
        'sections_BLEU': np.mean(sections_bleu) if sections_bleu else 0,
        'sections_METEOR': np.mean(sections_meteor) if sections_meteor else 0,
        'sections_BERT_Similarity': np.mean(sections_bert_score) if sections_bert_score else 0,
        'sections_Rouge_L': np.mean(sections_rouge_l) if sections_rouge_l else 0,
        'evaluated_sections_count': len(sections_bleu) if sections_bleu else 0
    }
    
    df = pd.DataFrame([avg_row])
    
    # Output path
    output_path = "../results/all_lin.csv"
    
    # Ensure results directory exists
    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Check if file exists and append or create new
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False, index=False, encoding='utf-8')
        print(f"\n📁 Results appended to: {output_path}")
    else:
        df.to_csv(output_path, mode='w', header=True, index=False, encoding='utf-8')
        print(f"\n📁 Results created and saved to: {output_path}")
    
    print(f"📊 Average evaluation results (based on {len(detailed_results)} samples)")
    print(f"📈 Total {len(df.columns)} evaluation metrics")