import json
import os
from transfer import *
import numpy as np
import nltk
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import re
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import warnings
import argparse
warnings.filterwarnings("ignore")
from data_load import *

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


nltk.data.path.append('/home/mczhang/zmc-dl/LLM/NTP/nltk_data')

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


def load_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)


def is_x_dot(s):
    pattern = r'^\d+\.$'
    return bool(re.match(pattern, s))
    
def extract_sentences(text):
    # 使用正则表达式去除小标题（假设小标题是以任意数量的“#”开头的行）
    cleaned_text = re.sub(r'^#+.*\n?', '', text, flags=re.MULTILINE)
    # cleaned_text = cleaned_text.replace('\n','')
    
    # 使用nltk的sent_tokenize进行句子分割
    sentences = sent_tokenize(cleaned_text)
    
    # 去除空字符串并去除句子首尾的空白字符
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    pro_sentences = []
    temp = ''
    for sen in sentences:
        if not is_x_dot(sen):
            pro_sentences.append(temp + sen)
            temp = ''
        else:
            temp = sen
            
    
    return pro_sentences



def calculate_sentence_similarity(sentence1: str, sentence2: str, 
                                model_name: str = "/home/mczhang/zmc-dl/LLM/NTP/SBERT") -> float:
    """
    使用SentenceBERT计算两个句子的相似度
    
    Args:
        sentence1 (str): 第一个句子
        sentence2 (str): 第二个句子
        model_name (str): SentenceBERT模型名称，默认为"all-MiniLM-L6-v2"
    
    Returns:
        float: 两个句子的余弦相似度 (范围: -1 到 1)
    """
    # 加载SentenceBERT模型并移动到GPU
    model = SentenceTransformer(model_name)
    model = model.to(device)
    
    # 编码句子
    with torch.no_grad():
        embeddings = model.encode([sentence1, sentence2], convert_to_tensor=True)
        embeddings = embeddings.to(device)
    
    # 计算余弦相似度
    similarity = torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))
    
    return float(similarity.cpu().numpy())
    
def batch_calculate_similarity(sentence_pairs: List[tuple], 
                             model_name: str = "/home/mczhang/zmc-dl/LLM/NTP/SBERT",
                             batch_size: int = 32) -> List[float]:
    """
    批量计算多对句子的相似度
    
    Args:
        sentence_pairs (List[tuple]): 句子对列表，每个元素为(sentence1, sentence2)
        model_name (str): SentenceBERT模型名称
        batch_size (int): 批处理大小，默认为32
    
    Returns:
        List[float]: 相似度列表
    """
    # 加载模型并移动到GPU
    model = SentenceTransformer(model_name)
    model = model.to(device)
    
    # 准备所有句子
    sentences1, sentences2 = zip(*sentence_pairs)
    all_sentences = list(sentences1) + list(sentences2)
    
    # 批量编码所有句子
    similarities = []
    with torch.no_grad():
        # 计算所有句子的嵌入
        embeddings = []
        for i in range(0, len(all_sentences), batch_size):
            batch = all_sentences[i:i + batch_size]
            batch_embeddings = model.encode(batch, convert_to_tensor=True)
            embeddings.append(batch_embeddings)
        
        # 将所有嵌入连接在一起
        all_embeddings = torch.cat(embeddings)
        
        # 分离出第一组和第二组句子的嵌入
        embeddings1 = all_embeddings[:len(sentences1)]
        embeddings2 = all_embeddings[len(sentences1):]
        
        # 计算相似度
        for emb1, emb2 in zip(embeddings1, embeddings2):
            similarity = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
            similarities.append(float(similarity.cpu().numpy()))
            
            # 打印句子对（如果需要）
            # print(f"计算相似度: {sentences1[len(similarities)-1]}")
            # print(f"          与: {sentences2[len(similarities)-1]}")
            # print(f"     相似度: {similarities[-1]:.4f}")
    
    return similarities

def get_simi(ori_introduction, gen_introduction):
    gen_sentences = extract_sentences(gen_introduction)
    ori_sentences = extract_sentences(ori_introduction)
    all_smi = []
    for sentence in gen_sentences:
        simi = batch_calculate_similarity([(sentence, ori_sentences[i]) for i in range(len(ori_sentences))])
        all_smi.append(np.mean(simi))
    print('均值',np.mean(all_smi))
    return all_smi

if __name__ == "__main__":
    loader = DataLoader("/home/mczhang/zmc-dl/LLM/NTP/paper_data/acl/2025/main")
    ori_data = loader.load_all()
    # ft = load_data('../writing_agents_results_gpt/gpt_elaborate/')
    data_path ='../writing_agents_results_gpt/gpt_elaborate/'
    gpt_file = [file for file in os.listdir(data_path) if file.endswith('.json')]
    print(len(gpt_file))
    # print(len(ft))
    # import shutil
    # c = 0
    # for item in ft.keys():
    #     # if item not in ori_data.keys():
    #     if item not in ori_data.keys():
    #         # c += 1
    #         # file_path = f'/home/mczhang/zmc-dl/LLM/NTP/outline_baseline/writing_agents_results_gpt/gpt_elaborate/{item}.json'
    #         # os.remove(file_path)
    #         # # shutil.rmtree(file_path)
    #         # print(f"File removed {file_path} successfully")
    # print(c)


        

        