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
            if 'introduction' in data.keys():
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
    cleaned_text = re.sub(r'^#+.*\n?', '', text, flags=re.MULTILINE)
    # cleaned_text = cleaned_text.replace('\n','')

    sentences = sent_tokenize(cleaned_text)

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

def get_simi(ori_introduction, gen_sentences):

    # gen_sentences = extract_sentences(gen_introduction)
    ori_sentences = extract_sentences(ori_introduction)
    # print(len(gen_sentences),len(ori_sentences))
    if len(gen_sentences)>0 and len(ori_sentences)>0:
        all_smi = []
        for sentence in gen_sentences:
            simi = batch_calculate_similarity([(sentence, ori_sentences[i]) for i in range(len(ori_sentences))])
            all_smi.append(np.max(simi))
        return all_smi
    elif len(gen_sentences)>0 and len(ori_sentences)==0:
        return [0 for i in range(len(gen_sentences))]
    else:
        return []


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Text similarity evaluation with CSV export')
    parser.add_argument('--generated_type', type = str, default = 'gpt_elaborate', help = 'path of generated introduction')
    args = parser.parse_args()
    gen_type = args.generated_type
    
    # loader = DataLoader("/home/mczhang/zmc-dl/LLM/NTP/paper_data/acl/2025/main")
    loader = DataLoader("/home/mczhang/zmc-dl/LLM/NTP/paper_data/cvpr/2025")
    ori_data_all = loader.load_all()
    all_items = []
    for item in ori_data_all.keys():
        if ori_data_all[item].abstract:
            all_items.append(item)

    gen_data = load_data(f'../writing_agents_results_cvpr/{gen_type}/')
    
    with open('cvpr_outline_2025.json','r',encoding='utf-8') as fp:
        ori_data = json.load(fp)
    results = {}


    # print(lenth)

    # sections identify
    SECTIONS = ['Background','Problem and Limitations of Existing Methods','Brief Method Overview and Summary of Main Results','Our Contributions'] 
    sections = ['background','problem_statement','method_overview','contributions']

    simis = {}
    for item in tqdm(all_items[::-1]):
        print(item)
        structure_data = load_json(f'../writing_agents_results_cvpr/{gen_type}_judge/{item}.json')
        structure_weight = structure_data['background'] + structure_data['problem_statement'] + structure_data['method_overview'] + structure_data['contributions']
        # count_1 = [part not in ['Method','Contributions'] for part in structure_data['background']]
        # count_2 = [part not in ['Method','Contributions'] for part in structure_data['problem_statement']]
        # count_3 = [part not in ['Background','Problem'] for part in structure_data['method_overview']]
        # count_4 = [part not in ['Background','Problem'] for part in structure_data['contributions']]
        count_1 = [part not in ['Method','Contributions','Problem'] for part in structure_data['background']]
        count_2 = [part not in ['Method','Contributions','Background'] for part in structure_data['problem_statement']]
        count_3 = [part not in ['Background','Problem','Contributions'] for part in structure_data['method_overview']]
        count_4 = [part not in ['Background','Problem','Method'] for part in structure_data['contributions']]
        # structure_weight = count_1 + count_2 + count_3 + count_4
        counts = [count_1,count_2,count_3,count_4]
        structure_weight = []
        # four part
        

        if gen_type not in ['base','ft','stage','qwen_outline','stage_ta','stage_content_only']:
            lacking = 0
            for s in sections[:3]:
                if gen_data[item]['new_sections'][s] == '':
                    lacking += 1
            print(lacking)
                    
            simi = [] 
            for i in [0,1,2,3]:
                if sections[i] in gen_data[item]['new_sections'].keys() and SECTIONS[i] in ori_data[item]['sections'].keys():
                    ori_part = ori_data[item]['sections'][SECTIONS[i]]
                    gen_part = extract_sentences(gen_data[item]['new_sections'][sections[i]])
                    structure_weight += counts[i]
                    temp_simi = get_simi(ori_part,gen_part)
                    simi += temp_simi
            simi = [simi[i]*structure_weight[i] for i in range(len(structure_weight))]
        else:
            lacking = 0
            for s in sections[:3]:
                if gen_data[item]['sections'][s] == '':
                    lacking += 1
            print(lacking)
            simi = [] 
            for i in [0,1,2,3]:
                if sections[i] in gen_data[item]['sections'].keys() and SECTIONS[i] in ori_data[item]['sections'].keys():
                    ori_part = ori_data[item]['sections'][SECTIONS[i]]
                    gen_part = extract_sentences(gen_data[item]['sections'][sections[i]])
                    structure_weight += counts[i]
                    temp_simi = get_simi(ori_part,gen_part)
                    simi += temp_simi
            simi = [simi[i]*structure_weight[i] for i in range(len(structure_weight))]



        lacking_pen = (4-lacking)/4
        print(np.mean(simi)*lacking_pen)
        if lacking_pen<1:
            print(np.mean(simi))
        simis[item] = np.mean(simi)*lacking_pen
        print(np.round(np.mean(list(simis.values())),3))        
    save_json(simis, f'evaluate_log/simi/{gen_type}_consistency.json')
    print(np.round(np.mean(list(simis.values())),3))  
    with open('temp.txt','w',encoding='utf-8') as fw:
        fp.write(str(np.round(np.mean(list(simis.values())),3)))
        