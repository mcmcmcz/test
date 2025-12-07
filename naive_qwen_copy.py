'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-06-18 16:36:59
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-07-20 14:41:14
FilePath: /zmc-dl/LLM/NTP/outline_baseline/naive_qwen.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessor
from peft import PeftModel
import re

def remove_references(text):
    
    # 处理各种引用格式
    # 1. 单个或多个作者(包括et al.)加年份的组合
    text = re.sub(r'\(\s*(?:[A-Za-z\s\-\.]+(?:\s+et\s+al\.)?(?:\s*;\s*[A-Za-z\s\-\.]+(?:\s+et\s+al\.)?)*),\s*\d{4}(?:,\s*\d{4})*\s*\)', '', text)
    
    # 2. 处理作者名称可能包含"and"或"&"的情况
    text = re.sub(r'\(\s*(?:[A-Za-z\s\-\.]+(?:\s+(?:and|&)\s+[A-Za-z\s\-\.]+)(?:\s*;\s*[A-Za-z\s\-\.]+(?:\s+(?:and|&)\s+[A-Za-z\s\-\.]+)?)*),\s*\d{4}(?:,\s*\d{4})*\s*\)', '', text)
    
    # 3. 处理组织名称作为作者的情况 (如 Council of Europe)
    text = re.sub(r'\(\s*(?:[A-Za-z\s\-\.]+(?:\s+of\s+[A-Za-z\s\-\.]+)?),\s*\d{4}(?:,\s*\d{4})*\s*\)', '', text)
    
    # 4. 处理多个引用组合的情况 (作者1, 年份; 作者2, 年份)
    text = re.sub(r'\(\s*(?:[A-Za-z\s\-\.]+(?:\s+et\s+al\.)?),\s*\d{4}(?:,\s*\d{4})*(?:\s*;\s*[A-Za-z\s\-\.]+(?:\s+et\s+al\.)?),\s*\d{4}(?:,\s*\d{4})*\s*\)', '', text)
    
    # 5. 处理只有组织缩写的情况 (如 ALTE, 2005; 2011)
    text = re.sub(r'\(\s*[A-Z]+,\s*\d{4}(?:;\s*\d{4})*\s*\)', '', text)
    
    # 6. 处理任何剩余的标准引用格式
    text = re.sub(r'\(\s*[A-Za-z\s\-\.]+,\s*\d{4}(?:,\s*\d{4})*\s*\)', '', text)
    
    # 去掉图片及其描述
    text = re.sub(r'!\[\]\S*\.jpg', '', text)
    text = re.sub(r'Figure \d+:.*?\n', '', text)
    text = re.sub(r'Table \d+:.*?\n', '', text)

    # 去掉多余的换行符和空格
    text = re.sub(r'\n\s*\n', '\n', text).strip()
    # text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\([^()]*\d{4}[^()]*\)', '', text)
    
    # 2. 常见学术引用的特定模式
    text = re.sub(r'\(\s*[A-Za-z\s\-\.]+(?:\s+et\s+al\.)?(?:\s*and\s*[A-Za-z\s\-\.]+)?,\s*\d{4}(?:[a-z])?(?:,\s*\d{4}(?:[a-z])?)*\s*\)', '', text)
    
    # 3. 用分号分隔的多作者模式
    text = re.sub(r'\(\s*(?:[A-Za-z\s\-\.]+(?:\s+et\s+al\.)?(?:,\s*\d{4}(?:[a-z])?)+(?:\s*;\s*[A-Za-z\s\-\.]+(?:\s+et\s+al\.)?(?:,\s*\d{4}(?:[a-z])?)+)*)\s*\)', '', text)
    
    # 4. 组织名称引用模式
    text = re.sub(r'\(\s*[A-Za-z][A-Za-z\s\-\.]+(?:\s+of\s+[A-Za-z\s\-\.]+)?,\s*\d{4}(?:[a-z])?(?:,\s*\d{4}(?:[a-z])?)*\s*\)', '', text)
    
    # 5. 缩写和首字母缩略词模式
    text = re.sub(r'\(\s*[A-Z][A-Za-z\s\-\.]*,\s*\d{4}(?:[a-z])?(?:,\s*\d{4}(?:[a-z])?)*\s*\)', '', text)
    
    # 6. 特定模式，处理类似(Wen et al., 2015a,b)的引用
    text = re.sub(r'\(\s*[A-Za-z\s\-\.]+(?:\s+et\s+al\.)?(?:,\s*\d{4}[a-z]?)+\s*\)', '', text)
    
    # 7. 处理多年份用分号分隔的引用
    text = re.sub(r'\(\s*[A-Za-z\s\-\.]+(?:\s+et\s+al\.)?(?:,\s*\d{4}(?:[a-z])?)+(?:\s*;\s*\d{4}(?:[a-z])?)+\s*\)', '', text)
    
    # 8. 处理包含特殊字符的作者名(如Dusˇek and Jurcˇı´cˇek, 2016)
    text = re.sub(r'\(\s*[A-Za-zˇ´\s\-\.]+(?:\s+and\s+[A-Za-zˇ´\s\-\.]+)?,\s*\d{4}(?:[a-z])?(?:,\s*\d{4}(?:[a-z])?)*\s*\)', '', text)
    text = text.replace("1 Introduction \n","").strip()
    return text

def count_lenth(text):
    """
    count lenth
    """
    lenth = len(text.split())
    return lenth

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="sequential")

    return tokenizer, model
    

class BanTokensProcessor(LogitsProcessor):
    def __init__(self, banned_token_ids):
        self.banned_token_ids = banned_token_ids
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 将禁止的token的概率设为负无穷，这样它们就不会被选中
        scores[:, self.banned_token_ids] = -float("inf")
        return scores

# 在生成时应用处理器
def model_generate(tokenizer, model, messages, banned_token_ids=[90, 92, 6667, 335, 7288]):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    logits_processor = [BanTokensProcessor(banned_token_ids)]
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        ** model_inputs,
        logits_processor = logits_processor,
        max_new_tokens=2048,
        repetition_penalty=1.2,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def get_user_prompt(abstract, figures, tables, ref):
    with open("../temp.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
    user_prompt = f"""
    {prompt}\n
    Given abstract: {abstract}\n
    Given figures: {figures}\n
    Given tables: {tables}\n
    Given references(These baseline references only exist in experiments): {ref}\n
    Please write a paper Introduction based on the above information. The Introduction should be well-structured, coherent, and follow the conventions of academic writing. Ensure that the introduction is original and does not contain any references to other works or authors. The paper should be written in a formal tone and should be suitable for submission to an academic conference.\n
    """
    return user_prompt


def format_messages(data):
    system = "You are an expert in academic paper writing. Please proceed with the academic writing in accordance with the relevant requirements. Please just write the Introduction section of the paper, and do not include any references to other works or authors. The paper should be written in a formal tone and should be suitable for submission to an academic conference."
    abstract = data['abstract']
    figures = data['figures']
    all_figures = [figures[item]['img_caption'] for item in list(figures.keys())]
    # print(all_figures)
    tables = data['tables']
    references = data['ref']
    messages = [
        {"role":"system", "content": system},
        {"role":"user", "content": get_user_prompt(abstract, all_figures, tables, references)},
    ]
    return messages

def load_process_data(json_path: str = "process_data.json") -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """示例使用"""
    # 初始化系统 - 可以选择使用API或本地模型
    # use_closeai = False  # 设置为False使用本地模型

    # model_id = "/home/tcsu/Qwen2.5-7B-Instruct/"  # 当use_closeai=False时需要提s供
    # model_id = "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/dpo_0628/"  # 当use_closeai=False时需要提供
    # model_id = "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/dpo_0708/"
    # model_id = "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/dpo_0714/"
    model_id = "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/dpo_0719-3/"
    tokenizer, model= load_model(model_id)
    # model = PeftModel.from_pretrained(model, "/home/mczhang/zmc-dl/LLM/NTP/sft_model", adapter_name="writing")
    # model = PeftModel.from_pretrained(model, "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/lora_results_sft_0625_qwen2.5-7b", adapter_name="writing")
    # model = PeftModel.from_pretrained(model, "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/saves/qwen/lora/dpo/checkpoint-315", adapter_name="writing")
    print(model_id)

    with open('results/case_infos.json', 'r', encoding='utf-8') as f:
        case_infos = json.load(f)
    cases = list(case_infos.keys())
    data_dir = '../paper_data/acl/2024/main/'
    # paper_lis = os.listdir(data_dir)
    evidences = {}
    # 使用加载的数据创建PaperEvidence对象
    for paper in cases:
        file_path = data_dir + paper + '/process_data.json'
        if os.path.exists(file_path):
            print(f"Processing {file_path}")
            data = load_process_data(file_path)
            messages = format_messages(data)
            intro = model_generate(tokenizer, model, messages)
            messages.append({"role": "assistant", "content": intro})
            lenth = count_lenth(intro)
            print('ori_lenth',lenth)
            if lenth>1200:
                messages.append({"role": "user", "content": f"The Introduction you wrote is too long. Please revise it—cutting unnecessary content—so it is between 600 and 1,200 words. Make sure to maintain the original meaning and structure."})
                intro = model_generate(tokenizer, model, messages)
            elif lenth<450:
                messages.append({"role": "user", "content": f"The Introduction you wrote is too short. Please revise it—adding necessary context, examples, and elaboration—so it falls within the 600–1,200-word range. Make sure to maintain the original meaning and structure."})
                intro = model_generate(tokenizer, model, messages)
            intro = remove_references(intro)
            lenth = count_lenth(intro)
            print('after_lenth',lenth)
            evidences[paper] = intro

    with open('results/qwen_dpo_lenth.json','w',encoding='utf-8') as fp:
        json.dump(evidences,fp)

        
        
if __name__ == "__main__":
    main() 