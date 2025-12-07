'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-06-18 16:36:59
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-08-24 19:02:23
FilePath: /zmc-dl/LLM/NTP/outline_baseline/naive_qwen.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import os
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
)
from peft import PeftModel
import re
from tqdm import tqdm


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", torch_dtype=torch.bfloat16, trust_remote_code=True)
    return tokenizer, model



# MODEL_PATH = "/home/tcsu/Qwen2.5-7B-Instruct/" 
MODEL_PATH = "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/qwen_sft_0902"
print(MODEL_PATH)

TOKENIZER, BASE_MODEL = load_model(MODEL_PATH)
HF_MODEL = BASE_MODEL       # 先给一个默认值，后面可被替换

chinese_token_ids = set()
for token, token_id in TOKENIZER.get_vocab().items():
    decoded = TOKENIZER.decode([token_id], skip_special_tokens=True)
    if re.search(r'[\u4e00-\u9fff]', decoded):  # 只要包含中文
        chinese_token_ids.add(token_id)
print(f"共找到 {len(chinese_token_ids)} 个解码后包含中文的 token")

class StrictChineseSuppressionLogitsProcessor(LogitsProcessor):
    def __init__(self, chinese_ids):
        self.chinese_ids = chinese_ids

    def __call__(self, input_ids, scores):
        # 直接将这些 token 的 logits 设为极小值
        scores[:, list(self.chinese_ids)] = -1e9
        return scores
        
global chinese_suppressor
chinese_suppressor = StrictChineseSuppressionLogitsProcessor(chinese_token_ids)

def model_generate(messages):

    text = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = TOKENIZER([text], return_tensors="pt").to(HF_MODEL.device)
    generated_ids = HF_MODEL.generate(
        **model_inputs,
        max_new_tokens=2304,
        logits_processor=[chinese_suppressor],
        # repetition_penalty=1.3,
        # length_penalty = 1.0
    )
    generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
    response = TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def get_user_prompt(title,abstract, figures, tables, ref):
    with open("../temp.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
    user_prompt = f"""{prompt}
    Given title: {title}
    Given abstract: {abstract}
    Given figures: {figures}
    Given tables: {tables}
    Given references(These baseline references only exist in experiments): {ref}
    Please write a paper Introduction based on the above information. The Introduction should be well-structured, coherent, and follow the conventions of academic writing. Ensure that the introduction is original and does not contain any references to other works or authors. The paper should be written in a formal tone and should be suitable for submission to an academic conference.\n
    """
    return user_prompt


def format_messages(data, high):
    
    system = "You are an expert in academic paper writing. Please proceed with the academic writing in accordance with the relevant requirements. Please just write the Introduction section of the paper, and do not include any references to other works or authors. The paper should be written in a formal tone and should be suitable for submission to an academic conference."
    title = data['title']
    abstract = data['abstract']
    figures = data['figures']
    tables = data['tables']
    references = data['ref']
    if not high:
        messages = [
            {"role":"system", "content": system},
            {"role":"user", "content": get_user_prompt(title, abstract, figures, tables, references)},
        ]
        return messages
    else:
        messages = [
            {f"role":"user","content": f"""You are an AI assistant tasked with generating a detailed and well-structured "Introduction" section of a research paper based on the provided title, abstract, and research materials. The abstract of the paper outlines its main objectives, methods, and potential contributions. Effectively integrate the given information to establish a clear research context, articulate the significance of existing gaps, and explicitly highlight the paper's methods and results as well as how it addresses these gaps through its novel contributions, and finally state the contributions.

**Important Format Requirements**:
- Your response MUST consist of EXACTLY FOUR PARAGRAPHS for the "Introduction".
- DO NOT deviate from this four-paragraph structure.
- Each paragraph must be between 100 and 150 words, totaling approximately 600 words.

**Structure**:
1. Paragraph 1: Broad overview of the research area, contextual insights from related materials, significance of the topic.
2. Paragraph 2: Specific problem or gap identified, supported by related materials.
3. Paragraph 3: Novel contributions of the target paper, including its methods and results, and how it addresses the gaps.
4. Paragraph 4: Summary of significance, potential impact, and research purpose.

**Style and Content Requirements**:
- Maintain a formal academic tone.
- Be as coherent and concise as possible, and directly related to the title and abstract.
- Use transitional phrases effectively.

**Citation Instructions**:
- Do not mention any citations. For example, "(Smith et al.)".

Target Paper:
Title: {title}
Abstract: {abstract}
Figures: {figures}
Tables: {tables}
References(These baseline references only exist in experiments): {references}

**Introduction**:"""}
        ]
        return messages

def load_process_data(json_path: str = "process_data.json") -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """示例使用"""
    data_dir = '../paper_data/cvpr/2025/'
    cases = os.listdir(data_dir)
    # paper_lis = os.listdir(data_dir)
    evidences = {}
    # 使用加载的数据创建PaperEvidence对象
    for paper in tqdm(cases):
        file_path = data_dir + paper + '/processed_data.json'
        if os.path.exists(file_path):
            print(f"Processing {file_path}")
            data = load_process_data(file_path)
            messages = format_messages(data, high=False)
            # print(messages)
            intro = model_generate(messages)
            evidences[paper] = intro
    with open('writing_agents_results_cvpr/naive_sft.json','w',encoding='utf-8') as fp:
        json.dump(evidences,fp)

        
        
if __name__ == "__main__":
    main() 