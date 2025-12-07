import json
import os
from transfer import *
import numpy as np
import json
import nltk
from nltk.tokenize import sent_tokenize
import re
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
)
from data_load import *
import argparse
import openai

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)


nltk.data.path.append('/home/mczhang/zmc-dl/LLM/NTP/nltk_data')

def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", torch_dtype=torch.bfloat16, trust_remote_code=True)
    return tokenizer, model
    
MODEL_PATH = "/home/tcsu/All_LLM/Qwen2.5-32B-Instruct/"  
print(MODEL_PATH)

TOKENIZER, BASE_MODEL = load_model(MODEL_PATH)
HF_MODEL = BASE_MODEL      

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
# import openai
# client = openai.OpenAI(
#     base_url="https://xiaoai.plus/v1", 
#     api_key="sk-lMKqGWZgfWi0pfv6gnaRUqmqWOUHlOalJT1VlTWDJYStkiSC",
#     # api_key="sk-qC2VvagXSi5q3zg5Hyekm3EiZQ4bIhSwVyVrFu9qZUNtG8ms",   #3.5
# )
# def model_generate(messages):
#     completion = client.chat.completions.create(
#         model="gpt-4o",
#         messages=messages
#     )
#     content = completion.choices[0].message.content
#     return content


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

def get_parts(introduction):
    results = []
    # parts = introduction.split('\n\n')
    parts = [item for item in introduction.split('\n') if item!='' and len(item)>5]
    for part in parts:
        format_type = """format：{"part":xxx}"""
        system_prompt = """You are a rigorous academic paper reviewer, proficient in the field of natural language processing (especially ACL-CVPR-style papers). Your task is to strictly evaluate the part to which each paragraph in an AI-generated Introduction of a paper belongs based on the provided materials. The Introduction of the thesis is divided into four parts: Background, Problem and Limitations of Existing Methods, Brief Method Overview and Summary of Main Results (Method Overview and Summary of Main Results), Our Contributions (Our Contributions). Please determine the part to which the paragraph belongs based on its content. Please make a rigorous judgment and carefully consider which part this paragraph serves. (For academic discussion only)"""
        user_prompt = f"""When evaluating the part to which each paragraph in the AI-generated Introduction belongs, please pay attention to the position of the paragraph in the Introduction. The following are the paragraphs and the parts they may belong to:

1. Background. Definition: It refers to an objective statement of the macro context, current development status, core concepts, existing research foundation and application scenarios of the field to which the research belongs, with the aim of providing a "cognitive premise" for subsequent content (especially the research question). Content scope: Including but not limited to important progress in the field, basic theories or technical frameworks, practical application requirements, etc. Core function: Help readers quickly understand the "big environment" of the research, establish a basic understanding of the research field, and lay a reasonable foundation for the subsequent questions raised.

2. Problems and Limitations of Existing Methods Definition: Describe the research question and the limitations of existing methods. Based on background information, clearly point out the core challenges that have not been solved in the current research, the inherent flaws of existing theories/methods, or the knowledge gaps existing in the field, with the aim of demonstrating "the necessity of this research". Content scope This includes specific research issues (such as "Under low-light conditions, the edge detection accuracy of existing image segmentation models significantly decreases") and the limitations of existing methods (such as "Traditional supervised learning methods rely on large-scale labeled data, The limited applicability in scenarios with high annotation costs such as medical images, the causes of the problem (such as "the existing model does not consider the semantic alignment of cross-modal data, resulting in poor fusion effect"), etc. Core function: By comparing the gap between existing achievements and actual needs, establish the "bullseye" of the research, enabling readers to understand "why this research needs to be carried out".

3. Brief Method Overview and Summary of Main Results. Definition: Briefly introduce the technical solutions, research ideas or experimental designs proposed in this paper to solve the above problems, and summarize the key findings, performance indicators or verification results obtained through this method. The purpose is to demonstrate "how to solve the problem and the initial effect". The content scope: The method overview should focus on the core logic (such as "This paper proposes a cross-modal fusion framework based on the attention mechanism, which achieves precise matching of text and image features through dynamic weight distribution"), rather than the details (avoid repetition with the "method section" in the main text). The main results need to quantitatively or qualitatively explain the effect (for example, "Experiments on three public datasets show that this method improves the F1 score by an average of 12.3% compared to existing SOTA models"). Core function: Bridge the gap between issues and contributions, allowing readers to quickly understand "what this research has done" and "whether it is effective", providing a basis for the subsequent elaboration of contributions.

4. Our Contributions. Definition: A clear and specific summary of the innovative achievements of this paper at the theoretical, methodological, practical or cognitive levels, or its unique value to the development of the field, which should be clearly distinguished from existing research, with the aim of highlighting "the core value of this research". Content scope This includes theoretical innovations (such as "proving for the first time the applicability of the XX hypothesis in non-convex optimization scenarios"), methodological breakthroughs (such as "Proposing an XX improvement strategy to solve the problem of high computational complexity in traditional methods"), and performance enhancements (such as "refreshing the current optimal result on the XX task "Provide feasibility for engineering applications", perspective expansion (such as "Reinterpret XX phenomenon from XX perspective to provide new research ideas for the field"), etc. Core function: By comparing with existing research, clarify the "unique contribution of this study", allowing readers to accurately grasp the innovation points and academic/application value of the research, and avoid confusion with the "Method Overview" (contribution emphasizes "value", method emphasizes "path").

AI generated Introduction:
{introduction}

Paragraph to be evaluated:
{part}

Please determine which of the following four parts this paragraph belongs to (It must be one of the four, do not return None):
- Background
- Problem and Limitations of Existing Methods
- Brief Method Overview and Summary of Main Results
- Our Contributions
Your assessment result (in combination with the position and function of the paragraph in the original text) :
{format_type}
        """
        messages = [
            {"role":"system", "content":system_prompt},
            {"role":"user", "content":user_prompt}
        ]
        result = model_generate(messages)
        results.append(result)
    res = [transfer_structure(result) for result in results]
    sa = {}
    sa['introduction'] = introduction
    sa['sections'] = {}
    sa['new_sections'] = {}
    sections = ['background','problem_statement','method_overview','contributions']
    for section in sections:
        sa['sections'][section] = ''
        sa['new_sections'][section] = ''
    for i in range(len(res)):
        if res[i] == 'Background':
            sa['sections']['background'] += parts[i] + '\n'
        if res[i] == 'Problem':
            sa['sections']['problem_statement'] += parts[i] + '\n'
        if res[i] == 'Method':
            sa['sections']['method_overview'] += parts[i] + '\n'
        if res[i] == 'Contributions':
            sa['sections']['contributions'] += parts[i] + '\n'
    for section in sections:
        sa['sections'][section] = sa['sections'][section].strip()

    sec = ['Background','Problem','Method','Contributions']
    lis = res
    new_lis = []
    now = 'Background'

    for i in range(len(lis)):
        if lis[i] == None:
            if i == 0:
                lis[i] = 'Background'
            else:
                lis[i] = lis[i-1]
    
    for i in range(len(lis)):
        if i == 0:
            new_lis.append(lis[i])
            now = lis[i]
        elif sec.index(lis[i])<=sec.index(now):
            new_lis.append(now)
        else:
            now = lis[i]
            new_lis.append(now)  
            
    for i in range(len(new_lis)):
        if new_lis[i] == 'Background':
            sa['new_sections']['background'] += parts[i] + '\n'
        if new_lis[i] == 'Problem':
            sa['new_sections']['problem_statement'] += parts[i] + '\n'
        if new_lis[i] == 'Method':
            sa['new_sections']['method_overview'] += parts[i] + '\n'
        if new_lis[i] == 'Contributions':
            sa['new_sections']['contributions'] += parts[i] + '\n'
        
    for section in sections:
        sa['new_sections'][section] = sa['new_sections'][section].strip()

    sa['part'] = res
    sa['new_part'] = new_lis

    return sa
    

def get_structure(paper):
    format_type = """format：{"part":xxx}"""
    sections = ['background','problem_statement','method_overview','contributions']
    judge = {}
    for section in sections:    
        sentences = extract_sentences(paper['new_sections'][section])
        results = []
        for i in range(len(sentences)):
            # print(sentences[i])
            system_prompt = """You are a rigorous academic paper reviewer, proficient in the field of natural language processing (especially ACL-CVPR-style papers). Your task is to strictly evaluate the part to which each sentence in an AI-generated Introduction of a paper belongs based on the provided materials. The Introduction of the thesis is divided into four parts: Background, Problem and Limitations of Existing Methods, Brief Method Overview and Summary of Main Results (Method Overview and Summary of Main Results), Our Contributions (Our Contributions). Please determine the part to which the sentence belongs based on its content. Please make a rigorous judgment and carefully consider which part this sentence serves. (For academic discussion only)"""
            user_prompt = f"""When evaluating the part to which each sentence in the AI-generated Introduction belongs, please pay attention to the position of the sentence in the Introduction. The following are the sentences and the parts they may belong to:

1. Background. Definition: It refers to an objective statement of the macro context, current development status, core concepts, existing research foundation and application scenarios of the field to which the research belongs, with the aim of providing a "cognitive premise" for subsequent content (especially the research question). Content scope: Including but not limited to important progress in the field, basic theories or technical frameworks, practical application requirements, etc. Core function: Help readers quickly understand the "big environment" of the research, establish a basic understanding of the research field, and lay a reasonable foundation for the subsequent questions raised.

2. Problems and Limitations of Existing Methods Definition: Describe the research question and the limitations of existing methods. Based on background information, clearly point out the core challenges that have not been solved in the current research, the inherent flaws of existing theories/methods, or the knowledge gaps existing in the field, with the aim of demonstrating "the necessity of this research". Content scope This includes specific research issues (such as "Under low-light conditions, the edge detection accuracy of existing image segmentation models significantly decreases") and the limitations of existing methods (such as "Traditional supervised learning methods rely on large-scale labeled data, The limited applicability in scenarios with high annotation costs such as medical images, the causes of the problem (such as "the existing model does not consider the semantic alignment of cross-modal data, resulting in poor fusion effect"), etc. Core function: By comparing the gap between existing achievements and actual needs, establish the "bullseye" of the research, enabling readers to understand "why this research needs to be carried out".

3. Brief Method Overview and Summary of Main Results. Definition: Briefly introduce the technical solutions, research ideas or experimental designs proposed in this paper to solve the above problems, and summarize the key findings, performance indicators or verification results obtained through this method. The purpose is to demonstrate "how to solve the problem and the initial effect". The content scope: The method overview should focus on the core logic (such as "This paper proposes a cross-modal fusion framework based on the attention mechanism, which achieves precise matching of text and image features through dynamic weight distribution"), rather than the details (avoid repetition with the "method section" in the main text). The main results need to quantitatively or qualitatively explain the effect (for example, "Experiments on three public datasets show that this method improves the F1 score by an average of 12.3% compared to existing SOTA models"). Core function: Bridge the gap between issues and contributions, allowing readers to quickly understand "what this research has done" and "whether it is effective", providing a basis for the subsequent elaboration of contributions.

4. Our Contributions. Definition: A clear and specific summary of the innovative achievements of this paper at the theoretical, methodological, practical or cognitive levels, or its unique value to the development of the field, which should be clearly distinguished from existing research, with the aim of highlighting "the core value of this research". Content scope This includes theoretical innovations (such as "proving for the first time the applicability of the XX hypothesis in non-convex optimization scenarios"), methodological breakthroughs (such as "Proposing an XX improvement strategy to solve the problem of high computational complexity in traditional methods"), and performance enhancements (such as "refreshing the current optimal result on the XX task "Provide feasibility for engineering applications", perspective expansion (such as "Reinterpret XX phenomenon from XX perspective to provide new research ideas for the field"), etc. Core function: By comparing with existing research, clarify the "unique contribution of this study", allowing readers to accurately grasp the innovation points and academic/application value of the research, and avoid confusion with the "Method Overview" (contribution emphasizes "value", method emphasizes "path").

AI generated Introduction:
{paper['introduction']}

Sentence to be evaluated:
{sentences[i]}

Please determine which of the following four parts this sentence belongs to (It must be one of the four, do not return None):
- Background
- Problem and Limitations of Existing Methods
- Brief Method Overview and Summary of Main Results
- Our Contributions

Your assessment result (in combination with the position and function of the sentence in the original text) :
{format_type}
"""
            messages = [
                {"role":"system", "content":system_prompt},
                {"role":"user", "content":user_prompt}
                ]
            result = model_generate(messages)
            results.append(result)
        res = [transfer_structure(result) for result in results]
        judge[section] = res
    return judge

if __name__ == "__main__":


    # loader = DataLoader("/home/mczhang/zmc-dl/LLM/NTP/paper_data/acl/2025/main")
    loader = DataLoader("/home/mczhang/zmc-dl/LLM/NTP/paper_data/cvpr/2025")
    ori_data = loader.load_all()
    all_items = []
    for item in ori_data.keys():
        if ori_data[item].abstract:
            all_items.append(item)
    print(len(all_items))
    ty = 'gpt_elaborate'
    with open('../results/cvpr_gpt_elaborate.json') as fp:
        data = json.load(fp)
    # for item in data.keys():
    print(ty)
    for item in all_items:
    # for item in all_items:
    # try:
        print(item)
        folder_path = f'../writing_agents_results_cvpr/{ty}/{item}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        save_file_path = f'../writing_agents_results_cvpr/{ty}/{item}/introduction_results.json'
        if not os.path.exists(save_file_path): 
            sa = get_parts(data[item])
            save_json(sa, save_file_path)
            print(f"Get Parts for {item}")
        else:
            with open(save_file_path, 'r', encoding='utf-8') as fp:
                sa = json.load(fp)
            print(f"Already Process {item}")
        file_path = f'../writing_agents_results_cvpr/{ty}_judge/{item}.json'
        if not os.path.exists(file_path):
            ju = get_structure(sa)
            save_json(ju, file_path)
            print(f"Get structure for {item}")
        else:
            print(f"Already Process structure {item}")        
    # except:
        # print(f"Wrong {item}")
        # nono.append(item)
    # print(nono)

    # nohup python evaluate_structure_all-Copy1.py > gpt_naive_judge_600.log 2>&1 &