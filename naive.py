import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re
from tqdm import tqdm

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="sequential")
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    return tokenizer, model, terminators
    
def model_generate(tokenizer, model, terminators, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048, 
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=terminators)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(response)
    return response


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
    # 初始化系统 - 可以选择使用API或本地模型
    # use_closeai = False  # 设置为False使用本地模型

    # model_id = "/home/tcsu/Qwen2.5-7B-Instruct/"  # 当use_closeai=False时需要提s供
    # model_id = "/home/mczhang/zmc-dl/LLM/Meta-Llama-3.1-8B-Instruct"
    model_id = "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/llama_sft_0902"
    # model_id = "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/dpo_0628/"  # 当use_closeai=False时需要提供
    # model_id = "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/dpo_0708/"
    # model_id = "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/dpo_0714/"
    # model_id = "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/dpo_0718-3/checkpoint-1625/"
    tokenizer, model, terminators = load_model(model_id)
    # model = PeftModel.from_pretrained(model, "/home/mczhang/zmc-dl/LLM/NTP/sft_model", adapter_name="writing")
    # model = PeftModel.from_pretrained(model, "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/lora_results_sft_0625_qwen2.5-7b", adapter_name="writing")
    # model = PeftModel.from_pretrained(model, "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/saves/qwen/lora/dpo/checkpoint-315", adapter_name="writing")
    print(model_id)

    # with open('results/case_infos.json', 'r', encoding='utf-8') as f:
    #     case_infos = json.load(f)
    # cases = list(case_infos.keys())
    data_dir = '../paper_data/acl/2025/main/'
    cases = os.listdir(data_dir)
    # paper_lis = os.listdir(data_dir)
    evidences = {}
    # 使用加载的数据创建PaperEvidence对象
    for paper in tqdm(cases):
        file_path = data_dir + paper + '/processed_data.json'
        if os.path.exists(file_path):
            # print(f"Processing {file_path}")
            data = load_process_data(file_path)
            messages = format_messages(data, high=True)
            intro = model_generate(tokenizer, model, terminators, messages)
            # intro = remove_references(intro)
            evidences[paper] = intro
    with open('results/naive_llama_great_prompt','w',encoding='utf-8') as fp:
        json.dump(evidences,fp)

        
        
if __name__ == "__main__":
    main() 
        
