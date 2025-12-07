import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
)

from peft import PeftModel
from utils.data_load import *
import re

def transfer(temp, path):
    introduction_results = {}
    outlines = {}
    o = []
    start = temp.index('<STAGE1>')
    end = temp.index('<STAGE2>')
    background = temp[start+9:end-8].replace("Contents for Background: \n","")
    
    start = temp.index('<STAGE3>')
    end = temp.index('<STAGE4>')
    problem = temp[start+9:end-8].replace("Contents for Problem and Limitations of Existing Methods: \n","")

    start = temp.index('<STAGE5>')
    end = temp.index('<STAGE6>')
    method = temp[start+9:end-8].replace("Contents for Brief Method Overview and Summary of Main Results: \n","")

    start = temp.index('<STAGE7>')
    contribution = temp[start+9:-8].replace("Contents for Our Contributions: \n","")
    
    sections = {
        "background":background,
        "problem_statement":problem,
        "method_overview":method,
        "contributions":contribution
    }
    introduction_results['sections'] = sections
    introduction_results['introduction'] = background +'\n'+ problem +'\n'+ method +'\n'+ contribution
    with open(path, "w", encoding="utf-8") as fw:
        json.dump(introduction_results, fw, ensure_ascii=False, indent=4)

def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", torch_dtype=torch.bfloat16, trust_remote_code=True)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    return tokenizer, model, terminators
    
MODEL_PATH = "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/llama_sft_stage_0911"  # replace with actual model path
print(MODEL_PATH)

TOKENIZER, BASE_MODEL, TERMINATORS = load_model(MODEL_PATH)
HF_MODEL = BASE_MODEL       # 先给一个默认值，后面可被替换
        


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
        pad_token_id=TOKENIZER.eos_token_id,
        eos_token_id=TERMINATORS)
    generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
    response = TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def model_regenerate(messages):

    text = TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = TOKENIZER([text], return_tensors="pt").to(HF_MODEL.device)
    generated_ids = HF_MODEL.generate(
        **model_inputs,
        max_new_tokens=2304, 
        temperature = 0.5,
        pad_token_id=TOKENIZER.eos_token_id,
        eos_token_id=TERMINATORS)
    generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
    response = TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def main():
   
    loader = DataLoader("../paper_data/acl/2025/main")
    all_data = loader.load_all()
    results = {}
    for item in all_data.items():
        paper = item[0]
        folder_path = f"writing_agents_results_llama/stage_sft/{paper}/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = f"writing_agents_results_llama/stage_sft/{paper}/introduction_results.json"
        if not os.path.exists(file_path): 
            research_data = item[1]
            messages = [
                {
                    "role":"system",
                    "content": """As an academic writing expert who has completed a research project and is currently in the paper-writing stage, please draft the introduction section based on the available materials.
            Follow the format of Outline → Content, first drafting the outline of this section and then the content.
            When writing, ensure logical coherence and smooth transitions, and use fluent and standard academic English.""",
                },
                {
                    "role":"user",
                    "content": f"""Compose the Introduction of an ACL paper based on the corresponding research materials. For each sub-section, first list the outline and then write the corresponding content of that section. You need to write four sections:
            1. Background: Provide the research background and the current status of the field. It usually appears at the beginning of the introduction to offer background information for the research question.
            2. Problem and Limitations of Existing Methods: Describe the research problem and the limitations of existing methods. It usually comes after the background, clearly pointing out the problems existing in current research.
            3. Brief Method Overview and Summary of Main Results: Briefly introduce the proposed method and summarize the main results.
            4. Our Contributions: Summarize the contributions of this paper, clearly stating the main contributions of the paper.
            Research materials:
            Title: {research_data.title}
            Abstract: {research_data.abstract}
            Baseline references: {research_data.baseline_references}
            Figure information: {research_data.figures}
            Table information: {research_data.tables}"""
                },
            #     {
            #         "role":"user",
            #         "content": f"""Compose the Introduction of an ACL paper based on the corresponding research materials. For each sub-section, first list the outline and then write the corresponding content of that section. You need to write four sections:
            # 1. Background: Provide the research background and the current status of the field. It usually appears at the beginning of the introduction to offer background information for the research question.
            # 2. Problem and Limitations of Existing Methods: Describe the research problem and the limitations of existing methods. It usually comes after the background, clearly pointing out the problems existing in current research.
            # 3. Brief Method Overview and Summary of Main Results: Briefly introduce the proposed method and summarize the main results.
            # 4. Our Contributions: Summarize the contributions of this paper, clearly stating the main contributions of the paper.
            # Research materials:
            # Title: {research_data.title}
            # Abstract: {research_data.abstract}"""
            #     },
            ]
            writing = model_generate(messages)
            sections = ["Background","Problem and Limitations of Existing Methods","Brief Method Overview and Summary of Main Results","Our Contributions"]
            special_tokens = [f'<STAGE{i}>' for i in range(8)] + [f'<END{i}>' for i in range(8)] + ["Contents for " + section for section in sections] + ["Outline for " + section for section in sections]
            for i in range(3):
                judge = True
                for token in special_tokens:
                    if token not in writing:
                        judge = False
                if not judge:
                    print(f"rewrite {paper}")
                    writing = model_regenerate(messages)
            judge = True
            for token in special_tokens:
                if token not in writing: 
                    judge = False  
            if not judge:
                print(f"Wrong {paper}")     
            results[paper] = writing
            try:
                transfer(writing, file_path)
                print(f"Successfully generate {paper}")
            except:
                print(f"False generate {paper}")

        else:
            print(f"Already generate {paper}")


if __name__ == "__main__":
    main() 