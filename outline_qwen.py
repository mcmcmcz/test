from typing import Dict, List, Optional
from dataclasses import dataclass
from zhipuai import ZhipuAI
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
MODEL_PATH = "/home/tcsu/Qwen2.5-7B-Instruct/"
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

class Agent:
    """基础智能体类"""
    def __init__(self, name, system_prompt, use_closeai):
        self.name = name
        self.system_prompt = system_prompt
        self.use_closeai = use_closeai
        
        if use_closeai:
            self.client = ZhipuAI(api_key="96bb7647b7f64f7c9cfe1e281651ecdf.yQ3W2XAwQZN81NZw")


    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        if self.use_closeai:
            completion = self.client.chat.completions.create(
                model="glm-4-flash",
                messages=messages
            )
            return completion.choices[0].message.content
        else:
            return model_generate(messages)

@dataclass
class PaperEvidence:
    """论文的证据材料"""
    title: str
    abstract: str
    figures: List[str]  # {"Figure X": {"img_path": str, "img_caption": str}}
    tables: List[List[str]]  # [["Table X: caption", "table_html_content"]]
    ref: List[Dict[str, str]]  # [{"model_name": str, "reference": str, "abstract": str}]

class OutlineAgent(Agent):
    """大纲生成智能体"""
    def __init__(self, use_closeai):
        super().__init__(
            name="OutlineGenerator",
            system_prompt="""You are an expert in creating academic paper outlines, specifically for research paper introductions.
            Your task is to create a clear, logical outline that follows the standard introduction structure.
            Focus on creating a comprehensive outline that captures all key aspects of the research.
            
            The outline should include the following parts:
            1. Background - Provide a brief overview of the macro background and current development status of the research field.
            2. Problem and Limitations of Existing Methods - Clearly identify the core problems and challenges in the current research. Analyze the deficiencies of existing research methods in addressing the problem.
            3. Brief Method Overview and Summary of Main Results- Summarize the research methods and key ideas used in this paper. Highlight the important conclusions and findings derived from the experiments.
            4. Our Contributions - Clearly list the innovations and values of this study.
            
            Use formal academic paper style without any markdown formatting or special characters.""",
            use_closeai=use_closeai,
        )

    def generate_outline(self, evidence: PaperEvidence) -> str:
        """生成论文大纲"""
        prompt = f'''Create a concise and focused outline for the introduction section based on the following research materials.
        The outline should be brief but comprehensive, focusing only on the most important points.

        Title: {evidence.title}

        Abstract:
        {evidence.abstract}

        Research materials:
        Figures: {evidence.figures}
        Tables: {evidence.tables}
        Baseline References: {evidence.ref}

        Requirements:
        1. Structure and Length Requirements:
           - Must follow exactly this 4-section structure
           - Each section should have 2-3 key points maximum
           - Total outline should not exceed 15 bullet points
           - Each bullet point should be 1-2 lines maximum
           - Keep points concise and focused

        2. Section Content Guidelines:
           Section 1: Background
           - Current state of the field (1-2 point)
           - Key challenges or gaps (1-2 point)

           Section 2: Problem and Limitations of Existing Methods
           - Main problem addressed (1-3 point)
           - Key limitations of existing methods (1-3 points)

           Section 3: Brief Method Overview and Summary of Main Results
           - Core approach (2-4 point)
           - Key findings (2-4 points)

           Section 4: Our Contributions
           - Main contributions (2-4 points)

        3. Writing Style:
           - Use clear, concise language
           - Focus on essential information only
           - Avoid redundancy
           - Maintain academic tone
           - Use active voice
           - Start each point with a strong verb

        4. Format Requirements:
           - Use "-" for bullet points
           - No sub-bullets
           - No numbering within sections
           - No markdown formatting
           - Consistent indentation

        Create a focused outline that captures the essence of this research:'''

        return self.generate(prompt)


class OutlineSystem:
    """大纲生成系统"""
    def __init__(self, use_closeai):
        self.outline_agent = OutlineAgent(use_closeai)
        self.introduction_writer = IntroductionWriter(use_closeai)
    
    def generate_outline(self, evidence: PaperEvidence) -> str:
        """生成论文大纲"""
        return self.outline_agent.generate_outline(evidence)
    
    def write_introduction(self, outline: str, evidence: PaperEvidence) -> str:
        """根据大纲生成介绍"""
        return self.introduction_writer.write_introduction(outline, evidence)
    
    def generate_introduction(self, evidence: PaperEvidence) -> tuple:
        """生成论文介绍"""
        # 生成大纲
        outline = self.generate_outline(evidence)
        
        # 根据大纲生成介绍
        introduction = self.write_introduction(outline, evidence)
        
        return outline, introduction

class IntroductionWriter(Agent):
    """介绍撰写智能体"""
    def __init__(self, use_closeai):
        super().__init__(
            name="IntroductionWriter",
            system_prompt="""You are an expert academic writer specializing in research paper introductions.
            Your writing should be detailed, well-structured, and maintain high academic standards.
            Focus on creating content that flows logically and builds a compelling narrative.
            
            Important Rules:
            1. Do not include any citations or references
            2. Do not use any citation formats
            3. Do not mention specific papers or authors
            4. Focus on the content without referencing other works, just write the main content
            5. Write as if describing the work in isolation""",
            use_closeai=use_closeai,
        )
        
    def write_introduction(self, outline: str, evidence: PaperEvidence) -> str:
        prompt = f'''
        Write a comprehensive 'Introduction' part based on the outline and the paper's abstract.
        Write the content directly as a continuous text without any formatting, section headers, or markdown.

        Outline:
        {outline}

        Paper Information:
        Title: {evidence.title}

        Abstract:
        {evidence.abstract}
        Figures: {evidence.figures}
        Tables: {evidence.tables}
        Baseline References: {evidence.ref}

        Requirements:
        1. Length Requirements:
           - Total length: 800-1200 words
           - 4-6 substantial paragraphs
           - Each paragraph: 150-250 words
           - No more than 5 sentences per paragraph
        
        2. Content Requirements:
           - Follow the outline structure exactly
           - Each section should be written only once
           - Maintain logical flow between paragraphs
           - Focus on essential information only
           - Avoid redundancy and repetition
           - Do not repeat the same information in different sections
        
        3. Writing Style:
           - Use formal academic language
           - Write complete sentences
           - Use active voice
           - Avoid informal language
           - No citations or references
           - Use precise technical terminology
           - Maintain scholarly tone throughout
           - Employ sophisticated vocabulary
           - Use complex sentence structures appropriately
        
        4. Important Rules:
           - Write as continuous text without any formatting
           - No special characters
           - Focus on content and clarity
           - Keep paragraphs concise and focused
           - Ensure each paragraph has a clear topic sentence
           - Use appropriate transitions between paragraphs
           - Maintain academic rigor throughout
           - Avoid colloquial expressions
           - Use precise and technical language
           - Do not repeat or rephrase the same content
           - Each point should be discussed only once

        Write the 'Introduction' part as a continuous text.'''

        return self.generate(prompt)

def load_process_data(json_path: str = "process_data.json") -> dict:
    """加载process_data.json文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """主函数"""
    # 初始化系统
    use_closeai = False
    # model_id = "/home/mczhang/zmc-dl/LLM/Meta-Llama-3.1-8B-Instruct/"

    # model = PeftModel.from_pretrained(model, "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/lora_results_sft_0618_qwen2.5-7b", adapter_name="writing")
    # print("sft")
    introduction_system = OutlineSystem(use_closeai)
    
    # data_dir = '../paper_data/acl/2025/main/'
    data_dir = '../paper_data/cvpr/2025/'
    cases = os.listdir(data_dir)
    print(len(cases))
    results = {}
    # 处理每个论文
    for paper in tqdm(cases):
        file_path = data_dir + paper + '/processed_data.json'
        if os.path.exists(file_path):
            print(f"\nProcessing {file_path}")
            try:
                data = load_process_data(file_path)
                evidence = PaperEvidence(
                    title=data["title"],
                    abstract=data["abstract"],
                    figures=data["figures"],
                    tables=data["tables"],
                    ref=data.get("ref", [])
                )
                
                # 生成介绍
                print(f"Generating introduction for {paper}")
                outline, introduction = introduction_system.generate_introduction(evidence)
                
                results[paper] = {}
                results[paper]['outline'] = outline
                results[paper]['introduction'] = introduction
                print(f"Successfully generated introduction for {paper}")
                
            except Exception as e:
                print(f"Error processing {paper}: {str(e)}")
                results[paper] = f"Error: {str(e)}"
    
    with open('results/cvpr_outline_base_1119.json', 'w', encoding='utf-8') as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print("Results saved successfully")

if __name__ == "__main__":
    main() 