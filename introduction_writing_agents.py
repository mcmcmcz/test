#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术论文Introduction多智能体写作系统
使用ZhipuAI API，通用化设计，适用于任何研究领域的论文写作
"""

import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse            # 新增
from utils.data_load import *
from transformers import LogitsProcessor

# ---- Shared HF model loading ----
MODEL_PATH = "/home/tcsu/Qwen2.5-7B-Instruct/"  # replace with actual model path
# MODEL_PATH = "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/dpo_0708/"  # DPO-1
print(MODEL_PATH)


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", torch_dtype=torch.bfloat16, trust_remote_code=True)
    return tokenizer, model


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
        


# ======== LoRA 适配器路径 ========
ADAPTER_PATHS = {
    "background_outline": "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/lora_results_outline_back_qwen",
    "background_content": "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/lora_results_content_back_qwen",
    "problem_outline": "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/lora_results_outline_prob_qwen",
    "problem_content": "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/lora_results_content_prob_qwen",
    "method_outline": "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/lora_results_outline_method_qwen",
    "method_content": "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/lora_results_content_method_qwen",
    "contributions_outline": "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/lora_results_outline_cont_qwen",
    "contributions_content": "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/lora_results_content_cont_qwen",
    # "integration": "/home/mczhang/zmc-dl/LLM/LLaMA-Factory/lora_results_integration",
}

def setup_model(generate_type: str = "ft"):
    """
    generate_type == 'ft'  加载所有 LoRA
    generate_type == 'base'  仅用 BASE_MODEL
    """
    global HF_MODEL, USE_LORA
    if generate_type != "ft":          # 只用基座
        HF_MODEL = BASE_MODEL
        USE_LORA = False
        return

    USE_LORA = True
    # -------- 加载 LoRA -----------
    first_name, first_path = next(iter(ADAPTER_PATHS.items()))
    HF_MODEL = PeftModel.from_pretrained(BASE_MODEL, first_path,
                                         adapter_name=first_name)
    for name, path in ADAPTER_PATHS.items():
        if name != first_name:
            HF_MODEL.load_adapter(path, adapter_name=name)


def model_generate(messages, adapter_name=None):
    """使用共享 HF_MODEL + 不同 LoRA 适配器生成文本"""
    # 只有模型支持 set_adapter 时才切换
    if adapter_name and hasattr(HF_MODEL, "set_adapter"):
        try:
            HF_MODEL.set_adapter(adapter_name)
        except ValueError:
            # 当模型未加载任何 LoRA 时忽略切换
            pass
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
    """智能体基类"""
    
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt
        self.client = None # Removed ZhipuAI client

    def generate(self, prompt, adapter_name: str):
        # Replaced ZhipuAI API call with HF model generation
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        return model_generate(messages, adapter_name)


class BackgroundAgent(Agent):
    """背景介绍智能体"""
    
    def __init__(self):
        system_prompt = """You are an English academic writing expert. You have completed a research project and are now in the writing phase. You are currently working on the Introduction section of the paper.
Task: Draft the "Background" subsection of paper Introduction(200–250 words).
Requirements:
1. Summarize macro background of the field.
2. Describe current development status.
3. Highlight importance and relevance.
4. Use concise formal academic English for academic writing."""
        
        super().__init__("Background Agent", system_prompt)
    
    def generate_outline(self, research_data: ResearchData) -> str:
        prompt = f"""Based on the following research materials, generate a writing outline for the "Background" subsection:
Title: {research_data.title}
Abstract: {research_data.abstract}
Baseline references: {research_data.baseline_references}
Figure information: {research_data.figures}

Please produce a clear outline containing NO MORE THAN FOUR concise bullet points that convey the key ideas and logical progression. Do NOT provide complete sentences or paragraph-level prose.
"""
        return self.generate(prompt, "background_outline")
    
    def write_content(self, research_data: ResearchData, outline: str) -> str:
        prompt = f"""Using the outline and research materials below, write the Background subsection of the paper's Introduction:

Outline:
{outline}

Research materials:
Title: {research_data.title}
Abstract: {research_data.abstract}
Baseline references: {research_data.baseline_references}
Figure information: {research_data.figures}

Please do NOT mention figures; they are only for internal reference.

Please draft the complete "Background" subsection. Remember you are a seasoned expert in academic writing and pay attention to the norms of academic writing. (Length 200–250 words, maximum 2 paragraphs)
"""
        return self.generate(prompt, "background_content")


class ProblemStatementAgent(Agent):
    """问题陈述和现有方法局限性智能体"""
    
    def __init__(self):
        system_prompt = """You are an English academic writing expert. You have completed a research project and are now in the writing phase. You are currently working on the Introduction section of the paper.
Task: Draft the "Problem Statement and Limitations of Existing Methods" subsection of paper Introduction. (150–250 words).
Requirements:
1. State core research problem and key challenges.
2. Critically analyze limitations of existing work.
3. Explain why gaps must be addressed.
4. Use concise formal academic English for academic writing."""
        
        super().__init__("Problem Statement Agent", system_prompt)
    
    def generate_outline(self, research_data: ResearchData) -> str:
        prompt = f"""Based on the following research materials, generate a writing outline for the "Problem Statement and Limitations of Existing Methods" subsection:
Title: {research_data.title}
Abstract: {research_data.abstract}
Baseline references: {research_data.baseline_references}
Figure information: {research_data.figures}

Please produce a clear outline containing NO MORE THAN FOUR concise bullet points that articulate the key issues and logical flow. Do NOT draft full sentences or paragraph-level text.
"""
        return self.generate(prompt, "problem_outline")
    
    def write_content(self, research_data: ResearchData, outline: str) -> str:
        prompt = f"""Using the outline and research materials below, write the "Problem Statement and Limitations of Existing Methods" subsection of the Introduction:

Outline:
{outline}

Research materials:
Title: {research_data.title}
Abstract: {research_data.abstract}
Baseline references: {research_data.baseline_references}
Figure information: {research_data.figures}

Please do NOT mention figures or tables; they are only for internal reference.

Please draft the complete Problem Statement and Limitations subsection. Remember you are a seasoned expert in academic writing and pay attention to the norms of academic writing. (Length 150–250 words)
"""
        return self.generate(prompt, "problem_content")


class MethodOverviewAgent(Agent):
    """方法概述和主要结果智能体"""
    
    def __init__(self):
        system_prompt = """You are an English academic writing expert. You have completed a research project and are now in the writing phase. You are currently working on the Introduction section of the paper.
Task: Draft the "Brief Method Overview and Summary of Main Results" subsection of paper Introduction. (250–450 words)
Requirements:
1. Briefly describe key ideas of the method. Use the original method names, do not tamper with them.
2. Summarize main experimental findings.
3. Emphasize novelty and effectiveness.
4. Use clear formal academic English for writing."""
        
        super().__init__("Method Overview Agent", system_prompt)
    
    def generate_outline(self, research_data: ResearchData) -> str:
        prompt = f"""Based on the following research materials, generate a writing outline for the "Brief Method Overview and Summary of Main Results" subsection:
Title: {research_data.title}
Abstract: {research_data.abstract}
Baseline references: {research_data.baseline_references}
Figure information: {research_data.figures}
Table information: {research_data.tables}

Please create a clear outline containing NO MORE THAN EIGHT concise bullet points that capture the essential ideas and logical sequencing. Refrain from sentence-level or paragraph-level drafting.
"""
        return self.generate(prompt, "method_outline")
    
    def write_content(self, research_data: ResearchData, outline: str) -> str:
        prompt = f"""Using the outline and research materials below, write the "Brief Method Overview and Summary of Main Results" subsection of the Introduction:

Outline:
{outline}

Research materials:
Title: {research_data.title}
Abstract: {research_data.abstract}
Baseline references: {research_data.baseline_references}
Figure information: {research_data.figures}
Table information: {research_data.tables}

Please do NOT mention figures or tables; they are only for internal reference.

Please draft the complete Method Overview and Main Results subsection. Remember you are a seasoned expert in academic writing and pay attention to the norms of academic writing. (Length 250–450 words)
"""
        return self.generate(prompt, "method_content")


class ContributionsAgent(Agent):
    """贡献总结智能体"""
    
    def __init__(self):
        system_prompt = """You are an academic writing expert. You have completed a research project and are now in the writing phase. You are currently working on the Introduction section of the paper.
Task: Draft the "Contributions" subsection (100–150 words).
Requirements:
1. Begin with the phrase: 'Our contributions are as follow:'
2. List main innovations concisely (bullet or numbered).
3. Highlight differences from prior work.
4. Use concise formal academic English for academic writing."""
        
        super().__init__("Contributions Agent", system_prompt)
    
    def generate_outline(self, research_data: ResearchData) -> str:
        prompt = f"""Based on the following research materials, generate a writing outline for the "Our Contributions" subsection:
Title: {research_data.title}
Abstract: {research_data.abstract}
Figure information: {research_data.figures}

Please produce a clear outline containing NO MORE THAN FOUR concise bullet points that summarise the principal contributions and their logical ordering. Avoid supplying full sentences or paragraph drafts.
"""
        return self.generate(prompt, "contributions_outline")
    
    def write_content(self, research_data: ResearchData, outline: str) -> str:
        prompt = f"""Using the outline and research materials below, write the "Our Contributions" subsection of the Introduction:

Outline:
{outline}

Research materials:
Title: {research_data.title}
Abstract: {research_data.abstract}
Figure information: {research_data.figures}

Please do NOT mention figures or tables; they are only for internal reference.

Please draft the complete Our Contributions subsection starting with "Our contributions are as follows:". (Length 100–150 words)
"""
        return self.generate(prompt, "contributions_content")


class IntegrationAgent(Agent):
    """整合智能体"""
    
    def __init__(self):
        system_prompt = """You are an English academic writing expert. You have completed a research project and are now in the writing phase. You are currently working on the Introduction section of the paper.
Task: You have written four subsections of Introduction. Now you need to integrate four subsections into a coherent Introduction (700–1100 words).
Requirements:
1. Ensure logical flow and smooth transitions.
2. Preserve the original text to the greatest extent possible, but remove redundancies and resolve inconsistencies.
3. Deliver polished academic English."""
        
        super().__init__("Integration Agent", system_prompt)
    
    def integrate_sections(self, background: str, problem_statement: str, 
                          method_overview: str, contributions: str, 
                          research_data: ResearchData):
        prompt = f"""Please integrate the following four subsections into a coherent Introduction (700–1100 words).

IMPORTANT:
1. Preserve the core ideas and key sentences; you may trim redundant phrases to improve cohesion.
2. Add concise transitions to ensure smooth logical flow.
3. Remove any references to figures or tables; visuals are for internal reference only.
4. Avoid prematurely describing the proposed method in the Background section.
5. Maintain scholarly tone and tight academic style.
6. Deliver 700–1100 words total.

Background:
{background}

Problem Statement and Limitations of Existing Methods:
{problem_statement}

Brief Method Overview and Summary of Main Results:
{method_overview}

Contributions:
{contributions}

Return only the final integrated Introduction.
Introduction: """
        intro = self.generate(prompt, "integration")
        return intro, prompt

    def revise_intro(self, introduction_text: str, feedback: str, counts: str, original_user_prompt: str) -> str:
        # Use multi-message context for better continuity
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": original_user_prompt},
            {"role": "assistant", "content": introduction_text},
            {"role": "user", "content": "Reviewer feedback:\n" + feedback + "Please revise the Introduction according to the reviewer's feedback. Maintain academic coherence and remove any mentions of figures or tables. The total word count should be between 700 and 1100 words. Do not include any content outside of the academic paper's Introduction, Just generate the final Introdution. The format should be:\n**Introduction**"}]
        return model_generate(messages)



class IntroductionWritingSystem:
    """Introduction写作系统主类"""
    
    def __init__(self, generate_type):
        self.generate_type = generate_type
        self.background_agent = BackgroundAgent()
        self.problem_agent = ProblemStatementAgent()
        self.method_agent = MethodOverviewAgent()
        self.contributions_agent = ContributionsAgent()
        self.integration_agent = IntegrationAgent()
    
    def write_introduction(self, research_data: ResearchData, paper_id:str, 
                          save_intermediate: bool = True) -> Dict[str, str]:
        """完整的Introduction写作流程"""
        
        # print("开始Introduction写作流程...")
        
        # 第一步：各智能体生成大纲
        # print("\n1. 生成各部分大纲...")
        background_outline = self.background_agent.generate_outline(research_data)
        problem_outline = self.problem_agent.generate_outline(research_data)
        method_outline = self.method_agent.generate_outline(research_data)
        contributions_outline = self.contributions_agent.generate_outline(research_data)
        
        if save_intermediate:
            self._save_outlines({
                "background": background_outline,
                "problem_statement": problem_outline,
                "method_overview": method_outline,
                "contributions": contributions_outline
            }, paper_id)
        
        # 第二步：根据大纲写作各部分内容
        # print("\n2. 撰写各部分内容...")
        background_content = self.background_agent.write_content(research_data, background_outline)
        problem_content = self.problem_agent.write_content(research_data, problem_outline)
        method_content = self.method_agent.write_content(research_data, method_outline)
        contributions_content = self.contributions_agent.write_content(research_data, contributions_outline)
        
        # 第三步：整合所有部分
        # print("\n3. 整合所有部分...")
        # integrated_introduction, prompt = self.integration_agent.integrate_sections(
        #     background_content, problem_content, method_content, 
        #     contributions_content, research_data
        # )
        integrated_introduction = background_content + '\n' + problem_content + '\n' + method_content + '\n' + contributions_content
        # 直接使用首次整合结果作为最终 Introduction
        final_introduction = integrated_introduction

        results = {
            "outlines": {
                "background": background_outline,
                "problem_statement": problem_outline,
                "method_overview": method_outline,
                "contributions": contributions_outline
            },
            "sections": {
                "background": background_content,
                "problem_statement": problem_content,
                "method_overview": method_content,
                "contributions": contributions_content
            },
            "introduction": final_introduction
        }
        
        if save_intermediate:
            self._save_results(results, paper_id)
        
        print("\nIntroduction写作完成!")
        return results
    
    def _save_outlines(self, outlines: Dict[str, str], paper_id: str):
        """保存大纲"""
        with open(f"writing_agents_results_cvpr/{self.generate_type}/{paper_id}/introduction_outlines.json", "w", encoding="utf-8") as f:
            json.dump(outlines, f, ensure_ascii=False, indent=2)
        # print("大纲已保存至 introduction_outlines.json")
    
    def _save_results(self, results: Dict[str, Any], paper_id: str):
        """保存所有结果"""
        with open(f"writing_agents_results_cvpr/{self.generate_type}/{paper_id}/introduction_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        # print("完整结果已保存至 introduction_results.json")


def load_research_data_from_json(file_path: str) -> ResearchData:
    """从JSON文件加载研究数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return ResearchData(
        title=data.get("title",""),
        abstract=data.get("abstract", ""),
        figures=data.get("figures", {}),
        tables=data.get("tables", []),
        baseline_references=data.get("baseline_references", [])
    )


def word_count(text: str) -> int:
    """Return approximate word count of a string."""
    # Split by whitespace using regex to count words reliably
    return len(text.split(' '))


def main():
    """主函数 - 直接使用example.json进行测试"""
    print("学术论文Introduction多智能体写作系统")
    print("="*50)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate_type", choices=["ft", "base"],
                        default="ft", help="ft=微调模型; base=纯基座")
    args = parser.parse_args()
    print(args.generate_type)
    setup_model(args.generate_type)        # ← 关键一步
    print(f"当前模式: {args.generate_type}")

    loader = DataLoader("../paper_data/cvpr/2025")
    all_data = loader.load_all()
    # 从JSON文件加载数据
    # research_data = load_research_data_from_json("example.json")

    for item in all_data.keys():
        paper_id = item
        if not os.path.exists(f"writing_agents_results_cvpr/{args.generate_type}/{paper_id}/introduction_results.json"):
                
            folder_path = f"writing_agents_results_cvpr/{args.generate_type}/{paper_id}/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            
            research_data = all_data[paper_id]
            # print("✓ 成功加载研究数据")
            
            # 创建写作系统
            writing_system = IntroductionWritingSystem(args.generate_type)
            # print("✓ 写作系统初始化成功")
            
            # 执行写作流程
            # print("\n开始执行Introduction写作流程...")
            results = writing_system.write_introduction(research_data, paper_id)
            
            # # 输出最终结果
            # print("\n" + "="*50)
            # print("最终生成的Introduction:")
            # print("="*50)
            # print(results["introduction"])
        
            # 保存最终结果到文件
            with open(f"writing_agents_results_cvpr/{args.generate_type}/{paper_id}/introduction.txt", "w", encoding="utf-8") as f:
                f.write(results["introduction"])
            # print(f"\n✓ 最终结果已保存至: introduction.txt")
            
            # # 显示统计信息
            # print(f"\n统计信息:")
            # print(f"- 最终Introduction长度: {word_count(results['introduction'])} 字")
            # print(f"- Background部分长度: {word_count(results['sections']['background'])} 字")
            # print(f"- Problem Statement部分长度: {word_count(results['sections']['problem_statement'])} 字")
            # print(f"- Method Overview部分长度: {word_count(results['sections']['method_overview'])} 字")
            # print(f"- Contributions部分长度: {word_count(results['sections']['contributions'])} 字")
            
            # print("\n✓ Introduction写作完成!")
            print(f"Successfully generate {paper_id}")
        else:
            print(f"Already generate {paper_id}")

if __name__ == "__main__":
    main() 