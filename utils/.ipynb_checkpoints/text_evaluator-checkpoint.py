from typing import Dict, List, Optional
from dataclasses import dataclass
from zhipuai import ZhipuAI
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LogitsProcessor
import re

def model_generate(tokenizer, model, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        # logits_processor=logits_processor,
        **model_inputs,
        max_new_tokens=2048,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    return response

def evaluate_introduction(generated_text: str, original_text: str = None) -> Dict[str, float]:
    """
    Evaluate three key metrics of the generated introduction: fluency, consistency, and logic
    
    Args:
        generated_text: The generated introduction text
        original_text: The original introduction text (for consistency evaluation)
        
    Returns:
        Dict[str, float]: A dictionary containing scores for three metrics, range 1-5
    """
    evaluator = TextEvaluator()
    
    # 评估流畅度
    fluency_score, _ = evaluator._evaluate_fluency(generated_text)
    
    # 评估一致性
    consistency_score, _ = evaluator._evaluate_consistency(generated_text, original_text)
    
    # 评估逻辑性
    logic_score, _ = evaluator._evaluate_logic(generated_text)
    
    return {
        "fluency": float(fluency_score),
        "consistency": float(consistency_score),
        "logic": float(logic_score)
    }

class TextEvaluator:
    """Text Quality Evaluator"""
    
    def __init__(self, tokenizer=None, model=None):
        
        self.tokenizer = tokenizer
        self.model = model
            
        # 定义描述性五级量表
        self.scale_mapping = {
            "fluency": {
                "very fluent": 5,
                "relatively fluent": 4, 
                "neither fluent nor unfluent": 3,
                "relatively unfluent": 2,
                "very unfluent": 1
            },
            "logic": {
                "very logical": 5,
                "relatively logical": 4,
                "neither logical nor illogical": 3,
                "relatively illogical": 2, 
                "very illogical": 1
            },
            "consistency": {
                "very consistent": 5,
                "relatively consistent": 4,
                "neither consistent nor inconsistent": 3,
                "relatively inconsistent": 2,
                "very inconsistent": 1
            }
        }

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """调用大语言模型"""
        return model_generate(self.tokenizer, self.model, messages)

    def _convert_descriptive_to_score(self, descriptive_score: str, metric_type: str) -> int:
        """将描述性评分转换为数字评分"""
        descriptive_score = descriptive_score.lower().strip()
        mapping = self.scale_mapping.get(metric_type, {})
        
        for desc, score in mapping.items():
            if desc in descriptive_score:
                return score
        
        # 如果没有匹配到，返回默认分数3
        return 3

    def _evaluate_fluency(self, generated_text: str) -> tuple[int, str]:
        """评估流畅度"""
        system_prompt = """You are a professional language fluency evaluation expert. Please assess the fluency of the given text.

Fluency Evaluation Criteria:
- very fluent (5 points): 
  * Natural sentence structure with elegant language expression
  * Accurate word choice with appropriate use of technical terms
  * Smooth transitions between sentences with natural flow
  * Perfect grammar without any errors
  * Reading feels effortless without any obstacles

- relatively fluent (4 points): 
  * Clear sentence structure with good expression
  * Generally accurate word choice with proper use of technical terms
  * Reasonable connections between sentences
  * Generally correct grammar with very few minor errors
  * Reading flows well with rare need for re-reading

- neither fluent nor unfluent (3 points):
  * Basic sentence structure complete but sometimes unnatural
  * Some inaccurate or unprofessional word choices
  * Sometimes stiff connections between sentences
  * Some grammatical errors but not affecting comprehension
  * Occasional need for pauses or re-reading

- relatively unfluent (2 points):
  * Often chaotic or incomplete sentence structure
  * Inaccurate word choice with improper use of technical terms
  * Stiff connections between sentences, lacking transitions
  * Obvious grammatical errors
  * Frequent need for pauses or re-reading

- very unfluent (1 point):
  * Severely chaotic or fragmented sentence structure
  * Chaotic word choice with incorrect use of technical terms
  * Almost no reasonable connections between sentences
  * Frequent grammatical errors severely affecting comprehension
  * Constant need for pauses, difficult to understand

Please respond in the following format only:
Reason: [Detailed scoring reason, 100-150 words]
Score: [Choose one of the five levels above]"""

        user_prompt = f"""Please evaluate the fluency of the following text:

{generated_text}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._call_llm(messages)
        
        for item in self.scale_mapping['fluency'].keys():
            if item in response.lower():
                return self.scale_mapping['fluency'][item]

    def _evaluate_logic(self, generated_text: str) -> tuple[int, str]:
        """评估逻辑性"""
        system_prompt = """You are a professional language logic evaluation expert. Please assess the logical coherence of the given text, focusing on logical relationships between sentences, coherence of argumentation, and orderliness of language expression. Do not evaluate the professional logic of the content itself.

Logic Evaluation Criteria:
- very logical (5 points): 
  * Very clear logical relationships between sentences with appropriate connectors and transitions
  * Well-defined paragraph hierarchy with clear central ideas in each paragraph
  * Orderly progression of arguments with tight correspondence
  * Accurate expression of causal and progressive logical relationships
  * Natural and smooth transitions between viewpoints without jumps in thinking

- relatively logical (4 points): 
  * Clear logical relationships between sentences with generally appropriate connectors
  * Complete paragraph structure with generally clear central ideas
  * Reasonable progression of arguments with general correspondence
  * Generally accurate expression of logical relationships with occasional imprecision
  * Generally natural transitions between viewpoints with few jumps

- neither logical nor illogical (3 points):
  * Basic logical relationships visible but sometimes with inappropriate connectors
  * Paragraph structure exists but lacks clear hierarchy
  * Argumentation progression is understandable but lacks tight correspondence
  * Sometimes unclear expression of logical relationships
  * Occasional stiff or jumping transitions between viewpoints

- relatively illogical (2 points):
  * Often unclear logical relationships with inappropriate connectors
  * Chaotic paragraph structure lacking clear central ideas
  * Chaotic progression of arguments lacking correspondence
  * Often incorrect or vague expression of logical relationships
  * Frequent jumping transitions between viewpoints, lacking coherence

- very illogical (1 point):
  * Almost no clear logical relationships, lacking necessary connectors
  * Completely chaotic paragraph structure with no hierarchy
  * No orderly argumentation, complete disconnection between parts
  * Severely incorrect or missing logical relationship expressions
  * Random jumps between viewpoints, completely lacking coherence

Please respond in the following format only:
Reason: [Detailed scoring reason, 100-150 words]
Score: [Choose one of the five levels above]"""

        user_prompt = f"""Please evaluate the logical coherence of the following text:

{generated_text}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._call_llm(messages)
        
        for item in self.scale_mapping['logic'].keys():
            if item in response.lower():
                return self.scale_mapping['logic'][item]

    def _evaluate_consistency(self, generated_text: str, original_text: str = None) -> tuple[int, str]:
        """评估一致性"""
        system_prompt = """You are a professional text consistency evaluation expert. Please assess the consistency between the generated text and the original paper, focusing on content accuracy, completeness, and fidelity.

Consistency Evaluation Criteria:
- very consistent (5 points): 
  * Completely preserves core ideas and key information from the original text
  * Accurately conveys the original research purpose and methods
  * Maintains the use and definition of technical terms from the original
  * No addition or omission of important information
  * Completely accurate summarization and paraphrasing of the original

- relatively consistent (4 points): 
  * Generally preserves core ideas and key information from the original text
  * Well conveys the original research purpose and methods
  * Most technical terms are used accurately
  * Minimal addition or omission of information, not affecting overall understanding
  * Generally accurate summarization and paraphrasing of the original

- neither consistent nor inconsistent (3 points):
  * Partially preserves core ideas and key information from the original text
  * Basically conveys the original research purpose and methods, but with some ambiguity
  * Sometimes inaccurate use of technical terms
  * Some addition or omission of information, slightly affecting understanding
  * Some deviations in summarization and paraphrasing of the original

- relatively inconsistent (2 points):
  * Many core ideas and key information missing or incorrect
  * Significant deviation in understanding of original research purpose and methods
  * Frequent inaccurate use of technical terms
  * Substantial addition or omission of information, affecting understanding
  * Obvious deviations in summarization and paraphrasing of the original

- very inconsistent (1 point):
  * Almost no preservation of core ideas and key information
  * Complete misunderstanding of original research purpose and methods
  * Severe errors in use of technical terms
  * Extensive addition or omission of information, content completely distorted
  * Completely inaccurate summarization and paraphrasing of the original

Please respond in the following format only:
Reason: [Detailed scoring reason, 100-150 words]
Score: [Choose one of the five levels above]"""

        if original_text:
            user_prompt = f"""Please evaluate the consistency between the generated text and the original paper:

[Generated Text]
{generated_text}

[Original Paper Text]
{original_text}"""
        else:
            user_prompt = f"""Please evaluate the internal consistency of the following text (as there is no original text for comparison, please assess the logical consistency within the text):

{generated_text}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self._call_llm(messages)
        
        for item in self.scale_mapping['consistency'].keys():
            if item in response.lower():
                return self.scale_mapping['consistency'][item]