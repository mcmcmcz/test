import json
import os
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
)
from peft import PeftModel
import re
from tqdm import tqdm
import re
import nltk
from nltk.tokenize import sent_tokenize
from text_evaluator import *
from data_load import *
import argparse            


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", torch_dtype=torch.bfloat16, trust_remote_code=True)
    return tokenizer, model



MODEL_PATH = "/home/tcsu/All_LLM/Qwen2.5-32B-Instruct/" 
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


def evaluate_introduction(evaluator:TextEvaluator, generated_text: str, original_text: str = None) -> Dict[str, float]:
    """
    Evaluate three key metrics of the generated introduction: fluency, consistency, and logic
    
    Args:
        generated_text: The generated introduction text
        original_text: The original introduction text (for consistency evaluation)
        
    Returns:
        Dict[str, float]: A dictionary containing scores for three metrics, range 1-5
    """
    
    # 评估流畅度
    fluency_score = evaluator._evaluate_fluency(generated_text)
    
    # 评估一致性
    consistency_score = evaluator._evaluate_consistency(generated_text, original_text)
    
    # 评估逻辑性
    logic_score = evaluator._evaluate_logic(generated_text)
    
    return {
        "fluency": float(fluency_score),
        "consistency": float(consistency_score),
        "logic": float(logic_score)
    }

class TextEvaluator:
    """Text Quality Evaluator"""
    
    def __init__(self, tokenizer=None, model=None):
        
        self.tokenizer = TOKENIZER
        self.model = BASE_MODEL
            
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
        return model_generate(messages)

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["ft", "base", "stage", "naive"],
                        default="ft", help="ft=微调模型; base=纯基座")
    args = parser.parse_args()
    print(args.type)
    type = args.type
    nltk.data.path.append('/home/mczhang/zmc-dl/LLM/NTP/nltk_data')
    LOCAL_MODEL_PATH = "./bert-large-uncased"  # 替换为你的本地模型路径
    model_path = "/home/tcsu/All_LLM/Qwen2.5-32B-Instruct/"
    tokenizer, model = load_model(model_path)
    evaluator = TextEvaluator(tokenizer, model)
    generated_dir = f"/home/mczhang/zmc-dl/LLM/NTP/outline_baseline/writing_agents_results/{type}/"
    save_dir = f"/home/mczhang/zmc-dl/LLM/NTP/outline_baseline/writing_agents_results/{type}_metrics_1/"
    loader = DataLoader("/home/mczhang/zmc-dl/LLM/NTP/paper_data/acl/2025/main")
    all_data = loader.load_all()
    if type not in ["ft", "base", "stage"]:
        print('process naive')
        with open('../results/naive_qwen.json', 'r', encoding='utf-8') as fp:
            generate_data_all = json.load(fp)
    for item in tqdm(all_data.keys()):
        try:
            gen_path = generated_dir + item + '/' + 'introduction_results.json'
            save_path = save_dir + f'{item}.json'
            if not os.path.exists(save_path):
                if type in ["ft", "base", "stage"]:
                    with open(gen_path, 'r', encoding='utf-8') as fp:
                        gen_data = json.load(fp)
                    gen_introduction = gen_data['introduction']
                else:
                    gen_introduction = generate_data_all[item]
                ori_introduction = all_data[item].introduction
                results = evaluate_introduction(evaluator, gen_introduction, ori_introduction)
                with open(save_path, 'w', encoding = 'utf-8') as fw:
                    json.dump(results, fw, ensure_ascii=True, indent=4)
                print(f"Evaluate {item}")
            else:
                print(f"Already evaluate {item}")
        except:
            print(f"Wrong {item}")