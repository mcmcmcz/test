"""
IntroductionOutline.py - 基于AutoSurvey架构的论文Introduction生成系统

该系统借鉴AutoSurvey的多智能体协作模式，专门用于根据给定材料生成高质量的论文Introduction。
主要组件：
1. MaterialParser: 解析JSON格式的输入材料
2. IntroductionOutliner: 生成Introduction结构大纲
3. IntroductionWriter: 撰写具体的Introduction内容
4. IntroductionGenerator: 主控制器，协调各组件工作
"""

import json
import re
import threading
import time
import torch
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.data_load import *
import argparse
from transformers import LogitsProcessor

GLOBAL_TOKENIZER = None
GLOBAL_MODEL = None

GLOBAL_tokens = 0


def load_model(model_path):
    """加载开源模型和tokenizer"""
    global GLOBAL_TOKENIZER, GLOBAL_MODEL
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="sequential"
    )
    
    GLOBAL_TOKENIZER = tokenizer
    GLOBAL_MODEL = model

    return tokenizer, model

GLOBAL_TOKENIZER, GLOBAL_MODEL = load_model('/home/tcsu/Qwen2.5-7B-Instruct/')

chinese_token_ids = set()
for token, token_id in GLOBAL_TOKENIZER.get_vocab().items():
    decoded = GLOBAL_TOKENIZER.decode([token_id], skip_special_tokens=True)
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
    """使用全局模型生成文本"""
    global GLOBAL_TOKENIZER, GLOBAL_MODEL
    
    text = GLOBAL_TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        logits_processor=[chinese_suppressor],
    )
    model_inputs = GLOBAL_TOKENIZER([text], return_tensors="pt").to(GLOBAL_MODEL.device)

    generated_ids = GLOBAL_MODEL.generate(
        **model_inputs,
        max_new_tokens=2048,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = GLOBAL_TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
    new_mes = messages.copy()
    new_mes.append({'role':'assistant','content':response})
    count_text = GLOBAL_TOKENIZER.apply_chat_template(
        new_mes,
        tokenize=False,
        add_generation_prompt=True,
    )    
    global GLOBAL_tokens
    GLOBAL_tokens = GLOBAL_tokens + len(GLOBAL_TOKENIZER(count_text)['input_ids'])
    return response


class LocalModel:
    
    def __init__(self):
        pass
    
    def chat(self, text, temperature=1):
        messages = [{"role": "user", "content": text}]
        return model_generate(messages)


def get_text_length(text):
    return len(text)


def get_word_count(text):
    return len(text.split())


@dataclass
class ResearchData:

    title: str=""
    abstract: str = ""
    figures: List[str] = None       # 修改为纯列表
    tables: List[Any] = None
    baseline_references: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.title is None:
            self.title = ""
        if self.abstract is None:
            self.abstract = ""
        if self.figures is None:
            self.figures = []           # 修改为空列表
        if self.tables is None:
            self.tables = []
        if self.baseline_references is None:
            self.baseline_references = []


class MaterialParser:

    
    def parse_json_data(self, data: Dict[str, Any]) -> ResearchData:
        return ResearchData(
            title=data.get('title',''),
            abstract=data.get('abstract', ''),
            figures=data.get('figures', []),        # 修改为列表
            tables=data.get('tables', []),
            baseline_references=data.get('ref', [])
        )
    
    def extract_key_concepts(self, material: ResearchData) -> List[str]:
        concepts = []
        
        # 从摘要中提取关键概念
        abstract_concepts = self._extract_concepts_from_text(material.abstract)
        concepts.extend(abstract_concepts)
        
        # 从图表标题中提取概念
        for figure in material.figures:
            if type(figure) == str:
                fig_concepts = self._extract_concepts_from_text(figure)
                concepts.extend(fig_concepts)
        
        # 从表格描述中提取概念
        for table in material.tables:
            if table:  # table[0] 通常是表格描述
                table_concepts = self._extract_concepts_from_text(table[0])
                concepts.extend(table_concepts)
        
        return list(set(concepts))  # 去重
    
    def _extract_concepts_from_text(self, text: str) -> List[str]:
        """从文本中提取关键概念（简化版本）"""
        # 提取大写字母开头的专业术语
        concepts = re.findall(r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\b', str(text))
        # 提取括号中的缩写
        abbreviations = re.findall(r'\(([A-Z]{2,})\)', text)
        concepts.extend(abbreviations)
        
        # 过滤常见词汇
        common_words = {'The', 'This', 'These', 'That', 'Those', 'We', 'Our', 'In', 'On', 'At', 'For', 'To', 'From', 'With', 'By', 'As', 'An', 'A', 'Table', 'Figure'}
        concepts = [c for c in concepts if c not in common_words and len(c) > 2]
        
        return concepts


class IntroductionOutliner:
    """Introduction大纲生成器 - 基于材料生成结构化大纲"""
    
    def __init__(self):
        self.model = LocalModel()
    
    def generate_outline(self, material: ResearchData, key_concepts: List[str]) -> str:
        """生成Introduction大纲"""
        
        # 第一步：生成粗略大纲
        rough_outline = self._generate_rough_outline(material, key_concepts)
        
        # 第二步：细化大纲结构
        detailed_outline = self._refine_outline_structure(rough_outline, material)
        
        # 第三步：生成最终大纲
        final_outline = self._generate_final_outline(detailed_outline, material)
        
        return final_outline
    
    def _generate_rough_outline(self, material: ResearchData, key_concepts: List[str]) -> str:
        """生成粗略的Introduction大纲"""
        
        prompt = self._create_rough_outline_prompt(material, key_concepts)

        
        outline = self.model.chat(prompt, temperature=0.7)

        
        return outline
    
    def _refine_outline_structure(self, rough_outline: str, material: ResearchData) -> str:
        """细化大纲结构"""
        
        prompt = self._create_structure_refinement_prompt(rough_outline, material)
        
        refined_outline = self.model.chat(prompt, temperature=0.5)
        
        return refined_outline
    
    def _generate_final_outline(self, detailed_outline: str, material: ResearchData) -> str:
        """生成最终大纲"""
        
        prompt = self._create_final_outline_prompt(detailed_outline, material)
        
        final_outline = self.model.chat(prompt, temperature=0.3)
        
        return final_outline
    
    def _create_rough_outline_prompt(self, material: ResearchData, key_concepts: List[str]) -> str:
        """创建粗略大纲生成提示"""
        
        concepts_text = ", ".join(key_concepts[:10])  # 限制概念数量
        
        prompt = f"""You are an expert academic writer tasked with creating a structured outline for a research paper Introduction section, do not write the title.

Given the following research materials:

Title: {material.title}

Abstract: {material.abstract}

Key Concepts: {concepts_text}

Figures: {material.figures}

Tables: {material.tables}

Baseline References: {material.baseline_references}

Please create a comprehensive outline for the Introduction section that follows academic writing conventions. The outline should include:

1. Background
2. Problem and Limitations of Existing Methods
3. Brief Method Overview and Summary of Main Results
4. Our Contributions


Format your response as:
<outline>
## Section 1: [Section Name]
Description: [Brief description of what this section covers]

## Section 2: [Section Name]  
Description: [Brief description of what this section covers]

...
</outline>

The outline should be logical, comprehensive, and suitable for a high-quality research paper Introduction."""

        return prompt
    
    def _create_structure_refinement_prompt(self, rough_outline: str, material: ResearchData) -> str:
        """创建结构细化提示"""
        
        prompt = f"""You are an expert academic editor. Please refine the following Introduction outline to ensure it follows the best practices for academic paper introductions, do not write the title.

Current outline:
{rough_outline}

Available research materials include:
- Abstract describing the main contribution
- Experimental figures showing results and methodology
- Performance comparison tables
- Baseline method references

Please refine this outline to:
1. Ensure logical flow from general background to specific contributions
2. Make sure each section has a clear purpose
3. Optimize the structure for maximum impact
4. Ensure smooth transitions between sections

Provide the refined outline in the same format:
<outline>
## Section 1: [Section Name]
Description: [Brief description]
...
</outline>"""

        return prompt
    
    def _create_final_outline_prompt(self, detailed_outline: str, material: ResearchData) -> str:
        """创建最终大纲生成提示"""
        
        prompt = f"""You are finalizing the Introduction outline for a research paper. Please create the final, polished version of the outline.

Current refined outline:
{detailed_outline}

Please provide the final outline with:
1. Clear section titles that are engaging and informative
2. Detailed descriptions of what each section should contain
3. Specific guidance on how to incorporate the available materials (figures, tables, baselines)
4. Logical flow and smooth transitions

Format as:
<final_outline>
## Section 1: [Final Section Name]
Description: [Detailed description including specific content guidance]
Materials to reference: [Which figures/tables/baselines to mention]

## Section 2: [Final Section Name]
Description: [Detailed description including specific content guidance]  
Materials to reference: [Which figures/tables/baselines to mention]

...
</final_outline>"""

        return prompt


class IntroductionWriter:
    """Introduction内容写作器 - 根据大纲撰写具体内容"""
    
    def __init__(self):
        self.model = LocalModel()
    
    def write_introduction(self, outline: str, material: ResearchData, target_length: int = 1500) -> str:
        """根据大纲撰写Introduction内容"""
        
        # 解析大纲
        sections = self._parse_outline(outline)
        
        # 顺序生成各个段落
        section_contents = self._generate_sections_sequential(sections, material, target_length)
        
        # 优化段落间的连贯性
        refined_contents = self._enhance_coherence(section_contents, material)
        
        # 组合最终Introduction
        final_introduction = self._assemble_introduction(refined_contents)
        
        return final_introduction
    
    def _parse_outline(self, outline: str) -> List[Dict[str, str]]:
        """解析大纲为结构化数据"""
        sections = []
        
        # 提取所有section
        section_pattern = r'## Section \d+: (.+?)\nDescription: (.+?)(?=\n## Section|\nMaterials to reference:|$)'
        matches = re.findall(section_pattern, outline, re.DOTALL)
        
        for i, (title, description) in enumerate(matches):
            # 提取materials信息
            materials_pattern = rf'## Section {i+1}:.*?Materials to reference: (.+?)(?=\n## Section|$)'
            materials_match = re.search(materials_pattern, outline, re.DOTALL)
            materials = materials_match.group(1).strip() if materials_match else ""
            
            sections.append({
                'title': title.strip(),
                'description': description.strip(),
                'materials': materials,
                'index': i
            })
        
        return sections
    
    def _generate_sections_sequential(self, sections: List[Dict[str, str]], material: ResearchData, target_length: int) -> List[str]:
        """顺序生成各个段落内容"""
        
        section_contents = []
        words_per_section = target_length // len(sections) if sections else target_length
        
        
        for i, section in enumerate(sections):
            content = self._write_section_content(section, material, words_per_section)
            section_contents.append(content)
            
            # 显示生成进度和长度信息
            word_count = get_word_count(content)
            char_count = get_text_length(content)
        
        return section_contents
    
    def _write_section_content(self, section: Dict[str, str], material: ResearchData, target_words: int) -> str:
        """撰写单个段落内容"""
        
        prompt = self._create_section_writing_prompt(section, material, target_words)
        
        content = self.model.chat(prompt, temperature=0.6)
        
        # 清理格式标记
        content = content.replace('<content>', '').replace('</content>', '').strip()
        
        return content
    
    def _create_section_writing_prompt(self, section: Dict[str, str], material: ResearchData, target_words: int) -> str:
        """创建段落写作提示"""
        
        prompt = f"""You are writing a section of a research paper Introduction. 

Section to write: "{section['title']}"
Section description: {section['description']}

Research context:
Title: {material.title}

Abstract: {material.abstract}

Figures: {material.figures}

Tables: {material.tables}

Baseline References: {material.baseline_references}

Requirements:
1. Write approximately {target_words} words
2. Use academic writing style appropriate for a top-tier conference/journal
3. Be specific and technical while remaining accessible
4. Reference the provided materials naturally when relevant
5. Maintain logical flow and clear argumentation
6. Do not include citations to external papers (focus on content only)

<content>
[Write the section content here]
</content>"""

        return prompt
    
    def _enhance_coherence(self, section_contents: List[str], material: ResearchData) -> List[str]:
        """增强段落间的连贯性"""
        
        enhanced_contents = section_contents.copy()
        
        # 优化相邻段落的连接
        for i in range(1, len(section_contents)):
            
            if i < len(section_contents) - 1:
                # 中间段落
                context = [section_contents[i-1], section_contents[i], section_contents[i+1]]
            else:
                # 最后一个段落
                context = [section_contents[i-1], section_contents[i], ""]
            
            enhanced_content = self._refine_paragraph_coherence(context, material)
            enhanced_contents[i] = enhanced_content
            
            # 显示优化后的长度信息
            word_count = get_word_count(enhanced_content)
            char_count = get_text_length(enhanced_content)
        
        return enhanced_contents
    
    def _refine_paragraph_coherence(self, context: List[str], material: ResearchData) -> str:
        """优化单个段落的连贯性"""
        
        prompt = f"""You are refining a paragraph in a research paper Introduction to improve coherence and flow.

Previous paragraph:
{context[0]}

Current paragraph to refine:
{context[1]}

Next paragraph:
{context[2] if context[2] else "(This is the last paragraph)"}

Please refine the current paragraph to:
1. Ensure smooth transition from the previous paragraph
2. Prepare for the next paragraph (if not the last)
3. Maintain all key information and technical content
4. Improve clarity and flow
5. Keep the same approximate length

Return only the refined paragraph without any additional formatting or comments."""

        refined_content = self.model.chat(prompt, temperature=0.4)
        
        return refined_content.strip()
    
    def _assemble_introduction(self, section_contents: List[str]) -> str:
        """组装最终的Introduction"""
        
        # 简单地用段落分隔符连接
        introduction = "\n\n".join(section_contents)
        
        # 清理多余的空行
        introduction = re.sub(r'\n{3,}', '\n\n', introduction)
        
        return introduction.strip()


class IntroductionGenerator:
    """主控制器 - 协调各组件生成Introduction"""
    
    def __init__(self):
        
        
        # 初始化各组件
        self.parser = MaterialParser()
        self.outliner = IntroductionOutliner()
        self.writer = IntroductionWriter()
    
    def generate_from_data(self, research_data: ResearchData, target_length: int = 1500, save_path: str = None) -> Dict[str, Any]:
        """从ResearchData对象生成Introduction"""
        
        material = research_data
        
        key_concepts = self.parser.extract_key_concepts(material)
        print(f"发现关键概念: {', '.join(key_concepts[:5])}{'...' if len(key_concepts) > 5 else ''}")
        
        outline = self.outliner.generate_outline(material, key_concepts)
        
        introduction = self.writer.write_introduction(outline, material, target_length)
        
        # 显示最终统计信息
        final_word_count = get_word_count(introduction)
        final_char_count = get_text_length(introduction)
        
        result = {
            'introduction': introduction,
            'outline': outline,
            'key_concepts': key_concepts,
            'statistics': {
                'word_count': final_word_count,
                'char_count': final_char_count,
                'target_length': target_length,
                'completion_rate': f"{(final_word_count/target_length*100):.1f}%"
            }
        }
        
        # 保存结果
        if save_path:
            self._save_results(result, save_path)

        
        return result
    

    
    def _save_results(self, result: Dict[str, Any], save_path: str):
        """保存生成结果"""
        
        # 保存为JSON格式
        if save_path.endswith('.json'):
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 保存为Markdown格式
        elif save_path.endswith('.md'):
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("# Generated Introduction\n\n")
                f.write(result['introduction'])
                f.write("\n\n## Outline\n\n")
                f.write(result['outline'])
                f.write(f"\n\n## Statistics\n\n")
                f.write(f"- Word count: {result['statistics']['word_count']}\n")
                f.write(f"- Character count: {result['statistics']['char_count']}\n")
                f.write(f"- Target length: {result['statistics']['target_length']}\n")
                f.write(f"- Completion rate: {result['statistics']['completion_rate']}\n")
        
        # 默认保存为文本格式
        else:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(result['introduction'])



def main():
    """示例用法"""
    
    parser = argparse.ArgumentParser(description='Generate Introduction from research materials using local models')
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = IntroductionGenerator()
    
    
    # 解析为ResearchData对象
    parser = MaterialParser()

    loader = DataLoader("../paper_data/cvpr/2025")
    all_data = loader.load_all()
    item = list(all_data.keys())[0]
    for item in list(all_data.keys()):
        paper_id = item
        if not os.path.exists(f"writing_agents_results_cvpr/outline/{paper_id}/introduction_results.json"):     
            folder_path = f"writing_agents_results_cvpr/outline/{paper_id}/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)            
            research_data = all_data[item]
            # 生成Introduction (使用ResearchData对象)
            result = generator.generate_from_data(
                research_data=research_data,
                target_length=700,
                save_path=f"writing_agents_results_cvpr/outline/{paper_id}/introduction_results.json"
            )
            print(f"Successfully generate {paper_id}")
            print('总长度',GLOBAL_tokens)
        else:
            print(f"Already generate {paper_id}")           



if __name__ == "__main__":
    main()
