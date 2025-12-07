import re
import json

def transfer_consistency(text):
    text = text.lower()  # 转换为小写以进行不区分大小写的匹配
    if "very consistent" in text:
        return 5
    elif "relatively consistent" in text:
        return 4
    elif "neither consistent nor inconsistent" in text or "partially consistent" in text:
        return 3
    elif "relatively inconsistent" in text:
        return 2
    elif "very inconsistent" in text:
        return 1
    else:
        return None

def transfer_rigorousness(text):
    text = text.lower()  # 转换为小写以进行不区分大小写的匹配
    if "very rigorous" in text:
        return 5
    elif "relatively rigorous" in text:
        return 4
    elif "neither rigorous nor unrigorous" in text or "partially rigorous" in text:
        return 3
    elif "relatively unrigorous" in text:
        return 2
    elif "very unrigorous" in text:
        return 1
    else:
        return None  # 如果没有匹配的关键词，返回None

def transfer_coherence(text):
    text = text.lower()  # 转换为小写以进行不区分大小写的匹配
    if "very coherent" in text:
        return 5
    elif "relatively coherent" in text:
        return 4
    elif "neither coherent nor incoherent" in text:
        return 3
    elif "relatively incoherent" in text:
        return 2
    elif "very incoherent" in text:
        return 1
    else:
        return None  # 如果没有匹配的关键词，返回None

def transfer_academic_quality(text):
    text = text.lower()  # 转换为小写以进行不区分大小写的匹配
    if "very high" in text:
        return 5
    elif "relatively high" in text:
        return 4
    elif "medium" in text:
        return 3
    elif "relatively low" in text:
        return 2
    elif "very low" in text:
        return 1
    else:
        return None  # 如果没有匹配的关键词，返回None


def extract_json_from_text(text):

    json_pattern = r'\{.*?\}'
    matches = re.findall(json_pattern, text, re.DOTALL)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue  # 跳过无效的JSON格式
    
    # 如果没有找到有效的JSON
    return None

def transfer_structure(text):
    text = str(extract_json_from_text(text))
    text = text.lower()  # 转换为小写以进行不区分大小写的匹配
    if "background" in text:
        return "Background"
    elif "problem and limitations of existing methods" in text:
        return "Problem"
    elif "brief method overview and summary of main results" in text:
        return "Method"
    elif "method overview" in text:
        return "Method"
    elif "our contributions" in text:
        return "Contributions"
    elif "contributions" in text:
        return "Contributions"
    else:
        return None  # 如果没有匹配的关键词，返回None