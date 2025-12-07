'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-08-02 16:37:34
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-08-02 16:39:12
FilePath: /LLM/NTP/utils/d.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# data_loader.py
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional


# ------------ 数据结构 ------------
@dataclass
class ResearchData:
    title: str = ""
    abstract: str = ""
    introduction: str = ""          # 新增字段
    figures: Dict[str, Any] = None
    tables: List[Any] = None
    baseline_references: List[Dict[str, str]] = None

    def __post_init__(self):
        if self.abstract is None:
            self.abstract = ""
        if self.introduction is None:
            self.introduction = ""
        if self.figures is None:
            self.figures = {}
        if self.tables is None:
            self.tables = []
        if self.baseline_references is None:
            self.baseline_references = []


def load_research_data_from_json(file_path: str) -> ResearchData:
    """从 JSON 文件加载研究数据"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ResearchData(
        title=data.get("title",""),
        abstract=data.get("abstract", ""),
        introduction=data.get("introduction", ""),   # 读取 introduction
        figures=data.get("figures", {}),
        tables=data.get("tables", []),
        baseline_references=data.get("baseline_references", [])
    )


# ------------ DataLoader ------------
class DataLoader:
    def __init__(self, base_dir: str):
        if not os.path.isdir(base_dir):
            raise ValueError(f"{base_dir} 不是一个有效目录")
        self.base_dir = base_dir

    def load_all(self) -> Dict[str, ResearchData]:
        """以子文件夹的相对路径为键，ResearchData 为值返回字典"""
        result = {}
        for sub_path in self._iter_sub_dirs():
            data = self._load_single(sub_path)
            if data is not None:
                key = os.path.relpath(sub_path, self.base_dir)
                key = key.replace(os.sep, "/")
                result[key] = data
        return result

    def load_one(self, sub_dir_name: str) -> Optional[ResearchData]:
        """加载指定子目录的数据"""
        sub_path = os.path.join(self.base_dir, sub_dir_name)
        return self._load_single(sub_path)

    # ---------------- 内部辅助 ----------------
    def _iter_sub_dirs(self):
        """遍历 base_dir 下的一级子目录"""
        for entry in os.scandir(self.base_dir):
            if entry.is_dir():
                yield entry.path

    @staticmethod
    def _load_single(sub_path: str) -> Optional[ResearchData]:
        """尝试加载单个子目录的 processed_data.json"""
        json_path = os.path.join(sub_path, "processed_data.json")
        if not os.path.isfile(json_path):
            return None
        try:
            return load_research_data_from_json(json_path)
        except Exception as e:
            print(f"[WARN] 跳过 {json_path}：{e}")
            return None


# ---------------- 简单测试 ----------------
if __name__ == "__main__":
    loader = DataLoader("../paper_data/acl/2025/main")
    all_data = loader.load_all()
