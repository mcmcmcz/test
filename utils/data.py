'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2025-07-08 12:22:00
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2025-07-15 15:24:10
FilePath: /zmc-dl/LLM/NTP/utils/data.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import json

if __name__ == "__main__":
    path = '../dpo_data/3/'
    lis = os.listdir(path)
    print(len(lis))
    all_data = []
    for file in lis:
        if file.endswith('.json'):
            file_path = os.path.join(path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if len(data['messages']) ==4:
                data['messages']['input'] = data['messages']['input'].replace('1,300 and 1,500', '600 and 1,200')
                new_data ={
                    "instruction": data['messages']['instruction'],
                    "input": data['messages']['input'],
                    "chosen": data['messages']['chosen'],
                    "rejected": data['messages']['rejected'],
                }
                all_data.append(new_data)
    with open('/home/mczhang/zmc-dl/LLM/LLaMA-Factory/data/dpo_0715_3.json', 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
