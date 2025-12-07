import json
import os
import argparse
import numpy as np

def get_data(path):
    with open(path,'r',encoding='utf-8') as fp:
        data = json.load(fp)
    return data


def judge_reasonable(items):
    item = items[1]
    lenth = len(item['background']) + len(item['problem_statement']) + len(item['method_overview']) + len(item['contributions'])
    # count_1 = [part not in ['Method','Contributions'] for part in item['background']]
    # count_2 = [part not in ['Method','Contributions'] for part in item['problem_statement']]
    # count_3 = [part not in ['Background','Problem'] for part in item['method_overview']]
    # count_4 = [part not in ['Background','Problem'] for part in item['contributions']]
    count_1 = [part not in ['Method','Contributions','Problem'] for part in item['background']]
    count_2 = [part not in ['Method','Contributions','Background'] for part in item['problem_statement']]
    count_3 = [part not in ['Background','Problem','Contributions'] for part in item['method_overview']]
    count_4 = [part not in ['Background','Problem','Method'] for part in item['contributions']]
    count = count_1 + count_2 + count_3 + count_4
    # return [np.mean(count_1), np.mean(count_2), np.mean(count_3), np.mean(count_4), sum(count)/lenth]
    results = {}
    if item['background']:
        results['background'] = np.mean(count_1)
    if item['problem_statement']:
        results['problem_statement'] = np.mean(count_2)
    if item['method_overview']:
        results['method_overview'] = np.mean(count_3)    
    if item['contributions']:
        results['contributions'] = np.mean(count_4) 
    if lenth == 0:
        results['all'] = 0
    else:
        results['all'] = sum(count)/lenth
    return results
    
if __name__ == "__main__":
    #  python test_structure_reasonable.py --evaluate_path naive_structure.json
    parser = argparse.ArgumentParser(description='Evaluating sturcture reasonable')
    parser.add_argument('--evaluate_path', type=str, help='NONE')
    args = parser.parse_args()
    data = get_data(args.evaluate_path)
    print(np.mean([judge_reasonable(item)['background'] for item in data.items() if 'background' in judge_reasonable(item)]))
    print(np.mean([judge_reasonable(item)['problem_statement'] for item in data.items() if 'problem_statement' in judge_reasonable(item)]))
    print(np.mean([judge_reasonable(item)['method_overview'] for item in data.items() if 'method_overview' in judge_reasonable(item)]))
    print(np.mean([judge_reasonable(item)['contributions'] for item in data.items() if 'contributions' in judge_reasonable(item)]))
    print(np.mean([judge_reasonable(item)['all'] for item in data.items()]))