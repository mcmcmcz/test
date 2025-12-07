python evaluate_cover_sections.py --generated_type qwen_outline_base > temp.log 2>&1 &


python evaluate_bleu.py --generated_path ../writing_agents_results/stage_0913/

cd /home/mczhang/zmc-dl/LLM/NTP/outline_baseline/writing_agents_results
python test_structure_reasonable.py --evaluate_path temp_structure.json

nohup python evaluate_structure.py --generated_type stage_ta > stru_llama_naive_great_prompt_0916.log 2>&1 &

python perplexity_similarity.py --generate_type qwen_naive_sft