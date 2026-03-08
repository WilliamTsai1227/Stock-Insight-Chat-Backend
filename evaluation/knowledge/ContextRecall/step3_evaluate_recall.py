import json
import os
import asyncio
from dotenv import load_dotenv

import sys
project_root = "/Users/william/Documents/project/Stock-Insight-Chat-Backend"
sys.path.append(project_root)

load_dotenv(os.path.join(project_root, ".env"))

DATASET_DIR = os.path.join(project_root, "evaluation/knowledge/ContextRecall/dataset")

async def calculate_recall():
    input_file = os.path.join(DATASET_DIR, "stage3_ready_for_eval.json")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total_recall_scores = []
    detailed_results = []
    
    print("🎯 開始計算 Context Recall@5...")
    
    for item in data:
        question = item["question"]
        target_ids = set(item["ground_truth_ids"])
        retrieved_ids = set([doc["id"] for doc in item["top5_retrieval"]])
        
        # 計算前 5 筆中有多少落在基準標註組 (Top-20 中 被 LLM 判定為相關的)
        matches = target_ids.intersection(retrieved_ids)
        
        # Recall@5 = (Top-5 中相關數) / (Top-20 中總相關數)
        denominator = len(target_ids)
        if denominator == 0:
            recall_score = 1.0  # 如果 20 個裡都沒相關的，前 5 個沒相關也算召回成功
        else:
            recall_score = len(matches) / denominator
            
        total_recall_scores.append(recall_score)
        
        detailed_results.append({
            "question": question,
            "target_relevant_count": denominator,
            "system_retrieved_relevant_count": len(matches),
            "recall_score": recall_score
        })
        print(f"完成計算: {question[:15]}... Recall: {recall_score:.2f}")

    overall_recall = sum(total_recall_scores) / len(total_recall_scores) if total_recall_scores else 0
    
    final_report = {
        "overall_context_recall": overall_recall,
        "detail_results": detailed_results
    }
    
    output_f_json = os.path.join(DATASET_DIR, "stage4_recall_final.json")
    with open(output_f_json, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
        
    print(f"\n📊 Context Recall 評估完成！總體召回率為: {overall_recall:.2f}")
    print(f"詳細報告已儲存至: {output_f_json}")

if __name__ == "__main__":
    asyncio.run(calculate_recall())
