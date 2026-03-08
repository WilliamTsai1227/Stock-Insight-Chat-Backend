import json
import os
import asyncio
import pandas as pd
from dotenv import load_dotenv

import sys
project_root = "/Users/william/Documents/project/Stock-Insight-Chat-Backend"
sys.path.append(project_root)

from app.custom_langchain.llm_model import OpenAIChat

load_dotenv(os.path.join(project_root, ".env"))

DATASET_DIR = os.path.join(project_root, "evaluation/knowledge/ContextPrecision/dataset")

async def evaluate_relevance(llm, question, context):
    """
    呼叫 LLM 判斷相關性並生成評語
    """
    prompt = f"""請擔任一個公正的評審。你的任務是判斷下方的「參考資料 (Context)」是否能幫助回答「問題 (Question)」。

問題：
{question}

參考資料：
{context}

請遵循以下格式回傳結果 (僅回傳 JSON 格式)：
{{
  "relevance": "relevant" 或 "irrelevant",
  "reason": "簡短的評語說明為什麼該資料相關或不相關"
}}
"""
    # 這裡直接呼叫模型
    response = await llm.model.ainvoke(prompt)
    try:
        # 清理 JSON 回傳字串中的 Markdown 標記
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_content)
        return result
    except Exception as e:
        print(f"解析 JSON 報錯: {e}, Content: {response.content}")
        return {"relevance": "irrelevant", "reason": "解析錯誤"}

async def evaluation_step():
    llm = OpenAIChat()
    
    # 讀取第二階段資料
    input_file = os.path.join(DATASET_DIR, "stage2_retrieved.json")
    with open(input_file, "r", encoding="utf-8") as f:
        retrieved_data = json.load(f)
    
    stage3_data = []
    total_precision_scores = []
    
    print("啟動 LLM 相關性評核... (不需參考答案模式)")
    
    for item in retrieved_data:
        question = item["question"]
        contexts = item["retrieved_contexts"]
        
        relevant_count = 0
        judgments = []
        
        print(f"正在評核問題: {question[:15]}...")
        
        for c in contexts:
            res = await evaluate_relevance(llm, question, c["content"])
            is_relevant = 1 if res["relevance"] == "relevant" else 0
            relevant_count += is_relevant
            
            judgments.append({
                "title": c["title"],
                "content_preview": c["content"][:50] + "...",
                "relevance": res["relevance"],
                "reason": res["reason"]
            })
            
        # 計算該題精準度 (Context Precision)
        # 定義：檢索出的資料中有幾篇是真的相關的
        precision_score = relevant_count / len(contexts) if contexts else 0
        total_precision_scores.append(precision_score)
        
        stage3_data.append({
            "question": question,
            "precision_score": precision_score,
            "judgments": judgments
        })

    # 計算總體指標
    overall_score = sum(total_precision_scores) / len(total_precision_scores) if total_precision_scores else 0
    
    final_output = {
        "overall_context_precision": overall_score,
        "detail_results": stage3_data
    }

    # 存入第三階段資料集 (JSON)
    output_f_json = os.path.join(DATASET_DIR, "stage3_evaluated.json")
    with open(output_f_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    print(f"\n第三階段資料集完成！總體檢索精準度為: {overall_score:.2f}")
    print(f"詳細報告已儲存至: {output_f_json}")

if __name__ == "__main__":
    asyncio.run(evaluation_step())
