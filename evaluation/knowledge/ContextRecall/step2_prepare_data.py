import json
import os
import asyncio
from dotenv import load_dotenv

import sys
project_root = "/Users/william/Documents/project/Stock-Insight-Chat-Backend"
sys.path.append(project_root)

from app.database import db
from app.custom_langchain.llm_model import OpenAIChat, OpenAIAppEmbeddings

load_dotenv(os.path.join(project_root, ".env"))

DATASET_DIR = os.path.join(project_root, "evaluation/knowledge/ContextRecall/dataset")
TOP_K_SYSTEM = 5

async def judge_relevance(llm, question, content):
    """
    LLM 幫忙判斷這筆資料是否屬於應該被找到的相關資料
    """
    prompt = f"""請擔任資深財經分析師。判斷這份資料是否對於回答該問題「極度關鍵」且「高度相關」。
問題：{question}
資料內容：{content}

請遵循以下格式回傳結果 (僅回傳 JSON 格式)：
{{
  "is_relevant": true 或 false
}}
"""
    response = await llm.model.ainvoke(prompt)
    try:
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_content)
        return result.get("is_relevant", False)
    except:
        return False

async def filtering_and_retrieval_step():
    db.connect()
    llm = OpenAIChat()
    embedder = OpenAIAppEmbeddings()
    
    input_file = os.path.join(DATASET_DIR, "stage2_top20_raw.json")
    with open(input_file, "r", encoding="utf-8") as f:
        stage2_data = json.load(f)
    
    stage3_data = []
    
    print(f"🕵️ 正在過濾基準標註組，並蒐集系統 Top-{TOP_K_SYSTEM} 的結果...")
    
    for item in stage2_data:
        question = item["question"]
        stock_code = item["stock_code"]
        top20_raw = item["top20_raw_docs"]
        
        # 1. 過濾出真正的基準標註集 (LLM 過濾)
        ground_truth_docs = []
        for doc in top20_raw:
            is_rel = await judge_relevance(llm, question, doc["content"])
            if is_rel:
                ground_truth_docs.append(doc)
                
        # 2. 蒐集系統真正會用到的 Top-5 片段
        vector = await embedder.aembed_query(question)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": os.getenv("MONGODB_VECTOR_INDEX", "vector_index"),
                    "path": "embedding",
                    "queryVector": vector,
                    "numCandidates": TOP_K_SYSTEM * 10,
                    "limit": TOP_K_SYSTEM
                }
            },
            {
                "$project": { "content": 1, "metadata": 1 }
            }
        ]
        if stock_code:
            pipeline[0]["$vectorSearch"]["filter"] = { "metadata.stock_list": stock_code }
            
        cursor = db.db["news_vector"].aggregate(pipeline)
        top5_retrieval = []
        async for doc in cursor:
            top5_retrieval.append({
                "id": str(doc["_id"]),
                "content": doc.get("content", ""),
                "title": doc.get("metadata", {}).get("title", "無標題")
            })
            
        stage3_data.append({
            "question": question,
            "relevant_target_count": len(ground_truth_docs),
            "ground_truth_ids": [d["id"] for d in ground_truth_docs],
            "top5_retrieval": top5_retrieval
        })
        print(f"完成數據蒐集: {question[:15]}... (目標相關數: {len(ground_truth_docs)})")

    output_file = os.path.join(DATASET_DIR, "stage3_ready_for_eval.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stage3_data, f, ensure_ascii=False, indent=2)
    
    print(f"📊 評估用數據集準備完成: {output_file}")
    db.close()

if __name__ == "__main__":
    asyncio.run(filtering_and_retrieval_step())
