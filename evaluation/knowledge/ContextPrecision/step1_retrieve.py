import json
import os
import asyncio
from dotenv import load_dotenv
from bson import ObjectId

# 確保能從根目錄讀取專案模組
import sys
project_root = "/Users/william/Documents/project/Stock-Insight-Chat-Backend"
sys.path.append(project_root)

from app.database import db
from app.custom_langchain.llm_model import OpenAIAppEmbeddings

load_dotenv(os.path.join(project_root, ".env"))

# 設定
TOP_K = 5  # 可在此自行調整
DATASET_DIR = os.path.join(project_root, "evaluation/knowledge/ContextPrecision/dataset")

async def retrieve_step():
    # 建立資料庫連線
    db.connect()
    embedder = OpenAIAppEmbeddings()
    
    # 讀取第一階段問題
    input_file = os.path.join(DATASET_DIR, "stage1_questions.json")
    with open(input_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    stage2_data = []
    
    print(f"開始執行向量檢索... (Top-K: {TOP_K})")
    
    for item in questions:
        query = item["question"]
        stock_code = item.get("stock_code")
        
        # 1. 將問題轉為向量
        vector = await embedder.aembed_query(query)
        
        # 2. 建立 MongoDB 向量搜尋 Pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": os.getenv("MONGODB_VECTOR_INDEX", "vector_index"),
                    "path": "embedding",
                    "queryVector": vector,
                    "numCandidates": TOP_K * 10,
                    "limit": TOP_K
                }
            },
            {
                "$project": {
                    "content": 1,
                    "metadata": 1,
                    "score": { "$meta": "vectorSearchScore" }
                }
            }
        ]
        
        # 加入過濾條件
        if stock_code:
            pipeline[0]["$vectorSearch"]["filter"] = { "metadata.stock_list": stock_code }
            
        collection = db.db["news_vector"]
        cursor = collection.aggregate(pipeline)
        
        retrieved_contexts = []
        async for doc in cursor:
            retrieved_contexts.append({
                "content": doc.get("content", ""),
                "score": doc.get("score", 0),
                "title": doc.get("metadata", {}).get("title", "無標題")
            })
            
        stage2_data.append({
            "question": query,
            "stock_code": stock_code,
            "retrieved_contexts": retrieved_contexts
        })
        print(f"完成檢索: {query[:15]}...")

    # 存入第二階段資料集
    output_file = os.path.join(DATASET_DIR, "stage2_retrieved.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stage2_data, f, ensure_ascii=False, indent=2)
    
    print(f"第二階段資料集製作完成: {output_file}")
    db.close()

if __name__ == "__main__":
    asyncio.run(retrieve_step())
