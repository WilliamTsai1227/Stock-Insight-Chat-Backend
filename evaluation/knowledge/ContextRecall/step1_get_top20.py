import json
import os
import asyncio
from dotenv import load_dotenv

import sys
project_root = "/Users/william/Documents/project/Stock-Insight-Chat-Backend"
sys.path.append(project_root)

from app.database import db
from app.custom_langchain.llm_model import OpenAIAppEmbeddings

load_dotenv(os.path.join(project_root, ".env"))

# 設定 K=20 作為基準回憶集 (Recall Set)
TOP_K_BASE = 20
DATASET_DIR = os.path.join(project_root, "evaluation/knowledge/ContextRecall/dataset")

async def get_top20_step():
    db.connect()
    embedder = OpenAIAppEmbeddings()
    
    input_file = os.path.join(DATASET_DIR, "stage1_questions.json")
    with open(input_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    stage2_data = []
    
    print(f"📡 製作基準標註集... (K={TOP_K_BASE})")
    
    for item in questions:
        query = item.get("question")
        stock_code = item.get("stock_code")
        
        vector = await embedder.aembed_query(query)
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": os.getenv("MONGODB_VECTOR_INDEX", "vector_index"),
                    "path": "embedding",
                    "queryVector": vector,
                    "numCandidates": TOP_K_BASE * 10,
                    "limit": TOP_K_BASE
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
        
        if stock_code:
            pipeline[0]["$vectorSearch"]["filter"] = { "metadata.stock_list": stock_code }
            
        cursor = db.db["news_vector"].aggregate(pipeline)
        
        top20_docs = []
        async for doc in cursor:
            top20_docs.append({
                "id": str(doc["_id"]),
                "content": doc.get("content", ""),
                "title": doc.get("metadata", {}).get("title", "無標題")
            })
            
        stage2_data.append({
            "question": query,
            "stock_code": stock_code,
            "top20_raw_docs": top20_docs
        })
        print(f"完成基準蒐集: {query[:15]}...")

    output_file = os.path.join(DATASET_DIR, "stage2_top20_raw.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stage2_data, f, ensure_ascii=False, indent=2)
    
    print(f"✨ 基準回憶集製作完成: {output_file}")
    db.close()

if __name__ == "__main__":
    asyncio.run(get_top20_step())
