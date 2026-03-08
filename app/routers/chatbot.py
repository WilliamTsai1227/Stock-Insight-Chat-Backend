from fastapi import APIRouter, Body
from app.database import db
from app.custom_langchain.llm_model import OpenAIChat, OpenAIAppEmbeddings
import os
from bson import ObjectId

# 遞迴處理 MongoDB 回傳的資料結構，確保 ObjectId 轉為字串
def convert_doc(doc):
    if isinstance(doc, list):
        return [convert_doc(x) for x in doc]
    if isinstance(doc, dict):
        return {k: convert_doc(v) for k, v in doc.items()}
    if isinstance(doc, ObjectId):
        return str(doc)
    return doc

router = APIRouter()

@router.post("/api/getAIResponse")
async def get_ai_response(request_body: dict = Body(...)):
    """
    專業二階段 RAG 流程：
    1. 定位：從 news_vector 找出最相關的 source_id。
    2. 提取：去原始 Collection (news / AI_news_analysis) 撈取全文。
    3. 生成：呼叫 OpenAI GPT-4o 產生精準回答。
    """
    user_query = request_body.get("query", "")
    stock_code = request_body.get("stock_code")
    top_k = 5 # 既然要撈全文，取 Top 5 原始文件通常已足夠

    # --- A. 向量搜尋 (第一階段：定位) ---
    embedder = OpenAIAppEmbeddings()
    vector = await embedder.aembed_query(user_query)

    pipeline = [
        {
            "$vectorSearch": {
                "index": os.getenv("MONGODB_VECTOR_INDEX", "vector_index"),
                "path": "embedding",
                "queryVector": vector,
                "numCandidates": top_k * 10,
                "limit": top_k
            }
        },
        {
            "$project": {
                "metadata": 1,
                "score": { "$meta": "vectorSearchScore" }
            }
        }
    ]
    
    if stock_code:
        pipeline[0]["$vectorSearch"]["filter"] = { "metadata.stock_list": stock_code }

    collection_vector = db.db["news_vector"]
    cursor = collection_vector.aggregate(pipeline)
    vector_results = [doc async for doc in cursor]

    # --- B. 提取原始全文 (第二階段：還原) ---
    unique_sources = {} # 用來去重，鍵值為 "collection:sid"
    for v_res in vector_results:
        meta = v_res.get('metadata', {})
        sid = meta.get('source_id')
        coll_name = meta.get('source_collection')
        
        if not sid or not coll_name:
            continue
            
        # 處理 MongoDB 可能回傳的 {'$oid': '...'} 格式或直接是 ObjectId 物件
        sid_str = str(sid['$oid']) if isinstance(sid, dict) and '$oid' in sid else str(sid)
        
        # 使用 "集合名:ID" 作為唯一鍵值，確保絕對不重複查詢
        unique_key = f"{coll_name}:{sid_str}"
        
        if unique_key not in unique_sources:
            unique_sources[unique_key] = {
                "id_str": sid_str,
                "collection": coll_name
            }

    full_docs_context = []
    for u_key, source_info in unique_sources.items():
        coll = db.db[source_info["collection"]]
        # 查詢原始文件
        raw_doc = await coll.find_one({"_id": ObjectId(source_info["id_str"])})
        
        if raw_doc:
            if source_info["collection"] == "news":
                # 原始新聞處理
                title = raw_doc.get("title", "無標題")
                content = raw_doc.get("content", "")
                full_docs_context.append(f"【新聞】{title}\n內容：{content}")
            
            elif source_info["collection"] == "AI_news_analysis":
                # AI 分析報告處理
                title = raw_doc.get("article_title", "AI 專題分析")
                summary = raw_doc.get("summary", "")
                news_pts = raw_doc.get("important_news", "")
                potential = raw_doc.get("potential_stocks_and_industries", "")
                
                analysis_text = f"主題：{title}\n摘要：{summary}\n重點：\n{news_pts}\n產業展望：\n{potential}"
                full_docs_context.append(f"【AI 分析】\n{analysis_text}")

    # --- C. 組合 Prompt 並呼叫 LLM ---
    context_text = "\n\n---\n\n".join(full_docs_context)
    
    if not context_text:
        return {"answer": "抱歉，資料庫中目前沒有相關資訊可以回答您的問題。", "source_count": 0}

    llm = OpenAIChat()
    response = await llm.ainvoke({
        "userEnterQuery": user_query,
        "knowledgeDoc": context_text
    })

    return {
        "answer": response,
        "source_count": len(full_docs_context),
        "sources": [doc.split('\n')[0] for doc in full_docs_context]
    }

@router.post("/api/vectorSearch")
async def vector_search(request_body: dict = Body(...)):
    """
    測試向量搜尋品質的 API。
    """
    query = request_body.get("query", "")
    top_k = request_body.get("top_k", 5)
    
    embedder = OpenAIAppEmbeddings()
    vector = await embedder.aembed_query(query)
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": os.getenv("MONGODB_VECTOR_INDEX", "vector_index"),
                "path": "embedding",
                "queryVector": vector,
                "numCandidates": top_k * 10,
                "limit": top_k
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
    
    collection = db.db["news_vector"]
    cursor = collection.aggregate(pipeline)
    
    # 確保所有結果中的 ObjectId 都已轉換
    results = [convert_doc(doc) async for doc in cursor]
    
    return {"results": results}
