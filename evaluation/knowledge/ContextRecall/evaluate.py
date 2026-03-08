import json
import asyncio
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall
import httpx
import os
from dotenv import load_dotenv

# 加載環境變數 (從專案根目錄)
load_dotenv(os.path.join(os.path.dirname(__file__), "../../../.env"))

# API 端點設定
BASE_URL = "http://localhost:8000"

async def get_rag_results(question, stock_code=""):
    """
    呼叫本地 API 獲取檢索結果與回答
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 呼叫 getAIResponse 獲取生成回答
        response = await client.post(
            f"{BASE_URL}/chatbot/api/getAIResponse",
            json={"query": question, "stock_code": stock_code},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"Error calling getAIResponse: {response.text}")
            return None
            
        data = response.json()
        
        # 呼叫向量搜尋端點獲取檢索到的上下文
        search_res = await client.post(
            f"{BASE_URL}/chatbot/api/vectorSearch",
            json={"query": question, "top_k": 5},
            headers={"Content-Type": "application/json"}
        )
        
        contexts = []
        if search_res.status_code == 200:
            contexts = [res["content"] for res in search_res.json().get("results", [])]
        else:
            print(f"Error calling vectorSearch: {search_res.text}")

        return {
            "question": question,
            "answer": data.get("answer", ""),
            "contexts": contexts
        }

async def run_evaluation():
    """
    執行 Context Recall 獨立評估
    """
    # 優先讀取自己目錄下的 dataset
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    test_set_path = os.path.join(dataset_dir, "test_set.json")
    
    # 如果自己下面沒有，則回退到上一層的通用測試集
    if not os.path.exists(test_set_path):
        test_set_path = os.path.join(os.path.dirname(__file__), "../test_set.json")

    if not os.path.exists(test_set_path):
        print(f"找不到測試集檔案: {test_set_path}")
        return

    with open(test_set_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    print(f"Context Recall: 開始為 {len(test_data)} 個問題收集資料...")
    
    for item in test_data:
        res = await get_rag_results(item["question"], item.get("stock_code", ""))
        if res:
            res["ground_truth"] = item["ground_truth"]
            results.append(res)

    if not results:
        print("沒有收集到任何有效資料。")
        return

    # 轉換為 Ragas 格式
    df = pd.DataFrame(results)
    dataset = Dataset.from_pandas(df)

    print("啟動 Ragas Context Recall 評估...")
    score = evaluate(dataset, metrics=[context_recall])
    
    output_df = score.to_pandas()
    print("\n--- Context Recall 評估結果 ---")
    print(output_df)
    
    # 儲存結果到自己的 dataset 目錄
    output_file = os.path.join(dataset_dir, "recall_results.csv")
    output_df.to_csv(output_file, index=False)
    print(f"\n評估完成！結果已儲存至: {output_file}")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
