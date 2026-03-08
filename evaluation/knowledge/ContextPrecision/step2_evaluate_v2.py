# 從 Ragas 0.4+ 開始，建議從各別模組引入以確保與 InstructorLLM 相容
from ragas.llms import llm_factory
from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
import json
import os
import asyncio
import pandas as pd
from dotenv import load_dotenv

import sys
project_root = "/Users/william/Documents/project/Stock-Insight-Chat-Backend"
sys.path.append(project_root)

# 載入環境變數
load_dotenv(os.path.join(project_root, ".env"))

DATASET_DIR = os.path.join(project_root, "evaluation/knowledge/ContextPrecision/dataset")

async def run_ragas_evaluation():
    # 1. 讀取第二階段資料
    input_file = os.path.join(DATASET_DIR, "stage2_retrieved.json")
    if not os.path.exists(input_file):
        print(f"錯誤：找不到第二階段資料集 {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        retrieved_data = json.load(f)
    
    # 2. 準備 Dataset (Ragas 0.4.3 使用 user_input 與 retrieved_contexts)
    prepared_data = {
        "user_input": [],
        "retrieved_contexts": []
    }
    
    for item in retrieved_data:
        prepared_data["user_input"].append(item["question"])
        # RAG 撈回來的多個片段組成一個列表
        contexts = [c["content"] for c in item["retrieved_contexts"]]
        prepared_data["retrieved_contexts"].append(contexts)
        
    df = pd.DataFrame(prepared_data)
    dataset = Dataset.from_pandas(df)
    
    print("啟動 Ragas v0.4.3 進行評估 (使用 ContextRelevance - 無須標準答案模式)...")
    
    # 3. 初始化 Ragas 控制器與指標 (Ragas 0.4+ 建議使用 llm_factory)
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm = llm_factory("gpt-4o", client=openai_client)
    
    # 【徹底修復 NaN 與 AttributeError 的關鍵】
    # Ragas 0.4.x 的 InstructorLLM 漏掉了 string-based 的生成方法，導致 Ragas 內部評分機制失效並返回 NaN
    # 我們在這裡手動補上這兩個方法，讓它能夠處理純文字生成請求
    if not hasattr(llm, "generate_text"):
        def generate_text(prompt, n=1, temperature=1e-7, stop=None, callbacks=None):
            # 建立一個簡單的文字回傳模擬
            from ragas.llms.base import CompletionResponse
            response = openai_client.chat.completions.create(
                model=llm.model,
                messages=[{"role": "user", "content": prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)}],
                temperature=temperature,
                n=n
            )
            return CompletionResponse(text=response.choices[0].message.content)
        llm.generate_text = generate_text

    if not hasattr(llm, "agenerate_text"):
        async def agenerate_text(prompt, n=1, temperature=1e-7, stop=None, callbacks=None):
            # 為了簡單起見，直接在 async wrapper 中呼叫 sync 版本 (或者直接實作 async)
            return llm.generate_text(prompt, n, temperature, stop, callbacks)
        llm.agenerate_text = agenerate_text
    
    # ContextRelevance 是 Ragas 提供的無須 reference Answer 的檢索評分指標
    # 目前 0.4.3 版必須從 collections 引入才能正確處理 InstructorLLM
    from ragas.metrics.collections.context_relevance import ContextRelevance
    cr_metric = ContextRelevance(llm=llm)
    
    # 4. 執行評估
    result = evaluate(
        dataset,
        metrics=[cr_metric]
    )
    
    # 5. 結果處理與輸出
    result_df = result.to_pandas()
    
    # 動態獲取指標欄位名稱 (Ragas 0.4 會將指標名稱作為欄位名)
    metric_cols = [col for col in result_df.columns if "relevance" in col or "relevancy" in col]
    metric_col = metric_cols[0] if metric_cols else "context_relevance"

    # 同步轉換回 JSON 結構供前台或後續分析使用
    stage3_detailed = []
    
    for _, row in result_df.iterrows():
        stage3_detailed.append({
            "question": row["user_input"],
            "context_relevancy_score": float(row[metric_col]),
            "contexts_preview": [c[:50] + "..." for c in row["retrieved_contexts"]]
        })
    
    final_output = {
        "overall_ragas_relevancy_score": float(result_df[metric_col].mean()),
        "detail_results": stage3_detailed
    }

    output_f_json = os.path.join(DATASET_DIR, "stage3_evaluated_ragas_v2.json")
    with open(output_f_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
        
    print(f"\nRagas 評估完成！總體相關度分數為: {final_output['overall_ragas_relevancy_score']:.2f}")
    print(f"詳細報告已儲存至: {output_f_json}")

if __name__ == "__main__":
    asyncio.run(run_ragas_evaluation())
