# Stock Insight Chat Backend 🚀

本專案是一個基於 **FastAPI** 框架的生成式 AI RAG (Retrieval-Augmented Generation) 後端應用。專為股票市場資訊檢索、分析與智慧問答設計，深度整合了 MongoDB Atlas Vector Search 與 OpenAI GPT-4o。

---

## 🌟 核心特色

- **二階段 RAG (Two-stage RAG)**: 完美結合向量搜尋與原始內容提取。
  1. **定位**: 透過向量搜尋精準找出相關新聞片段。
  2. **還原**: 根據 ID 回到原始集合 (`news` / `AI_news_analysis`) 撈取全文，避免片段化資訊。
- **智慧去重與整合**: 自動識別並合併來自同一新聞來源的片段，節省 Token 並提供完整上下文。
- **多維度檢索**: 支援 `stock_code` 個股預過濾，縮小範圍並提高回答相關性。
- **自動序列化優化**: 內建遞迴工具函式，自動解決 MongoDB `ObjectId` 與 JSON 序列化的相容性問題。

---

## 🏗️ 專案架構

```text
Backend/
├── app/
│   ├── app.py              # FastAPI 應用入口與路徑掛載
│   ├── database.py         # MongoDB 連接池管理
│   ├── routers/
│   │   ├── base_router.py  # 基礎路由配置
│   │   └── chatbot.py      # RAG 核心邏輯、搜尋 API
│   └── custom_langchain/
│       └── llm_model.py    # OpenAI LLM 與 Embedding 封裝
├── requirements.txt        # 專案相依套件
├── .env                    # 環境變數設定
└── README.md               # 專案說明文件
```

---

## 🛠️ 安裝與啟動

### 1. 環境準備
建議使用 Python 3.11+
```bash
# 建立虛擬環境
python -m venv venv
source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt
```

### 2. 環境變數設定
在根目錄建立 `.env` 檔案：
```env
MONGO_URI=your_mongodb_atlas_uri
MONGO_DB_NAME=stock_insight
MONGODB_VECTOR_INDEX=vector_index
OPENAI_API_KEY=sk-your-openai-key
```

### 3. 啟動伺服器
```bash
uvicorn app.app:app --reload
```

---

## 📡 API 規格

> **重要提醒**: 所有 `POST` 請求皆必須在 Header 帶上 `"Content-Type: application/json"`。

### 1. RAG AI 問答 (Main API)
- **Endpoint**: `POST /chatbot/api/getAIResponse`
- **功能**: 執行完整 RAG 流程，從多個資料來源（新聞、分析報告）生成回答。
- **Body**:
  ```json
  {
    "query": "台積電最近的營收表現與散熱供應鏈的連動關係？",
    "stock_code": "2330"
  }
  ```

### 2. 純向量搜尋 (Debug API)
- **Endpoint**: `POST /chatbot/api/vectorSearch`
- **功能**: 用於驗證檢索品質，返回相似度分數與 `metadata`。
- **Body**:
  ```json
  {
    "query": "半導體產業展望",
    "top_k": 3
  }
  ```

---

## 🧪 測試指令 (cURL 範例)

**測試 RAG 問答：**
```bash
curl -X POST "http://localhost:8000/chatbot/api/getAIResponse" \
     -H "Content-Type: application/json" \
     -d '{"query": "台積電表現如何？", "stock_code": "2330"}'
```

**測試向量搜尋：**
```bash
curl -X POST "http://localhost:8000/chatbot/api/vectorSearch" \
     -H "Content-Type: application/json" \
     -d '{"query": "AI 伺服器散熱", "top_k": 3}'
```

---

## 📊 RAG 評估系統 (Context Precision)

專案內建了一套基於 **Ragas** 概念的檢索精準度評估工具，位於 `evaluation/knowledge/ContextPrecision/`。該工具專注於測試「向量搜尋是否撈回了真正相關的資料」，且 **不需要標準答案 (Ground Truth)**。

### 1. 三階段資料集設計
| 階段 | 檔案名稱 | 內容 |
| :--- | :--- | :--- |
| **第一階段** | `stage1_questions.json` | 包含 20 題模擬股市檢索的問題（如台積電毛利、ASML 預測等）。 |
| **第二階段** | `stage2_retrieved.json` | 執行檢索腳本後，記錄下每一題拿回來的 Top-K 文本片段。 |
| **第三階段** | `stage3_evaluated.json` | 讓 LLM (GPT-4o) 擔任裁判，為每一片段給出「相關性判斷」與「具體評語」，並計算總分。 |

### 2. 如何執行評估
請確保伺服器環境變數 (.env) 已設定完成，且資料庫中有對應資料。

**第一步：執行向量檢索 (製作第二階段資料)**
```bash
python evaluation/knowledge/ContextPrecision/step1_retrieve.py
```
> 您可以在腳本中調整 `TOP_K` 變數來測試不同的檢索數量對精準度的影響。

**第二步：執行 LLM 閱卷 (製作第三階段評分報告)**

執行以下指令啟動 AI 評分：
```bash
python evaluation/knowledge/ContextPrecision/step2_evaluate.py
```

*   **評分模式**: 採用 **Without Ground Truth (無須標準答案)** 模式，直接由 GPT-4o 判斷相關性。
*   **核心優勢**: 
    1. **極度穩定**: 自定義評核邏輯，不受第三方套件 (如 Ragas) 版本更新導致的 API 變動影響。
    2. **具備詳細評語 (Reason)**: 這是本系統最強大的地方。AI 會針對每一條檢索資料給出「為什麼相關」或「為什麼是無用雜訊」的具體理由，協助開發者精準優化 Embedding 與 Chunking 策略。
*   **輸出檔案**: `evaluation/knowledge/ContextPrecision/dataset/stage3_evaluated.json`。

### 3. 評估指標定義

*   **Context Precision (檢索精準度)**: 衡量檢索出的 K 筆資料中，真正與問題相關（對回答有幫助）的資料比例。
*   此系統能協助診斷：
    2. Top-K 設定是否過大導致雜訊過多？
    3. $vectorSearch 的過濾器是否運作正常？

---

## 📝 開發筆記
- **ObjectId 處理**: 由於 MongoDB 回傳的 `_id` 是物件格式，專案使用 `convert_doc` 函式遞迴將其轉為字串，確保 FastAPI 輸出時不會報錯。
- **去重鍵值**: 去重邏輯採用 `"collection:id"` 格式，確保在跨集合檢索時資料的唯一性。
- **自動化評估**: 獨立的評估腳本設計，不干擾主 API 運行，適合整合進 CI/CD 或開發週期的品質檢查。
