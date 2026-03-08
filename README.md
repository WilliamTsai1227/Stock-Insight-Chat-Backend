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

## 📝 開發筆記
- **ObjectId 處理**: 由於 MongoDB 回傳的 `_id` 是物件格式，專案使用 `convert_doc` 函式遞迴將其轉為字串，確保 FastAPI 輸出時不會報錯。
- **去重鍵值**: 去重邏輯採用 `"collection:id"` 格式，確保在跨集合檢索時資料的唯一性。
