from typing import Any, List
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class OpenAIChat:
    """
    使用 OpenAI GPT-4o 模型進行生成。
    """
    def __init__(self, model_name: str = "gpt-4o"):
        self.model = ChatOpenAI(
            model=model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3
        )
        
    async def ainvoke(self, input_data: Any, config: Any = None) -> str:
        query = input_data.get("userEnterQuery", "")
        knowledge = input_data.get("knowledgeDoc", "")
        
        prompt = query
        if knowledge:
            prompt = f"請根據以下參考資料回答問題。如果資料中沒有提到，請回答不知道。\n\n參考資料：\n{knowledge}\n\n問題：{query}\n答案："
            
        response = await self.model.ainvoke(prompt, config)
        return response.content

class OpenAIAppEmbeddings:
    """
    使用 OpenAI text-embedding-3-small 模型進行向量化。
    """
    def __init__(self, model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
    async def aembed_query(self, text: str) -> List[float]:
        return await self.embeddings.aembed_query(text)
