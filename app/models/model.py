from typing import Any, List, Optional
from pydantic import BaseModel, Field

class AIChatRequest(BaseModel):
    query: str
    query_type: str = "general"

class AIChatResponse(BaseModel):
    answer: str
    status: str = "success"
