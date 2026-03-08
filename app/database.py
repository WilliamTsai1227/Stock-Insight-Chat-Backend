import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

class MongoDB:
    def __init__(self):
        self.client = None
        self.db = None

    def connect(self):
        """
        初始化連接池。
        Motor 的 AsyncIOMotorClient 本身就內建了連線池管理。
        """
        mongo_uri = os.getenv("MONGO_URI")
        db_name = os.getenv("MONGO_DB_NAME", "stock_insight")
        
        # minPoolSize: 最小連線數, maxPoolSize: 最大連線數
        self.client = AsyncIOMotorClient(
            mongo_uri, 
            maxPoolSize=10, 
            minPoolSize=1
        )
        self.db = self.client[db_name]
        print(f"Connected to MongoDB: {db_name}")

    def close(self):
        if self.client:
            self.client.close()
            print("MongoDB connection closed")

# 建立全域實體
db = MongoDB()
