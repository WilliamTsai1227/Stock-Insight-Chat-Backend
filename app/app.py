from fastapi import FastAPI
from app.routers import chatbot, base_router
from app.database import db

app = FastAPI(title="RAG Backend")

@app.on_event("startup")
async def startup_event():
    db.connect()

@app.on_event("shutdown")
async def shutdown_event():
    db.close()

app.include_router(base_router.router)
app.include_router(chatbot.router, prefix="/chatbot")

@app.get("/")
async def root():
    return {"message": "RAG Backend is running"}
