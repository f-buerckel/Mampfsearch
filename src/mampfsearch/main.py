import logging

from fastapi import FastAPI
from contextlib import asynccontextmanager
from mampfsearch.utils import config
from mampfsearch.routes import maintenance, ingest, lectures

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_model = config.get_embedding_model()
    qdrant_client = config.get_qdrant_client()
    ollama_client = config.get_llm_client()
    yield

app = FastAPI(
    title="MampfSearch API",
    description="API for MampfSearch - a search engine for lecture videos",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(maintenance.router)
app.include_router(ingest.router)
app.include_router(lectures.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the MampfSearch API"}
