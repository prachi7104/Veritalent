import os
import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure the root of the project is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.data_access.candidate_repository import load as load_candidates
from backend.data_access.feature_store_repository import load as load_features
from backend.services.retrieval_service import load as load_retrieval
from backend.services.ranking_service import load as load_ranking
from backend.services.explainability_service import load as load_explainability

from backend.api.routes import search, candidate, compare, rerank, scenarios, health

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Veritalent Backend API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Initializing backend services...")
    load_candidates()
    load_features()
    load_retrieval()
    load_ranking()
    load_explainability()
    logger.info("Backend services initialized successfully.")

app.include_router(health.router, tags=["Health"])
app.include_router(search.router, tags=["Search"])
app.include_router(candidate.router, tags=["Candidate"])
app.include_router(compare.router, tags=["Compare"])
app.include_router(rerank.router, tags=["Rerank"])
app.include_router(scenarios.router, tags=["Scenarios"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
