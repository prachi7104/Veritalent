# Veritalent Backend API

This directory contains the production backend system for Veritalent Candidate Discovery.

## Startup
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
This loads the feature store, candidate dict, dense index, and LightGBM model into memory.

## Architecture
Memory-resident data layout guarantees zero I/O per search request.
