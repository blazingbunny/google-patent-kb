#!/usr/bin/env python3
"""
Patent Search Service
=====================
FastAPI app that loads a turbovec index + SQLite metadata and serves
semantic patent search. Runs on argus (CPU-only, no GPU needed).

Usage
-----
  # Install
  pip install fastapi uvicorn turbovec sentence-transformers numpy

  # Run (loads index + metadata)
  python query_service.py --index ./data/patent_index.tvim \
                          --meta ./data/patent_meta.db \
                          --model BAAI/bge-large-en-v1.5

  # Direct uvicorn module startup does not load an index;
  # endpoints will return 503 until engine loading is added to lifespan config.
"""

import argparse
import json
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("patent-search")


# ---------------------------------------------------------------------------
# Search Engine
# ---------------------------------------------------------------------------

class PatentSearchEngine:
    """Loads index + embedding model + metadata. Handles search."""

    def __init__(
        self,
        index_path: str,
        meta_path: str,
        model_name: str,
        allowlist_limit: Optional[int] = None,
    ):
        log.info(f"Loading embedding model: {model_name}")
        import sentence_transformers
        self.model = sentence_transformers.SentenceTransformer(
            model_name, device="cpu"
        )
        log.info(f"  Dimension: {self.model.get_sentence_embedding_dimension()}")

        log.info(f"Loading turbovec index: {index_path}")
        from turbovec import IdMapIndex
        # NOTE: IdMapIndex.load(path) MUST be the classmethod form.
        # The instance-method form IdMapIndex().load(path) silently returns
        # 0 vectors in turbovec 0.7.0 (the instance is discarded by __init__).
        self.index = IdMapIndex.load(index_path)
        log.info(f"  Vectors: {len(self.index)}")

        log.info(f"Loading metadata: {meta_path}")
        self.meta_path = meta_path
        self.allowlist_limit = allowlist_limit
        self._conn = sqlite3.connect(meta_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        log.info("  Ready")

    def search(
        self,
        query: str,
        k: int = 10,
        cpc_filter: Optional[str] = None,
        assignee_filter: Optional[str] = None,
        chunk_type: Optional[str] = None,
        years: Optional[tuple] = None,
        publication_number: Optional[str] = None,
    ) -> dict:
        """Full search: embed → vector search → metadata join."""
        t0 = time.time()

        # 1. Embed query
        emb = self.model.encode([query], normalize_embeddings=True)
        query_vec = np.asarray(emb, dtype=np.float32)
        t_embed = time.time() - t0

        # 2. Optional: get allowlist from metadata filters
        allowlist = None
        if any([cpc_filter, assignee_filter, chunk_type, years, publication_number]):
            allowed_ids = self._resolve_filters(
                cpc_filter, assignee_filter, chunk_type, years, publication_number
            )
            if allowed_ids is not None:
                allowlist = np.array(allowed_ids, dtype=np.uint64)
                if len(allowlist) == 0:
                    return {
                        "query": query,
                        "results": [],
                        "timing": {"embed_ms": round(t_embed * 1000),
                                   "search_ms": 0, "total_ms": round((time.time()-t0)*1000)},
                        "total_results": 0,
                    }

        # 3. Vector search
        t1 = time.time()
        kwargs = {"k": k}
        if allowlist is not None:
            kwargs["allowlist"] = allowlist
        scores, ids = self.index.search(query_vec, **kwargs)
        t_search = time.time() - t1

        # 4. Fetch metadata for results
        results = self._fetch_chunks(ids[0], scores[0])

        total_ms = (time.time() - t0) * 1000

        return {
            "query": query,
            "results": results,
            "timing": {
                "embed_ms": round(t_embed * 1000, 1),
                "search_ms": round(t_search * 1000, 1),
                "total_ms": round(total_ms, 1),
            },
            "total_results": len(results),
            "filters_applied": {
                "cpc": cpc_filter,
                "assignee": assignee_filter,
                "chunk_type": chunk_type,
                "years": years,
            },
        }

    def _resolve_filters(self, cpc, assignee, chunk_type, years, publication_number=None):
        """Resolve metadata filters to chunk_id allowlist."""
        clauses = []
        params = []

        if chunk_type:
            clauses.append("c.chunk_type = ?")
            params.append(chunk_type)
        if years:
            clauses.append("p.publication_date BETWEEN ? AND ?")
            params.append(f"{years[0]}-01-01")
            params.append(f"{years[1]}-12-31")
        if publication_number is not None:
            clauses.append("REPLACE(p.patent_number, '-', '') = ?")
            params.append(publication_number.replace("-", ""))

        # For assignee, CPC, years, or publication_number, filter patents first
        patent_join = ""
        if cpc or assignee or years or publication_number:
            if cpc:
                clauses.append("p.cpc_codes LIKE ?")
                params.append(f"%{cpc}%")
            if assignee:
                clauses.append("p.assignee LIKE ?")
                params.append(f"%{assignee}%")
            patent_join = "JOIN patents p ON c.patent_number = p.patent_number"

        if not clauses:
            return None

        where = " AND ".join(clauses)
        sql = f"SELECT c.chunk_id FROM chunks c {patent_join} WHERE {where}"
        if self.allowlist_limit is not None:
            sql += " LIMIT ?"
            params.append(self.allowlist_limit)

        cur = self._conn.execute(sql, params)
        rows = cur.fetchall()
        return [r["chunk_id"] for r in rows]

    def _fetch_chunks(self, chunk_ids, scores):
        """Fetch chunk + patent metadata for search results."""
        if not len(chunk_ids):
            return []

        # Build lookup: chunk_id → score
        score_map = {int(cid): float(score)
                     for cid, score in zip(chunk_ids, scores)}

        placeholders = ",".join("?" for _ in chunk_ids)
        sql = f"""
            SELECT c.chunk_id, c.patent_number, c.chunk_type, c.chunk_label,
                   c.chunk_text,
                   p.title, p.assignee, p.filing_date, p.publication_date,
                   p.ipc_codes, p.cpc_codes, p.kind_code
            FROM chunks c
            JOIN patents p ON c.patent_number = p.patent_number
            WHERE c.chunk_id IN ({placeholders})
        """
        cur = self._conn.execute(sql, [int(i) for i in chunk_ids])
        rows = cur.fetchall()

        # Sort by original search score order
        result_map = {}
        for r in rows:
            result_map[r["chunk_id"]] = {
                "chunk_id": r["chunk_id"],
                "patent_number": r["patent_number"],
                "chunk_type": r["chunk_type"],
                "chunk_label": r["chunk_label"],
                "chunk_text": r["chunk_text"][:500],  # preview
                "title": r["title"],
                "assignee": r["assignee"],
                "filing_date": r["filing_date"],
                "publication_date": r["publication_date"],
                "ipc_codes": json.loads(r["ipc_codes"]) if r["ipc_codes"] else [],
                "cpc_codes": json.loads(r["cpc_codes"]) if r["cpc_codes"] else [],
                "kind_code": r["kind_code"],
                "similarity": None,
            }

        results = []
        for cid in chunk_ids:
            cid = int(cid)
            if cid in result_map:
                r = result_map[cid]
                r["similarity"] = round(score_map[cid], 4)
                results.append(r)

        return results


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

engine: Optional[PatentSearchEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Startup: already loaded via CLI or env
    yield
    # Shutdown
    engine = None


app = FastAPI(
    title="Patent Knowledge Base",
    version="1.0.0",
    description="Semantic search over US patents using turbovec",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    k: int = 10
    cpc: Optional[str] = None
    assignee: Optional[str] = None
    chunk_type: Optional[str] = None
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    publication_number: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: list
    timing: dict
    total_results: int
    filters_applied: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "vectors": len(engine.index) if engine else 0}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if engine is None:
        raise HTTPException(503, "Search engine not loaded")
    years = None
    if req.year_start and req.year_end:
        years = (req.year_start, req.year_end)
    return engine.search(
        query=req.query,
        k=req.k,
        cpc_filter=req.cpc,
        assignee_filter=req.assignee,
        chunk_type=req.chunk_type,
        years=years,
        publication_number=req.publication_number,
    )


@app.get("/search")
def search_get(
    q: str = Query(..., description="Search query"),
    k: int = Query(10, description="Number of results"),
    cpc: Optional[str] = Query(None, description="CPC class filter"),
    assignee: Optional[str] = Query(None, description="Assignee name filter"),
    chunk_type: Optional[str] = Query(None, description="chunk type: claim, claims_all, description, title_abstract"),
    year_start: Optional[int] = Query(None),
    year_end: Optional[int] = Query(None),
    publication_number: Optional[str] = Query(None, description="Filter by patent number (any format)"),
):
    if engine is None:
        raise HTTPException(503, "Search engine not loaded")
    years = (year_start, year_end) if year_start and year_end else None
    return engine.search(q, k=k, cpc_filter=cpc, assignee_filter=assignee,
                         chunk_type=chunk_type, years=years,
                         publication_number=publication_number)


@app.get("/patent/{patent_number}")
def get_patent(patent_number: str):
    """Get full patent + all chunks."""
    if engine is None:
        raise HTTPException(503, "Search engine not loaded")
    cur = engine._conn.execute(
        "SELECT * FROM patents WHERE REPLACE(patent_number, '-', '') = ?",
        (patent_number.replace("-", ""),)
    )
    patent = cur.fetchone()
    if not patent:
        raise HTTPException(404, f"Patent {patent_number} not found")

    cur = engine._conn.execute(
        "SELECT chunk_id, chunk_type, chunk_label, chunk_text FROM chunks "
        "WHERE patent_number = ? ORDER BY chunk_id",
        (patent_number,)
    )
    chunks = [dict(r) for r in cur.fetchall()]

    return {
        "patent": dict(patent),
        "chunks": chunks,
    }


@app.get("/stats")
def stats():
    """Index statistics."""
    if engine is None:
        raise HTTPException(503, "Search engine not loaded")
    cur = engine._conn.execute("SELECT COUNT(*) AS n FROM patents")
    n_patents = cur.fetchone()["n"]
    cur = engine._conn.execute("SELECT COUNT(*) AS n FROM chunks")
    n_chunks = cur.fetchone()["n"]
    cur = engine._conn.execute(
        "SELECT chunk_type, COUNT(*) AS n FROM chunks GROUP BY chunk_type"
    )
    by_type = {r["chunk_type"]: r["n"] for r in cur.fetchall()}
    return {
        "n_patents": n_patents,
        "n_chunks": n_chunks,
        "chunks_by_type": by_type,
        "vectors_in_index": len(engine.index),
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Patent Search Service")
    parser.add_argument("--index", required=True, help="Path to .tvim index")
    parser.add_argument("--meta", required=True, help="Path to patent_meta.db")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5",
                        help="Embedding model (default: bge-large-en-v1.5)")
    parser.add_argument(
        "--allowlist-limit",
        type=int,
        default=0,
        help="Maximum filtered chunk IDs to pass to turbovec; 0 means unlimited (default)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    global engine
    allowlist_limit = args.allowlist_limit if args.allowlist_limit > 0 else None
    engine = PatentSearchEngine(
        args.index,
        args.meta,
        args.model,
        allowlist_limit=allowlist_limit,
    )

    import uvicorn
    log.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
