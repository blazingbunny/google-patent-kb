#!/usr/bin/env python3
"""
Patent Knowledge Base Pipeline
==============================
Query Google Patents Public Datasets (BigQuery) → chunk → embed → turbovec index.

Usage
-----
  # Test with 1000 patents first
  python pipeline.py --limit 1000 --output ./data/test

  # Full US corpus (runs on vast.ai GPU instance)
  python pipeline.py --output ./data/us_full

  # By technology class
  python pipeline.py --cpc G06F --output ./data/g06f

  # Resume is not safe yet; use a clean output directory for reruns

Requirements (vast.ai instance)
--------------------------------
  pip install google-cloud-bigquery turbovec sentence-transformers numpy tqdm

Authentication
--------------
  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

  The service account needs: bigquery.jobs.create, bigquery.readsessions.create
  on the `patents-public-data` project (public datasets don't need your own data).
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # BigQuery
    project: Optional[str] = None          # Your GCP project (for billing)
    dataset: str = "patents-public-data"   # Public dataset project
    table: str = "patents.publications"    # Main patent table
    limit: Optional[int] = None            # For testing — limit rows
    cpc: list[str] = None                  # Filter by CPC classes (e.g. ["G06F16", "G06N"])
    country: str = "US"                    # Jurisdiction
    years: Optional[tuple] = None          # (start, end) e.g. (2015, 2026)
    batch_size: int = 5_000               # Rows per BigQuery page
    max_chunks_per_batch: int = 64_000    # Chunks before embed + index flush

    # Embedding
    model_name: str = "BAAI/bge-large-en-v1.5"
    device: str = "cuda"                  # Torch device: "cuda" or "cpu"
    # Alternative models:
    #   "sentence-transformers/all-MiniLM-L6-v2"    → 384-dim, fast
    #   "BAAI/bge-base-en-v1.5"                     → 768-dim, good
    #   "BAAI/bge-large-en-v1.5"                    → 1024-dim, best quality
    embed_batch_size: int = 512            # GPU batch size for embedding
    chunk_max_tokens: int = 512           # Max tokens per description chunk

    # Index
    bit_width: int = 4                     # 2 or 4 bits per dimension
    index_path: str = "patent_index.tvim"
    metadata_path: str = "patent_meta.db"
    checkpoint_path: str = "pipeline_checkpoint.json"
    resume: bool = False                  # Reserved until safe resume is implemented

    # Output
    output_dir: str = "./data"


# ---------------------------------------------------------------------------
# BigQuery — query patents
# ---------------------------------------------------------------------------

def build_query(cfg: Config) -> str:
    """Build the BigQuery SQL for fetching patent data."""
    # Schema notes (verified 2025-06-07):
    #   - cpc and ipc are top-level RECORD REPEATED, not under "classifications"
    #   - publication_date / filing_date are INTEGER (YYYYMMDD), not DATE
    #   - assignee / inventor are STRING REPEATED (arrays), not single strings
    #   - Table has NO partitioning or clustering; every query scans full table
    #   - Dry-run before real query -- cost monitoring is caller's responsibility
    selects = """
        publication_number,
        country_code,
        kind_code,
        (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1) AS title,
        (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1) AS abstract,
        (SELECT text FROM UNNEST(claims_localized) WHERE language = 'en' LIMIT 1) AS claims,
        (SELECT text FROM UNNEST(description_localized) WHERE language = 'en' LIMIT 1) AS description,
        ARRAY(SELECT code FROM UNNEST(ipc)) AS ipc_codes,
        ARRAY(SELECT code FROM UNNEST(cpc)) AS cpc_codes,
        CAST(CAST(filing_date AS STRING) AS DATE FORMAT 'YYYYMMDD') AS filing_date,
        CAST(CAST(publication_date AS STRING) AS DATE FORMAT 'YYYYMMDD') AS publication_date,
        ARRAY_TO_STRING(assignee, ', ') AS assignee,
        ARRAY_TO_STRING(inventor, ', ') AS inventor
    """

    wheres = [
        "country_code = @country",
        "kind_code IN ('A1', 'B1', 'B2')",
    ]
    if cfg.cpc:
        cpc_clauses = " OR ".join(
            f"EXISTS(SELECT 1 FROM UNNEST(cpc) AS c WHERE c.code LIKE @cpc{i})"
            for i in range(len(cfg.cpc))
        )
        wheres.append(f"({cpc_clauses})")
    if cfg.years:
        wheres.append("CAST(publication_date AS STRING) BETWEEN @start_date AND @end_date")

    where_clause = "\n            AND ".join(wheres)

    limit_clause = ""
    if cfg.limit:
        limit_clause = f"\n        LIMIT {cfg.limit}"

    return f"""
        WITH selected AS (
            SELECT {selects}
            FROM `{cfg.dataset}.{cfg.table}`
            WHERE {where_clause}
        )
        SELECT *
        FROM selected
        WHERE (description IS NOT NULL OR claims IS NOT NULL)
        {limit_clause}
    """


def build_query_params(cfg: Config) -> dict:
    """Build BigQuery query parameters."""
    params = {"country": cfg.country}
    if cfg.cpc:
        for i, c in enumerate(cfg.cpc):
            params[f"cpc{i}"] = f"{c}%"
    if cfg.years:
        # publication_date is INT64; CAST to STRING in WHERE, then to DATE in SELECT
        params["start_date"] = f"{cfg.years[0]}0101"
        params["end_date"] = f"{cfg.years[1]}1231"
    return params


def stream_patents(cfg: Config) -> Generator[dict, None, None]:
    """Stream patent rows from BigQuery. Returns dicts with extracted fields."""
    from google.cloud import bigquery

    client = bigquery.Client(project=cfg.project)
    query = build_query(cfg)
    params = build_query_params(cfg)
    dataset_id = f"{cfg.project}.patent_kb_temp"

    log.info("Querying BigQuery...")
    log.info(f"  Dataset: {cfg.dataset}.{cfg.table}")
    log.info(f"  Filter:  country={cfg.country}")
    if cfg.cpc:
        log.info(f"  CPC:     {', '.join(cfg.cpc)}")
    if cfg.years:
        log.info(f"  Years:   {cfg.years[0]}-{cfg.years[1]}")
    if cfg.limit:
        log.info(f"  Limit:   {cfg.limit}")
    log.info(f"  Query:\n{query}")

    # Estimate cost
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(k, "STRING", v)
            for k, v in params.items()
        ],
        dry_run=True,
    )
    dry = client.query(query, job_config=job_config)
    log.info(f"  Data scanned: {dry.total_bytes_processed / 1e9:.1f} GB "
             f"(free tier: 1 TB/month)")

    # For large result sets, use a temporary destination table.
    # Without a LIMIT, the REST API response limit (~10 GB) can be exceeded.
    job_config.dry_run = False
    if cfg.limit and cfg.limit < 10000:
        # Small query — stream directly
        job = client.query(query, job_config=job_config)
        result = job.result()
    else:
        # Large query — use temporary destination table to avoid response limit
        temp_table_id = f"patent_kb_results_{uuid.uuid4().hex[:8]}"
        dest_project = cfg.project
        if not dest_project:
            dest_project = client.project
        if not dest_project:
            dest_project = "ss-fleet-498508"
        dataset_ref = bigquery.DatasetReference(dest_project, "patent_kb_temp")
        table_ref = dataset_ref.table(temp_table_id)
        job_config.destination = table_ref
        job_config.write_disposition = "WRITE_TRUNCATE"
        # Ensure the temp dataset exists in US location
        ds = bigquery.Dataset(dataset_ref)
        ds.location = "US"
        try:
            client.create_dataset(ds, exists_ok=True)
        except Exception:
            pass  # dataset may already exist
        # With destination table, allow_large_results is enabled by default
        job = client.query(query, job_config=job_config)
        log.info(f"  Writing large results to {dataset_id}.{temp_table_id}...")
        job.result()
        log.info("  Reading results from temp table...")
        # Use TableReference object (not string) for reliable get_table + list_rows
        dest_table_ref = dataset_ref.table(temp_table_id)
        dest_table = client.get_table(dest_table_ref)
        log.info(f"  Table found: {dest_table.table_id}, rows={dest_table.num_rows}")
        rows = client.list_rows(dest_table, page_size=500)
        result = rows  # RowIterator supports .pages
        # Clean up temp table
        try:
            client.delete_table(dest_table_ref)
        except Exception:
            pass

    total = 0
    for page in result.pages:
        for row in page:
            total += 1
            yield {
                "publication_number": row.get("publication_number", ""),
                "country_code": row.get("country_code", ""),
                "kind_code": row.get("kind_code", ""),
                "title": row.get("title") or "",
                "abstract": row.get("abstract") or "",
                "claims": row.get("claims") or "",
                "description": row.get("description") or "",
                "ipc_codes": row.get("ipc_codes") or [],
                "cpc_codes": row.get("cpc_codes") or [],
                "filing_date": row.get("filing_date") or "",
                "publication_date": row.get("publication_date") or "",
                "assignee": row.get("assignee") or "",
                "inventor": row.get("inventor") or "",
            }

    log.info(f"  Total patents returned: {total}")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_claims(claims_text: str, patent_number: str) -> list[dict]:
    """Split claims text into individual claim chunks."""
    # Claims are typically separated by newlines and numbered
    # "1. A method..." or "claim 1: ..." or just numbered paragraphs
    if not claims_text:
        return []

    # Normalize line endings
    text = claims_text.replace("\r\n", "\n")

    # Try to split on claim boundaries
    # Common patterns: "1.", "1) ", "Claim 1.", etc.
    claim_pattern = re.compile(
        r'(?:^|\n)\s*(?:Claim\s+)?(\d+)[\.\)]\s*',
        re.IGNORECASE | re.MULTILINE
    )

    matches = list(claim_pattern.finditer(text))
    chunks = []

    if len(matches) <= 1:
        # Couldn't split — treat as a single claim chunk
        text_clean = text.strip()
        if text_clean:
            chunks.append({
                "patent_number": patent_number,
                "chunk_type": "claims_all",
                "chunk_label": "All Claims",
                "chunk_text": text_clean,
            })
        return chunks

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        claim_text = text[start:end].strip()
        if claim_text:
            claim_num = m.group(1)
            chunks.append({
                "patent_number": patent_number,
                "chunk_type": "claim",
                "chunk_label": f"Claim {claim_num}",
                "chunk_text": claim_text,
            })

    return chunks


def chunk_description(
    desc_text: str, patent_number: str, max_tokens: int = 512
) -> list[dict]:
    """Split description into roughly even chunks by paragraph groups."""
    if not desc_text:
        return []

    text = desc_text.replace("\r\n", "\n")
    # Split into paragraphs (double newlines)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if not paragraphs:
        # Fallback: split by single newlines
        paragraphs = [l.strip() for l in text.split("\n") if l.strip()]
    if not paragraphs:
        return []

    # Group paragraphs into chunks of approximately max_tokens
    # Rough estimate: 1 token ≈ 4 chars for English
    char_budget = max_tokens * 4
    chunks = []
    current_group = []
    current_chars = 0
    section_num = 0

    for para in paragraphs:
        para_chars = len(para) + 1  # +1 for separator
        if current_chars + para_chars > char_budget and current_group:
            section_num += 1
            text = "\n\n".join(current_group)
            chunks.append({
                "patent_number": patent_number,
                "chunk_type": "description",
                "chunk_label": f"Description §{section_num}",
                "chunk_text": text,
            })
            current_group = [para]
            current_chars = para_chars
        else:
            current_group.append(para)
            current_chars += para_chars

    if current_group:
        section_num += 1
        text = "\n\n".join(current_group)
        chunks.append({
            "patent_number": patent_number,
            "chunk_type": "description",
            "chunk_label": f"Description §{section_num}",
            "chunk_text": text,
        })

    return chunks


def chunk_patent(patent: dict, cfg: Config) -> list[dict]:
    """Split a single patent into searchable chunks."""
    chunks = []
    pn = patent["publication_number"]

    # 1. Title + abstract (one chunk)
    title_abstract = f"{patent['title']}\n\n{patent['abstract']}".strip()
    if title_abstract:
        chunks.append({
            "patent_number": pn,
            "chunk_type": "title_abstract",
            "chunk_label": "Title & Abstract",
            "chunk_text": title_abstract,
        })

    # 2. Individual claims
    if patent["claims"]:
        chunks.extend(chunk_claims(patent["claims"], pn))

    # 3. Description sections
    if patent["description"]:
        chunks.extend(
            chunk_description(patent["description"], pn, cfg.chunk_max_tokens)
        )

    return chunks


# ---------------------------------------------------------------------------
# Metadata database (SQLite)
# ---------------------------------------------------------------------------

class MetadataDB:
    """Stores patent metadata and chunk text for retrieval."""

    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=OFF")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS patents (
                patent_number TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                assignee TEXT,
                inventor TEXT,
                filing_date TEXT,
                publication_date TEXT,
                country TEXT,
                kind_code TEXT,
                ipc_codes TEXT,
                cpc_codes TEXT
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patent_number TEXT NOT NULL REFERENCES patents(patent_number),
                chunk_type TEXT NOT NULL,
                chunk_label TEXT,
                chunk_text TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_patent
                ON chunks(patent_number);
            CREATE INDEX IF NOT EXISTS idx_chunks_type
                ON chunks(chunk_type);
        """)
        self.conn.commit()

    def insert_patent(self, patent: dict):
        self.conn.execute(
            """INSERT OR IGNORE INTO patents
               (patent_number, title, abstract, assignee, inventor,
                filing_date, publication_date, country, kind_code,
                ipc_codes, cpc_codes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                patent["publication_number"],
                patent["title"],
                patent["abstract"],
                patent["assignee"],
                patent["inventor"],
                patent["filing_date"],
                patent["publication_date"],
                patent["country_code"],
                patent["kind_code"],
                json.dumps(patent["ipc_codes"]),
                json.dumps(patent["cpc_codes"]),
            ),
        )

    def insert_chunks(self, chunks: list[dict]) -> list[int]:
        """Insert chunks and return their auto-generated chunk_ids."""
        ids = []
        for c in chunks:
            cur = self.conn.execute(
                "INSERT INTO chunks (patent_number, chunk_type, chunk_label, chunk_text) "
                "VALUES (?, ?, ?, ?)",
                (c["patent_number"], c["chunk_type"],
                 c["chunk_label"], c["chunk_text"]),
            )
            ids.append(cur.lastrowid)
        return ids

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

class Checkpoint:
    """Track advisory progress checkpoints for failed-run diagnostics."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"patents_processed": 0, "chunks_indexed": 0,
                "last_patent_number": None}

    def save(self, patents: int, chunks: int, last_pn: Optional[str] = None):
        self.data["patents_processed"] = patents
        self.data["chunks_indexed"] = chunks
        if last_pn:
            self.data["last_patent_number"] = last_pn
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    @property
    def patents_processed(self) -> int:
        return self.data.get("patents_processed", 0)

    @property
    def chunks_indexed(self) -> int:
        return self.data.get("chunks_indexed", 0)


# ---------------------------------------------------------------------------
# Embedding + Indexing
# ---------------------------------------------------------------------------

class EmbedIndexPipeline:
    """Batch-embed chunks and add to turbovec index."""

    def __init__(self, cfg: Config, meta_db: MetadataDB):
        self.cfg = cfg
        self.meta = meta_db

        log.info(f"Loading embedding model: {cfg.model_name}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            cfg.model_name,
            device=cfg.device,
        )
        self.dim = self.model.get_sentence_embedding_dimension()
        log.info(f"  Embedding dimension: {self.dim}")

        log.info(f"Creating turbovec index (dim={self.dim}, bit_width={cfg.bit_width})")
        from turbovec import IdMapIndex
        self.index = IdMapIndex(dim=self.dim, bit_width=cfg.bit_width)

        # Buffer for current batch
        self.buffer_chunks: list[dict] = []
        self.buffer_texts: list[str] = []
        self.total_patents = 0
        self.total_chunks = 0

    def add(self, patent: dict):
        """Process one patent: chunk, buffer, flush if batch is full."""
        chunks = chunk_patent(patent, self.cfg)
        if not chunks:
            return 0

        # Insert patent metadata
        self.meta.insert_patent(patent)

        # Get chunk IDs from metadata DB
        chunk_ids = self.meta.insert_chunks(chunks)

        # Add texts to embed buffer
        for chunk, cid in zip(chunks, chunk_ids):
            self.buffer_chunks.append({"id": cid, **chunk})
            self.buffer_texts.append(chunk["chunk_text"])

        self.total_patents += 1
        self.total_chunks += len(chunks)

        # Flush when buffer is full
        if len(self.buffer_texts) >= self.cfg.max_chunks_per_batch:
            self._flush()

        # Commit metadata periodically
        if self.total_patents % 10_000 == 0:
            self.meta.commit()

        return len(chunks)

    def _flush(self):
        """Embed buffer and add vectors to turbovec index."""
        if not self.buffer_texts:
            return

        log.info(f"  Embedding batch of {len(self.buffer_texts)} chunks...")
        t0 = time.time()
        embeddings = self.model.encode(
            self.buffer_texts,
            batch_size=self.cfg.embed_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        elapsed = time.time() - t0
        log.info(f"  Done in {elapsed:.1f}s ({len(self.buffer_texts)/elapsed:.0f} chunks/sec)")

        # Build id array
        ids = np.array([c["id"] for c in self.buffer_chunks], dtype=np.uint64)
        vectors = np.asarray(embeddings, dtype=np.float32)

        try:
            self.index.add_with_ids(vectors, ids)
        except Exception as e:
            log.error(f"turbovec add failed: {e}")
            # If duplicate IDs caused the issue, add one at a time
            log.info("  Falling back to per-chunk add...")
            for vec, cid in zip(vectors, ids):
                try:
                    self.index.add_with_ids(
                        vec.reshape(1, -1),
                        np.array([cid], dtype=np.uint64),
                    )
                except Exception:
                    pass  # Skip duplicates

        self.buffer_chunks = []
        self.buffer_texts = []
        self.meta.commit()

    def flush(self):
        """Flush remaining buffer and save index."""
        if self.buffer_texts:
            self._flush()

        out_dir = Path(self.cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        index_path = out_dir / self.cfg.index_path
        log.info(f"Saving index to {index_path}")
        self.index.write(str(index_path))
        log.info(f"Index saved: {len(self.index)} vectors")

        meta_path = out_dir / self.cfg.metadata_path
        log.info(f"Metadata saved to {meta_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(cfg: Config):
    log.info("=" * 60)
    log.info("Patent Knowledge Base Pipeline")
    log.info("=" * 60)

    # Prevent torch multi-thread deadlocks on CPU where thread pool init
    # conflicts with sentence-transformers internal parallelism
    if cfg.device == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if cfg.resume:
        raise SystemExit(
            "--resume is not implemented safely yet. Use a clean output "
            "directory or delete the partial output before rebuilding."
        )

    out_dir = Path(cfg.output_dir)
    index_path = out_dir / cfg.index_path
    meta_path = out_dir / cfg.metadata_path
    checkpoint_path = out_dir / cfg.checkpoint_path
    existing_outputs = [p for p in (index_path, meta_path, checkpoint_path) if p.exists()]
    if existing_outputs:
        existing = ", ".join(str(p) for p in existing_outputs)
        raise SystemExit(
            "Output directory is not clean; refusing to append duplicate chunks/vectors. "
            f"Existing files: {existing}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    meta = MetadataDB(str(meta_path))
    checkpoint = Checkpoint(str(checkpoint_path))
    pipeline = EmbedIndexPipeline(cfg, meta)

    patents_stream = stream_patents(cfg)

    try:
        with tqdm(desc="Patents processed", unit="pat") as pbar:
            for patent in patents_stream:
                pipeline.add(patent)
                pbar.update(1)

                # Save checkpoint every 5K patents
                if pipeline.total_patents % 5_000 == 0:
                    checkpoint.save(
                        pipeline.total_patents,
                        pipeline.total_chunks,
                        patent["publication_number"],
                    )
                    pbar.set_postfix(
                        chunks=pipeline.total_chunks,
                        last=patent["publication_number"][:20],
                    )

    except KeyboardInterrupt:
        log.warning("Interrupted — flushing and saving partial index...")
    except Exception:
        log.exception("Pipeline failed")
    finally:
        pipeline.flush()
        checkpoint.save(pipeline.total_patents, pipeline.total_chunks)
        meta.close()

    log.info("=" * 60)
    log.info(f"Done. {pipeline.total_patents} patents → {pipeline.total_chunks} chunks indexed")
    log.info(f"Index: {out_dir / cfg.index_path}")
    log.info(f"Metadata: {out_dir / cfg.metadata_path}")
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a patent knowledge base from Google Patents Public Datasets"
    )
    # BigQuery filters
    parser.add_argument("--project", help="Your GCP project ID (for billing)")
    parser.add_argument("--limit", type=int, help="Limit patents (testing)")
    parser.add_argument("--cpc", action="append", help="Filter by CPC class (e.g. G06F16). Repeat for multiple.")
    parser.add_argument("--country", default="US", help="Jurisdiction (default: US)")
    parser.add_argument("--years", nargs=2, type=int, metavar=("START", "END"),
                        help="Date range e.g. 2020 2026")

    # Output
    parser.add_argument("--output", "-o", default="./data",
                        help="Output directory (default: ./data)")
    parser.add_argument("--resume", action="store_true",
                        help="Reserved for future safe resume support; currently exits with an error")

    # Runtime
    parser.add_argument("--device", default="cuda",
                        help="Torch device for embedding (default: cuda). Use 'cpu' for CPU-only runs.")

    # Index params
    parser.add_argument("--bits", type=int, default=4, choices=[2, 4],
                        help="Quantization bits (2 or 4, default: 4)")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5",
                        help="Embedding model (default: bge-large-en-v1.5)")

    args = parser.parse_args()

    cfg = Config(
        project=args.project,
        limit=args.limit,
        cpc=args.cpc,
        country=args.country,
        years=tuple(args.years) if args.years else None,
        bit_width=args.bits,
        model_name=args.model,
        device=args.device,
        output_dir=args.output,
        resume=args.resume,
    )

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
