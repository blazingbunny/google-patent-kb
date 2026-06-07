#!/usr/bin/env python3
"""
Merge multiple patent knowledge base outputs into a single unified index.

Reads each source directory's patent_meta.db, deduplicates patents, merges
chunks with remapped IDs, then re-embeds all chunk text into a single
turbovec index.

Usage
-----
  python merge.py \\
    --source ./data/test \\
    --source ./data/curated \\
    --target ./data/combined \\
    --model BAAI/bge-large-en-v1.5 \\
    --device cuda

  # CPU merge (slower but no GPU needed)
  python merge.py --source A --source B --target C --device cpu

Why re-embed?
  turbovec's IdMapIndex does not expose stored vector IDs, so we cannot
  extract and remap vectors from separate indexes. Re-embedding from the
  SQLite chunk text is the only way to produce a single coherent index.
  For ~50K chunks this takes ~30s on GPU; for ~3M chunks ~5min.
"""

import argparse
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("merge")


def read_source(source_dir: Path) -> tuple[list[dict], list[dict], int]:
    """Read patents and chunks from a source metadata DB.

    Returns (patents, chunks, max_chunk_id) where max_chunk_id is the
    highest chunk_id in this source (used for ID remapping across sources).
    """
    db_path = source_dir / "patent_meta.db"
    if not db_path.exists():
        raise SystemExit(f"Source {source_dir} has no patent_meta.db")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    patents = []
    for row in conn.execute("SELECT * FROM patents ORDER BY patent_number"):
        patents.append(dict(row))

    chunks = []
    for row in conn.execute("SELECT * FROM chunks ORDER BY chunk_id"):
        chunks.append(dict(row))

    conn.close()

    max_chunk_id = max(c["chunk_id"] for c in chunks) if chunks else 0
    log.info(f"  Source {source_dir.name}: {len(patents):,} patents, "
             f"{len(chunks):,} chunks (max chunk_id={max_chunk_id})")

    return patents, chunks, max_chunk_id


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple patent knowledge base outputs into one"
    )
    parser.add_argument(
        "--source", "-s", action="append", required=True, dest="sources",
        help="Source output directory (repeat for each source)",
    )
    parser.add_argument(
        "--target", "-t", required=True,
        help="Target output directory for merged result",
    )
    parser.add_argument(
        "--model", default="BAAI/bge-large-en-v1.5",
        help="Embedding model (default: BAAI/bge-large-en-v1.5)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Torch device (default: cuda; use 'cpu' for CPU-only)",
    )
    parser.add_argument(
        "--bits", type=int, default=4, choices=[2, 4],
        help="Quantization bits per dimension (default: 4)",
    )
    parser.add_argument(
        "--embed-batch-size", type=int, default=512,
        help="Batch size for embedding (default: 512)",
    )
    args = parser.parse_args()

    if len(args.sources) < 2:
        raise SystemExit("Need at least 2 --source directories to merge")

    target_dir = Path(args.target)
    target_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Read all sources ----
    log.info("=" * 60)
    log.info(f"Merging {len(args.sources)} sources into {target_dir}")
    log.info("=" * 60)

    all_patents: dict[str, dict] = {}  # patent_number → dict (first wins)
    all_chunks: list[dict] = []         # chunks with remapped chunk_ids
    chunk_id_offset = 0

    for src in args.sources:
        src_path = Path(src)
        if not src_path.is_dir():
            raise SystemExit(f"Source path does not exist: {src_path}")

        log.info(f"Reading: {src_path}")
        patents, chunks, max_cid = read_source(src_path)

        # Deduplicate patents (first source wins).
        # Track which patent_numbers this source contributes so we can
        # also exclude their chunks when the patent is a duplicate.
        new_patents_this_source: set[str] = set()
        for p in patents:
            if p["patent_number"] not in all_patents:
                all_patents[p["patent_number"]] = p
                new_patents_this_source.add(p["patent_number"])

        # Only keep chunks whose patent was accepted (not a duplicate).
        # Without this guard, overlapping patents produce duplicate chunk
        # vectors in the merged index.
        n_skipped = 0
        for c in chunks:
            if c["patent_number"] not in new_patents_this_source:
                n_skipped += 1
                continue
            c["chunk_id"] = c["chunk_id"] + chunk_id_offset
            all_chunks.append(c)

        if n_skipped:
            log.info(f"  Skipped {n_skipped} chunks from duplicate patents")

        chunk_id_offset += max_cid

    n_patents = len(all_patents)
    n_chunks = len(all_chunks)
    log.info(f"\nDedup summary:")
    log.info(f"  Unique patents:  {n_patents:,}")
    log.info(f"  Total chunks:    {n_chunks:,}")

    if n_chunks == 0:
        raise SystemExit("No chunks to merge — nothing to do.")

    # ---- 2. Write merged metadata DB ----
    log.info("Writing merged metadata DB...")
    meta_path = target_dir / "patent_meta.db"

    conn = sqlite3.connect(str(meta_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")

    conn.executescript("""
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
            chunk_id INTEGER PRIMARY KEY,
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

    for p in all_patents.values():
        conn.execute(
            """INSERT OR IGNORE INTO patents
               (patent_number, title, abstract, assignee, inventor,
                filing_date, publication_date, country, kind_code,
                ipc_codes, cpc_codes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                p["patent_number"],
                p.get("title") or "",
                p.get("abstract") or "",
                p.get("assignee") or "",
                p.get("inventor") or "",
                p.get("filing_date") or "",
                p.get("publication_date") or "",
                p.get("country") or "",
                p.get("kind_code") or "",
                p.get("ipc_codes") or "[]",
                p.get("cpc_codes") or "[]",
            ),
        )

    for c in all_chunks:
        conn.execute(
            "INSERT INTO chunks (chunk_id, patent_number, chunk_type, "
            "chunk_label, chunk_text) VALUES (?, ?, ?, ?, ?)",
            (c["chunk_id"], c["patent_number"], c["chunk_type"],
             c["chunk_label"], c["chunk_text"]),
        )

    conn.commit()
    conn.close()
    db_mb = meta_path.stat().st_size / 1_000_000
    log.info(f"  Wrote {meta_path} ({db_mb:.0f} MB)")

    # ---- 3. Load embedding model ----
    log.info(f"Loading embedding model: {args.model}")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model, device=args.device)
    dim = model.get_sentence_embedding_dimension()
    log.info(f"  Dimension: {dim}")

    # ---- 4. Embed all chunks ----
    chunk_texts = [c["chunk_text"] for c in all_chunks]
    chunk_ids = np.array([c["chunk_id"] for c in all_chunks], dtype=np.uint64)

    log.info(f"Embedding {len(chunk_texts):,} chunks...")
    t0 = time.time()
    embeddings = model.encode(
        chunk_texts,
        batch_size=args.embed_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    elapsed = time.time() - t0
    chunks_per_sec = len(chunk_texts) / elapsed if elapsed > 0 else 0
    log.info(f"  Done in {elapsed:.1f}s ({chunks_per_sec:.0f} chunks/sec)")

    vectors = np.asarray(embeddings, dtype=np.float32)

    # ---- 5. Build turbovec index ----
    log.info(f"Building turbovec index (dim={dim}, bit_width={args.bits})...")
    from turbovec import IdMapIndex

    index = IdMapIndex(dim=dim, bit_width=args.bits)
    index.add_with_ids(vectors, chunk_ids)

    index_path = target_dir / "patent_index.tvim"
    index.write(str(index_path))
    idx_mb = index_path.stat().st_size / 1_000_000
    log.info(f"  Wrote {index_path} ({idx_mb:.0f} MB, "
             f"{len(index):,} vectors)")

    log.info("=" * 60)
    log.info("Merge complete!")
    log.info(f"  Patents:  {n_patents:,}")
    log.info(f"  Chunks:   {n_chunks:,}")
    log.info(f"  Index:    {len(index):,} vectors")
    log.info(f"  Metadata: {meta_path}")
    log.info(f"  Index:    {index_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
