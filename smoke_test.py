#!/usr/bin/env python3
"""Quick smoke test for patent KB deployment on argus."""
import sqlite3
import sys
import os
sys.path.insert(0, os.path.expanduser("~/patent_kb"))

from query_service import PatentSearchEngine

print("Loading index...")
index = PatentSearchEngine(
    index_path="./patent_index.tvim",
    meta_path="./patent_meta.db",
    model_name="BAAI/bge-large-en-v1.5",
)
print(f"Index: {len(index.index):,} vectors")

conn = sqlite3.connect("./patent_meta.db")
cnt_p = conn.execute("SELECT COUNT(*) FROM patents").fetchone()[0]
cnt_c = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
conn.close()
print(f"Patents in DB: {cnt_p:,}")
print(f"Chunks in DB: {cnt_c:,}")

# Run a search
result = index.search("neural network transformer attention mechanism", k=3)
hits = result["results"]
print(f"\nTop-3 results: {len(hits)}")
for r in hits:
    print(f"  {r['patent_number']}: {r['title'][:80]}... (sim={r['similarity']:.4f})")

print("\n✅ Smoke test passed!")
