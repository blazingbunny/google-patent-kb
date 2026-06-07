# google-patent-kb — Specification

## 1. Overview

Build a semantic search knowledge base over US patents using the Google Patents
Public Datasets on BigQuery. The build runs once on a temporary GPU instance,
produces a local turbovec vector index plus SQLite metadata store, and serves
semantic search from argus without external query-time APIs.

### 1.1 Target corpus

The first production corpus is intentionally narrow:

- **Search engine patents:** CPC prefix `G06F16`
- **Machine-learning / AI overlap:** CPC prefix `G06N`
- **Natural-language processing overlap:** CPC prefix `G06F40`
- **Jurisdiction:** US publications only
- **Publication kinds:** `A1`, `B1`, `B2` once the BigQuery query is corrected

Expected scale is roughly **728K patents** and **6.7M chunks** (based on
measured 9.2 chunks/patent from a test build).

### 1.2 Goals

1. **Semantic patent search** — find patents by meaning, not only keyword match.
2. **Filtered retrieval** — constrain search by CPC class, assignee, publication
   date range, and chunk type.
3. **Self-hosted serving** — no external API dependency at query time.
4. **Cheap initial build** — target <$10 for the first search/ML corpus.
5. **Scalable architecture** — the same pipeline should scale to the full US
   corpus after resume, filtering, and storage limits are hardened.

### 1.3 Non-goals

- Not a Google Patents replacement: no citations graph, family expansion, legal
  status, prosecution history, or full public UI.
- Not real-time: corpus updates are batch rebuilds or append-only increments.
- Not multi-tenant: deployment is for a single user/team.
- No frontend in this phase: FastAPI only.
- No paid embedding API in the default serving path.

---

## 2. System Architecture

```text
BigQuery: patents-public-data.patents.publications
    │
    │  column-limited query over US patent records
    ▼
Temporary vast.ai GPU instance
    │
    │  stream rows → chunk text → embed chunks → write turbovec + SQLite
    ▼
Argus CPU server
    │
    │  FastAPI query service + local embedding model + turbovec search
    ▼
HTTP API clients
```

### 2.1 Build/serve split

The system deliberately separates expensive build-time work from cheap serving:

- **Build:** vast.ai RTX 4090/3090, short-lived, GPU embedding throughput.
- **Serve:** argus, Xeon E-2388G, 125 GB RAM, AVX-512, no GPU.

The output contract between the two is:

| File | Purpose |
|---|---|
| `patent_index.tvim` | turbovec `IdMapIndex` containing chunk embeddings keyed by `chunk_id` |
| `patent_meta.db` | SQLite metadata and chunk text keyed by the same `chunk_id` values |
| `pipeline_checkpoint.json` | Build progress state; currently advisory, not a reliable resume mechanism |

---

## 3. Data Pipeline

### 3.1 BigQuery source

**Source table:** `patents-public-data.patents.publications`

**Fields extracted:**

| Field | Type | Source |
|---|---:|---|
| `publication_number` | string | Direct column |
| `country_code` | string | Direct column |
| `kind_code` | string | Direct column |
| `title` | string | English `title_localized` entry |
| `abstract` | string | English `abstract_localized` entry |
| `claims` | string | English `claims_localized` entry |
| `description` | string | English `description_localized` entry |
| `assignee` | string/repeated-derived | Direct column as returned by BigQuery client |
| `inventor` | string/repeated-derived | Direct column as returned by BigQuery client |
| `filing_date` | string | `YYYYMMDD` integer cast to string |
| `publication_date` | string | `YYYYMMDD` integer cast to string |
| `ipc_codes` | string array | `ipc.code` (from `UNNEST(ipc)`) |
| `cpc_codes` | string array | `cpc.code` (from `UNNEST(cpc)`) |

### 3.2 BigQuery filters

Default production filters:

- `country_code = 'US'`
- `kind_code IN ('A1', 'B1', 'B2')`
- at least one of English claims or English description is present
- optional CPC prefix filters combined with **OR**, not AND
- optional publication-year range

CPC filters must be a **single BigQuery pass** using repeated `--cpc` flags:

```bash
python pipeline.py \
  --cpc G06F16 \
  --cpc G06N \
  --cpc G06F40 \
  --output ./data/search_plus
```

This avoids duplicate patents when a publication belongs to multiple selected
CPC classes. Running separate pipeline jobs per CPC class is not allowed for the
same output directory because it can duplicate chunks and vectors.

**Extracting specific patents by number:**

```bash
python pipeline.py \
  --publication-numbers US-7603345-B2,US-20240211660-A1 \
  --output ./data/curated
```

The `--publication-numbers` flag accepts a comma-separated list of publication
numbers in any format (hyphenated or not). Hyphens are stripped on both the
CLI side and the BigQuery SQL side for matching.

**Incremental mode:**

```bash
python pipeline.py \
  --cpc G06F16 \
  --output ./data/search_plus \
  --incremental
```

`--incremental` queries the existing SQLite metadata DB for already-ingested
patent numbers and adds BigQuery WHERE clauses to skip them. This avoids
re-processing and is safe for ongoing updates as long as the output directory
already contains a valid index+metadata pair.

### 3.3 Pre-flight checklist (status)

The following were identified as blockers before the first full run.
Current status as of the 5K test build (2026-06-07):

1. ~~**BigQuery alias filtering:** the SQL must not reference `description` or
   `claims` aliases in the `WHERE` clause.~~ **Done** — the query wraps
   the SELECT in a CTE (`WITH selected AS (...)`) and filters in the outer WHERE,
   which BigQuery supports. Verified by successful 5K build query.
2. ~~**Kind-code filter:** the implementation should add `kind_code IN ('A1','B1','B2')`.~~
   **Done** — present in the SQL since the first implementation.
3. **Resume semantics:** the original `--resume` flag loaded checkpoint state
   unconditionally and did not skip already-processed BigQuery rows. **Replaced**
   by `--incremental` (see §12.2), which queries existing SQLite metadata for
   already-ingested patent numbers and skips them before the BigQuery pass.
   The standalone `merge.py` tool combines multiple SQLite DBs + TVIM files
   by re-embedding from scratch (necessary due to turbovec API limitations).
4. **Filter allowlist cap:** resolved below default. ✅ (See §8.4.)
5. **Service startup mode:** resolved — production starts via
   `python query_service.py --index ... --meta ...` under systemd.
6. **Clean-output safety check:** the pipeline refuses to start if the output
   directory already contains index or metadata files. This is still in place.
7. **Python version on argus:** deferred — Ubuntu 22.04 / Python 3.10 until
   October 2026 EOL.

### 3.4 BigQuery cost (measured)

The `patents-public-data.patents.publications` table has no partitioning or
clustering. Every query scans the full table regardless of LIMIT or WHERE
filters. A column-limited US query scans approximately **1.5 TB**.

**Measured:** a CPC-filtered query scanning G06F16+G06N+G06F40 scanned
**1,511.3 GB** (confirmed 2026-06-07). The unfiltered result set is
**727,783 rows** for these three CPC classes combined.

At BigQuery on-demand pricing ($5/TB for the first 1 TB, then ~$5-7/TB after
free tier), expect:

- **CPC-filtered corpus** (~728K patents): ~$10-20 (full scan minus free tier)
- **Full US corpus** (~11M patents): same scan cost (same table scanned once)

Mitigations:

1. **Always dry-run first** — `pipeline.py` prints the estimated bytes before
   running the real query. Cancel if cost exceeds the monthly budget.
2. **Batch with `--limit`** for development — keep iteration under the 1 TB
   free tier during tuning.
3. **Future: export to GCS** — a one-time export of selected columns to
   Parquet/JSONL in GCS avoids per-query scan costs for the build pipeline.
4. **Future: partitioned clone** — create a yearly-partitioned clone of the
   subset and query against that.

---

## 4. Chunking Strategy

### 4.1 Chunk types

Each patent produces one or more chunks:

| Type | Count per patent | Use |
|---|---:|---|
| `title_abstract` | 0-1 | High-signal overview chunk for broad relevance |
| `claim` | N | Per-claim legal scope and claim-language search |
| `claims_all` | 0-1 fallback | Used only when individual claim splitting fails |
| `description` | N | Technical implementation detail and terminology recall |

The API must accept all four stored chunk types. Documentation examples should
prefer `claim`, `description`, and `title_abstract`, but `claims_all` is a real
fallback type and must not be hidden from consumers.

### 4.2 Claim splitting decision

BigQuery claim text should be assumed **good enough for an initial regex-based
split**, but not guaranteed clean enough to trust blindly.

Initial splitter:

```text
(?:^|\n)\s*(?:Claim\s+)?(\d+)[\.)]\s*
```

Decision:

- Use regex splitting for M0.
- Preserve a `claims_all` fallback when fewer than two claim boundaries are
  found.
- Add pre-flight sampling before the first full run: sample at least 50 patents
  across `A1`, `B1`, and `B2`; inspect split counts and first/last chunks.
- Continue if at least 90% split into plausible individual claims.
- If the success rate is lower, switch to safer `claims_all` chunking for the
  first build rather than blocking the project on a perfect parser.

Rationale: individual claims improve retrieval granularity, but search remains
usable if a minority of patents fall back to all-claims chunks.

### 4.3 Description chunking decision

Descriptions should be included in the initial index. They improve recall for
implementation details, terminology, and non-claim phrasing that users are
likely to search for.

Risk: description chunks are longer and noisier than claims, so they may crowd
out claim results for legal-scope queries.

Mitigation for M0:

- Preserve `chunk_type` in metadata.
- Expose `chunk_type` filtering in the API.
- Evaluate search quality by reporting results both with all chunks and with
  `chunk_type=claim` for known patent queries.
- Do **not** down-weight descriptions in the first implementation; turbovec
  returns raw similarity. If descriptions dominate poor results, add reranking
  or chunk-type weighting later.

### 4.4 Description chunk size

Description chunks target roughly 512 tokens, approximated as 2048 characters.
Paragraph grouping is preferred over hard token windows because it keeps local
technical context together.

If a single paragraph exceeds the budget, the current implementation keeps it
as one oversized chunk. That is acceptable for M0 but should be measured during
pre-flight sampling.

### 4.5 Chunk count estimates (measured)

Measured from a `--limit 5000` test build on 2026-06-07:

| Metric | Measured | Notes |
|---|---:|---:|
| Chunks per patent | 9.2 | 5,000 patents → 45,949 chunks |
| Title/abstract per patent | 1.0 | Every sampled patent had title+abstract |
| Claims per patent | ~4.5 | Varies by patent; claims_all fallback used when regex split finds <2 boundaries |

Estimated full-corpus projections:

| Corpus | Patents | Estimated chunks |
|---|---:|---:|
| `G06F16` search-engine corpus | ~200K | ~1.8M |
| `G06F16 ∪ G06N ∪ G06F40` | ~728K | ~6.7M |
| Full US corpus | ~11M | ~100M |

---

## 5. Embedding Model

### 5.1 Default: `BAAI/bge-large-en-v1.5`

| Property | Value |
|---|---|
| Dimensions | 1024 |
| Query-time dependency | Local CPU model |
| Build-time dependency | Local GPU model |
| License | MIT |
| Role | Default M0 embedding model |
| Build throughput (measured) | **61 chunks/sec** on RTX 4090 (46K chunks in 750s) |

Decision: use `bge-large-en-v1.5` for the first production build.

Rationale:

- It is local, cheap, and compatible with self-hosted serving.
- It avoids query-time API costs and privacy concerns.
- It is strong enough to validate the product shape before paid model work.

### 5.2 Alternatives

| Model | Dim | Use case |
|---|---:|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast smoke tests and tiny development corpora |
| `BAAI/bge-base-en-v1.5` | 768 | Lower CPU latency and smaller index if bge-large is too slow |
| `BAAI/bge-large-en-v1.5` | 1024 | Default quality-first local model |
| `voyage-law-2` | API | Future benchmark/rerank candidate, not M0 default |

### 5.3 Voyage/legal model decision

Do **not** benchmark `voyage-law-2` before the first local build. The first
unknown is whether the corpus, chunking, and turbovec serving path work at all.
A paid model benchmark before that would optimize the wrong layer.

Revisit legal/patent-specific embeddings after M0 if manual evaluation shows
poor recall on claim-language queries. The correct benchmark point is after a
known-query eval set exists and the local baseline has measured recall@10/MRR.

---

## 6. Vector Index and Metadata

### 6.1 turbovec index

Use turbovec `IdMapIndex` with `chunk_id` as the external vector ID.

| Parameter | Default | Notes |
|---|---:|---|
| `bit_width` | 4 | Default for argus RAM budget |
| `dim` | 1024 | Derived from embedding model |
| ID type | `uint64` | SQLite `chunk_id` values |

### 6.2 Quantization tradeoff

| Bit width | Vector size at 1024 dim | 10M vectors | Expected recall tradeoff |
|---|---:|---:|---|
| 2 | 256 bytes | ~2.6 GB | Lower recall, smaller/faster |
| 4 | 512 bytes | ~5.1 GB | Better recall, still compact |

Decision: start with 4-bit. Switch to 2-bit only if measured full-corpus memory
pressure leaves too little headroom for SQLite cache, OS cache, and the
embedding model.

### 6.3 SQLite metadata

SQLite stores:

- patent-level metadata in `patents`
- chunk text and chunk labels in `chunks`
- JSON-encoded IPC/CPC arrays

Minimum indexes:

- `chunks(patent_number)`
- `chunks(chunk_type)`
- additional indexes should be added before broad filtered serving:
  - `patents(publication_date)`
  - potentially normalized CPC/assignee tables if LIKE scans are too slow

### 6.4 Deduplication decision

For M0, deduplication is handled by **single-pass BigQuery selection** over all
requested CPC prefixes. Do not run separate builds into the same output.

Future hardening:

- enforce a unique chunk identity such as `(patent_number, chunk_type, chunk_label)`
- skip existing patents on resume before inserting chunks
- load and append to existing turbovec index only when append semantics are
  explicitly tested

---

## 7. Checkpointing and Resume

### 7.1 Current state

The current checkpoint records:

- patents processed
- chunks indexed
- last patent number

However, this is not a complete resume system. A safe resume must satisfy all
of the following:

1. avoid inserting duplicate chunks into SQLite,
2. avoid inserting duplicate vectors into turbovec,
3. preserve already-built vectors or rebuild from a clean metadata state,
4. resume BigQuery processing without a full expensive scan when practical,
5. make interruption state obvious in logs and checkpoint metadata.

The current code enforces a clean-output safety check: the pipeline refuses
to start if index or metadata files already exist in the output directory.
This prevents accidental state corruption during M0 while resume remains
unsafe.

### 7.2 M0 decision

For the first search/ML corpus, prefer **clean restart over resume** unless the
pipeline is fixed and tested on a small interrupted run.

Before claiming resume support, run this test:

1. Build `--limit 1000` into a fresh output directory.
2. Interrupt during processing or force an exception after at least one flush.
3. Rerun with the intended resume command.
4. Verify patent count, chunk count, vector count, and duplicate chunk identity.
5. Verify search works against chunks created before and after interruption.

### 7.3 Full-corpus future

For full US scale, export BigQuery results to durable storage first, then process
locally with file/object offsets. That makes resume cheap and deterministic.
Streaming directly from BigQuery is acceptable for M0 but not ideal for a
multi-hour full-corpus build.

---

## 8. Query Service

### 8.1 Search flow

1. Embed query on CPU using the same model family as the index.
2. Resolve optional metadata filters to a chunk allowlist.
3. Search turbovec, optionally constrained by allowlist.
4. Fetch chunk previews and patent metadata from SQLite.
5. Return ranked results with timing information.

### 8.2 API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Basic readiness and vector count |
| `GET` | `/stats` | Patent/chunk counts, chunk-type distribution, and vector count from the loaded index |
| `GET` | `/search?q=...` | Browser/curl-friendly quick search |
| `POST` | `/search` | Structured search with filters |
| `GET` | `/patent/{number}` | Full patent metadata and all stored chunks |

### 8.3 POST `/search`

```json
{
  "query": "neural network transformer",
  "k": 10,
  "cpc": "G06N",
  "assignee": "Google",
  "chunk_type": "claim",
  "year_start": 2018,
  "year_end": 2024
}
```

`chunk_type` may be one of `title_abstract`, `claim`, `claims_all`, or
`description`.

### 8.4 Filtering mechanism

M0 uses SQLite to resolve filters into chunk IDs, then passes those IDs as a
turbovec allowlist.

The allowlist has no default size cap (`--allowlist-limit 0`, unlimited).
Set `--allowlist-limit 100000` to cap resolution if SQLite scans for broad
filters (e.g. unfiltered CPC or date ranges exceeding a million chunks) become
slow.

For production with frequent broad CPC filters, consider normalizing CPC and
assignee into separate tables (see §12.4) rather than relying on JSON `LIKE`
scans.

Filter semantics:

### 8.5 Publication number normalization

The search endpoint normalizes publication numbers by stripping hyphens before
comparison:

```
Input: "US-7603345-B2"   →  "US7603345B2"
Input: "US7603345B2"     →  "US7603345B2"
```

Both the BigQuery SQL and the query service use `REPLACE(column, '-', '')`
on both sides of the comparison. The CLI also strips hyphens from
`--publication-numbers` input before passing to BigQuery parameters.

This means any input format (hyphenated or not) works at every entry point.

### 8.6 Security and access

The query service binds to `127.0.0.1` only and is firewalled by UFW. It is
not directly reachable over the public internet. Access requires either:

- **Localhost** on argus itself
- **Tailscale SSH tunnel** from a machine on the same Tailscale network:

  ```bash
  ssh -L 8081:localhost:8081 -N adrian@100.124.16.75
  # Then: curl http://localhost:8081/search?q=...
  ```

This ensures the service is only accessible to authorized Tailscale users
without exposing it to the public internet.

### 8.7 Performance targets

Targets for the initial `G06F16 ∪ G06N ∪ G06F40` corpus on argus:

| Metric | Target | Notes |
|---|---:|---|
| P50 search latency | <150 ms | Unfiltered, warm service |
| P95 search latency | <300 ms | Includes CPU query embedding |
| Throughput | 10+ QPS | Single process, bge-large CPU embedding |
| Query embedding | <100 ms | Must be measured on argus |
| Vector search | <20 ms | Expected to be much faster than embedding |

Previous sub-100ms end-to-end targets are plausible but should be treated as
aspirational until measured on the actual corpus and machine.

---

## 9. Deployment Plan

### 9.1 Build on vast.ai — measured performance

Instance target:

- RTX 4090 preferred; RTX 3090 acceptable
- at least 100 GB disk for first corpus
- PyTorch image with CUDA available
- expected cost: around $11 for the first corpus (confirmed: $0.38/hr × ~30h)

**Measured throughput** from a `--limit 5000` test build (2026-06-07):

| Phase | Duration | Rate |
|---|---:|---:|
| BigQuery scan (1.5 TB) | ~7 min | ~727K patents in ~20 min (estimated) |
| Patent chunking (CPU) | ~6 min | ~830 patents/min |
| Embedding (RTX 4090) | 750s | 61 chunks/sec |
| **Total per 5K patents** | **~13 min** | **~0.4 patents/sec** |

**Projected full build** (728K patents, ~6.7M chunks):

| Metric | Estimate | Notes |
|---|---:|---:|
| Embedding time | ~30.5h | 6.7M / (61 chunks/s) / 3600 |
| BigQuery + chunking | ~3h | Overlaps partially with embedding (streaming pipeline) |
| **Total wall time** | **~30h** | |
| **GPU cost** | **~$11.40** | $0.38/hr × 30h |
| Index size | ~3.4 GB | 500 bytes/chunk × 6.7M |
| Metadata DB size | ~67 GB | 10 KB/chunk × 6.7M (will need compression or pruning) |

*Note: The vast.ai RTX 4090 instance used for the 5K test has since been
destroyed. A new GPU instance must be provisioned for the full build.*

Pipeline CLI flags:

```bash
# Default production filters
python pipeline.py \
  --cpc G06F16 --cpc G06N --cpc G06F40 \
  --output ./data/search_plus

# Extract specific patents by number
python pipeline.py \
  --publication-numbers US-7603345-B2,US-20240211660-A1 \
  --output ./data/curated

# Incremental: skip already-ingested patents
python pipeline.py \
  --cpc G06F16 \
  --output ./data/search_plus \
  --incremental

# Merge two separate runs
python merge.py \
  --dirs ./data/search_plus ./data/curated \
  --output ./data/merged
```

Build sequence:

1. Install dependencies.
2. Configure BigQuery service-account credentials.
3. Run a dry-run/smoke test with `--limit 10` and inspect chunks.
4. Run `--limit 1000` and verify DB/index consistency.
5. Run the first CPC-filtered production build.
6. Copy index and metadata to argus.

### 9.2 Serve on argus

Start the service with explicit index and metadata paths, binding to localhost
for security:

```bash
python query_service.py \
  --index ./data/search_plus/patent_index.tvim \
  --meta ./data/search_plus/patent_meta.db \
  --model BAAI/bge-large-en-v1.5 \
  --allowlist-limit 0 \
  --host 127.0.0.1 \
  --port 8081
```

`--allowlist-limit` sets the maximum number of chunk IDs that SQLite can pass
to turbovec for filtered searches. `0` (default) means unlimited. Set to e.g.
`100000` to cap broad CPC/date-range filters from saturating SQLite scans.

`--host 127.0.0.1` binds to localhost only. The service is behind UFW and only
reachable from argus itself or via a Tailscale SSH tunnel:

```bash
# From a Tailscale-connected client:
ssh -L 8081:localhost:8081 -N adrian@100.124.16.75
curl http://localhost:8081/health
```

Systemd runs the same entrypoint. See `DEPLOY.md` for the full unit definition.

---

## 10. Verification Strategy

### 10.1 Pre-flight pipeline verification

Before the first full build:

1. **SQL dry run:** confirm bytes scanned and query validity.
2. **Tiny build:** `--limit 10`; manually inspect title, claims, description,
   and chunk labels.
3. **Chunk quality sample:** at least 50 patents across publication kinds;
   record claim split success rate.
4. **Small index build:** `--limit 1000`; verify:
   - patent count > 0
   - chunk count > patent count
   - vector count equals chunk count
   - `/stats` matches SQLite and index counts
5. **Known-query smoke tests:** run the API and inspect top-k results.

### 10.2 Known-query manual tests

Initial smoke queries:

| Query | Expected behavior |
|---|---|
| `spam detection documents` | Should surface Anna Patterson / Google spam or document scoring patents if present in corpus |
| `PageRank` | Should surface PageRank-related Google patents if present in corpus |
| `machine learning ranking` | Should surface learning-to-rank / ML ranking patents |
| `query expansion synonyms` | Should surface search query expansion patents |
| `natural language search intent` | Should surface NLP/search-intent patents in `G06F40`/`G06N` overlap |

Do not require exact patent numbers until the corpus membership has been
verified. Some canonical patents may be outside the selected CPC prefixes or
missing full English text.

### 10.3 Measured test results

A `--limit 5000` test build was completed on 2026-06-07 (see §9.1 for timing).

**Verification results:**

| Check | Result |
|---|---:|
| Chunk count > patent count | 45,949 > 5,000 ✓ |
| Vector count = chunk count | 45,949 = 45,949 ✓ |
| turbovec index searchable | 5/5 semantic queries return relevant results ✓ |
| Metadata DB accessible | All chunk types retrievable via chunk_id join ✓ |
| BigQuery streaming | Direct query works for <10K rows, destination table for larger sets ✓ |
| Multi-CPC deduplication | Single-pass query, no duplicate patents ✓ |

**Known-query results:**

| Query | Top result relevance | Source |
|---|---:|---:|
| "information retrieval database search indexing" | US-2011161260-A1 "User-driven index selection" | G06F16 ✓ |
| "neural network attention transformer model" | US-2023177338-A1 "Small and fast transformer model" | G06N ✓ |
| "XML document schema parsing validation" | US-2013061133-A1 "Markup language schema error correction" | G06F40 ✓ |
| "natural language processing text classification" | US-2014279761-A1 "Document Coding Computer System" | G06F40 ✓ |

### 10.4 Automated eval future

After M0, create 20-50 query → known relevant patent pairs and track:

- recall@10
- MRR
- result mix by chunk type
- filtered vs unfiltered quality
- bge-large vs bge-base vs legal/patent-specific alternatives if needed

---

## 11. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| BigQuery SQL mismatch with real schema | Build fails immediately | Run dry-run and `--limit 10` before renting long GPU time |
| Claim parsing quality lower than expected | Claim search is noisy | Keep `claims_all`; sample before full build |
| Description chunks dominate results | Lower precision | Use `chunk_type=claim`; add reranking/weighting later |
| Resume duplicates chunks/vectors | Corrupt index/metadata alignment | Treat resume as unsafe until tested; use clean output dirs |
| Broad filter allowlist truncates results | Incorrect filtered search | Already fixed: code defaults to unlimited; set `--allowlist-limit` only if SQLite scans are slow |
| CPU embedding slower than expected | Higher latency/lower QPS | Measure on argus; switch to bge-base or cache queries if needed |
| turbovec `IdMapIndex.load()` instance-method bug (v0.7.0) | Index loads as empty (0 vectors) | Always use class method `IdMapIndex.load(path)`; never `IdMapIndex().load(path)` |
| SQLite LIKE over JSON CPC arrays is slow | Slow filters | Normalize CPC/assignee metadata if measured slow |
| Full US index exceeds practical RAM | Serving instability | Use 2-bit quantization or serve narrower corpora first |
| Python 3.10 reaches EOL (Oct 2026) | `google-api-core` drops support, blocking BigQuery access from argus | Upgrade argus to Ubuntu 24.04 (Python 3.12) before Oct 2026; no code changes needed |

---

## 12. Future Extensions

### 12.1 Cross-encoder reranking

Rerank top-50 vector results with a local cross-encoder such as
`BAAI/bge-reranker-v2-m3` if top-10 precision is insufficient. Add only after
baseline latency and quality are measured.

### 12.2 Incremental updates

Implemented. The pipeline's `--incremental` flag queries the existing SQLite
metadata DB for already-ingested patent numbers, skips them in the BigQuery
pass, and appends new chunks and vectors. Standalone `merge.py` combines
multiple SQLite DBs + TVIM files by re-embedding all chunks from scratch
(necessary because turbovec's `IdMapIndex` API does not expose vector/ID
extraction for append).

Workflow for ongoing updates:

```bash
# Run pipeline with --incremental to skip already-ingested patents
python pipeline.py \
  --cpc G06F16 --cpc G06N --cpc G06F40 \
  --output ./data/search_plus \
  --incremental

# Or merge two separate builds into one index
python merge.py --dirs ./data/run1 ./data/run2 --output ./data/merged
```

### 12.3 Hybrid BM25 + vector search

Add SQLite FTS5 over chunk text and combine keyword and vector results. This is
likely valuable for exact technical phrases, inventor names, rare acronyms, and
patent-number-adjacent searches.

### 12.4 Better metadata model

Normalize repeated metadata into separate tables:

- `patent_cpc(patent_number, cpc_code)`
- `patent_ipc(patent_number, ipc_code)`
- `patent_assignee(patent_number, assignee)`

This will make filters faster and less error-prone than JSON `LIKE` scans.

### 12.5 Multi-jurisdiction corpus

Extend beyond US only after M0 quality is proven. EP/WO/JP coverage introduces
claim-format variation, non-English text, multilingual embedding choices, and a
larger index.

---

## 13. Resolved Section 7 Questions

1. **Claim text quality:** proceed with regex splitting plus `claims_all`
   fallback. Validate with a 50-patent sample before full run.
2. **Description relevance:** include descriptions in M0. Use `chunk_type`
   filters to diagnose noise; defer weighting/reranking until measured.
3. **Multi-CPC deduplication:** use a single BigQuery pass with repeated `--cpc`
   flags. Do not run one pipeline per CPC class into the same output.
4. **Checkpoint resumption speed:** resume is not safe enough for M0 claims.
   Use clean restarts for the first corpus unless interruption tests pass; use
   file/object export offsets for future full-corpus builds.
5. **Embedding model benchmark:** do not benchmark `voyage-law-2` before M0.
   Build the local bge-large baseline first, then benchmark alternatives only
   against a real eval set if quality is insufficient.
