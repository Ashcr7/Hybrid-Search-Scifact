# Hybrid Search RAG on BEIR SciFact

A step-by-step project that builds a retrieval pipeline over the [BEIR](https://github.com/beir-cellar/beir) **SciFact** dataset, compares different retrieval strategies, and connects the best one to an LLM (ChatGPT) for question answering.

The goal is to measure what each component actually adds, so every method is evaluated on the same queries with the same metrics:

1. Dense search only (baseline)
2. BM25 (keyword search)
3. Hybrid search (dense + BM25)
4. Hybrid search + reranker
5. RAG with ChatGPT on top of the best retriever

## Status

| Step | Status |
|---|---|
| Load SciFact corpus, queries and test qrels | Done |
| Dense retrieval (Ollama `mxbai-embed-large` + Chroma) | Done |
| Evaluation of dense retrieval (Recall@k, k=2 to 6) | Done |
| BM25 retrieval | Done |
| Hybrid retrieval (dense + BM25) | Done (`EnsembleRetriever`, weighted RRF; Recall@20 = 0.917) |
| Cross-encoder reranker | Done. `ms-marco-MiniLM-L-6-v2` was below hybrid at every k (likely a domain mismatch); `bge-reranker-base` with a pool of 10 beats hybrid at k=3 to 8 (about +0.02 to +0.03 at k=4 to 7) |
| Comparison of all methods | Done (see Results and Findings) |
| ChatGPT integration | Done (hybrid + `bge-reranker-base`, top 6, `gpt-4o-mini`) |

## Dataset

SciFact is a scientific claim-verification dataset from BEIR.

| File | Content |
|---|---|
| `scifact/corpus.jsonl` | 5,183 abstracts (`_id`, `title`, `text`) |
| `scifact/queries.jsonl` | 1,109 claims (`_id`, `text`, `metadata`) |
| `scifact/qrels/test.tsv` | Relevance labels: `query-id`, `corpus-id`, `score` |

Only the queries listed in `qrels/test.tsv` (300 queries) are used for evaluation. The remaining queries belong to the training split and are skipped.

## Approach

### Documents

Each abstract becomes one LangChain `Document`:

```python
Document(
    page_content=f'{doc["title"]}. {doc["text"]}',   # embedded
    metadata={"corpus_id": doc["_id"]},              # returned with results, not embedded
)
```

- The title and text are embedded together, because the title usually names the subject that the claims refer to.
- The `corpus_id` lives in the metadata, so every retrieved document can be matched to the IDs in `test.tsv`. It is never part of the embedded text.
- Abstracts are short (roughly 200 to 300 words), so they are **not split into chunks**. The benchmark labels whole documents, and chunking would put several chunks of one document into the top-k.

### Dense retrieval

- **Embeddings:** `mxbai-embed-large` served locally through Ollama (`langchain-ollama`).
- **Vector store:** Chroma, persisted in `./Chroma-mxbai`.
- **Queries:** only the query text is embedded. The query ID is used as the key for storing results, and the query metadata (which holds the supporting-sentence annotations) is ignored.

Embedding all 5,183 abstracts on CPU takes a long time (about 1 second per document in my runs), so the notebook:

- embeds in small batches (32 at a time) instead of one huge request, which crashed the Ollama runner;
- is **resumable**: it reads the `corpus_id`s already stored in Chroma and embeds only the missing documents.

## Evaluation

For each test query the retriever returns the top-k documents. The retrieved `corpus_id`s are compared with the relevant IDs in `test.tsv`.

- **Recall@k:** the fraction of a query's relevant documents found in the top k, averaged over the 300 test queries.

The same metric code is used for every method, so the numbers are directly comparable.

## Results

Recall@k over the 300 SciFact test queries:

| Method | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 |
|---|---|---|---|---|---|---|---|---|---|
| Dense (mxbai-embed-large) | 0.687 | 0.719 | 0.761 | 0.777 | 0.822 | | | | |
| BM25 (lowercase + punctuation stripped) | 0.619 | 0.677 | 0.704 | 0.730 | 0.746 | 0.749 | 0.762 | 0.769 | 0.774 |
| Hybrid (RRF, 0.5/0.5, pool of 30 each) | 0.697 | 0.743 | 0.771 | 0.789 | 0.816 | 0.831 | 0.843 | 0.862 | 0.864 |
| Hybrid + reranker (ms-marco-MiniLM-L-6-v2, pool of 20) | 0.651 | 0.699 | 0.741 | 0.770 | 0.789 | 0.805 | 0.823 | 0.841 | 0.856 |
| Hybrid + reranker (bge-reranker-base, pool of 10) | 0.691 | 0.750 | 0.798 | 0.817 | 0.837 | 0.851 | 0.856 | 0.859 | 0.864 |

Dense Recall was computed for k=2 to 6 only.

### Findings

- **Dense beats BM25.** Dense is ahead at every k from 2 to 6, by about 0.07 (for example 0.761 vs 0.704 at k=4). BM25 also flattens out after k=6, which suggests it misses claims that use different wording from the abstract.
- **Hybrid is clearly better than BM25 and only slightly better than dense.** Hybrid is ahead of dense by 0.01 to 0.02 at k=2 to 5, but slightly behind at k=6 (0.816 vs 0.822). With 300 queries this small gap is not conclusive. At k=10, hybrid reaches 0.864 against 0.774 for BM25. Dense was not run beyond k=6, so there is no dense number to compare with at k=10.
- **A reranker only helps if it suits the domain.** `ms-marco-MiniLM-L-6-v2` (trained on web search) scored below hybrid at every k, for example 0.741 vs 0.771 at k=4. `bge-reranker-base` on the same hybrid candidates (pool of 10) scored above hybrid at k=3 to 8, with gains of about +0.02 to +0.03 at k=4 to 7 (0.798 vs 0.771 at k=4). At k=2, and at k=9 to 10, it is level with hybrid or marginally behind.
- **The pool limits what a reranker can do.** A reranker only reorders the candidates it is given. With a pool of 10, Recall@10 is identical to hybrid by construction (0.8636), and with a pool of 20 the Recall@20 of 0.9167 matches hybrid exactly, which also served as a check that the reranking code is correct.
- **Cost.** BM25 and the vector search take seconds. Reranking is the slowest step: with MiniLM and a pool of 20, 300 queries took about 9 minutes on CPU, and `bge-reranker-base` is larger and slower still.

**Chosen pipeline:** hybrid retrieval, a pool of 10, `bge-reranker-base`, top 6 passed to the LLM. It had the best recall in the k=4 to 7 range, where an LLM context of a few abstracts is practical.

**Limitations**
- Only Recall@k was measured. It does not reward placing a relevant document higher within the top k, so the effect of the reranker on ranking order is not fully captured.
- 300 test queries is a small sample, so differences of about 0.01 should not be read as real.
- Only one embedding model, one set of RRF weights (0.5 / 0.5) and one simple BM25 tokenizer were tried.

## RAG with ChatGPT

The final pipeline answers a scientific claim from the retrieved abstracts:

1. **Hybrid retrieval:** BM25 and dense (`mxbai-embed-large`) fused with weighted RRF (`EnsembleRetriever`, pools of 30 each), cut to 10 candidates.
2. **Reranking:** `BAAI/bge-reranker-base` re-scores the 10 candidates and the top 6 are kept.
3. **Generation:** `gpt-4o-mini` (temperature 0) receives the 6 abstracts and returns a verdict (`VALID`, `INVALID` or `NOT ENOUGH INFO PRESENT`), a short explanation and the citation ID. The prompt tells the model to use only the given context.

It is built with LangChain's `create_stuff_documents_chain` and `create_retrieval_chain`, with the retrieval and reranking steps wrapped in one `RunnableLambda`.

## Setup

**Requirements:** Python 3.12+, [Ollama](https://ollama.com), and about an hour of CPU time for the first embedding run.

```bash
# 1. Pull the embedding model
ollama pull mxbai-embed-large

# 2. Install the Python packages
pip install langchain-core langchain-text-splitters langchain-chroma langchain-ollama langchain-community langchain-openai rank_bm25 sentence-transformers python-dotenv ipykernel
```

Place the SciFact files in `scifact/` as shown above, then run `main.ipynb` from top to bottom. The first run builds the Chroma database. Later runs load it from disk.

For the ChatGPT step, add your key to a local `.env` file (never commit it):

```
OPENAI_API_KEY=...
```

## Project structure

```
Hybrid_search/
├── main.ipynb        # data loading, embedding, retrieval, evaluation
├── scifact/          # corpus, queries and qrels
├── Chroma-mxbai/     # persisted vector store (generated)
├── pyproject.toml
└── README.md
```

## Notes

- Vector search in Chroma returns distances, so a lower score means a closer match. Keep this in mind when combining scores in the hybrid step.
- Use the same `k` for every method when comparing results.
