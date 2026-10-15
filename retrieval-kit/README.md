# retrieval-kit (`retrievekit`)

Interchangeable retrieval strategies behind a single call, `retrieve(query, k, filters)`, plus a benchmark that compares them on your own data.

```python
from retrievekit import Index

idx = Index.build("chunks.jsonl", "index/")       # embeds once via llmgw
idx = Index.load("index/")

hits = idx.retriever("hybrid").retrieve("how do I reset the thermostat", k=5,
                                         filters={"source": ["manual.pdf", "troubleshooting.pdf"]})
for h in hits:
    print(h.score, h.metadata["source"], h.metadata.get("page"), h.text[:80])
```

## Building blocks
| piece | notes |
|---|---|
| `BM25Retriever(docs)` | lexical search, so it handles exact product names, error codes and SKUs |
| `DenseRetriever.build(docs, llm)` | in-memory cosine search; query and document task types for Gemini |
| `ChromaRetriever.open(path, collection)` | the same interface over an existing Chroma store |
| `HybridRetriever([bm25, dense])` | fused with reciprocal rank fusion |
| `MultiQuery(llm, n)` / `HyDE(llm)` | query transforms |
| `LLMReranker(llm)` / `CrossEncoderReranker()` | rerankers; the cross-encoder needs `retrieval-kit[cross-encoder]` |
| `Pipeline(retriever, transforms, reranker)` | runs transforms, fuses results, then reranks |

Mode strings put these together for you: `bm25`, `dense`, `hybrid`, `hybrid+mq`, `hybrid+hyde`, `hybrid+llmrerank`, `hybrid+mq+cerank`, and so on.

Filters are either dicts (`{"source": "x.pdf"}` or `{"page": [1, 2]}`) or any `fn(metadata) -> bool`. Dict filters are vectorised (a filtered query over 200k chunks takes about 30 ms instead of 70 ms), so prefer them to functions on large indexes. Hybrid search and query transforms run their sub-searches in parallel.

## Benchmark
The qrels file is JSONL with rows like `{"query": "...", "relevant_ids": ["chunk-id", ...]}`. Rows in the format `synth-data` produces also work.

**Comparing chunkers.** Chunk ids only mean something for the chunking that made them. Rows can also carry `"relevant_spans": [{"source", "start", "end"}]`, the answer's position in the extracted document text (`synth-data` writes these). When present, a retrieved chunk counts as relevant if it covers at least half of the shorter of itself and the span, whatever chunker produced it, and each span counts once. So the same qrels can score a headings-based index against a 600-character one. Offsets refer to the extracted text: changing the chunker is fine, changing how text is extracted (a different PDF reader) is not.

```
retrievekit build chunks.jsonl -o index
retrievekit search index "error E4 after changing batteries" --mode hybrid+llmrerank
retrievekit bench index qrels.jsonl --modes bm25,dense,hybrid,hybrid+mq,hybrid+llmrerank -k 10
```
The output is a table with one row per mode and columns `recall@k`, `hit@k`, `mrr`, `ndcg@k` and `latency_ms`.
