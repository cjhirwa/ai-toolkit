# embedding-viz (`embviz`)

Shows what is in your corpus, what is duplicated, and which user questions it has no good answer for.

```
embviz map index/ --color source                       # interactive scatter plot; hover shows chunk text
embviz map index/ --color cluster --clusters 15 --queries eval.jsonl   # overlay eval questions as ×
embviz dupes index/ --threshold 0.95 -o dupes.csv      # same PDF ingested twice, boilerplate pages
embviz coverage index/ real_questions.jsonl --threshold 0.6   # questions with no nearby chunk
```

The input can be a `retrieval-kit` index directory, which reuses its stored embeddings, or a `chunks.jsonl`, which gets embedded through llmgw.

| what | how |
|---|---|
| 2-D projection | UMAP if `embedding-viz[umap]` is installed, otherwise PCA; t-SNE needs `[sklearn]` |
| clusters | spherical k-means on cosine similarity, implemented directly so it needs no extra dependencies |
| duplicates | pairs whose cosine similarity is at least the threshold, computed in blocks so large corpora fit in memory |
| coverage | each query's most similar chunk, lowest similarity first |

Points to look for on the map:
- query × marks far from any chunk usually mean missing content, so the fix is new documents rather than tuning retrieval;
- one dense cluster coming from many sources means repeated boilerplate;
- a source that is spread across several clusters probably mixes topics that could be split into separate documents.

```python
from embviz import load_corpus, reduce, corpus_map, save_html
c = load_corpus("index/")
save_html(corpus_map(c, reduce(c.vectors)), "map.html")
```
