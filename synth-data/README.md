# synth-data (`synthdata`)

Builds an evaluation set from your own chunks: it generates question and answer pairs, has a critic model filter them, and removes near-duplicates. The output works directly with `eval-harness` (answer quality) and `retrieval-kit` (retrieval quality).

```
synthdata gen chunks.jsonl -o eval.jsonl --qrels qrels.jsonl \
  --sample 60 --per-chunk 2 --multi-hop 15 --unanswerable 10 \
  --persona "a first-time customer with no technical background" --language English

evalh run --task my_rag.py:answer --dataset eval.jsonl --metrics correctness,faithfulness
retrievekit bench index qrels.jsonl --modes bm25,hybrid,hybrid+llmrerank
```

## Question kinds
| kind | what it tests |
|---|---|
| `factual` | looking up a fact stated in one chunk |
| `reasoning` | applying facts from one chunk, for example to a short case scenario |
| `multi_hop` | combining two related chunks from the same source |
| `unanswerable` | on-topic questions the material does not answer; the expected answer is a refusal, which tests for hallucination |

## Quality controls
- A separate critic model scores every item 1–5 on *grounded*, *standalone* and *realistic* for the persona. Items below `--min-score` (default 4) on any of the three are dropped.
- Questions whose word sets overlap by at least 80% (Jaccard) are treated as duplicates and removed.
- `synthdata gen` prints what happened to each kind of question: asked for, made, rejected by the critic (and on which criterion), failed, duplicates and kept, and marks kinds where nothing was kept. In Python, pass `report={}` to `generate()`. Small local models often keep only a fraction, so read it before trusting the set's balance.
- Use a critic that differs from the generator (`--critic-model`); the CLI notes when they are the same model, since a model rarely rejects its own questions.
- Each row records its source chunk ids and the critic's reasoning in `metadata`, so you can audit it.
- Each row also records `spans`, the answer passage's source and character offsets, and `--qrels` passes them on as `relevant_spans`, so the retrieval labels stay valid if you re-chunk the corpus.

Read through a sample of the output before relying on it. Synthetic sets tend to over-represent easy, lexically matching questions, so mix in real user questions where you have them.
