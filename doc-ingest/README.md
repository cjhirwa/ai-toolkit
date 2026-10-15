# doc-ingest (`docingest`)

Takes PDFs, DOCX, HTML, Markdown, text and scanned images, and outputs clean chunks in `chunks.jsonl`. Each chunk keeps its source, page, character offsets and heading path, so retrieval results can always be traced back to where they came from.

```python
from docingest import ingest, HeadingChunker, LLMOCR, write_jsonl

chunks = ingest(["data/manuals/"], HeadingChunker(size=1200, overlap=150), ocr=LLMOCR())
write_jsonl(chunks, "chunks.jsonl")
```

Each line of the output looks like this:

```json
{"id": "3f2a…-0007", "doc_id": "3f2a…", "text": "Installation > Wiring\n\nConnect the red wire…",
 "metadata": {"source": "manual.pdf", "title": "manual", "page": 12, "chunk_index": 7,
              "start": 18231, "end": 19302, "heading_path": "Installation > Wiring"}}
```

`source` is the path relative to the folder you passed (`manual.pdf`, `appendix/wiring.pdf`), or the file name when you pass a file, so it stays the same wherever you run the command.

## Loaders
- **PDF**: uses PyMuPDF if installed (`doc-ingest[pdf]`), otherwise pypdf. When a page has no text layer and an OCR backend is set, the page is rendered at 200 dpi and OCR'd (needs PyMuPDF; see [OCR](#ocr)).
  PDFs have no heading markup, so with PyMuPDF the loader infers headings from typography and writes them as Markdown `#` lines, which `HeadingChunker` turns into `heading_path`. Lines noticeably larger than the body text become headings, ranked by size; short standalone bold lines become the next level. Cover-page titles, running headers, page numbers and pull quotes are ignored. Pass `headings=False` to `load_pdf` to turn it off. Without PyMuPDF, PDFs get no headings.
- **DOCX**: headings become `#` Markdown headings, list items become `-` bullets, and tables become Markdown tables.
- **HTML**: removes scripts and navigation, and turns `h1`–`h6` into Markdown headings.
- **Images**: require an OCR backend.

## OCR

OCR is **off by default** (`--ocr none`). Without it, only text that is already in the file is extracted: pages that are scanned images come out empty and are left out of the chunks. Ingestion warns about them, for example:

```
WARNING guide.pdf: 24/253 pages have no text layer; pass an OCR backend
```

When you see that warning, turn OCR on.

Only pages with almost no text (under 25 characters) are sent to OCR; pages that already have text are not. Several pages are sent at once: 4 for cloud models, 1 for Ollama (a local server answers one request at a time). Set `DOCINGEST_OCR_CONCURRENCY` to change it, for example lower it if your provider returns `429` rate-limit errors.

### Backends
| backend | CLI | runs where | good at | needs |
|---|---|---|---|---|
| `LLMOCR()` | `--ocr llm` | the model in `DOCINGEST_OCR_MODEL` (default `gemini:gemini-3.6-flash`), through llmgw | tables, handwriting, mixed layouts; keeps the original language, does not translate | `doc-ingest[pdf]` and a provider key |
| `TesseractOCR(lang="eng")` | `--ocr tesseract` | locally, free | clean printed text | `doc-ingest[tesseract]` and the [tesseract](https://github.com/tesseract-ocr/tesseract) program on your `PATH` |

### Set up
OCR works on images of pages, and only PyMuPDF can render a PDF page to an image. PyMuPDF is not installed by default, so install an extra first:

```
uv sync --all-packages --extra pdf          # for --ocr llm
uv sync --all-packages --extra tesseract    # for --ocr tesseract (includes PyMuPDF)
uv run docingest run -o data/chunks.jsonl --ocr llm
```

Keep passing the extra: a plain `uv sync` removes packages that are not in the default install, PyMuPDF included. In another project, install `doc-ingest[pdf]` instead.

PyMuPDF also extracts text more accurately than the default `pypdf`, so it is worth installing even without OCR.

If you pass `--ocr` without PyMuPDF installed, the pages that have text are still ingested and a warning tells you to install `doc-ingest[pdf]`; the scanned pages are left out.

**A failed OCR page never costs the whole document.** If the OCR model fails on a page after its retries (provider overloaded, timeout), that page is left out with a warning naming it, and the rest of the document is kept:

```
WARNING guide.pdf: OCR failed on page 14, leaving it out: 503 UNAVAILABLE ...
```

Re-run later to fill those pages in.

Chunks whose text came from OCR have `"ocr": true` in their metadata, so you can review them or filter them out, for example to check OCR quality before indexing.

### Things to know
- **Choosing the OCR model.** `DOCINGEST_OCR_MODEL` is the model that does OCR, set on its own in `.env`. It is independent of `LLMGW_DEFAULT_MODEL`: changing the model used for answers, judging and everything else never changes OCR, and the reverse. In Python, pass `LLMOCR(model="openai:gpt-4o-mini")` or `LLMOCR(llm=LLM(...))`. The model must accept images: Gemini and OpenAI vision models do, as do Ollama vision models such as `llava`.
- **Cost and privacy.** `--ocr llm` makes one vision call per image-only page, and those page images are sent to the provider of `DOCINGEST_OCR_MODEL`. Use `--ocr tesseract`, or an Ollama vision model in `DOCINGEST_OCR_MODEL`, to keep documents on your machine.
- **Licence.** PyMuPDF is licensed under the AGPL, unlike this toolkit (MIT). That is why it is an optional extra: check that its licence suits your project before shipping it.
- **Caching.** `--ocr llm` results are cached in `.llmgw/ocr-cache.sqlite`, keyed by the page image's hash, the model and the prompt, so re-running ingestion does not pay to read the same page twice. The cache stores the extracted page text, so treat it like the documents themselves; set `DOCINGEST_OCR_CACHE=0` to turn it off or to a path to move it.
- **Images** (`.png`, `.jpg`, ...) always need an OCR backend; without one they are skipped.
- **Your own backend.** Anything with an `image_to_text(image: bytes, mime: str) -> str` method can be passed as `ocr=`.

## Chunkers
| chunker | what it does |
|---|---|
| `RecursiveChunker(size, overlap)` | splits on paragraph, then line, then sentence, then word, and packs the pieces up to `size` |
| `FixedChunker(size, overlap)` | fixed-size windows that only break between words |
| `HeadingChunker(size, overlap)` | one chunk per section, splits long sections, adds `heading_path` and prepends it to the text |
| `SemanticChunker(llm, max_size)` | breaks where the embedding distance between neighbouring sentences jumps |

Every chunker returns character spans, so you can write your own and pass it in.

## CLI
```
docingest run data/ -o chunks.jsonl --chunker headings --size 1200 --overlap 150 --ocr llm
docingest run -o chunks.jsonl      # no path: reads the folder in DOCS_DIR (.env)
docingest stats chunks.jsonl
```
Chunks with identical text (after normalizing whitespace and case) are removed, which handles the same PDF saved twice.

Files larger than `DOCINGEST_MAX_FILE_MB` (default 200) and PDFs with more than `DOCINGEST_MAX_PAGES` pages (default 5000) are skipped with a warning, so one huge or malformed upload cannot exhaust memory or stall a run. Set either to `0` to remove the limit.
