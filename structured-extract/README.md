# structured-extract (`sextract`)

Turns documents into validated Pydantic records. It covers single texts, documents too long for one prompt, and whole folders of files.

```python
from pydantic import BaseModel
from sextract import extract, extract_long, extract_files, write_csv

class LineItem(BaseModel):
    name: str
    quantity: int
    unit_price: float | None = None

class Invoice(BaseModel):
    vendor: str
    invoice_date: str
    items: list[LineItem]

inv = extract(text, Invoice)                 # one call; validation errors go back to the model to fix
inv = extract_long(big_pdf_text, Invoice)    # extract per chunk, then merge
results = list(extract_files(["invoices/"], Invoice, instructions="Prices in EUR"))
write_csv(results, "invoices.csv")
```

## How long documents are handled
1. The text is split into chunks of about 12k characters.
2. Each chunk is extracted into a *partial* version of the schema, where every field is optional.
3. The partials are merged: list fields are concatenated with duplicates removed, and single-value fields take the first non-null value.
4. If two chunks disagree on a single-value field, or a required field is still missing, one final LLM call reconciles the merged record.

The system prompt tells the model to leave fields empty rather than guess, and to copy numbers and units exactly. Extraction runs at temperature 0.

## CLI
```
sextract run invoices/ --schema schemas.py:Invoice -o out.jsonl --csv out.csv --ocr llm -m gemini:gemini-3.6-flash
```
Files are loaded with `doc-ingest`, so PDF, DOCX, HTML and scans (with `--ocr`) all work. `-m` sets the extraction model; OCR uses `DOCINGEST_OCR_MODEL` (see the [doc-ingest OCR notes](../doc-ingest/README.md#ocr)).
