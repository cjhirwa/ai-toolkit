import pytest

from docingest import (
    Document,
    FixedChunker,
    HeadingChunker,
    PageSpan,
    RecursiveChunker,
    SemanticChunker,
    chunk_document,
    ingest,
    read_jsonl,
    write_jsonl,
)
from llmgw import LLM, FakeProvider

TEXT = """# Installation

Installation takes about 20 minutes. Turn off power at the breaker first.

## Wiring

Connect the red wire to the R terminal. Label each wire before removing the old unit. Call an electrician if unsure.

## Mounting

Use the included wall plate and screws. Keep the unit away from heaters and direct sunlight.

# Warranty

The warranty covers defects for two years. Keep your receipt."""


def test_recursive_respects_size_and_overlap():
    text = " ".join(f"word{i}." for i in range(400))
    spans = RecursiveChunker(200, 50).split(text)
    assert all(b - a <= 200 for a, b, _ in spans)
    assert len(spans) > 5
    # consecutive chunks overlap
    assert spans[1][0] < spans[0][1]
    # every character of the text is covered
    assert spans[0][0] == 0 and spans[-1][1] == len(text)


def test_fixed_breaks_on_words():
    text = "alpha beta gamma delta " * 20
    spans = FixedChunker(50, 0).split(text)
    assert all(b - a <= 50 for a, b, _ in spans)
    words = set(text.split())
    assert all(set(text[a:b].split()) <= words for a, b, _ in spans)  # no word cut in half


def test_heading_chunker_paths():
    spans = HeadingChunker(size=500, overlap=0).split(TEXT)
    paths = [m.get("heading_path") for _, _, m in spans]
    assert "Installation > Wiring" in paths and "Warranty" in paths
    wiring = next(TEXT[a:b] for a, b, m in spans if m.get("heading_path") == "Installation > Wiring")
    assert "red wire" in wiring and "wall plate" not in wiring


def test_chunk_document_pages_and_prefix():
    p1 = "# Installation\n\nInstallation takes about 20 minutes."
    p2 = "## Wiring\n\nConnect the red wire."
    text = p1 + "\n\n" + p2
    doc = Document("d1", "x.pdf", text, [PageSpan(1, 0, len(p1)), PageSpan(2, len(p1) + 2, len(text))], {"source": "x.pdf"})
    chunks = chunk_document(doc, HeadingChunker(size=500, overlap=0))
    first, last = chunks[0], chunks[-1]
    assert first.text.startswith("# Installation")  # no duplicated heading prefix
    assert last.metadata["page"] == 2 and last.metadata["heading_path"] == "Installation > Wiring"
    assert last.text.startswith("Installation\n\n## Wiring")  # parent context added
    long_section = "# A\n\n" + "word " * 300
    mid = chunk_document(Document("d2", "x.md", long_section), HeadingChunker(size=400, overlap=0))[1]
    assert mid.text.startswith("A\n\n")  # mid-section chunk gets the full path


def test_semantic_chunker_with_fake_embeddings():
    llm = LLM("fake:x", provider=FakeProvider(), log_dir=False)
    spans = SemanticChunker(llm, max_size=300, min_size=20, breakpoint_percentile=70).split(TEXT)
    assert len(spans) >= 2 and all(b - a <= 300 for a, b, _ in spans)


def test_ingest_files_roundtrip(tmp_path):
    (tmp_path / "a.md").write_text(TEXT, encoding="utf-8")
    (tmp_path / "b.html").write_text("<html><title>T</title><body><h1>Warranty</h1><p>Keep your receipt for 2 years.</p><script>x</script></body></html>", encoding="utf-8")
    (tmp_path / "dup.md").write_text(TEXT, encoding="utf-8")
    chunks = ingest([tmp_path], HeadingChunker(400, 0))
    sources = {c.metadata["source"] for c in chunks}
    assert "b.html" in sources
    assert not any("dup.md" in c.metadata["source"] for c in chunks)  # deduped identical content
    html_chunk = next(c for c in chunks if c.metadata["source"].endswith("b.html"))
    assert "Keep your receipt" in html_chunk.text and "x" not in html_chunk.text.split()
    out = tmp_path / "c.jsonl"
    write_jsonl(chunks, out)
    assert [c.id for c in read_jsonl(out)] == [c.id for c in chunks]


def test_source_is_relative_to_input_folder(tmp_path):
    (tmp_path / "guides").mkdir()
    (tmp_path / "guides" / "wiring.md").write_text("Connect the red wire to the R terminal.", encoding="utf-8")
    (tmp_path / "warranty.txt").write_text("The warranty covers defects for two years.", encoding="utf-8")
    assert {c.metadata["source"] for c in ingest([tmp_path])} == {"guides/wiring.md", "warranty.txt"}
    assert {c.metadata["source"] for c in ingest([tmp_path / "warranty.txt"])} == {"warranty.txt"}


def test_cli_reads_docs_dir_from_env(tmp_path, monkeypatch):
    from docingest.cli import main

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "warranty.txt").write_text("The warranty covers defects for two years.", encoding="utf-8")
    monkeypatch.setenv("DOCS_DIR", str(tmp_path / "docs"))
    out = tmp_path / "chunks.jsonl"
    main(["run", "-o", str(out)])
    assert [c.metadata["source"] for c in read_jsonl(out)] == ["warranty.txt"]


def test_llm_ocr_model_selection(monkeypatch):
    import llmgw.client
    from docingest import LLMOCR

    monkeypatch.setitem(llmgw.client._INSTANCES, "fake", FakeProvider(["page text"]))  # undone after the test
    monkeypatch.setenv("LLMGW_LOG", "0")
    monkeypatch.setenv("LLMGW_DEFAULT_MODEL", "fake:default")
    monkeypatch.delenv("DOCINGEST_OCR_MODEL", raising=False)
    assert LLMOCR().llm.model == "gemini:gemini-3.6-flash"  # own default, not LLMGW_DEFAULT_MODEL
    monkeypatch.setenv("DOCINGEST_OCR_MODEL", "fake:vision")
    ocr = LLMOCR()
    assert ocr.llm.model == "fake:vision"
    assert ocr.image_to_text(b"png-bytes") == "page text"
    assert LLMOCR(model="fake:explicit").llm.model == "fake:explicit"


def _pdf_with_blank_middle_page(path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for text in ["Page one explains how to install the thermostat.", None, "Page three covers the warranty terms."]:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(path)
    return path


class _FailingOCR:
    def image_to_text(self, image, mime="image/png"):
        raise RuntimeError("503 UNAVAILABLE")


class _FakeOCR:
    def image_to_text(self, image, mime="image/png"):
        return "Page two was scanned and read by OCR."


def test_failed_ocr_page_keeps_the_rest_of_the_pdf(tmp_path, caplog):
    from docingest import load

    path = _pdf_with_blank_middle_page(tmp_path / "manual.pdf")
    doc = load(path, _FailingOCR())
    assert "install the thermostat" in doc.text and "warranty terms" in doc.text
    assert [s.page for s in doc.pages] == [1, 3]
    assert "OCR failed on page 2" in caplog.text
    assert "scanned" in load(path, _FakeOCR()).text


def test_ocr_without_pymupdf_ingests_text_pages(tmp_path, caplog, monkeypatch):
    import sys

    from docingest import load

    path = _pdf_with_blank_middle_page(tmp_path / "manual.pdf")
    monkeypatch.setitem(sys.modules, "pymupdf", None)  # import pymupdf now raises ImportError
    doc = load(path, _FakeOCR())
    assert "install the thermostat" in doc.text and "warranty terms" in doc.text
    assert "install doc-ingest[pdf]" in caplog.text


def test_blank_pages_are_reported_without_ocr(tmp_path, caplog):
    from docingest import load

    load(_pdf_with_blank_middle_page(tmp_path / "manual.pdf"))
    assert "1/3 pages have no text layer" in caplog.text


def test_chunks_from_ocr_pages_are_marked(tmp_path):
    path = _pdf_with_blank_middle_page(tmp_path / "manual.pdf")
    chunks = ingest([path], RecursiveChunker(60, 0), _FakeOCR())
    marked = {c.metadata["page"] for c in chunks if c.metadata.get("ocr")}
    assert marked == {2}
    assert all("ocr" not in c.metadata for c in chunks if c.metadata["page"] != 2)


def test_ocr_pages_run_concurrently_and_keep_their_place(tmp_path):
    import time

    pymupdf = pytest.importorskip("pymupdf")
    from docingest import load

    doc = pymupdf.open()
    for i in range(1, 7):  # pages 2, 4 and 6 are blank
        page = doc.new_page()
        if i % 2:
            page.insert_text((72, 72), f"Printed page {i} has enough text to skip OCR.")
    doc.save(tmp_path / "scan.pdf")

    class SlowOCR:
        concurrency = 3

        def image_to_text(self, image, mime="image/png"):
            time.sleep(0.3)
            return "Scanned text read by OCR."

    start = time.perf_counter()
    result = load(tmp_path / "scan.pdf", SlowOCR())
    assert time.perf_counter() - start < 0.8  # three 0.3 s pages in parallel, not 0.9 s in series
    assert [s.page for s in result.pages] == [1, 2, 3, 4, 5, 6]
    assert result.metadata["ocr_pages"] == [2, 4, 6]


def test_llm_ocr_concurrency_defaults(monkeypatch):
    from docingest import LLMOCR

    monkeypatch.delenv("DOCINGEST_OCR_CONCURRENCY", raising=False)
    assert LLMOCR(model="ollama:gemma3:4b").concurrency == 1
    assert LLMOCR(model="gemini:gemini-3.6-flash").concurrency == 4
    monkeypatch.setenv("DOCINGEST_OCR_CONCURRENCY", "8")
    assert LLMOCR(model="ollama:gemma3:4b").concurrency == 8
    assert LLMOCR(model="ollama:gemma3:4b", concurrency=2).concurrency == 2


def _typeset_pdf(path):
    """Cover page, then two chapters: big chapter titles (one wrapped over lines ending in
    commas), bold subheadings, a bold three-line quote, body text and a running header."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    body = "The thermostat measures the room temperature and switches the heating on or off. " * 3

    def page_with(items):
        page = doc.new_page()
        y = 60
        for text, size, font in items:
            page.insert_text((72, y), text, fontsize=size, fontname=font)
            y += size * 2.2

    page_with([("ACME THERMOSTAT GUIDE", 40, "helv"), ("Second Edition", 30, "helv")])
    for _, title in [(1, ["GETTING STARTED"]), (2, ["WIRING, POWER,", "AND SAFETY"])]:
        page_with([("Acme Thermostat Guide", 9, "helv")] + [(t, 24, "helv") for t in title]
                  + [(body[i:i + 90], 11, "helv") for i in range(0, 240, 90)]
                  + [("Before you begin", 11, "hebo")]
                  + [(body[i:i + 90], 11, "helv") for i in range(0, 240, 90)]
                  + [("Never open the unit while it is powered,", 11, "hebo"),
                     ("because the terminals carry mains voltage", 11, "hebo"),
                     ("and can give you a serious shock.", 11, "hebo")]
                  + [(body[i:i + 90], 11, "helv") for i in range(0, 240, 90)])
    for _ in range(2):  # more pages so the running header counts as repeated
        page_with([("Acme Thermostat Guide", 9, "helv")] + [(body[i:i + 90], 11, "helv") for i in range(0, 240, 90)])
    doc.save(path)
    return path


def test_pdf_headings_are_inferred_from_typography(tmp_path):
    from docingest import load

    doc = load(_typeset_pdf(tmp_path / "guide.pdf"))
    spans = HeadingChunker(size=2000, overlap=0).split(doc.text)
    paths = {m.get("heading_path") for _, _, m in spans} - {None}
    assert "GETTING STARTED > Before you begin" in paths
    assert "WIRING, POWER, AND SAFETY > Before you begin" in paths  # wrapped title joined
    assert not any("ACME THERMOSTAT GUIDE" in p or "Second Edition" in p for p in paths)  # one-off cover sizes
    assert not any("Acme Thermostat Guide" in p for p in paths)  # running header
    assert not any("Never open the unit" in p for p in paths)  # bold quote is not a heading
    from docingest.loaders import load_pdf

    assert "#" not in load_pdf(tmp_path / "guide.pdf", headings=False).text


def test_ocr_results_are_cached_by_page_image(tmp_path, monkeypatch):
    import llmgw.client
    from docingest import LLMOCR

    calls = []

    def read(req):
        calls.append(1)
        return "Scanned text"

    monkeypatch.setitem(llmgw.client._INSTANCES, "fake", FakeProvider(read))
    monkeypatch.setenv("DOCINGEST_OCR_CACHE", str(tmp_path / "ocr.sqlite"))
    monkeypatch.setenv("LLMGW_LOG", "0")
    assert LLMOCR(model="fake:vision").image_to_text(b"page-1") == "Scanned text"
    assert LLMOCR(model="fake:vision").image_to_text(b"page-1") == "Scanned text"  # new run, same page
    LLMOCR(model="fake:vision").image_to_text(b"page-2")
    assert len(calls) == 2
    monkeypatch.setenv("DOCINGEST_OCR_CACHE", "0")
    assert LLMOCR(model="fake:vision").llm.cache is None


def test_oversized_inputs_are_skipped_not_loaded(tmp_path, monkeypatch, caplog):
    (tmp_path / "big.txt").write_text("x" * 3_000_000, encoding="utf-8")
    (tmp_path / "small.txt").write_text("The warranty covers defects for two years.", encoding="utf-8")
    monkeypatch.setenv("DOCINGEST_MAX_FILE_MB", "1")
    assert {c.metadata["source"] for c in ingest([tmp_path])} == {"small.txt"}
    assert "over DOCINGEST_MAX_FILE_MB" in caplog.text

    path = _pdf_with_blank_middle_page(tmp_path / "manual.pdf")
    monkeypatch.setenv("DOCINGEST_MAX_PAGES", "2")
    from docingest.loaders import load_pdf

    with pytest.raises(ValueError, match="over DOCINGEST_MAX_PAGES"):
        load_pdf(path)
