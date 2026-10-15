import pytest


@pytest.fixture(autouse=True)
def _no_disk_caches(monkeypatch):
    """Tests must not share or leave behind the on-disk embedding and OCR caches;
    tests that exercise a cache point it at tmp_path explicitly."""
    monkeypatch.setenv("LLMGW_EMBED_CACHE", "0")
    monkeypatch.setenv("DOCINGEST_OCR_CACHE", "0")
