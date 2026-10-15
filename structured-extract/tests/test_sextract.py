import json

from pydantic import BaseModel

from llmgw import LLM, FakeProvider
from sextract import ExtractionResult, extract, extract_files, extract_long, merge_partials, write_csv


class LineItem(BaseModel):
    name: str
    quantity: str


class Invoice(BaseModel):
    vendor: str
    items: list[LineItem]


def fake(*responses):
    return LLM("fake:x", provider=FakeProvider(list(responses)), log_dir=False)


def test_extract_single():
    llm = fake(json.dumps({"vendor": "Acme", "items": [{"name": "Thermostat", "quantity": "1"}]}))
    inv = extract("Acme ... Thermostat x1", Invoice, llm=llm)
    assert inv.items[0].name == "Thermostat"


def test_merge_partials_lists_and_conflicts():
    merged, conflicts = merge_partials(Invoice, [
        {"vendor": "Acme", "items": [{"name": "Thermostat", "quantity": "1"}]},
        {"vendor": None, "items": [{"name": "Thermostat", "quantity": "1"}, {"name": "Wall plate", "quantity": "2"}]},
        {"vendor": "Acme Ltd", "items": []},
    ])
    assert [i["name"] for i in merged["items"]] == ["Thermostat", "Wall plate"]
    assert conflicts == {"vendor": ["Acme", "Acme Ltd"]}


def test_extract_long_chunks_and_resolves():
    text = ("Thermostat x1 at 129.00. " * 30) + "\n\n" + ("Wall plate x2 at 9.50. " * 30)
    llm = fake(
        json.dumps({"vendor": "Acme", "items": [{"name": "Thermostat", "quantity": "1"}]}),
        json.dumps({"vendor": None, "items": [{"name": "Wall plate", "quantity": "2"}]}),
    )
    inv = extract_long(text, Invoice, llm=llm, chunk_size=1000, overlap=0)
    assert inv.vendor == "Acme" and [i.name for i in inv.items] == ["Thermostat", "Wall plate"]


def test_extract_files_and_csv(tmp_path):
    (tmp_path / "a.md").write_text("# Invoice\nThermostat x1", encoding="utf-8")
    llm = fake(json.dumps({"vendor": "Acme", "items": [{"name": "Thermostat", "quantity": "1"}]}))
    results = list(extract_files([tmp_path], Invoice, llm=llm))
    assert results[0].data["vendor"] == "Acme"
    write_csv(results + [ExtractionResult("bad", None, "err")], tmp_path / "o.csv")
    content = (tmp_path / "o.csv").read_text(encoding="utf-8-sig")
    assert content.splitlines()[0] == "source,vendor,items" and "Thermostat" in content
