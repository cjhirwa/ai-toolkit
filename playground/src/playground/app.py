"""Side-by-side LLM playground. Run with:  uv run playground   (or: streamlit run app.py)"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from llmgw import LLM

HISTORY = Path(os.getenv("PLAYGROUND_HISTORY", ".playground/history.jsonl"))
DEFAULT_MODELS = os.getenv(
    "PLAYGROUND_MODELS", "gemini:gemini-3.6-flash,openai:gpt-4o-mini,ollama:llama3.2"
).split(",")

st.set_page_config(page_title="LLM Playground", layout="wide")


def registry():
    try:
        from promptreg import Registry

        reg = Registry(os.getenv("PROMPTS_DIR", "prompts"))
        return reg if reg.names() else None
    except Exception:  # noqa: BLE001
        return None


def column_config(i: int, reg) -> dict:
    with st.sidebar.expander(f"Column {i + 1}", expanded=i < 2):
        model = st.selectbox("Model", DEFAULT_MODELS + ["custom…"], index=min(i, len(DEFAULT_MODELS) - 1), key=f"m{i}")
        if model == "custom…":
            model = st.text_input("provider:model", key=f"mc{i}")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.2, 0.05, key=f"t{i}")
        max_tokens = st.number_input("Max output tokens", 0, 32000, 0, 256, key=f"mt{i}", help="0 = provider default")
        prompt_ref, system = None, ""
        if reg:
            options = ["(free text)"] + [f"{n}@v{v}" for n in reg.names() for v in reg.versions(n)]
            prompt_ref = st.selectbox("Prompt", options, key=f"p{i}")
            prompt_ref = None if prompt_ref == "(free text)" else prompt_ref
        if not prompt_ref:
            system = st.text_area("System prompt", key=f"s{i}", height=100)
    return {"model": model, "temperature": temperature, "max_tokens": int(max_tokens) or None,
            "system": system or None, "prompt": prompt_ref}


def build_messages(cfg: dict, user_input: str, variables: dict, reg) -> tuple[str, str | None]:
    if cfg["prompt"] and reg:
        r = reg.get(cfg["prompt"]).render(**{"question": user_input, "input": user_input, **variables})
        return r.user, r.system
    return user_input, cfg["system"]


def run_one(cfg: dict, user_input: str, variables: dict, reg) -> dict:
    t0 = time.perf_counter()
    try:
        llm = LLM(cfg["model"], temperature=cfg["temperature"], max_tokens=cfg["max_tokens"])
        prompt, system = build_messages(cfg, user_input, variables, reg)
        resp = llm.complete(prompt, system=system)
        return {"text": resp.text, "latency_ms": resp.latency_ms, "in": resp.usage.input_tokens,
                "out": resp.usage.output_tokens, "cost": resp.usage.cost_usd}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "latency_ms": (time.perf_counter() - t0) * 1000}


def save_history(record: dict) -> None:
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


reg = registry()
st.sidebar.title("Playground")
n_cols = st.sidebar.slider("Columns", 1, 4, 2)
configs = [column_config(i, reg) for i in range(n_cols)]
vars_text = st.sidebar.text_area("Template variables (JSON)", "{}", help="Extra variables for registry prompts")
try:
    variables = json.loads(vars_text or "{}")
except json.JSONDecodeError:
    st.sidebar.error("Variables must be JSON")
    variables = {}

tab_single, tab_dataset, tab_history = st.tabs(["Compare", "Dataset", "History"])

with tab_single:
    user_input = st.text_area("Input", height=140, placeholder="Ask something…")
    if st.button("Run", type="primary", disabled=not user_input.strip()):
        with st.spinner("Running…"), ThreadPoolExecutor(max_workers=n_cols) as pool:
            results = list(pool.map(lambda c: run_one(c, user_input, variables, reg), configs))
        cols = st.columns(n_cols)
        for col, cfg, res in zip(cols, configs, results):
            with col:
                st.markdown(f"**{cfg['model']}**" + (f" · `{cfg['prompt']}`" if cfg["prompt"] else ""))
                if "error" in res:
                    st.error(res["error"])
                else:
                    cost = f" · ${res['cost']:.5f}" if res["cost"] is not None else ""
                    st.caption(f"{res['latency_ms']:.0f} ms · {res['in']}→{res['out']} tok{cost}")
                    st.markdown(res["text"])
        save_history({"ts": datetime.now().isoformat(timespec="seconds"), "input": user_input,
                      "configs": configs, "results": results})

with tab_dataset:
    st.caption("Upload JSONL or CSV with an `input` column (and optionally `expected`). Each row runs through every column.")
    up = st.file_uploader("Dataset", type=["jsonl", "csv"])
    limit = st.number_input("Rows", 1, 200, 10)
    if up and st.button("Run dataset"):
        if up.name.endswith(".csv"):
            df = pd.read_csv(up)
        else:
            df = pd.DataFrame([json.loads(l) for l in up.getvalue().decode("utf-8").splitlines() if l.strip()])
        df = df.head(int(limit))
        out = df[[c for c in ("id", "input", "expected") if c in df.columns]].copy()
        progress = st.progress(0.0)
        for j, cfg in enumerate(configs):
            label = cfg["model"] + (f" [{cfg['prompt']}]" if cfg["prompt"] else "")
            with ThreadPoolExecutor(max_workers=6) as pool:
                res = list(pool.map(lambda x, cfg=cfg: run_one(cfg, str(x), variables, reg), df["input"]))
            out[label] = [r.get("text", r.get("error")) for r in res]
            progress.progress((j + 1) / len(configs))
        st.dataframe(out, use_container_width=True, height=600)
        st.download_button("Download CSV", out.to_csv(index=False).encode("utf-8"), "playground_results.csv")

with tab_history:
    if HISTORY.exists():
        rows = [json.loads(l) for l in HISTORY.read_text(encoding="utf-8").splitlines()[-50:]][::-1]
        for r in rows:
            with st.expander(f"{r['ts']} · {r['input'][:80]}"):
                for cfg, res in zip(r["configs"], r["results"]):
                    st.markdown(f"**{cfg['model']}** — {res.get('latency_ms', 0):.0f} ms")
                    st.text(res.get("text") or res.get("error"))
    else:
        st.info("No runs yet.")
