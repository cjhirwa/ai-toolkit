# playground

A Streamlit app that runs the same input through up to four model and prompt configurations side by side, with latency, token and cost figures for each.

```
uv run playground
```

- **Compare**: one input, N columns. Each column has its own model, temperature, max tokens, and either a free-text system prompt or a version from `prompt-registry`.
- **Dataset**: upload a JSONL or CSV file with an `input` column. Every row runs through every column, and the results appear in one table you can download as CSV. This is a quick way to look at outputs before writing proper `evalh` metrics.
- **History**: the last 50 comparisons, saved in `.playground/history.jsonl`.

## Configuration
| env var | default |
|---|---|
| `PLAYGROUND_MODELS` | `gemini:gemini-3.6-flash,openai:gpt-4o-mini,ollama:llama3.2` |
| `PROMPTS_DIR` | `prompts` (the prompt picker appears only when this folder has prompts) |
| `PLAYGROUND_HISTORY` | `.playground/history.jsonl` |

Registry prompts receive the input as both `question` and `input`, plus any variables from the sidebar JSON box.
