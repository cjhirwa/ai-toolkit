# mcp-servers (`toolkit_mcp`)

Three MCP servers that make the toolkit available to Claude Code, Claude Desktop, `agent-loop`, or any other MCP client.

| server | tools | config |
|---|---|---|
| `mcp-retrieval` | `search(query, k, source, mode)`, `get_passage(id)`, `list_sources()` | `RETRIEVAL_INDEX_DIR`, `RETRIEVAL_MODE` and the provider key used to embed queries |
| `mcp-evals` | `list_runs`, `run_summary`, `compare_runs`, `failing_examples` | `EVALH_DB` |
| `mcp-prompts` | `list_prompts`, `get_prompt`, `render_prompt`, `diff_prompt` | `PROMPTS_DIR` |

## Configuration
Each server loads the toolkit's `.env` on startup, so set `RETRIEVAL_INDEX_DIR`, `RETRIEVAL_MODE`, `EVALH_DB`, `PROMPTS_DIR` and your provider key there (see `.env.example`). Relative paths are resolved from the toolkit folder when you start the server with `uv run --directory`, as below. A variable set in the client's `env` block overrides `.env`.

## Use from Claude Code
Add a `.mcp.json` to the project that should use them, replacing `/path/to/ai-toolkit` with where you cloned this repository:

```json
{
  "mcpServers": {
    "docs": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/ai-toolkit", "mcp-retrieval"]
    },
    "evals": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/ai-toolkit", "mcp-evals"]
    }
  }
}
```

Or add it from the command line:

```
claude mcp add docs -- uv run --directory /path/to/ai-toolkit mcp-retrieval
```

After that, you can ask questions like *"search the docs for the warranty period and cite pages"* or *"compare the last two eval runs and explain the regressions"*.

## HTTP
Every server can also be served over streamable HTTP instead of stdio, for remote or multi-client use:

```
mcp-retrieval --http --port 8765
```

Set `MCP_AUTH_TOKEN` (in `.env` or the environment) and every HTTP request must send `Authorization: Bearer <token>`; others get `401`. The servers refuse to start on any host other than localhost without a token, because the tools expose your documents and eval data:

```
MCP_AUTH_TOKEN=$(openssl rand -hex 32) mcp-retrieval --http --host 0.0.0.0 --port 8765
```

Over stdio (what Claude Code and Desktop use) no token is needed: only the client that launched the server can talk to it.

## Testing a server
```
npx @modelcontextprotocol/inspector uv run mcp-retrieval
```

The tool functions are ordinary Python, so `tests/` calls them directly.
