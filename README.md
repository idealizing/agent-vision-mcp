# agent-vision-mcp

<!-- mcp-name: io.github.idealizing/agent-vision-mcp -->

[English](README.md) | [中文](README_CN.md)

Give MCP-compatible AI agents image analysis, metadata inspection, cropping,
OCR, and image comparison through any OpenAI-compatible vision model.

## Features

- Analyze screenshots, charts, documents, UI, objects, and general images.
- Inspect image dimensions and metadata without calling a model.
- Crop and zoom into regions using normalized coordinates.
- Extract visible text with a VLM or an optional dedicated OCR model.
- Compare two to four images.
- Accept public URLs, local files, data URLs, and Base64 images.
- Run locally over the standard MCP stdio transport.

## Claude Code

### Requirements

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- An OpenAI-compatible vision API endpoint and API key

`uvx` downloads the published package from PyPI into an isolated environment
and runs it. It does not use the source code in your current directory and
does not permanently install the package into your system Python.

### Add To Claude Code

The command below configures Claude Code to start `agent-vision-mcp` from PyPI:

```bash
claude mcp add --scope user agent-vision \
  --env UV_DEFAULT_INDEX=https://pypi.org/simple \
  VISION_API_KEY="your-api-key" \
  VISION_BASE_URL="https://your-provider.example/v1" \
  VISION_MODEL_ID="your-vision-model" \
  -- uvx agent-vision-mcp
```

Use `UV_DEFAULT_INDEX=https://pypi.org/simple` when your local PyPI mirror has
not synchronized the latest release.

Verify the connection:

```bash
claude mcp get agent-vision
claude mcp list
```

Then start Claude Code and ask:

```text
Use vision_capabilities to show the available vision tools.
```

Analyze a local image:

```text
Use vision_inspect on /data/example.png, then use vision_analyze to describe it.
```

By default, local image access is limited to `/data` and `/tmp`. Add another
directory with:

```bash
claude mcp remove --scope user agent-vision

claude mcp add --scope user agent-vision \
  --env UV_DEFAULT_INDEX=https://pypi.org/simple \
  VISION_API_KEY="your-api-key" \
  VISION_BASE_URL="https://your-provider.example/v1" \
  VISION_MODEL_ID="your-vision-model" \
  VISION_ALLOWED_PATHS="/data,/tmp,/home/your-user/Pictures" \
  -- uvx agent-vision-mcp
```

### Dedicated OCR Model

Without dedicated OCR configuration, `vision_extract_text` uses the configured
vision model. To use a separate OCR model:

```bash
claude mcp add --scope user agent-vision \
  --env UV_DEFAULT_INDEX=https://pypi.org/simple \
  VISION_API_KEY="your-vision-api-key" \
  VISION_BASE_URL="https://your-provider.example/v1" \
  VISION_MODEL_ID="your-vision-model" \
  OCR_ENABLED=true \
  OCR_API_KEY="your-ocr-api-key" \
  OCR_BASE_URL="https://your-provider.example/v1" \
  OCR_MODEL_ID="your-ocr-model" \
  -- uvx agent-vision-mcp
```

Never commit real API keys to Git.

## Other MCP Clients

Use this stdio configuration with MCP clients that accept JSON configuration:

```json
{
  "mcpServers": {
    "agent-vision": {
      "command": "uvx",
      "args": ["agent-vision-mcp"],
      "env": {
        "UV_DEFAULT_INDEX": "https://pypi.org/simple",
        "VISION_API_KEY": "your-api-key",
        "VISION_BASE_URL": "https://your-provider.example/v1",
        "VISION_MODEL_ID": "your-vision-model"
      }
    }
  }
}
```

## Tools

| Tool | Purpose |
| --- | --- |
| `vision_analyze` | Analyze an image with task-specific prompts |
| `vision_inspect` | Read image dimensions, format, size, and mode |
| `vision_crop_analyze` | Crop and analyze a normalized image region |
| `vision_extract_text` | Extract visible text using OCR or the VLM |
| `vision_compare` | Compare two to four images |
| `vision_capabilities` | Show server configuration and limits |

## Response format

Every tool returns a JSON string. Clients must `json.loads` the result
before reading any field. All top-level keys are always present (even when
empty), so consumers can iterate the envelope without `dict.get(...)`
guards.

### Success envelope

```json
{
  "schema_version": "1.0",
  "ok": true,
  "tool": "vision_analyze",
  "task": "general",
  "model": "...",
  "source": null,
  "sources": [],
  "result": {},
  "warnings": [],
  "raw_model_output": null,
  "error": null
}
```

| Field | Type | When set |
| --- | --- | --- |
| `schema_version` | `string` | Always. Currently `"1.0"`. |
| `ok` | `bool` | Always. `true` on success, `false` on failure. |
| `tool` | `string` | Always. The tool name (e.g. `vision_analyze`). |
| `task` | `string \| null` | The `task` argument when the tool takes one; `null` for `vision_capabilities` and `vision_extract_text`. |
| `model` | `string \| null` | The configured model identifier (e.g. `glm-4v-flash`). Set even on failure when the tool knew it. |
| `source` | `SourceMeta \| null` | Single-image tools. `null` for `vision_compare` and `vision_capabilities`. |
| `sources` | `SourceMeta[]` | `vision_compare` only: one entry per input image. Empty for all other tools. |
| `result` | `object` | Tool-specific (see below). `null` on failure. |
| `warnings` | `string[]` | Always a list (empty on success). Soft-failure notes (e.g. `vision_extract_text` falling back from OCR to VLM). |
| `raw_model_output` | `object \| null` | Sanitized provider response when `include_raw=true`; `null` otherwise. |
| `error` | `ErrorPayload \| null` | `null` on success. Populated on failure. |

`SourceMeta` fields: `type` (`url` / `file` / `data_url` / `base64`),
`mime_type`, `width`, `height`, `size_bytes`, `source_ref` (only when
`include_source_ref=true`; redacted to `host/path` for URLs or `basename`
for files; `null` for data URLs and base64).

### Failure envelope

```json
{
  "schema_version": "1.0",
  "ok": false,
  "tool": "vision_analyze",
  "task": "general",
  "model": "...",
  "source": null,
  "sources": [],
  "result": null,
  "warnings": [],
  "raw_model_output": null,
  "error": {
    "code": "INVALID_INPUT",
    "message": "Input is not a valid supported image",
    "retryable": false,
    "details": {}
  }
}
```

`error.code` values: `INVALID_INPUT`, `IMAGE_TOO_LARGE`, `UNSUPPORTED_FORMAT`,
`SECURITY_ERROR`, `PROVIDER_ERROR`, `TIMEOUT`, `INTERNAL_ERROR`.
`retryable=true` means the caller may try the same call again.

### Per-tool `result` shape

| Tool | `result` keys |
| --- | --- |
| `vision_analyze` | `summary`, `observations[]`, `inferences[]`, `uncertainties[]`, `suggested_followups[]` |
| `vision_extract_text` | `text`, `blocks[]`, `layout_preserved`, `unclear_segments[]` |
| `vision_compare` | `summary`, `differences[]`, `same_elements[]` |
| `vision_crop_analyze` | `crop: {x, y, width, height}`, `summary`, `observations[]` |
| `vision_inspect` | `width`, `height`, `format`, `mime_type`, `mode`, `size_bytes`, `has_transparency`, `source_type` |
| `vision_capabilities` | `server`, `version`, `vlm_provider`, `ocr_provider`, `ocr_enabled`, `tools`, `supports`, `limits`, `task_types` |

Arrays that are not yet parsed from model output are returned as empty
arrays (no fabricated structure). `observations`, `inferences`, and
`differences` are empty in the current release; only `summary` carries
the model's free-form text.

### Multi-image input

`vision_compare` accepts 2–4 images. The envelope reports them in
`sources: [SourceMeta, ...]` (one entry per input, in input order).
`source` is `null` for multi-image tools. All other image tools accept a
single image and use `source`; `sources` is `[]`.

### Opt-in flags

- `include_raw: bool = False` — when `true`, `raw_model_output` contains a
  sanitized subset of the provider response:
  `{model, response_metadata: {model_name, finish_reason, system_fingerprint},
   usage_metadata: {input_tokens, output_tokens, total_tokens}}`. HTTP
  headers, request IDs, signed URLs, and raw exception text are dropped
  before reaching the envelope. Off by default to keep responses small
  and to avoid leaking auth material.
- `include_source_ref: bool = False` — when `true`, `source.source_ref`
  is populated with a redacted reference: `host/path` for URLs (query
  string stripped, including signed tokens) or `basename` for local
  files. `data_url` and base64 inputs always return `null` for
  `source_ref`. Off by default to avoid leaking paths and signed URLs.

## URL Handling

`VISION_URL_MODE` controls remote-image handling:

- `auto` passes URLs through for analysis and comparison, but downloads them
  when inspection, cropping, or OCR requires image bytes.
- `passthrough` prefers URL passthrough, except for tools that require bytes.
- `download` always downloads and verifies remote images before model calls.

Downloads are streamed with byte limits, redirects are security checked, and
downloaded or encoded inputs are verified as supported images.

## Troubleshooting

If Claude Code cannot find the PyPI package:

```bash
UV_DEFAULT_INDEX=https://pypi.org/simple uvx --refresh agent-vision-mcp
```

If the MCP server does not connect:

```bash
claude mcp get agent-vision
uvx agent-vision-mcp
```

If you change the Claude Code configuration:

```bash
claude mcp remove --scope user agent-vision
```

Then add it again with the updated values.

## Development

```bash
git clone https://github.com/idealizing/agent-vision-mcp.git
cd agent-vision-mcp
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
.venv/bin/python -m unittest discover -s tests -v
```

## License

MIT
