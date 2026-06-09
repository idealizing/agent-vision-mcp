# agent-vision-mcp

<!-- mcp-name: io.github.idealizing/agent-vision-mcp -->

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
