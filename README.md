# agent-vision-mcp

<!-- mcp-name: io.github.idealizing/agent-vision-mcp -->

`agent-vision-mcp` exposes image analysis, inspection, cropping, OCR, and comparison
tools through the Model Context Protocol.

## Quickstart

Run the published package without installing it permanently:

```bash
uvx agent-vision-mcp
```

Example MCP client configuration:

```json
{
  "mcpServers": {
    "agent-vision": {
      "command": "uvx",
      "args": ["agent-vision-mcp"],
      "env": {
        "VISION_API_KEY": "your-api-key",
        "VISION_BASE_URL": "https://your-provider.example/v1",
        "VISION_MODEL_ID": "your-vision-model"
      }
    }
  }
}
```

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
.venv/bin/agent-vision-mcp
```

Configure an OpenAI-compatible multimodal endpoint with `VISION_API_KEY`,
`VISION_BASE_URL`, and `VISION_MODEL_ID`.

## URL Handling

`VISION_URL_MODE` controls how remote images are sent to the model:

- `auto` passes URLs through for analysis and comparison, but downloads them
  when inspection, cropping, or OCR requires image bytes.
- `passthrough` prefers URL passthrough, except for tools that require bytes.
- `download` always downloads and verifies remote images before model calls.

Downloads are streamed with byte limits, redirects are security checked, and
all downloaded or encoded inputs are verified as supported images.
URL passthrough relies on the configured model provider to fetch URLs safely;
use `download` when the provider is not trusted to enforce outbound-network
restrictions.

Dedicated OCR is disabled by default. Set `OCR_ENABLED=true` and configure the
`OCR_*` variables to use a separate OCR model; otherwise OCR uses the VLM.

## Run Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## License

MIT
