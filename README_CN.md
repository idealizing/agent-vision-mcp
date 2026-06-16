# agent-vision-mcp

<!-- mcp-name: io.github.idealizing/agent-vision-mcp -->

[English](README.md) | [中文](README_CN.md)

为兼容 MCP 协议的 AI Agent 提供图像分析、元数据查看、裁剪、OCR
和多图对比能力，可对接任何兼容 OpenAI 接口的视觉模型。

## 功能

- 支持截屏、图表、文档、UI、物体识别与通用图像分析。
- 无需调用模型即可查看图片尺寸与元数据。
- 使用归一化坐标裁剪并放大局部区域。
- 通过视觉模型或可选的专用 OCR 模型提取可见文字。
- 支持 2–4 张图片的对比。
- 接受公网 URL、本地文件、data URL 与 Base64 图片。
- 通过标准 MCP stdio 传输在本地运行。

## Claude Code

### 环境要求

- Python 3.10 或更新版本
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- 一个兼容 OpenAI 的视觉模型 API 端点与密钥

`uvx` 会从 PyPI 拉取已发布的包到独立环境中运行。它**不会**使用你
当前目录下的源码，也不会把包永久安装到系统 Python 中。

### 添加到 Claude Code

下面的命令会配置 Claude Code 从 PyPI 启动 `agent-vision-mcp`：

```bash
claude mcp add --scope user agent-vision \
  --env UV_DEFAULT_INDEX=https://pypi.org/simple \
  VISION_API_KEY="your-api-key" \
  VISION_BASE_URL="https://your-provider.example/v1" \
  VISION_MODEL_ID="your-vision-model" \
  -- uvx agent-vision-mcp
```

当你的本地 PyPI 镜像还未同步最新版本时，使用
`UV_DEFAULT_INDEX=https://pypi.org/simple`。

验证连接：

```bash
claude mcp get agent-vision
claude mcp list
```

然后启动 Claude Code 并输入：

```text
使用 vision_capabilities 查看可用的视觉工具。
```

分析本地图片：

```text
对 /data/example.png 使用 vision_inspect，再用 vision_analyze 描述它。
```

默认情况下，本地图片访问仅限于 `/data` 与 `/tmp`。要添加其他目录：

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

### 专用 OCR 模型

未配置专用 OCR 时，`vision_extract_text` 会使用已配置的视觉模型。
要使用单独的 OCR 模型：

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

永远不要把真实的 API 密钥提交到 Git。

## 其他 MCP 客户端

对接受 JSON 配置的 MCP 客户端使用下面的 stdio 配置：

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

## 工具列表

| 工具 | 用途 |
| --- | --- |
| `vision_analyze` | 使用任务相关的提示词分析图片 |
| `vision_inspect` | 查看图片尺寸、格式、大小与模式 |
| `vision_crop_analyze` | 裁剪并分析归一化坐标指定的区域 |
| `vision_extract_text` | 使用 OCR 或视觉模型提取可见文字 |
| `vision_compare` | 对比 2 到 4 张图片 |
| `vision_capabilities` | 显示服务端配置与限制 |

## 响应格式

每个工具都返回一段 JSON 字符串。客户端必须先 `json.loads` 解析，
再读取其中的字段。所有顶层字段始终存在（即使为空），因此消费方
无需 `dict.get(...)` 防御即可遍历 envelope。

### 成功 envelope

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

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `schema_version` | `string` | 始终存在，当前为 `"1.0"`。 |
| `ok` | `bool` | 始终存在。成功为 `true`，失败为 `false`。 |
| `tool` | `string` | 始终存在。工具名（如 `vision_analyze`）。 |
| `task` | `string \| null` | 接受 `task` 参数时为该参数；`vision_capabilities` 与 `vision_extract_text` 为 `null`。 |
| `model` | `string \| null` | 已配置的模型标识（如 `glm-4v-flash`）。即使失败，工具已知时也会填上。 |
| `source` | `SourceMeta \| null` | 单图工具使用。`vision_compare` 与 `vision_capabilities` 为 `null`。 |
| `sources` | `SourceMeta[]` | 仅 `vision_compare` 使用：每张输入图一项。其他工具为空数组。 |
| `result` | `object` | 各工具自有 schema（见下）。失败时为 `null`。 |
| `warnings` | `string[]` | 始终为数组（成功时为空）。软失败提示（例如 `vision_extract_text` 从 OCR 回退到视觉模型）。 |
| `raw_model_output` | `object \| null` | 当 `include_raw=true` 时为脱敏后的 provider 响应；否则为 `null`。 |
| `error` | `ErrorPayload \| null` | 成功时为 `null`，失败时填充。 |

`SourceMeta` 字段：`type`（`url` / `file` / `data_url` / `base64`）、
`mime_type`、`width`、`height`、`size_bytes`、`source_ref`（仅当
`include_source_ref=true` 时填充；URL 脱敏为 `host/path`，文件脱敏为
`basename`；data URL 与 base64 始终为 `null`）。

### 失败 envelope

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

`error.code` 取值：`INVALID_INPUT`、`IMAGE_TOO_LARGE`、
`UNSUPPORTED_FORMAT`、`SECURITY_ERROR`、`PROVIDER_ERROR`、`TIMEOUT`、
`INTERNAL_ERROR`。`retryable=true` 表示调用方可以重试同一次请求。

### 各工具的 `result` 结构

| 工具 | `result` 字段 |
| --- | --- |
| `vision_analyze` | `summary`、`observations[]`、`inferences[]`、`uncertainties[]`、`suggested_followups[]` |
| `vision_extract_text` | `text`、`blocks[]`、`layout_preserved`、`unclear_segments[]` |
| `vision_compare` | `summary`、`differences[]`、`same_elements[]` |
| `vision_crop_analyze` | `crop: {x, y, width, height}`、`summary`、`observations[]` |
| `vision_inspect` | `width`、`height`、`format`、`mime_type`、`mode`、`size_bytes`、`has_transparency`、`source_type` |
| `vision_capabilities` | `server`、`version`、`vlm_provider`、`ocr_provider`、`ocr_enabled`、`tools`、`supports`、`limits`、`task_types` |

尚未从模型输出解析的数组会以空数组返回（不伪造结构）。当前版本中
`observations`、`inferences`、`differences` 都为空，仅 `summary` 携带
模型的自由文本。

### 多图输入

`vision_compare` 接受 2–4 张图片。envelope 在 `sources: [SourceMeta, ...]`
中按输入顺序报告每张图片。多图工具的 `source` 为 `null`。
其他图像工具只接受单张图片，使用 `source`，`sources` 为 `[]`。

### 可选开关

- `include_raw: bool = False` —— 为 `true` 时，`raw_model_output` 包
  含脱敏后的 provider 响应子集：
  `{model, response_metadata: {model_name, finish_reason,
  system_fingerprint}, usage_metadata: {input_tokens, output_tokens,
  total_tokens}}`。HTTP 头、请求 ID、签名 URL、原始异常文本在进入
  envelope 之前就已经被剔除。默认关闭，避免响应过大并防止认证信息
  泄露。
- `include_source_ref: bool = False` —— 为 `true` 时，`source.source_ref`
  会被填充为脱敏引用：URL 为 `host/path`（剥去 query string，包括签名
  token），本地文件为 `basename`。data URL 与 base64 输入的
  `source_ref` 始终为 `null`。默认关闭，避免泄露路径与签名 URL。

## URL 处理

`VISION_URL_MODE` 控制远端图片的处理方式：

- `auto` —— 分析与对比场景透传 URL，但当 inspect、裁剪或 OCR 需要
  图像字节时会下载。
- `passthrough` —— 优先透传 URL，但需要字节的工具除外。
- `download` —— 总是先下载并校验远端图片再调用模型。

下载以流式进行并设了字节上限，重定向会经过安全检查，下载或编码
的输入会被校验为受支持的图像。

## 故障排查

如果 Claude Code 找不到 PyPI 包：

```bash
UV_DEFAULT_INDEX=https://pypi.org/simple uvx --refresh agent-vision-mcp
```

如果 MCP 服务连不上：

```bash
claude mcp get agent-vision
uvx agent-vision-mcp
```

如果修改了 Claude Code 的配置：

```bash
claude mcp remove --scope user agent-vision
```

然后用新值重新添加。

## 开发

```bash
git clone https://github.com/idealizing/agent-vision-mcp.git
cd agent-vision-mcp
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
.venv/bin/python -m unittest discover -s tests -v
```

## 许可证

MIT
