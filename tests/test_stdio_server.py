"""End-to-end tests for the stdio MCP entry point."""

import os
import sys
import unittest
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).parent.parent


class StdioServerTest(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_server_initializes_and_lists_tools(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(PROJECT_ROOT),
                "VISION_API_KEY": "test",
                "VISION_BASE_URL": "https://example.com/v1",
                "VISION_MODEL_ID": "test",
                "VISION_TRANSPORT": "stdio",
            }
        )
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "agent_vision_mcp.server"],
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                result = await session.initialize()
                tools = await session.list_tools()

        self.assertEqual(result.serverInfo.name, "agent-vision-mcp")
        self.assertEqual(
            [tool.name for tool in tools.tools],
            [
                "vision_analyze",
                "vision_inspect",
                "vision_crop_analyze",
                "vision_extract_text",
                "vision_compare",
                "vision_capabilities",
            ],
        )
