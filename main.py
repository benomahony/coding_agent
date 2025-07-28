import asyncio
from os import environ
import subprocess

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.models.gemini import GeminiModel

GEMINI_API_KEY = (
    subprocess.run(
        args=["op", "read", "op://employee/povmeksro7vsc5xhdufg7mpp4q/credential"],
        capture_output=True,
    )
    .stdout.decode("utf-8")
    .strip()
)
assert GEMINI_API_KEY, "GEMINI_API_KEY environment variable is not set"

model = GeminiModel(
    "gemini-2.5-pro-preview-05-06", provider=GoogleGLAProvider(api_key=GEMINI_API_KEY)
)

run_python = MCPServerStdio(
    "deno",
    args=[
        "run",
        "-N",
        "-R=node_modules",
        "-W=node_modules",
        "--node-modules-dir=auto",
        "jsr:@pydantic/mcp-run-python",
        "stdio",
    ],
)
internet_search = MCPServerStdio(command="uvx", args=["duckduckgo-mcp-server"])
jira = MCPServerStdio(
    command="uv",
    args=["run", "/Users/benomahony/Code/open_source/jira_cli_mcp/main.py"],
    tool_prefix="jira",
    env=dict(environ),
)
code_reasoning = MCPServerStdio(
    command="npx",
    args=["-y", "@mettamatt/code-reasoning"],
    tool_prefix="code_reasoning",
)
desktop_commander = MCPServerStdio(
    command="npx",
    args=["-y", "@wonderwhy-er/desktop-commander"],
    tool_prefix="desktop_commander",
)
# duckdb = MCPServerStdio(
#     command="uvx",
#     args=["mcp-server-motherduck", "--db-path", ":memory:"],
#     tool_prefix="duckdb",
# )
context7 = MCPServerStdio(
    command="npx", args=["-y", "@upstash/context7-mcp"], tool_prefix="context"
)
puppeteer = MCPServerStdio(
    command="npx", args=["-y", "puppeteer-mcp-server"], env={}, tool_prefix="env"
)
python_tools = MCPServerStdio(
    command="uvx", args=["python-tools-mcp"], tool_prefix="Python_Tools"
)
audio = MCPServerStdio(
    command="uv",
    args=[
        "run",
        "--directory=/Users/benomahony/Code/open_source/csm-audio-mcp",
        "src/server.py",
    ],
    env={"UV_PYTHON": "3.12"},
)
agent = Agent(
    model=model,
    mcp_servers=[
        run_python,
        audio,
        python_tools,
        internet_search,
        code_reasoning,
        jira,
        context7,
        desktop_commander,
        puppeteer,
        # duckdb,
    ],
)


@agent.tool_plain()
def commit_message():
    diff = subprocess.run(
        "git diff --staged", shell=True, capture_output=True
    ).stdout.decode("utf-8")
    prompt = f" You are an expert at following the Conventional Commit specification. Given the git diff listed below, please generate a commit message for me: {diff}"
    return agent.run_sync(user_prompt=prompt).output


async def main():
    try:
        async with agent.run_mcp_servers():
            await agent.to_cli()

    except (RuntimeError, asyncio.CancelledError) as e:
        if "cancel scope" in str(e) or "already running" in str(e):
            pass
        else:
            raise


if __name__ == "__main__":
    asyncio.run(main())
