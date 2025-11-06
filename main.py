import asyncio
from os import environ
import subprocess
import logfire
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.openai import OpenAIModel

from pydantic_ai.providers.openai import OpenAIProvider


class Settings(BaseSettings):
    gemini_api_key: str = Field(default="")

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def get_api_key(cls, v: str) -> str:
        if v:
            return v

        result = subprocess.run(
            args=["op", "read", "op://employee/povmeksro7vsc5xhdufg7mpp4q/credential"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

        raise ValueError("GEMINI_API_KEY not found in environment or 1Password")


settings = Settings()
ollama_model = OpenAIModel(
    model_name="qwen3:32b",
    provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
)
gemini_model = GeminiModel(
    "gemini-2.5-pro-preview-05-06",
    provider=GoogleGLAProvider(api_key=settings.gemini_api_key),
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
context7 = MCPServerStdio(
    command="npx", args=["-y", "@upstash/context7-mcp"], tool_prefix="context"
)

agent = Agent(
    model=gemini_model,
    toolsets=[
        run_python,
        internet_search,
        code_reasoning,
        jira,
        context7,
        desktop_commander,
    ],
)


@agent.tool_plain()
def run_tests() -> str:
    """Run tests using uv."""
    result = subprocess.run(
        ["uv", "run", "pytest", "-xvs", "tests/"], capture_output=True, text=True
    )
    return result.stdout


@agent.tool_plain()
def commit_message():
    diff = subprocess.run(
        "git diff --staged", shell=True, capture_output=True
    ).stdout.decode("utf-8")
    prompt = f" You are an expert at following the Conventional Commit specification. Given the git diff listed below, please generate a commit message for me: {diff}"
    return agent.run_sync(user_prompt=prompt).output


async def main():
    async with agent:
        environ["OTEL_SERVICE_NAME"] = "coding-agent"
        environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
        _ = logfire.configure(send_to_logfire=False)
        logfire.instrument_pydantic_ai()
        logfire.instrument_httpx(capture_all=True)
        await agent.to_cli()


if __name__ == "__main__":
    asyncio.run(main())
