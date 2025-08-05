# src/services/agent.py

import os
from agents import Agent, Runner, function_tool, ModelSettings
from src.config import settings
from src.services.tools import get_latest_metrics, send_alert

# Ensure the SDK picks up your key
os.environ["OPENAI_API_KEY"] = settings.openai_api_key

# Wrap stubs as callable tools
@function_tool
def metrics_tool(subject_id: str, window_mins: int = 60) -> dict:
    return get_latest_metrics(subject_id, window_mins)

@function_tool
def alert_tool(subject_id: str, message: str, severity: str = "info") -> dict:
    return send_alert(subject_id, message, severity)

# Define your Agent with O3 (reasoning) and temp=1.0
t1d_agent = Agent(
    name="T1D Assistant",
    instructions=(
        "You are a kind, pediatric diabetes assistant. "
        "Use the metrics and alert tools for proactive feedback."
    ),
    model="o3",
    model_settings=ModelSettings(temperature=1.0),
    tools=[metrics_tool, alert_tool],
)

async def run_agent_async(subject_id: str, user_text: str) -> dict:
    """
    Async entrypoint for the agent. Uses Runner.run() under the hood,
    which works inside an existing event loop (unlike run_sync).  [oai_citation:0‡OpenAI GitHub](https://openai.github.io/openai-agents-python/ref/run/?utm_source=chatgpt.com)
    """
    result = await Runner.run(
        t1d_agent,
        user_text,
        context={"subject_id": subject_id},
    )

    # For now, ignore function-calls. You can inspect result.new_items later.
    tool_used = None
    tool_result = None

    return {
        "reply": result.final_output,
        "tool_used": None,
        "tool_result": None,
    }