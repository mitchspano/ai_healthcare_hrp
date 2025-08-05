# server/services/agent.py

import os
from agents import Agent, Runner, function_tool, ModelSettings
from server.config import settings
from server.services.tools import (
    get_latest_metrics,
    send_alert,
    get_model_info,
    predict_glucose_levels,
)
from server.services.conversation_service import conversation_service

# Ensure the SDK picks up your key
os.environ["OPENAI_API_KEY"] = settings.openai_api_key


# Wrap stubs as callable tools
@function_tool
def metrics_tool(subject_id: str, window_mins: int = 60) -> dict:
    return get_latest_metrics(subject_id, window_mins)


@function_tool
def alert_tool(subject_id: str, message: str, severity: str = "info") -> dict:
    return send_alert(subject_id, message, severity)


@function_tool
def model_info_tool() -> dict:
    """Get information about the loaded diabetes LSTM model."""
    return get_model_info()


@function_tool
def glucose_prediction_tool(subject_id: str, historical_data: str = None) -> dict:
    """Predict glucose levels using the diabetes LSTM model."""
    return predict_glucose_levels(subject_id, historical_data)


# Define your Agent with GPT-4o and temp=1.0
t1d_agent = Agent(
    name="T1D Assistant",
    instructions=(
        "You are a kind, pediatric diabetes assistant with access to a trained LSTM model for glucose prediction. "
        "Use the metrics, alert, model info, and glucose prediction tools for proactive feedback. "
        "When asked about glucose predictions, use the glucose_prediction_tool. "
        "When asked about the AI model, use the model_info_tool. "
        "You have access to conversation history to provide context-aware responses. "
        "Use this context to remember previous questions and provide more personalized assistance."
    ),
    model="gpt-4o",
    model_settings=ModelSettings(temperature=1.0),
    tools=[metrics_tool, alert_tool, model_info_tool, glucose_prediction_tool],
)


async def run_agent_async(
    subject_id: str, user_text: str, conversation_id: str = None
) -> dict:
    """
    Async entrypoint for the agent. Uses Runner.run() under the hood,
    which works inside an existing event loop (unlike run_sync).
    """
    # Get or create conversation
    conversation = conversation_service.get_or_create_conversation(
        subject_id, conversation_id
    )

    # Add user message to conversation
    conversation_service.add_message(conversation.id, "user", user_text)

    # Get conversation context
    context = conversation_service.get_conversation_context(
        conversation.id, max_messages=10
    )

    # Prepare the full message with context
    if context:
        full_message = f"{context}\n\nCurrent message: {user_text}"
    else:
        full_message = user_text

    result = await Runner.run(
        t1d_agent,
        full_message,
        context={"subject_id": subject_id, "conversation_id": conversation.id},
    )

    # Add assistant response to conversation
    conversation_service.add_message(conversation.id, "assistant", result.final_output)

    # For now, ignore function-calls. You can inspect result.new_items later.
    tool_used = None
    tool_result = None

    return {
        "reply": result.final_output,
        "tool_used": None,
        "tool_result": None,
        "conversation_id": conversation.id,
    }
