import logging
from typing import Dict, List, Any, AsyncGenerator, Generator, Optional, Union

from src.config import TEAM_MEMBERS
from src.graph import build_graph
from langchain_community.adapters.openai import convert_message_to_dict
from src.constants import STREAMING_LLM_AGENTS, EventType
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level is INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def enable_debug_logging():
    """Enable debug level logging for more detailed execution information."""
    logging.getLogger("src").setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)

# Create the graph
graph = build_graph()

# Cache for coordinator messages
coordinator_cache = []
MAX_CACHE_SIZE = 3


async def run_agent_workflow(
    user_input_messages: List[Dict[str, Any]],
    debug: bool = False,
    deep_thinking_mode: bool = False,
    search_before_planning: bool = False,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Run the agent workflow with the given user input.

    Args:
        user_input_messages: The user request messages
        debug: If True, enables debug level logging
        deep_thinking_mode: If True, enables deep thinking mode
        search_before_planning: If True, performs search before planning

    Returns:
        The final state after the workflow completes
    """
    if not user_input_messages:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    logger.info(f"Starting workflow with user input: {user_input_messages}")

    workflow_id = str(uuid.uuid4())

    # Reset coordinator cache at the start of each workflow
    global coordinator_cache, is_handoff_case
    coordinator_cache = []
    is_handoff_case = False
    is_workflow_triggered = False
    last_event_data = None
    async for event in graph.astream_events(
        {
            "TEAM_MEMBERS": TEAM_MEMBERS,
            "messages": user_input_messages,
            "deep_thinking_mode": deep_thinking_mode,
            "search_before_planning": search_before_planning,
        },
        version="v2",
    ):
        kind, data, name, node, langgraph_step, run_id = _extract_event_data(event)
        last_event_data = data

        # Process events and generate output data
        for ydata in _process_event(
            kind,
            data,
            name,
            node,
            workflow_id,
            langgraph_step,
            run_id,
            user_input_messages,
        ):
            if ydata:
                if ydata.get("event") == "start_of_workflow":
                    is_workflow_triggered = True
                yield ydata

    # Handle workflow completion - Fix for using yield from in async functions
    for final_event in _generate_final_events(
        workflow_id, last_event_data, is_workflow_triggered
    ):
        yield final_event


def _extract_event_data(
    event: Dict[str, Any],
) -> tuple[str, Dict[str, Any], str, str, str, str]:
    """Extract key data from events"""
    kind = event.get("event")
    data = event.get("data")
    name = event.get("name")
    metadata = event.get("metadata", {})

    node = ""
    if metadata.get("checkpoint_ns") is not None:
        node = metadata.get("checkpoint_ns").split(":")[0]

    langgraph_step = ""
    if metadata.get("langgraph_step") is not None:
        langgraph_step = str(metadata["langgraph_step"])

    run_id = ""
    if event.get("run_id") is not None:
        run_id = str(event["run_id"])

    return kind, data, name, node, langgraph_step, run_id


def _process_event(
    kind: str,
    data: Dict[str, Any],
    name: str,
    node: str,
    workflow_id: str,
    langgraph_step: str,
    run_id: str,
    user_input_messages: List[Dict[str, Any]],
) -> Generator[Dict[str, Any], None, None]:
    """Process events and return corresponding output data"""
    # Handle chain start events
    if kind == EventType.CHAIN_START.value and name in STREAMING_LLM_AGENTS:
        yield from _handle_chain_start(
            name, workflow_id, langgraph_step, user_input_messages
        )

    # Handle chain end events
    elif kind == EventType.CHAIN_END.value and name in STREAMING_LLM_AGENTS:
        yield from _handle_chain_end(name, workflow_id, langgraph_step)

    # Handle chat model start events
    elif kind == EventType.CHAT_MODEL_START.value and node in STREAMING_LLM_AGENTS:
        yield from _handle_chat_model_start(node)

    # Handle chat model end events
    elif kind == EventType.CHAT_MODEL_END.value and node in STREAMING_LLM_AGENTS:
        yield from _handle_chat_model_end(node)

    # Handle chat model stream events
    elif kind == EventType.CHAT_MODEL_STREAM.value and node in STREAMING_LLM_AGENTS:
        yield from _handle_chat_model_stream(data, node)

    # Handle tool start events
    elif kind == EventType.TOOL_START.value and node in TEAM_MEMBERS:
        yield from _handle_tool_start(node, name, data, workflow_id, run_id)

    # Handle tool end events
    elif kind == EventType.TOOL_END.value and node in TEAM_MEMBERS:
        yield from _handle_tool_end(node, name, data, workflow_id, run_id)

    return None


def _handle_chain_start(
    name: str,
    workflow_id: str,
    langgraph_step: str,
    user_input_messages: List[Dict[str, Any]],
) -> Generator[Dict[str, Any], None, None]:
    """Handle chain start events"""
    # If it's the planner, generate workflow start event
    if name == "planner":
        yield {
            "event": "start_of_workflow",
            "data": {"workflow_id": workflow_id, "input": user_input_messages},
        }

    yield {
        "event": "start_of_agent",
        "data": {
            "agent_name": name,
            "agent_id": f"{workflow_id}_{name}_{langgraph_step}",
        },
    }


def _handle_chain_end(
    name: str, workflow_id: str, langgraph_step: str
) -> Generator[Dict[str, Any], None, None]:
    """Handle chain end events"""
    yield {
        "event": "end_of_agent",
        "data": {
            "agent_name": name,
            "agent_id": f"{workflow_id}_{name}_{langgraph_step}",
        },
    }


def _handle_chat_model_start(node: str) -> Generator[Dict[str, Any], None, None]:
    """Handle chat model start events"""
    yield {
        "event": "start_of_llm",
        "data": {"agent_name": node},
    }


def _handle_chat_model_end(node: str) -> Generator[Dict[str, Any], None, None]:
    """Handle chat model end events"""
    yield {
        "event": "end_of_llm",
        "data": {"agent_name": node},
    }


def _handle_chat_model_stream(
    data: Dict[str, Any], node: str
) -> Generator[Dict[str, Any], None, None]:
    """Handle chat model stream events"""
    global coordinator_cache, is_handoff_case

    content = data["chunk"].content

    # Handle empty content
    if content is None or content == "":
        if not data["chunk"].additional_kwargs.get("reasoning_content"):
            return

        yield {
            "event": "message",
            "data": {
                "message_id": data["chunk"].id,
                "delta": {
                    "reasoning_content": (
                        data["chunk"].additional_kwargs["reasoning_content"]
                    )
                },
            },
        }
        return

    # Handle coordinator messages
    if node == "coordinator":
        yield from _handle_coordinator_message(data, content)
    else:
        # Handle messages from other agents
        yield {
            "event": "message",
            "data": {
                "message_id": data["chunk"].id,
                "delta": {"content": content},
            },
        }


def _handle_tool_start(node, name, data, workflow_id, run_id):
    """Handle tool start events"""
    yield {
        "event": "tool_call",
        "data": {
            "tool_call_id": f"{workflow_id}_{node}_{name}_{run_id}",
            "tool_name": name,
            "tool_input": data.get("input"),
        },
    }


def _handle_tool_end(node, name, data, workflow_id, run_id):
    """Handle tool end events"""
    yield {
        "event": "tool_call_result",
        "data": {
            "tool_call_id": f"{workflow_id}_{node}_{name}_{run_id}",
            "tool_name": name,
            "tool_result": data["output"].content if data.get("output") else "",
        },
    }


def _generate_final_events(
    workflow_id: str, data: Dict[str, Any], is_workflow_triggered: bool
) -> Generator[Dict[str, Any], None, None]:
    """Generate workflow end events"""
    if is_workflow_triggered:
        yield {
            "event": "end_of_workflow",
            "data": {"workflow_id": workflow_id},
        }

    yield {
        "event": "final_session_state",
        "data": {
            "messages": [
                convert_message_to_dict(msg)
                for msg in data["output"].get("messages", [])
            ],
        },
    }


def _handle_coordinator_message(
    data: Dict[str, Any], content: str
) -> Generator[Dict[str, Any], None, None]:
    """Handle coordinator messages"""
    global coordinator_cache, is_handoff_case

    # Cache not full, continue caching
    if len(coordinator_cache) < MAX_CACHE_SIZE:
        coordinator_cache.append(content)
        cached_content = "".join(coordinator_cache)

        # Check if it's a handoff case
        if cached_content.startswith("handoff"):
            is_handoff_case = True
            return

        # Cache not full, continue waiting
        if len(coordinator_cache) < MAX_CACHE_SIZE:
            return

        # Cache full, send cached message
        yield {
            "event": "message",
            "data": {
                "message_id": data["chunk"].id,
                "delta": {"content": cached_content},
            },
        }

    # Cache full and not a handoff case, send message directly
    elif not is_handoff_case:
        yield {
            "event": "message",
            "data": {
                "message_id": data["chunk"].id,
                "delta": {"content": content},
            },
        }
