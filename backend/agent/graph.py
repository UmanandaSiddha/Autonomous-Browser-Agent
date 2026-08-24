from langgraph.graph import END, START, StateGraph

from .nodes import (
    check_auth,
    extract_emails,
    handle_error,
    prepare_retry,
    summarize_emails,
    validate_digest,
)
from .state import AgentState


def route_after_auth(state: AgentState):
    print(
        f"[ROUTER] Auth state = {state['authenticated']}"
    )

    if state["authenticated"]:
        return "extract_emails"

    return "error"


def route_after_validation(state: AgentState):
    print(
        f"[ROUTER] Validation error = {state['error']}"
    )

    print(
        f"[ROUTER] Retry count = {state['retry_count']}"
    )
    
    if state["error"] is None:
        return "success"

    if state["retry_count"] < 2:
        return "retry"

    return "error"


def build_graph():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("check_auth", check_auth)
    graph.add_node("extract_emails", extract_emails)
    graph.add_node("summarize_emails", summarize_emails)
    graph.add_node("validate_digest", validate_digest)
    graph.add_node("prepare_retry", prepare_retry)
    graph.add_node("error", handle_error)

    # Start
    graph.add_edge(START, "check_auth")

    # Authentication routing
    graph.add_conditional_edges(
        "check_auth",
        route_after_auth,
        {
            "extract_emails": "extract_emails",
            "error": "error",
        },
    )

    # Main workflow
    graph.add_edge(
        "extract_emails",
        "summarize_emails",
    )

    graph.add_edge(
        "summarize_emails",
        "validate_digest",
    )

    # Validation routing
    graph.add_conditional_edges(
        "validate_digest",
        route_after_validation,
        {
            "success": END,
            "retry": "prepare_retry",
            "error": "error",
        },
    )

    # Retry
    graph.add_edge(
        "prepare_retry",
        "summarize_emails",
    )

    # Error
    graph.add_edge("error", END)

    return graph.compile()