from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from .nodes import (
    handle_error,
    prepare_retry,
    route_after_validation,
    summarize_emails,
    validate_digest,
)

from .state import AgentState


def build_graph():

    graph = StateGraph(
        AgentState
    )

    # --------------------------------------------------
    # Nodes
    # --------------------------------------------------

    graph.add_node(
        "summarize_emails",
        summarize_emails,
    )

    graph.add_node(
        "validate_digest",
        validate_digest,
    )

    graph.add_node(
        "prepare_retry",
        prepare_retry,
    )

    graph.add_node(
        "error",
        handle_error,
    )

    # --------------------------------------------------
    # Start
    # --------------------------------------------------

    graph.add_edge(
        START,
        "summarize_emails",
    )

    # --------------------------------------------------
    # Summary → Validation
    # --------------------------------------------------

    graph.add_edge(
        "summarize_emails",
        "validate_digest",
    )

    # --------------------------------------------------
    # Validation → Success / Retry / Error
    # --------------------------------------------------

    graph.add_conditional_edges(
        "validate_digest",
        route_after_validation,
        {
            "success": END,
            "retry": "prepare_retry",
            "error": "error",
        },
    )

    # --------------------------------------------------
    # Retry → Summary
    # --------------------------------------------------

    graph.add_edge(
        "prepare_retry",
        "summarize_emails",
    )

    # --------------------------------------------------
    # Error
    # --------------------------------------------------

    graph.add_edge(
        "error",
        END,
    )

    return graph.compile()