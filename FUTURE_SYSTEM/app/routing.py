from typing import Optional, Tuple


class RequestOrchestrator:
    """
    Orchestrator copied to match v3 routing behavior.
    Routes between Impact Agent and Script Reviewer.
    """

    IMPACT_AGENT = "impact agent"
    SCRIPT_REVIEWER = "script reviewer"

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
        from google import genai  # type: ignore

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def _create_routing_prompt(self, user_prompt: str) -> str:
        return f"""You are a routing system that determines which agent should handle a user's request.

You have TWO agents available:

1. **Impact Agent**: Handles questions about:
   - Impact, outcomes, effects, consequences, results
   - Benefits, changes, influence of actions/policies/programs
   - Analysis of societal, environmental, or business impacts
   - "What if" scenarios and their implications

2. **Script Reviewer**: Handles questions about:
   - Reviewing, analyzing, or critiquing scripts (movie, TV, theater)
   - Code review and programming help
   - Document review and editing
   - Writing feedback and improvements

USER QUERY: {user_prompt}

Based on the user's query, determine which agent should handle this request.

RESPOND WITH ONLY ONE OF THESE TWO OPTIONS (exactly as written):
- "impact agent"
- "script reviewer"

Your response should contain ONLY the agent name, nothing else."""

    def _fallback_routing(self, user_prompt: str) -> str:
        user_lower = (user_prompt or "").lower()

        script_keywords = [
            "script",
            "review",
            "code",
            "analyze",
            "critique",
            "feedback",
            "document",
            "written",
            "check",
            "screenplay",
            "program",
            "function",
            "bug",
            "error",
            "syntax",
        ]

        impact_keywords = [
            "impact",
            "effect",
            "outcome",
            "result",
            "consequence",
            "benefit",
            "change",
            "influence",
            "affect",
            "what if",
            "implications",
            "happens",
            "caused by",
        ]

        script_score = sum(1 for keyword in script_keywords if keyword in user_lower)
        impact_score = sum(1 for keyword in impact_keywords if keyword in user_lower)

        if script_score > impact_score:
            return self.SCRIPT_REVIEWER
        return self.IMPACT_AGENT


def keyword_route(prompt: str) -> Tuple[Optional[str], int, int]:
    text = (prompt or "").lower()

    script_keywords = [
        "script",
        "review",
        "screenplay",
        "dialogue",
        "plot",
        "character",
        "code",
        "bug",
        "function",
        "error",
        "syntax",
        "refactor",
        "document",
    ]
    impact_keywords = [
        "impact",
        "outcome",
        "consequence",
        "benefit",
        "harm",
        "society",
        "community",
        "ethics",
        "policy",
        "what if",
        "implication",
    ]

    script_score = sum(1 for k in script_keywords if k in text)
    impact_score = sum(1 for k in impact_keywords if k in text)

    if script_score > 0 and impact_score == 0:
        return RequestOrchestrator.SCRIPT_REVIEWER, script_score, impact_score
    if impact_score > 0 and script_score == 0:
        return RequestOrchestrator.IMPACT_AGENT, script_score, impact_score

    return None, script_score, impact_score


def safe_route(prompt: str, orchestrator: RequestOrchestrator) -> Tuple[str, str]:
    kw_agent, s_score, i_score = keyword_route(prompt)
    if kw_agent is not None:
        return kw_agent, f"keyword(script={s_score},impact={i_score})"

    try:
        routing_prompt = orchestrator._create_routing_prompt(prompt)
        response = orchestrator.client.models.generate_content(
            model=orchestrator.model_name,
            contents=routing_prompt,
        )
        decision = (getattr(response, "text", "") or "").strip().lower()

        if orchestrator.IMPACT_AGENT in decision:
            return orchestrator.IMPACT_AGENT, "llm"
        if orchestrator.SCRIPT_REVIEWER in decision:
            return orchestrator.SCRIPT_REVIEWER, "llm"

        fb = orchestrator._fallback_routing(prompt)
        return fb, f"fallback_format(script={s_score},impact={i_score})"

    except Exception as e:
        fb = orchestrator._fallback_routing(prompt)
        return fb, f"fallback_error:{type(e).__name__}(script={s_score},impact={i_score})"


def choose_route(prompt: str, prefer_llm: bool, client=None, model: str = "") -> Tuple[str, str]:
    # Keep public signature stable for app/ui.py.
    api_key = getattr(client, "api_key", "") if client is not None else ""

    if prefer_llm and api_key:
        try:
            orchestrator = RequestOrchestrator(
                api_key=api_key,
                model_name=model or "gemini-2.5-flash-lite",
            )
            return safe_route(prompt, orchestrator)
        except Exception:
            pass

    # No LLM available -> deterministic fallback aligned with v3 fallback style.
    kw_agent, s_score, i_score = keyword_route(prompt)
    if kw_agent is not None:
        return kw_agent, f"keyword(script={s_score},impact={i_score})"

    # Use fallback routing scores without extra LLM call.
    user_lower = (prompt or "").lower()
    script_keywords = [
        "script",
        "review",
        "code",
        "analyze",
        "critique",
        "feedback",
        "document",
        "written",
        "check",
        "screenplay",
        "program",
        "function",
        "bug",
        "error",
        "syntax",
    ]
    impact_keywords = [
        "impact",
        "effect",
        "outcome",
        "result",
        "consequence",
        "benefit",
        "change",
        "influence",
        "affect",
        "what if",
        "implications",
        "happens",
        "caused by",
    ]
    script_score = sum(1 for keyword in script_keywords if keyword in user_lower)
    impact_score = sum(1 for keyword in impact_keywords if keyword in user_lower)
    if script_score > impact_score:
        return RequestOrchestrator.SCRIPT_REVIEWER, f"fallback_no_llm(script={script_score},impact={impact_score})"
    return RequestOrchestrator.IMPACT_AGENT, f"fallback_no_llm(script={script_score},impact={impact_score})"

