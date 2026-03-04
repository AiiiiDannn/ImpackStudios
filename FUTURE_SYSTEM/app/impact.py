def build_impact_prompt(user_prompt: str, script_text: str, humanity_def: str) -> str:
    script_block = script_text.strip() if script_text else "No script uploaded."
    return f"""You are the Impact Agent.

Humanity-uplifting definition:
{humanity_def}

Task:
Provide practical impact guidance for the request.
Output sections:
1) Objective
2) Audience
3) Messaging strategy
4) Risks and mitigations
5) Next 3 concrete actions

User request:
{user_prompt}

Script/context:
{script_block}
"""


def format_impact_markdown(raw: str, route_mode: str) -> str:
    return (
        f"**Routed Agent:** `impact agent`\n\n"
        f"**Route Mode:** `{route_mode}`\n\n"
        f"### Impact Agent Output\n{raw}"
    )

