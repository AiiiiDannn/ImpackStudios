# ImpactStudio Script Reviewer Integration Guide

## Scope

This test folder is now reduced to one backend path:

- Gemini `Script Reviewer`
- FastAPI endpoint
- TSX frontend rendering the `review` result card

## Files

- [FINAL_ImpactStudio_FullSystem_v6.ipynb](/Users/aiiiidannn/Documents/Peter%20the%20Anteater/Courses/2025%20FALL/IN4MATX%20191A/ImpackStudios/System_Architecture/Studio_Workflows/FastAPI_Gemini_Test/FINAL_ImpactStudio_FullSystem_v6.ipynb)
  Minimal notebook for dependency install, API key setup, script-review prompt definition, and Gemini smoke test.

- [SECTION_E_API_SERVER.py](/Users/aiiiidannn/Documents/Peter%20the%20Anteater/Courses/2025%20FALL/IN4MATX%20191A/ImpackStudios/System_Architecture/Studio_Workflows/FastAPI_Gemini_Test/SECTION_E_API_SERVER.py)
  FastAPI server that reads script text, calls Gemini, normalizes the response, and streams SSE events.

- [impact-studios-app.tsx](/Users/aiiiidannn/Documents/Peter%20the%20Anteater/Courses/2025%20FALL/IN4MATX%20191A/ImpackStudios/System_Architecture/Studio_Workflows/FastAPI_Gemini_Test/impact-studios-app.tsx)
  Frontend wired to the FastAPI server and expecting the script review output shape.

## API Contract

### `GET /api/health`

Example:

```json
{
  "status": "ok",
  "reviewer_ready": true,
  "model": "gemini-2.5-flash-lite",
  "agent": "Script Reviewer"
}
```

### `POST /api/analyze`

Accepts `multipart/form-data`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `story_text` | string | no* | The user request or pasted script text |
| `mission_text` | string | no | Optional mission context |
| `file` | file | no* | `.pdf`, `.docx`, `.txt`, `.md` |
| `chat_history` | string | no | JSON array of `{role, content}` |

`story_text` or `file` must be present.

## SSE Flow

The server streams progress events in this order:

1. `routing`
2. `routed`
3. `working`
4. `working`
5. `done`

The final `done` event contains this result format:

```json
{
  "stage": "done",
  "agent": "Script Reviewer",
  "route_mode": "fixed(script-review)",
  "result": {
    "type": "review",
    "verdict": "Yes",
    "score": 4,
    "benefits": [
      "Deepens empathy through grounded emotional conflict",
      "Shows a clear path toward repair and accountability"
    ],
    "risks": [
      "Some beats need clearer pacing",
      "Secondary character motivations could be sharper"
    ],
    "rationale": "This script has a constructive emotional core and can become more effective with tighter execution."
  }
}
```

This matches the review card structure used by the frontend prototype.

## Gemini Output Rule

The backend instructs Gemini to return JSON with exactly:

- `verdict`
- `score`
- `benefits`
- `risks`
- `rationale`

The API normalizes the response before sending it to the frontend.

## Recommended Run Order

1. Run the notebook smoke test.
2. Copy `CELL E1`, `CELL E2`, and `CELL E3` from [SECTION_E_API_SERVER.py](/Users/aiiiidannn/Documents/Peter%20the%20Anteater/Courses/2025%20FALL/IN4MATX%20191A/ImpackStudios/System_Architecture/Studio_Workflows/FastAPI_Gemini_Test/SECTION_E_API_SERVER.py) into Colab.
3. Launch the server and copy `API_PUBLIC_URL`.
4. Set `API_URL` in [impact-studios-app.tsx](/Users/aiiiidannn/Documents/Peter%20the%20Anteater/Courses/2025%20FALL/IN4MATX%20191A/ImpackStudios/System_Architecture/Studio_Workflows/FastAPI_Gemini_Test/impact-studios-app.tsx).
5. Run the frontend and submit a review request.
