# User Interaction Flow (End-to-End)

## What the user experiences
1. User opens UI (chat-like page).
2. User enters text (and optionally uploads file + mission context).
3. User clicks submit.
4. System returns one answer (with optional sources and score).

## What the system does before answering
1. **Input normalization**
   - Parse text/file into one canonical payload.
   - Basic validation (empty input, file parse errors).

2. **Routing decision**
   - Orchestrator classifies request:
     - `impact` (ethics/community impact analysis)
     - `script-review` (code/script/document review)

3. **Impact path (if routed to impact)**
   - Run retrieval on indexed docs (RAG top-k chunks).
   - Build final prompt with:
     - system rules
     - user submission
     - optional mission
     - retrieved context chunks
   - Call fine-tuned LoRA model.
   - Parse/normalize output fields.

4. **Script-review path (if routed to script-review)**
   - Build reviewer prompt.
   - Call Gemini backend.
   - Normalize output fields.

5. **Response packaging**
   - Add metadata (agent used, optional sources, runtime info).
   - Return to UI for display.

## What happens after response
1. UI appends turn to chat history.
2. Optional logging:
   - route decision
   - latency
   - model/backend used
3. User can continue conversation or start new chat.

## Sequence Diagram (Text)

1. `UI -> Orchestrator: route_and_run(user_input, mission, file)`
2. `Orchestrator -> RouterModel: classify intent`
3. `RouterModel -> Orchestrator: impact | script-review`
4. If `impact`:
   - `Orchestrator -> RAGIndex: search(user_input, top_k)`
   - `RAGIndex -> Orchestrator: chunks`
   - `Orchestrator -> FineTunedModel: generate(prompt_with_chunks)`
   - `FineTunedModel -> Orchestrator: impact_answer`
5. If `script-review`:
   - `Orchestrator -> Gemini: generate(reviewer_prompt)`
   - `Gemini -> Orchestrator: review_answer`
6. `Orchestrator -> UI: normalized_response`

## Recommended User-Facing Behavior
- Always show which agent responded (`Impact Agent` or `Script Reviewer`).
- On impact responses, show retrieved sources when available.
- On routing uncertainty, default to impact only if confidence is low and request is ambiguous.
