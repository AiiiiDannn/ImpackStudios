# Impact Studios System Architecture

## Goals
- Route each user request to the correct specialist path.
- Keep Impact analysis grounded (RAG + mission context).
- Use the fine-tuned model only where it adds value (Impact path).
- Keep UI independent from model/backend details.

## Layered Design

### 1) Interface Layer (UI)
- Responsibility:
  - Collect user input (text + optional file).
  - Show response, sources, and chat history.
- Current source:
  - `ImpackStudios/FineTuneLLM/Impact_RAG_v2.ipynb` (Gradio UI section).
- Rule:
  - UI should call one orchestration endpoint/function only.

### 2) Orchestration Layer (Router + Workflow)
- Responsibility:
  - Decide target agent: `impact` vs `script-review`.
  - Build standardized request envelope.
  - Trigger the correct backend pipeline.
- Current source:
  - `ImpackStudios/FineTuneLLM/Orchestrator_Pipeline.ipynb`.
- Needed adjustment:
  - Keep routing in orchestrator.
  - Move generation out of router internals and into backend-specific handlers.

### 3) Retrieval Layer (RAG)
- Responsibility:
  - Parse uploaded/reference documents.
  - Chunk + embed + FAISS search.
  - Return top-k evidence chunks.
- Current source:
  - `ImpackStudios/FineTuneLLM/Impact_RAG_v2.ipynb` (`RAGIndex`, `search`, `answer_question`).
- Rule:
  - Retrieval outputs structured context, not final answer text.

### 4) Model Layer
- Impact Model Path:
  - Fine-tuned LoRA model for Impact evaluation.
  - Current source: `ImpackStudios/FineTuneLLM/usemodelkaggle.ipynb` (to be renamed to colab-oriented name).
- Script Reviewer Path:
  - Gemini (or alternate general model) for code/script/document review.

### 5) Output Formatting Layer
- Responsibility:
  - Normalize outputs from different paths into one schema.
- Recommended response schema:
  - `agent`: `impact` | `script-review`
  - `answer`: string
  - `sources`: list of `{file, chunk_id, score}`
  - `meta`: `{latency_ms, model_name, route_reason}`

## Recommended Repository Organization

```
ImpackStudios/
  System_Architecture/
    ARCHITECTURE.md
    USER_FLOW.md
  FineTuneLLM/
    Orchestrator_Pipeline.ipynb
    Impact_RAG_v2.ipynb
    usemodel_colab.ipynb           # rename from usemodelkaggle.ipynb
    FineTuneLLM.ipynb
    my_finetuned_llm/
```

## Integration Contract (Minimal)

### Router input
- `user_text`: str
- `mission`: str (optional)
- `attachments`: list (optional)

### Impact handler input
- `user_text`
- `mission`
- `retrieved_context` (top-k chunks)

### Impact handler output
- `answer`
- `score` (if available)
- `sources`

### Script-review handler input
- `user_text`

### Script-review handler output
- `answer`

## Practical Implementation Order
1. Rename `usemodelkaggle.ipynb` -> `usemodel_colab.ipynb`.
2. Refactor orchestrator to route only (no direct generation in router core).
3. Add two handler functions:
   - `handle_impact(...)`
   - `handle_script_review(...)`
4. Let `handle_impact` call RAG first, then fine-tuned model.
5. Keep UI bound to one function: `route_and_run(...)`.
