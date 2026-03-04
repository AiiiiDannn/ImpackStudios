# System Architecture - Runnable Skeleton

This folder contains a runnable routed system that reuses your notebook logic:
- Router (impact vs script-review)
- Impact handler (RAG + Gemini generation)
- Script-review handler (Gemini generation)
- Gradio app entrypoint

## Structure

- `app.py`: Gradio entrypoint
- `requirements.txt`: dependencies
- `system_arch/config.py`: env/settings
- `system_arch/router.py`: routing logic + keyword fallback
- `system_arch/orchestrator.py`: route + dispatch
- `system_arch/services/rag_service.py`: file loading, chunking, embedding, FAISS
- `system_arch/services/generation_service.py`: Gemini generation and context rendering
- `system_arch/handlers/impact_handler.py`: impact pipeline
- `system_arch/handlers/script_handler.py`: script reviewer pipeline

## Quick Start (Colab or local)

1. Install deps:
   - `pip install -r ImpackStudios/System_Architecture/requirements.txt`
2. Set env var:
   - `export GOOGLE_API_KEY=...`
3. Run:
   - `python ImpackStudios/System_Architecture/app.py`

## Notes

- This skeleton keeps your current decision: impact path uses RAG.
- It is intentionally modular so you can later swap impact generation to your fine-tuned model without changing UI or router.
