# FUTURE_SYSTEM Modular Layout

This folder is the modular Colab-ready version of the final workflow.

## File Map

- `requirements.txt`: minimal dependencies.
- `run_colab.py`: single entrypoint for Colab launch.
- `app/config.py`: centralized models + runtime defaults.
- `app/keys.py`: API key cleaning and validation.
- `app/drive_utils.py`: Drive helpers and upload text extraction.
- `app/llm_clients.py`: `google-genai` client wrapper.
- `app/routing.py`: keyword + LLM orchestrator routing.
- `app/summary.py`: long-script summary gate utilities.
- `app/reviewer.py`: reviewer prompt, parsing, verdict/score alignment.
- `app/impact.py`: impact-agent prompt and formatter.
- `app/ui.py`: Gradio UI and pipeline wiring.

## Run In Colab

Recommended entrypoint: open [launch_colab.ipynb](/Users/aiiiidannn/Documents/Peter%20the%20Anteater/Courses/2025%20FALL/IN4MATX%20191A/ImpackStudios/FUTURE_SYSTEM/launch_colab.ipynb) in Colab and run all cells.

Before running, add the shared `FUTURE_SYSTEM` folder as a shortcut under `My Drive`.

```python
%cd /content/drive/MyDrive/FUTURE_SYSTEM
!pip install -q -r requirements.txt
# In notebook Step 4, input GOOGLE_API_KEY via hidden getpass
# In notebook Step 5, input HF_TOKEN via hidden getpass
# Then launch UI (Step 6; if GPU is missing, local reviewer will fallback to Gemini)

from run_colab import main
main()
```

## What This Gives You

- Keeps the same high-level behavior as notebook flow:
  - prompt + optional script upload
  - orchestrator route selection
  - summary gate for long scripts
  - script reviewer / impact agent response
- Makes future edits safer:
  - prompt changes only in `reviewer.py` / `impact.py`
  - routing logic only in `routing.py`
  - UI changes only in `ui.py`

## Notes

- Impact route + summary gate use Gemini (`google-genai`).
- Script reviewer route uses local adapter model (Unsloth + PEFT) and needs `HF_TOKEN`.
- If GPU is unavailable in Colab runtime, local reviewer auto-falls back to Gemini.
- Adapter folder (`my_finetuned_llm`) is required and validated.
