from app.config import AppConfig
from app.ui import build_interface


def main(share: bool = True):
    import gradio as gr

    # In Colab/Jupyter, re-running launch cells can leave stale servers/event loops.
    # Always close previous instances before creating a new app.
    try:
        gr.close_all()
    except Exception:
        pass

    config = AppConfig()
    demo = build_interface(config)
    demo.queue(default_concurrency_limit=1).launch(
        share=share,
        debug=True,
        prevent_thread_lock=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
