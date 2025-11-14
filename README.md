# Fine-tuned LLM Demo on Kaggle Notebook

Please read this simple step-by-step guide to load your fine-tuned LLM and run a Gradio demo on Kaggle.

## 1. Create a Kaggle Account

Go to **[https://www.kaggle.com](https://www.kaggle.com)** and create an account (or log in).

## 2. Create a New Notebook

1. On the left sidebar, click **+ Create -> Notebook**.
2. A new notebook opens.
3. Click the notebook title (top left) to rename it.

## 3. Enable GPU

To run a large 8B model, you must enable GPU.
But **Kaggle requires phone number verification first**.

### 3.1 Phone verification

Inside the notebook:

1. Look at the settings area on the right side of the notebook.
2. You can see `Input`,`Output`, `Table of Contents`, `Session Options`, etc., under the `Notebook` title.
3. Click `Session Options`, find this small message under the `Add Tags` button:

> “Want more power? Access GPU/TPU at no cost or turn on an internet connection. Get phone verified.”

4. Click the link -> verify your phone number.
5. Close the notebook, then reopen it.

Now your account can use GPUs.

### 3.2 Two ways to enable GPU

#### **Method A: From Settings bar**

- Click **Settings** under the notebook title
- Choose **Accelerator -> GPU T4×2** or **GPU P100**.

#### **Method B: From right-hand sidebar**

In the right sidebar, open **Session options** -> choose an **Accelerator**.

### 3.3 Which GPU should you choose? (Simple explanation)

| GPU       | Memory      | Speed  | When to use                             |
| --------- | ----------- | ------ | --------------------------------------- |
| **T4 x2** | ~30GB total | Faster | ⭐ Best for LLM inference & Gradio demo |
| **P100**  | ~16GB       | Slower | Only if T4 is unavailable               |

👉 **Recommendation: choose T4×2.**
You don't have to choose this until you are going to load the model (STEP 8)

> Got these from ChatGPT. I just used T4 x2, which worked pretty well.
> There should be a 30hrs quota for our free account. You may link you Colab Pro account to your Kaggle account for a higher quota of GPU (I think it should be 45hrs, and Google Colab Pro is available for our student email linked account)

## 4. Install Required Packages

Delete the default code cell, add a new one, and run:

```python
!pip install -q unsloth peft accelerate bitsandbytes gradio
```

This installs everything needed to load the model and run the Gradio interface. This actually takes a long time (we have unsloth and gradio in here), because it's going to save a few GB files to your Kaggle cloud. I waited around 5 min for have everything setup. No worries about the red ERROR message like "pip's dependency" things, they don't matter as far as what I will go through.

## 5. Upload Out Fine-tuned Model as a Kaggle Dataset

You should have our model file downloaded to your device, which you may find it`:

```
FineTuneLLM > my_finetuned_llm.zip
```

### 5.1 Create the dataset

1. On Kaggle, open the left sidebar (the icon with 3 lines) -> click **Datasets**
   (or visit [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets))
2. Click **+ New Dataset**
3. Click **Browse files**
4. Upload `my_finetuned_llm.zip`
5. Give it any title, e.g. **my-finetuned-llm**
6. Save the dataset

Kaggle will automatically extract the zip file.

> I have not tried uploading from Github Repo or Remote URL, but you can take a try. I guess it won't hurt anything except for that your ADAPTER_DIR may be changed a little bit, which I'm going to discuss later.

## 6. Add the Model Dataset to Your Notebook

Go back to your notebook.

1. On the right sidebar, at the top, click **+ Add Input**
2. Find our dataset **my-finetuned-llm** (or any other name you gave to it)
3. Click the **⊕** button
4. Wait for it to mount
5. Close the panel

Now the dataset appears under the **Input** (under **DATASETS**, with a triangle and a teal icon on the left)

### 6.1 Copy the dataset path (MAK)

Hover your mouse over the dataset name, where you will see a **copy icon** on the right. Click it.

Kaggle copies a path like (which appears on my device):

```
/kaggle/input/my-finetuned-llm
```

I call this the **Model Address in Kaggle (MAK)**.

## 7. Set the Model Path in the Notebook

Add a new code cell and paste:

```python
# ============================================================
# 1. Set Kaggle dataset path
# ============================================================
ADAPTER_DIR = "/kaggle/input/my-finetuned-llm"   # Replace with your MAK if needed
```

## 8. Load the Base Model + LoRA Adapter

Add another code cell:

```python
from unsloth import FastLanguageModel
from peft import PeftModel
import torch

# ============================================================
# 2. Load 4-bit base model
# ============================================================
BASE_ID = "unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    BASE_ID,
    max_seq_length = 2048,
    load_in_4bit = True,
)
print("=" * 50)
print("Base model loaded")
print("=" * 50)

# ============================================================
# 3. Load LoRA adapter
# ============================================================
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
print("=" * 50)
print("LoRA loaded")
print("=" * 50)

# ============================================================
# 4. Switch to inference mode
# ============================================================
FastLanguageModel.for_inference(model)
print("=" * 50)
print("Model ready for inference")
print("=" * 50)

model.eval()
print("\n\n")
print("=" * 50)
print("Fine-tuned model is ready!")
print("=" * 50)
```

If it prints those "=" wrapped texts, everything is correct. There could be a few message like `Unable to register cuDNN/cuBLAS/cuFFT`. No worries about them. As long as you have the message starting from 🦥 (patch process starts) and can see something like `model.safetensors: 100%`, `tokenizer.json: 100%`, our model is loading or loaded.

## 9. Run the Gradio Interface (GPU required)

After enabling GPU and loading the model, run:

```python
import gradio as gr
import torch
import re
from threading import Lock

# -------------------------
# YOUR MODEL + TOKENIZER
# -------------------------
tok = tokenizer
mdl = model
mdl.eval()

gen_lock = Lock()

# -------------------------
# Format the model output
# -------------------------
def format_output(raw):
    sections = {
        "verdict": r"(?i)verdict[:\-]\s*(.*)",
        "score": r"(?i)score[:\-]\s*([0-9\.]+)",
        "benefits": r"(?i)benefits?[:\-](.*?)(?=risks?:|overall|conclusion|$)",
        "risks": r"(?i)risks?[:\-](.*?)(?=overall|conclusion|$)",
        "overall": r"(?i)(overall|rationale|analysis)[:\-](.*)",
    }

    formatted = {}
    for key, pattern in sections.items():
        m = re.search(pattern, raw, re.S)
        if m:
            text = m.group(1).strip()
        else:
            text = ""
        formatted[key] = text

    return formatted


# -------------------------
# Evaluation Function
# -------------------------
history = []

def evaluate_story(story, context):
    with gen_lock:
        if not story or story.strip() == "":
            return "Please enter a story.", 0, history

        template = """Below is an instruction that describes a task.

### Instruction:
Evaluate whether the following idea uplifts humanity. Provide:
- Verdict (Yes/No or Positive/Negative)
- Score from 0 to 1
- Benefits
- Risks
- Overall rationale
Be objective and specific.

### Input:
{story}

### Additional Context:
{context}

### Response:
"""

        prompt = template.format(
            story = story.strip(),
            context = context.strip() if context else "None"
        )

        inputs = tok(prompt, return_tensors="pt").to("cuda")

        outputs = mdl.generate(
            **inputs,
            max_new_tokens = 400,
            temperature   = 0.4,
            top_p         = 0.95,
            do_sample     = True,
        )

        raw_text = tok.decode(outputs[0], skip_special_tokens=True)

        # Clean prefix if exists
        if "### Response:" in raw_text:
            raw_text = raw_text.split("### Response:")[1].strip()

        parsed = format_output(raw_text)

        # Score handling
        try:
            score = float(parsed["score"])
        except:
            score = 0

        # Save history
        history.append({
            "story": story,
            "score": score,
            "verdict": parsed["verdict"],
            "output": raw_text[:600] + ("..." if len(raw_text) > 600 else "")
        })

        display_text = f"""
### Verdict
{parsed["verdict"]}

### Score
{parsed["score"]}

### Benefits
{parsed["benefits"]}

### Risks
{parsed["risks"]}

### Overall Rationale
{parsed["overall"]}
""".strip()

        return display_text, score, history


# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🌍 Humanity Uplift Evaluator — Kaggle Demo")
    gr.Markdown("Enter an idea / story. The model evaluates whether it uplifts humanity.")

    with gr.Row():
        story = gr.Textbox(
            label="Story / Idea",
            placeholder="Enter the narrative or idea here...",
            lines=8
        )
        context = gr.Textbox(
            label="Additional Context (optional)",
            placeholder="Studio mission, values, situation...",
            lines=4
        )

    btn = gr.Button("Evaluate", variant="primary")

    with gr.Row():
        output = gr.Markdown(label="Model Output")
        score_bar = gr.Slider(label="Score (0 to 1)", minimum=0, maximum=1, value=0, interactive=False)

    history_box = gr.JSON(label="History (previous evaluations)")

    btn.click(
        evaluate_story,
        inputs=[story, context],
        outputs=[output, score_bar, history_box]
    )

demo.launch(share=True)
```

This will print a public URL such as:

```
Running on local URL:  http://127.0.0.1:xxxx
Running on public URL: https://xxxx.gradio.live
```

Click the link you'd like to go, and our model is ready to use in the gradio interface.

You may use this prompt, which I asked ChatGPT to provide with me. I think it will score at around .90 to .95.

> A coalition of teachers, engineers, and local volunteers launches a free community learning center that offers open-access STEM workshops, AI literacy classes, and mentorship for students from underserved backgrounds.
> The program runs year-round, provides laptops and materials at no cost, and pairs each student with a trained mentor who supports both academic growth and emotional well-being. Alumni return to become new mentors, creating a self-sustaining cycle of empowerment.
> The center is designed to be inclusive for learners of all ages and abilities, and all curriculum materials are open-source so that other communities can replicate the model.
> Evaluate whether this initiative uplifts humanity, provide a score from 0 to 1, explain benefits and risks, and give a concise recommendation for scaling.
