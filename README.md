# Current: SetUp Docker Environment

This guide covers the manual setup for building and running the development environment.

_(For the simpler, automatic setup, see the "Dev Containers" guide in the main README)._

## Prerequisites

1.  [Git](https://git-scm.com/downloads)
2.  [Docker Desktop](https://www.docker.com/products/docker-desktop/)
3.  [Visual Studio Code](https://code.visualstudio.com/)
4.  The **"Dev Containers"** extension in VS Code.

---

### Step 1: Clone & Build

1.  **Open Docker Desktop:** Ensure Docker Desktop is running in the background.

2.  **Clone the Repo:**

    `cd` into the directory where you want to store your project (like your `IN4MATX 191A` folder).

    ‼️Just to remember that the `Dockerfile`, `requirements.txt`, and `.devcontainer` folder should be placed at the root (the top-level) of your cloned project folder (ImpackStudios), not inside any sub-folder.

    ```bash
    git clone [https://github.com/AiiiiDannn/ImpackStudios.git](https://github.com/AiiiiDannn/ImpackStudios.git)
    cd ImpackStudios
    ```

3.  **Build the Image:**
    - In your terminal (inside the `ImpackStudios` folder), run the build command.
    - This command is the **same for both Mac and PC**.
    ```bash
    docker build -t ai-studio .
    ```
    - Wait for the build to complete. This may take several minutes. (For me, it was around 4 minutes)

### Step 2: Run the Container (from Docker Desktop)

1.  **Find Image:**

    - In Docker Desktop, go to the **"Images"** tab.
    - You will see your new `ai-studio` image.

2.  **Run Image:**

    - Click the **"Run"** button (play icon) next to the `ai-studio` image.
    - An "Optional Settings" (or "Run new container") window will appear.

3.  **Configure Settings:**

    - **Container Name:** Give it a clear name (e.g., `ai-studio-dev`).

    - **Ports:**

      - Map `Host Port` **7860** to `Container Port` **7860**.
      - Click `+` (if needed), and map `Host Port` **8000** to `Container Port` **8000**.

    - **Volumes (Most Important):**

      - **Path 1 (Code):**
        - `Host Path`: Click "Browse" and select your `ImpackStudios` folder.
        - `Container Path`: Type `/workspace`
        - _(**Explanation:** This creates a **live mirror**. Any file you edit in your local `ImpackStudios` folder will instantly update inside the container's `/workspace`, and vice-versa.)_
      - **Path 2 (Cache):**
        - `Host Path`: Click "Browse" and select the `hf_cache` folder _inside_ your `ImpackStudios` folder.
        - `Container Path`: Type `/workspace/hf_cache`
        - _(**Explanation:** This folder **saves large AI models** from Hugging Face. By mapping it, models are saved to your host machine, preventing re-downloads every time you restart the container.)_

    - **Environment Variables:**

      - (Optional) Add any API keys you need (e.g., `GOOGLE_API_KEY`).

    - **For PC/GPU Users:** `To Eddie: I'm not sure if it is correct. Please check it on your PC with GPU available
      - **For PC/GPU Users (Eddie):**
        - In the "Run new container" window, scroll down and click the **"Advanced settings"** (or "Host config") dropdown.
        - This will expand a new section, often with tabs. Click the **"Resources"** tab.
        - Find the **"GPU"** section and enable **"GPU resource access"**.
        - _(This adds the `--gpus all` flag and gives the container access to your PC's NVIDIA GPU for fine-tuning.)_

4.  **Start Container:** Click the final **"Run"** button to start your container.
5.  After all these steps are completed, you can direct to the `Containers` tab, where a live container is running. Please check if its name matches what you entered in the settings before you `Run`.

### Step 3: Connect VS Code

1.  **Attach to Container:**

    - Open VS Code.
    - Click the green `><` icon in the bottom-left corner.
    - Select **"Attach to Running Container..."**
    - Choose your container (e.g., `/ai-studio-dev`)

2.  **Open Folder:**
    - A new VS Code window will open, connected to the container.
    - It will prompt you to "Open Folder". The default path might be `/root`.
    - Delete `/root` and type `/workspace`. (You can also check what other folders are, but `/workspace/` is where we will be working for the project)
    - Click "OK".

You are now working inside the container. The `/workspace` folder is a direct mirror of your local `ImpackStudios` folder. Any file you save locally will instantly appear inside the container, and vice-versa.
