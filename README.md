      
# AI Educational Video Generator

This Python script automatically generates educational videos on a given topic. It uses the Google Gemini API to create a script, Google Text-to-Speech (gTTS) for narration, and MoviePy to assemble the video with synchronized text visuals and audio.

## Features

*   **AI-Powered Content:** Leverages Google Gemini to generate educational scripts.
*   **Text-to-Speech:** Uses gTTS for clear audio narration of the generated script.
*   **Automated Video Creation:** Combines text overlays and audio into an MP4 video file using MoviePy.
*   **Dynamic Text Display:** Automatically wraps text and paginates longer paragraphs for readability.
*   **Visual Enhancements:** Includes a title bar, progress bar, and page numbers.
*   **Font Handling:** Attempts to find common system fonts (Arial, DejaVu Sans) and falls back to a default font.
*   **Title & Credits:** Adds simple opening title and closing credits screens.
*   **Optimized Performance:** Uses concurrent processing to speed up audio generation and frame rendering.
*   **Error Handling:** Includes basic fallback mechanisms and error reporting during generation.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Python:** Version 3.7 or higher. You can download it from [python.org](https://www.python.org/).
2.  **pip:** Python's package installer (usually comes with Python).
3.  **FFmpeg:** This is essential for MoviePy to process video and audio.
    *   **Installation:**
        *   **Linux (Debian/Ubuntu):**
            ```bash
            sudo apt update
            sudo apt install ffmpeg
            ```
        *   **macOS (using Homebrew):**
            ```bash
            brew install ffmpeg
            ```
        *   **Windows:** Download from the [official FFmpeg website](https://ffmpeg.org/download.html) (choose a build for Windows). After downloading, **you must add the `bin` directory** (containing `ffmpeg.exe`) to your system's **PATH environment variable**. Search for "Edit the system environment variables" in Windows settings.
    *   **Verification:** Open a *new* terminal or command prompt and type `ffmpeg -version`. If it shows the version details, FFmpeg is correctly installed and in your PATH.

## Setup & Installation

Follow these steps to set up the project environment:

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory-name>
    ```
    *(Replace `<your-repository-url>` and `<repository-directory-name>` with your actual repo details)*

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment to isolate project dependencies.
    *   On macOS and Linux:
        ```bash
        python3 -m venv venv
        ```
    *   On Windows:
        ```bash
        python -m venv venv
        ```

3.  **Activate the Virtual Environment:**
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows (Command Prompt):
        ```bash
        .\venv\Scripts\activate
        ```
    *   On Windows (PowerShell):
        ```bash
        .\venv\Scripts\Activate.ps1
        ```
    *(Your terminal prompt should now indicate that you are in the `(venv)` environment.)*

4.  **Install Dependencies:**
    Install all the required Python packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Google Gemini API Key:**
    *   You need a Google API key enabled for the Gemini API. You can get one from the [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   **Set the API Key:**
        *   **(Recommended) Environment Variable:** Set an environment variable named `GOOGLE_API_KEY`. This is more secure than hardcoding it.
            *   *macOS/Linux:* `export GOOGLE_API_KEY='YOUR_API_KEY_HERE'` (add this to your `.bashrc` or `.zshrc` for persistence).
            *   *Windows (Command Prompt):* `set GOOGLE_API_KEY=YOUR_API_KEY_HERE` (session-specific).
            *   *Windows (PowerShell):* `$env:GOOGLE_API_KEY='YOUR_API_KEY_HERE'` (session-specific).
            *   For persistent environment variables on Windows, search for "Edit the system environment variables".
        *   **(Less Secure) Directly in Script:** If you cannot use environment variables, replace the placeholder `'YOUR_API_KEY_HERE'` in the script (`.py` file) with your actual API key. **Warning:** Avoid committing your API key directly into version control (like Git).
            ```python
            # In the script:
            GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or 'YOUR_API_KEY_HERE' # Replace placeholder if needed
            ```

2.  **FFmpeg Path (Optional):**
    *   The script relies on MoviePy, which usually automatically detects `ffmpeg` if it's in your system's PATH (see Prerequisites).
    *   If MoviePy cannot find `ffmpeg`, you might need to explicitly tell it where `ffmpeg` is located. Uncomment and modify the following line near the top of the script:
        ```python
        # change_settings({"FFMPEG_BINARY": "/path/to/your/ffmpeg"}) # Adjust the path as needed
        ```
        *(Replace `/path/to/your/ffmpeg` with the actual path to the `ffmpeg` executable, e.g., `/usr/bin/ffmpeg` on Linux or `C:/ffmpeg/bin/ffmpeg.exe` on Windows).*

## Usage

1.  **Activate Virtual Environment:** Make sure your virtual environment is activated (you should see `(venv)` in your terminal prompt).
    ```bash
    # Example for macOS/Linux:
    source venv/bin/activate
    # Example for Windows CMD:
    .\venv\Scripts\activate
    ```

2.  **Run the Script:**
    Execute the main Python script from your terminal:
    ```bash
    python your_script_name.py
    ```
    *(Replace `your_script_name.py` with the actual filename of the Python script, e.g., `video_generator.py`)*

3.  **Enter Topic:** The script will prompt you to enter the educational topic you want the video to be about:
    ```
    Enter educational topic: [Your Topic Here]
    ```
    Type your topic and press Enter.

4.  **Wait for Generation:** The script will then:
    *   Generate the script using Gemini.
    *   Generate audio using gTTS.
    *   Render video frames with text.
    *   Combine everything into the final video.
    This process can take several minutes depending on the content length and your computer's speed. You will see progress messages in the terminal.

5.  **Output:** Once completed, the video file will be saved in the same directory as the script, named like `Your_Topic_Name_educational_video.mp4`.

## Dependencies (`requirements.txt`)

Create a file named `requirements.txt` in the root of your project directory with the following content:

    

IGNORE_WHEN_COPYING_START
Use code with caution.Markdown
IGNORE_WHEN_COPYING_END

google-generativeai
gTTS
Pillow
moviepy
numpy
torch
pydub

      
You can install these using `pip install -r requirements.txt` after activating your virtual environment.

## Troubleshooting & Notes

*   **`ffmpeg: command not found` or MoviePy errors:** Ensure FFmpeg is installed correctly and added to your system's PATH (see Prerequisites). If issues persist, try setting the `FFMPEG_BINARY` path explicitly in the script (see Configuration).
*   **API Key Errors (`PERMISSION_DENIED`, `API key not valid`, etc.):**
    *   Double-check that your API key is correct and pasted without extra spaces.
    *   Ensure the Gemini API is enabled for your key in your Google Cloud project or AI Studio.
    *   Verify you haven't exceeded any API usage quotas.
    *   Check your internet connection.
*   **Font Errors (`IOError: cannot open resource`):** The script tries common locations for Arial and DejaVu Sans. If these aren't found, it uses a basic default font. On minimal Linux systems, you might need to install fonts: `sudo apt-get install ttf-dejavu` (Debian/Ubuntu) or similar for your distribution.
*   **Memory Issues (`MemoryError`):** Video processing, especially concatenation, can be memory-intensive. If you encounter errors, try closing other applications or potentially reducing the `max_workers` variable in the `generate_video` function call within the script (though this might slow down processing). The script includes some memory management (`gc.collect`), but complex videos can still demand significant RAM.
*   **gTTS Errors:** gTTS requires an active internet connection to synthesize speech. Check your network if audio generation fails.
*   **Video Display (`display(Video(...))`):** This line uses `IPython.display` and will only automatically display the video inline if you are running the script within a Jupyter Notebook or a compatible environment like VS Code Notebooks. It won't display the video if run from a standard terminal. The video file is still created successfully.
