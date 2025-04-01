#------general video for VS Code----------

import torch
import numpy as np
import os
import google.generativeai as genai
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import *
from moviepy.config import change_settings
import concurrent.futures
from IPython.display import Video, display  # Keep this for potential VS Code notebook execution
import re
import time
import tempfile
import wave
import threading
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import gc  # Import gc and queue here as requested by user in the original colab code
import queue
import multiprocessing

# Configure MoviePy -  It's better to let MoviePy auto-detect ffmpeg if it's in PATH.
# If you face issues, you might need to specify the path explicitly.
# change_settings({"FFMPEG_BINARY": "/usr/bin/ffmpeg"})  # Uncomment and adjust if needed

# --- Gemini API Setup ---
# IMPORTANT: Set your GOOGLE_API_KEY as an environment variable or directly here (replace with your actual API key)
# It's recommended to use environment variables for security.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or 'YOUR_API_KEY_HERE'  # Replace 'YOUR_API_KEY_HERE' with your actual key if not using env variable
genai.configure(api_key=GOOGLE_API_KEY)

# --- Model Initialization (Improved) ---
def initialize_gemini_model():
    """Initializes the Gemini model, handling potential model name variations."""
    try:
        # First try listing models to see what's available
        available_models = [m.name for m in genai.list_models()]
        print(f"Available models: {available_models}")

        # Prioritize Gemini 1.5 Pro if available, then fall back to other versions.
        if any('gemini-1.5-pro' in model_name for model_name in available_models):
            gemini_model_name = next(model_name for model_name in available_models if 'gemini-1.5-pro' in model_name)
        elif any('gemini-pro' in model_name for model_name in available_models):
             gemini_model_name = next(model_name for model_name in available_models if 'gemini-pro' in model_name)
        else:
            gemini_model_name = 'gemini-pro' #last fallback

        print(f"Using {gemini_model_name}")
        return genai.GenerativeModel(gemini_model_name)

    except Exception as e:
        print(f"Error initializing Gemini: {e}")
        print("Trying direct model name as a last resort...")
        try:
            # Fallback if listing fails (might be an older API version)
            model = genai.GenerativeModel('gemini-pro')
            print("Using gemini-pro with direct name")
            return model
        except Exception as e2:
            print(f"Final error: {e2}")
            raise SystemExit("No suitable Gemini model found. Check your API key and network connection.")

# --- Content Generation ---
def generate_content_gemini(prompt, model=None):
    """Generates educational content using Gemini API optimized for video narration."""
    if model is None:
        model = initialize_gemini_model()

    try:
        print("Sending request to Gemini API...")
        response = model.generate_content(
            f"""Create an educational script about '{prompt}' for students.

            Guidelines:
            1. The script should be 5 minutes long when read aloud (approximately 750-800 words total).
            2. Structure the content into 3-5 clear, concise sections.
            3. Each section should contain multiple short paragraphs (2-3 sentences each).
            4. Use simple language and explain concepts clearly.
            5. Include an introduction, main content with key points, and a conclusion.
            6. Focus on educational value and engagement.
            7. DO NOT use a generic template. Respond directly to the topic with specific details, facts, and examples related to '{prompt}'. Avoid vague or repetitive statements.
            8. DO NOT use asterisks (*) in the text as they will be read aloud.

            Format the response as a series of paragraphs separated by blank lines.
            Each section should be separated by two blank lines.
            """,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=800,
                top_p=0.95,
            )
        )
        print("Successfully received response from Gemini API")
        if response.text:
             return response.text
        else:
            print("Gemini returned an empty response. Falling back.")
            return generate_fallback_content(prompt)
    except Exception as e:
        print(f"Gemini API error: {e}")
        print("Using fallback content generation...")
        return generate_fallback_content(prompt)


def generate_fallback_content(topic):
    """Generate basic fallback content if the Gemini API fails."""
    return f"""
    Introduction to {topic}.

    {topic} is a fascinating subject with many applications in our daily lives.

    Let's explore some key aspects of {topic} that make it important to understand.

    In conclusion, {topic} remains a critical area of study with ongoing developments.

    Thank you for learning about {topic} today.
    """

# --- Text Processing ---
def clean_text_for_speech(text):
    """Remove asterisks and other characters that shouldn't be read aloud."""
    # Remove asterisks
    text = re.sub(r'\*', '', text)

    # Replace excessive whitespace with single spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove markdown formatting characters that might cause issues
    text = re.sub(r'[#_~`]', '', text)

    # Replace common abbreviations with their full forms to improve speech
    text = re.sub(r'\bi\.e\.\s', 'that is, ', text, flags=re.IGNORECASE)
    text = re.sub(r'\be\.g\.\s', 'for example, ', text, flags=re.IGNORECASE)

    return text.strip()

def clean_and_split_content(content):
    """Clean and split content into sections and paragraphs."""
    if not content:
        return []

    # Remove extra whitespace and normalize line endings
    content = re.sub(r'\s+', ' ', content).strip()

    # Split into sections (using double blank lines)
    sections = re.split(r'\n\s*\n\s*\n', content)

    result = []
    for section in sections:
        # Split each section into paragraphs (using single blank lines)
        paragraphs = re.split(r'\n\s*\n', section)

        # Further processing of paragraphs
        clean_paragraphs = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if not p[-1] in '.!?':
                p += '.'
            clean_paragraphs.append(p)
        result.extend(clean_paragraphs)

    return result

# --- Font and Text Rendering ---
def get_system_font():
    """Get an available system font with caching."""
    if hasattr(get_system_font, 'cached_fonts'):
        return get_system_font.cached_fonts

    system_fonts = [
        ("arial.ttf", "arialbd.ttf"), # Windows/Common
        ("Arial.ttf", "Arial Bold.ttf"), # Windows/Common (alternative)
        ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"), # Linux
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"), # Linux (explicit path)
        ("/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"), # Linux (alternative path)
        ("/usr/share/fonts/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/liberation/LiberationSans-Bold.ttf") # Linux (Liberation fonts)
    ]

    for regular, bold in system_fonts:
        try:
            reg_font = ImageFont.truetype(regular, 36)
            bold_font = ImageFont.truetype(bold, 36)
            get_system_font.cached_fonts = (reg_font, bold_font)
            return reg_font, bold_font
        except IOError:
            continue

    # Fallback to default
    default_font = ImageFont.load_default()
    get_system_font.cached_fonts = (default_font, default_font)
    return default_font, default_font

def get_font_text_width(text, font):
    """Get text width using the appropriate method based on PIL/Pillow version."""
    try:
        return font.getlength(text)
    except AttributeError:
        try:
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0]
        except AttributeError:
            try:
                return font.getsize(text)[0]
            except AttributeError:
                return len(text) * (font.size // 2)

def get_font_text_height(text, font):
    """Get text height using the appropriate method based on PIL/Pillow version."""
    try:
        metrics = font.getmetrics()
        return metrics[0] + metrics[1]
    except AttributeError:
        try:
            bbox = font.getbbox(text)
            return bbox[3] - bbox[1]
        except AttributeError:
            try:
                return font.getsize(text)[1]
            except AttributeError:
                return font.size + 4

# --- Image and Frame Creation ---
def create_text_frame(width=1280, height=720, title=""):
    """Creates a blank video frame with the specified dimensions."""
    img = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw title bar
    if title:
        draw.rectangle([(0, 0), (width, 60)], fill=(70, 130, 180))
        regular_font, bold_font = get_system_font()
        title_width = get_font_text_width(title, bold_font)
        draw.text(((width - title_width) // 2, 10), title, (255, 255, 255), font=bold_font)

    # Draw progress bar at bottom
    draw.rectangle([(0, height-20), (width, height)], fill=(240, 240, 240))

    return img, draw

def render_text_page(text, current_time, duration, page_number=0, max_lines=13, width=1280, height=720, title=""):
    """Renders a page of text with proper line wrapping and paging."""
    img, draw = create_text_frame(width=width, height=height, title=title)

    # Get fonts
    regular_font, _ = get_system_font()

    # Prepare text for rendering
    words = text.split()
    lines = []
    current_line = []
    max_line_width = width - 80  # Margins

    # Create lines by wrapping text
    for word in words:
        test_line = ' '.join(current_line + [word])
        line_width = get_font_text_width(test_line, regular_font)

        if line_width <= max_line_width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:  # Add the last line
        lines.append(' '.join(current_line))

    # Calculate which page to show based on current time and total duration
    total_pages = (len(lines) + max_lines - 1) // max_lines  # Ceiling division

    # If page_number is provided, use it directly
    if page_number < 0:
        # Calculate page based on time
        page_progress = current_time / duration
        current_page = min(int(page_progress * total_pages), total_pages - 1)
    else:
        current_page = page_number

    # Determine start and end lines for the current page
    start_line = current_page * max_lines
    end_line = min(start_line + max_lines, len(lines))

    # Draw the text for the current page
    y_start = 80  # Start below title bar
    line_height = 45

    for i in range(start_line, end_line):
        draw.text((40, y_start + (i - start_line) * line_height), lines[i], (0, 0, 0), font=regular_font)

    # Update progress bar
    if duration > 0:
        progress_width = int((current_time / duration) * width)
        draw.rectangle([(0, height - 20), (progress_width, height)], fill=(70, 130, 180))

    # Add page number indicator
    page_text = f"Page {current_page + 1}/{total_pages}"
    page_width = get_font_text_width(page_text, regular_font)
    draw.text((width - page_width - 20, height - 45), page_text, (100, 100, 100), font=regular_font)

    return np.array(img)

# --- Audio Generation ---
def generate_audio(text, idx, temp_dir):
    """Generates audio for a given text using gTTS, with special character handling."""
    try:
        # Clean the text for speech synthesis
        speech_text = clean_text_for_speech(text)

        # Generate audio
        tts = gTTS(text=speech_text, lang='en', slow=False)
        filename = os.path.join(temp_dir, f"audio_{idx}.mp3")
        tts.save(filename)

        # Create and return the audio clip
        return filename, AudioFileClip(filename)
    except Exception as e:
        print(f"Audio generation error for paragraph {idx}: {e}")
        return None, None

# --- Video Processing ---
def process_paragraph(args):
    """Processes a single paragraph to create a video clip with paged text and audio."""
    idx, para, temp_dir, prompt, total_duration, start_time, error_queue = args

    try:
        # Generate audio for the paragraph
        audio_file, audio_clip = generate_audio(para, idx, temp_dir)
        if audio_clip is None:
            print(f"Audio generation failed for paragraph {idx}, using fallback duration")
            # Fallback: Estimate duration based on word count
            audio_duration = max(3, len(para.split()) / 150 * 60)  # ~150 words per minute
        else:
            audio_duration = audio_clip.duration

        # Calculate how many pages this paragraph will take
        regular_font, _ = get_system_font()
        words = para.split()
        lines = []
        current_line = []
        max_line_width = 1280 - 80  # Margins

        for word in words:
            test_line = ' '.join(current_line + [word])
            line_width = get_font_text_width(test_line, regular_font)

            if line_width <= max_line_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:  # Add the last line
            lines.append(' '.join(current_line))

        max_lines = 13
        total_pages = (len(lines) + max_lines - 1) // max_lines  # Ceiling division

        # If the paragraph is short enough to fit on one page
        if total_pages == 1:
            # Create a simple clip with the paragraph text
            def make_frame(t):
                return render_text_page(
                    para,
                    t,
                    audio_duration,
                    page_number=0,
                    title=f"{prompt} - Educational Video"
                )

            clip = VideoClip(make_frame, duration=audio_duration)
            if audio_clip:
                clip = clip.set_audio(audio_clip)
            return clip

        # If the paragraph needs multiple pages
        else:
            # Calculate time per page
            time_per_page = audio_duration / total_pages

            # Create a clip that shows different pages over time
            def make_frame(t):
                current_page = min(int(t / time_per_page), total_pages - 1)
                return render_text_page(
                    para,
                    t,
                    audio_duration,
                    page_number=current_page,
                    title=f"{prompt} - Educational Video"
                )

            clip = VideoClip(make_frame, duration=audio_duration)
            if audio_clip:
                clip = clip.set_audio(audio_clip)
            return clip
    except Exception as e:
        error_queue.put((idx, str(e)))
        print(f"Error processing paragraph {idx}: {e}")
        return None

# --- Main Video Generation ---
def generate_video(prompt, output_file="educational_video.mp4", max_workers=3):
    """Generates the complete educational video with text and audio synchronization."""
    print(f"Generating content about: {prompt}")
    start_time = time.time()

    # Initialize model once and reuse
    model = initialize_gemini_model()

    # Create temporary directory for audio files
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Generate content using Gemini
            content = generate_content_gemini(prompt, model)
            if content is None or content.strip() == "":
                print("Content generation failed completely.")
                raise ValueError("Failed to generate any content.")

            # Clean and split the content
            paragraphs = clean_and_split_content(content)

            if not paragraphs:
                print("No paragraphs extracted. Using emergency content.")
                # Emergency content splitting
                words = content.split()
                paragraphs = []
                for i in range(0, len(words), 20):
                    chunk = ' '.join(words[i:i+20])
                    if chunk:
                        paragraphs.append(chunk)

            # Calculate total word count to estimate video length
            total_words = sum(len(p.split()) for p in paragraphs)
            estimated_minutes = total_words / 150  # Assuming 150 words per minute
            estimated_duration = estimated_minutes * 60  # Convert to seconds

            print(f"Content processed: {len(paragraphs)} paragraphs, {total_words} words")
            print(f"Estimated video length: {estimated_minutes:.1f} minutes")

            # Process paragraphs in parallel
            print("Processing video frames and audio...")

            # Calculate estimated start times for each paragraph
            estimated_start_times = [0]
            for i, para in enumerate(paragraphs[:-1]):
                words = len(para.split())
                est_duration = words / 150 * 60  # ~150 words per minute
                estimated_start_times.append(estimated_start_times[-1] + est_duration)

            # Create a queue to track errors
            error_queue = queue.Queue()

            # Create arguments for parallel processing
            args_list = [
                (i, p, temp_dir, prompt, estimated_duration, estimated_start_times[i], error_queue)
                for i, p in enumerate(paragraphs)
            ]

            # Process in batches to avoid memory issues
            num_paragraphs = len(paragraphs)
            batch_size = min(max_workers * 2, num_paragraphs)  # Adjust batch size based on workers
            all_clips = []

            for i in range(0, num_paragraphs, batch_size):
                batch_args = args_list[i:i+batch_size]

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    batch_clips = list(executor.map(process_paragraph, batch_args))

                # Add valid clips to our collection
                all_clips.extend([c for c in batch_clips if c is not None])

                # Check for errors
                while not error_queue.empty():
                    idx, error = error_queue.get()
                    print(f"Error in paragraph {idx}: {error}")

                # Free up memory
                gc.collect()

            if not all_clips:
                raise ValueError("No valid clips generated")

            print(f"Successfully generated {len(all_clips)} video clips")

            # Create final video
            print("Combining clips and rendering final video...")
            final = concatenate_videoclips(all_clips)

            # Add opening title and closing credits
            try:
                # Create opening title
                title_img, title_draw = create_text_frame(width=1280, height=720, title="")
                regular_font, bold_font = get_system_font()
                title_text = f"Educational Video: {prompt}"
                subtitle_text = "Created with StudentHolic AI"

                # Draw title text
                title_y = 300
                title_width = get_font_text_width(title_text, bold_font)
                title_draw.text(((1280 - title_width) // 2, title_y), title_text, (0, 0, 0), font=bold_font)

                # Draw subtitle
                subtitle_y = title_y + 60
                subtitle_width = get_font_text_width(subtitle_text, regular_font)
                title_draw.text(((1280 - subtitle_width) // 2, subtitle_y), subtitle_text, (70, 130, 180), font=regular_font)

                # Create title clip with fade
                title_clip = ImageClip(np.array(title_img)).set_duration(4)
                title_clip = title_clip.fadein(1).fadeout(1)

                # Create closing credits
                credits_img, credits_draw = create_text_frame(width=1280, height=720, title="")
                credits_text = "Thank You for Watching"
                credits_width = get_font_text_width(credits_text, bold_font)
                credits_draw.text(((1280 - credits_width) // 2, 300), credits_text, (0, 0, 0), font=bold_font)

                credits_subtitle = f"Educational Video on {prompt}"
                credits_subtitle_width = get_font_text_width(credits_subtitle, regular_font)
                credits_draw.text(((1280 - credits_subtitle_width) // 2, 360), credits_subtitle, (70, 130, 180), font=regular_font)

                credits_clip = ImageClip(np.array(credits_img)).set_duration(3)
                credits_clip = credits_clip.fadein(1).fadeout(1)

                # Add title and credits to video
                final = concatenate_videoclips([title_clip, final.fadein(0.5), credits_clip])
            except Exception as e:
                print(f"Error creating title/credits: {e}, skipping...")

            # Write to file with optimized settings
            final.write_videofile(
                output_file,
                fps=24,
                threads=max(2, max_workers),
                preset='ultrafast',
                audio_codec='aac',
                bitrate='1500k',
                logger=None
            )

            elapsed_time = time.time() - start_time
            print(f"Video generation completed in {elapsed_time:.1f} seconds")
            print(f"Video saved to: {output_file}")

            return output_file

        except Exception as e:
            print(f"Error in video generation: {e}")
            import traceback
            traceback.print_exc()
            raise

# --- Main Execution ---
def main():
    # Get user input
    user_prompt = input("Enter educational topic: ")
    output_file = f"{user_prompt.replace(' ', '_')}_educational_video.mp4"

    # Determine optimal number of workers based on CPU cores
    optimal_workers = max(1, min(3, multiprocessing.cpu_count() - 1))

    try:
        video_path = generate_video(user_prompt, output_file, max_workers=optimal_workers)
        print(f"Video saved to: {video_path}")
        # Try to display the video if in a notebook environment (like VS Code notebook)
        try:
            display(Video(video_path))
        except:
            print("Video created successfully but cannot be displayed in this environment (if not in a notebook).")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
