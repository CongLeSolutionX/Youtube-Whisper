import os
import re
import warnings
import gradio as gr
import whisper
from download_video import download_mp3_yt_dlp

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Global model cache
models = {}

def is_valid_youtube_url(url):
    """
    Validates whether the provided URL is a valid YouTube link.
    """
    youtube_regex = (
        r"^(https?://)?(www\.)?(youtube\.com|youtu\.?be)/.+$"
    )
    return re.match(youtube_regex, url) is not None

def download_video_info(youtube_url):
    """
    Downloads audio from the YouTube URL and retrieves video information.

    Parameters:
    - youtube_url (str): The URL of the YouTube video.

    Returns:
    - audio_path (str): Path to the downloaded audio file.
    - title (str): Title of the YouTube video.
    - thumbnail_url (str): URL of the video thumbnail.
    - error_message (str): Error message if any issues occur.
    """
    if not is_valid_youtube_url(youtube_url):
        return None, "Invalid YouTube URL.", None

    try:
        audio_path, title, thumbnail_url = download_mp3_yt_dlp(youtube_url)
        if not audio_path:
            return None, "Failed to download audio.", None
        return audio_path, title, thumbnail_url
    except Exception as e:
        print(f"Error downloading video info: {e}")
        return None, "An error occurred while downloading the video.", None

def transcribe_audio(audio_path, model_size="base", language=None):
    """
    Transcribes the audio file using the specified Whisper model.

    Parameters:
    - audio_path (str): Path to the audio file.
    - model_size (str): Size of the Whisper model to use.
    - language (str): Language code for transcription.

    Returns:
    - transcription (str): The transcribed text.
    """
    try:
        if model_size not in models:
            models[model_size] = whisper.load_model(model_size)
        model = models[model_size]

        options = {}
        if language:
            options['language'] = language

        result = model.transcribe(audio_path, **options)
        return result['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        return "An error occurred during transcription."
def get_video_info_and_transcribe(youtube_url, model_size="base", language=None, progress=gr.Progress()):
    """
    Fetches video info and transcribes audio, updating progress.
    """
    progress(0, "Validating URL...")
    audio_path, title, thumbnail_url = download_video_info(youtube_url)
    if not audio_path:
        return title, None, "", None

    progress(50, "Transcribing audio...")
    transcription = transcribe_audio(audio_path, model_size, language)

    # Save transcription to a file for download
    transcription_file = "transcription.txt"
    with open(transcription_file, "w", encoding="utf-8") as f:
        f.write(transcription)

    # Clean up downloaded audio file
    if os.path.exists(audio_path):
        os.remove(audio_path)

    progress(100, "Done")
    return title, thumbnail_url, transcription, transcription_file

with gr.Blocks() as app:
    gr.Markdown("# YouTube Video Transcriber")
    gr.Markdown("Transcribe YouTube videos using OpenAI's Whisper model.")

    with gr.Row():
        youtube_url = gr.Textbox(
            label="YouTube URL",
            placeholder="Enter the YouTube video URL here",
            info="Supports standard YouTube links (e.g., https://www.youtube.com/watch?v=VIDEO_ID)."
        )

    with gr.Row():
        model_size = gr.Dropdown(
            choices=["tiny", "base", "small", "medium", "large"],
            value="base",
            label="Model Size",
            info="Select the Whisper model size. Larger models are more accurate but require more resources."
        )
        language = gr.Textbox(
            label="Language (Optional)",
            placeholder="e.g., 'en' for English",
            info="Specify the language code. Leave empty for auto-detection."
        )

    transcribe_button = gr.Button("Transcribe")

    with gr.Row():
        title_output = gr.Textbox(
            label="Video Title",
            interactive=False
        )

    thumbnail_output = gr.Image(
        label="Thumbnail",
        interactive=False
    )

    transcription_output = gr.Textbox(
        label="Transcription",
        lines=10,
        max_lines=30,
        interactive=False
    )

    download_button = gr.File(label="Download Transcription")

    transcribe_button.click(
        fn=get_video_info_and_transcribe,
        inputs=[youtube_url, model_size, language],
        outputs=[title_output, thumbnail_output, transcription_output, download_button]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)