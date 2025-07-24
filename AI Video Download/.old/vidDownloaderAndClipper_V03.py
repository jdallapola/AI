import os
import subprocess
import re
import pathlib
import shutil
import requests
import csv
import textwrap
import time
import logging
import json
from io import StringIO

from requests.exceptions import ReadTimeout, ConnectionError
from faster_whisper import WhisperModel

# Hide console windows on Windows (used by run_quiet)
CREATE_NO_WINDOW = 0x08000000

def run_quiet(cmd):
    """Run external command quietly (hides output)."""
    return subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
        check=False
    ).returncode == 0

# ── USER CONFIG ─────────────────────────────────────────────────────────────
URLS = [
    "https://www.youtube.com/watch?v=6_m6a0zeLq0",
]
YT_DLP = shutil.which("yt-dlp") or r"C:\Users\nunes\AppData\Roaming\Python\Python313\Scripts\yt-dlp.exe"
FFMPEG = shutil.which("ffmpeg") or r"C:\tools\ffmpeg\bin\ffmpeg.exe"
TMP_DIR = pathlib.Path.cwd() / "temp_audio"
TRANSCRIPT_DIR = pathlib.Path.cwd() / "transcripts"
CLIP_DIR = pathlib.Path.cwd() / "clips"
TMP_DIR.mkdir(exist_ok=True)
TRANSCRIPT_DIR.mkdir(exist_ok=True)
CLIP_DIR.mkdir(exist_ok=True)

# Use built-in Whisper model variants: tiny, base, small, medium, large-v2
WHISPER_MODEL = "medium"

# Mistral (Ollama) config
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "mistral:latest"

# ── LOGGING CONFIG ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ── WHISPER TRANSCRIPTION HELPERS ────────────────────────────────────────────

def download_audio(url: str, dest: pathlib.Path) -> bool:
    """Download the best audio using yt-dlp."""
    return run_quiet([YT_DLP, "-f", "bestaudio", "-o", str(dest), url])


def whisper_transcribe(audio_path: pathlib.Path):
    """Transcribe audio with Whisper locally, returning timestamped segments."""
    try:
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="float16")
    except ValueError:
        logger.warning("float16 not supported; falling back to int8")
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(audio_path),
        language="pt",
        beam_size=5,
        word_timestamps=True
    )
    return [{"start": s.start, "end": s.end, "text": s.text} for s in segments]


def write_srt(segments, srt_path: pathlib.Path):
    """Write Whisper segments to an SRT file with proper timing."""
    def fmt_time(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        ms = int((sec - int(sec)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            start = fmt_time(seg["start"])
            end = fmt_time(seg["end"])
            text = seg["text"].strip().replace("\n", " ")
            f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")

# ── MISTRAL CLIP SELECTION HELPERS ───────────────────────────────────────────

def fix_timestamp(ts: str, max_seconds: int) -> str:
    """Ensure HH:MM:SS doesn't exceed video length."""
    h, m, s = [int(x) for x in ts.split(":")]
    total = h*3600 + m*60 + s
    total = min(total, max_seconds-1)
    h = total//3600
    m = (total % 3600)//60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def trim_clip(src: pathlib.Path, dst: pathlib.Path, start: str, end: str):
    """Trim a video clip from src to dst using ffmpeg."""
    run_quiet([
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-ss", start, "-to", end,
        "-i", str(src), "-c", "copy", str(dst)
    ])




def ask_mistral_complete(text, max_chars=8000, retries=1, timeout=240):
    """
    Identify up to 3 concise, punchy, substance-rich clips ideal for social media.

    Criteria:
    - Focus on political science insights (Rubens is a political scientist); humor allowed if it boosts engagement.
    - Select 1–3 standout clips based on qualitative value; allow fewer if fewer truly stand out.
    - Each clip should deliver a complete thought: start at a sentence beginning and end at its conclusion.
    - Reason must be a concise hook sentence explaining why the chosen clip is engaging.
    - Output ONLY a TSV with header 'start\tend\treason'; tabs only, no commas or quotes.
    - Timestamp format: HH:MM:SS.
    """
    system_msg = (
        "You are a social-media editor.\n"
        "Analyze the transcript and choose the 1–3 moments that are most concise, punchy, and rich in content—perfect for social media.\n"
        "Each clip must start at a full-sentence beginning and end at a full-sentence ending.\n"
        "The reason field should be a concise hook explaining why the chosen clip is engaging.\n"
        "Output ONLY a TSV with header 'start\tend\treason' and the clip rows, using tabs only.\n"
    )
    prompt = textwrap.shorten(text, max_chars, placeholder=" […]")

    body = {
        "model": MODEL_NAME,
        "system": system_msg,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "format": "text"}
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=body, timeout=timeout)
            outer = resp.json()
            resp_text = outer.get("response", "").lstrip()
            print(f"\n--- DEBUG INNER RESPONSE (attempt {attempt}) ---")
            print(resp_text)
            print("--- END DEBUG ---\n")

            lines = resp_text.splitlines()
            if not lines or '	' not in lines[0]:
                raise ValueError(f"Invalid TSV header: {lines[0] if lines else ''}")

            reader = csv.reader(StringIO(resp_text), delimiter='\t')
            rows = [[cell.strip() for cell in row] for row in reader]
            header = [h.lower() for h in rows[0]]
            if header != ['start', 'end', 'reason']:
                raise ValueError(f"Bad header: {rows[0]}")
            count = len(rows) - 1
            if count < 1 or count > 3:
                raise ValueError(f"Expected 1-3 clips, got {count}")

            clips = []
            for row in rows[1:]:
                start, end, reason = row
                if not re.match(r"^\d{2}:\d{2}:\d{2}$", start) or not re.match(r"^\d{2}:\d{2}:\d{2}$", end):
                    raise ValueError(f"Invalid timestamp: {row}")
                clips.append({"start": start, "end": end, "reason": reason})
            return clips
        except Exception as e:
            print(f"⚠️ attempt {attempt}: {e}; retrying…")
            body['system'] += "\nEnsure TSV format, tabs only, sentence boundaries, and 1-3 clips."
            time.sleep(1)

    print("❌ mistral failed to supply valid TSV after retries.")
    return None


# ── MAIN EXECUTION ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    for url in URLS:
        vid = re.search(r"(?:v=|be/)([A-Za-z0-9_-]{11})", url).group(1)
        audio_file = TMP_DIR / f"{vid}.m4a"
        transcript_file = TRANSCRIPT_DIR / f"{vid}.srt"
        video_file = TMP_DIR / f"{vid}.mp4"

        logger.info(f"Processing video {vid}")

        # Download audio + video
        if not audio_file.exists():
            logger.info(f"Downloading audio for {vid}")
            if not download_audio(url, audio_file):
                logger.error(f"Audio download failed for {vid}")
                continue
        if not video_file.exists():
            logger.info(f"Downloading video for {vid}")
            run_quiet([YT_DLP, "-f", "mp4", "-o", str(video_file), url])

        # Handle transcript
        if transcript_file.exists():
            resp = input(f"Use existing transcript for {vid}? [y/N]: ")
            if resp.strip().lower().startswith('y'):
                logger.info(f"Loading existing transcript for {vid}")
                text_lines = []
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.isdigit():
                            continue
                        text_lines.append(line)
                transcript_text = " ".join(text_lines)
                logger.info(f"Loaded transcript text ({len(transcript_text)} chars)")
            else:
                logger.info(f"Re-transcribing audio for {vid}")
                segments = whisper_transcribe(audio_file)
                write_srt(segments, transcript_file)
                transcript_text = " ".join(s['text'] for s in segments)
        else:
            logger.info(f"Transcribing audio for {vid}")
            segments = whisper_transcribe(audio_file)
            write_srt(segments, transcript_file)
            transcript_text = " ".join(s['text'] for s in segments)
           
      
        # Ask Mistral for clips
        clips = ask_mistral_complete(transcript_text)
        if not clips:
            logger.warning(f"No clips returned for {vid}")
            continue

        # Trim and save clips
        duration = int(segments[-1]["end"]) + 1
        for idx_clip, clip in enumerate(clips, start=1):
            start = fix_timestamp(clip["start"], duration)
            end = fix_timestamp(clip["end"], duration)
            dst = CLIP_DIR / f"{vid}_{idx_clip}_{start.replace(':','-')}_{end.replace(':','-')}.mp4"
            trim_clip(video_file, dst, start, end)
            logger.info(f"Saved clip: {dst}")

    logger.info("All done.")
