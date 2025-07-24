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

from io import StringIO
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
    "https://www.youtube.com/watch?v=UHZDGvTXsLw",
]

# 
# URLS = [
#         "https://www.youtube.com/watch?v=cZqP6ESiHE4",
#         "https://www.youtube.com/watch?v=3gpAtG2Tkr4",
#         "https://www.youtube.com/watch?v=6_m6a0zeLq0",
#         "https://www.youtube.com/watch?v=UHZDGvTXsLw",
#         "https://www.youtube.com/watch?v=RZHVTTtK6D0",
#         "https://www.youtube.com/watch?v=qTnc8GXe0BY",
#         "https://www.youtube.com/watch?v=eCv0Pe9nn9U",
#         "https://www.youtube.com/watch?v=_KjoH9JqEcU",
#         "https://www.youtube.com/watch?v=2NgBRi4LnQk",
#         "https://www.youtube.com/watch?v=xbHDJtNecmA",
#     ]


YT_DLP = shutil.which("yt-dlp") or r"C:\Users\nunes\AppData\Roaming\Python\Python313\Scripts\yt-dlp.exe"
FFMPEG = shutil.which("ffmpeg") or r"C:\tools\ffmpeg\bin\ffmpeg.exe"
TMP_DIR = pathlib.Path.cwd() / "temp_audio"
TRANSCRIPT_DIR = pathlib.Path.cwd() / "transcripts"
CLIP_DIR = pathlib.Path.cwd() / "clips"
TMP_DIR.mkdir(exist_ok=True)
TRANSCRIPT_DIR.mkdir(exist_ok=True)
CLIP_DIR.mkdir(exist_ok=True)

# Use built-in model (tiny, base, small, medium, large-v2)
WHISPER_MODEL = "base"

# Mistral config
oLLAMA_URL = "http://localhost:11434/api/generate"
MISTRAL_MODEL = "mistral"

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
    """Transcribe audio locally with Whisper, returning timestamped segments."""
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), language="pt")
    return [{"start": s.start, "end": s.end, "text": s.text} for s in segments]


def write_srt(segments, srt_path: pathlib.Path):
    """Write segments to an SRT file with proper timing."""
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
    """Normalize HH:MM:SS to avoid overshooting video length."""
    h, m, s = [int(x) for x in ts.split(":")]
    total = h*3600 + m*60 + s
    total = min(total, max_seconds-1)
    h = total//3600
    m = (total%3600)//60
    s = total%60
    return f"{h:02d}:{m:02d}:{s:02d}"


def trim_clip(src: pathlib.Path, dst: pathlib.Path, start: str, end: str):
    run_quiet([
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-ss", start, "-to", end,
        "-i", str(src), "-c", "copy", str(dst)
    ])


def ask_mistral_complete(text: str, max_chars=3000, retries=3, timeout=240):
    """
    Ask Mistral for up to 3 clips: start<tab>end<tab>reason
    """
    system_msg = (
        "You are a professional social-media video editor.\n"
        "Select up to 3 self-contained clips from the INTERVIEWEE only.\n"
        "Clips 30-90s (120s max if needed), full sentences.\n"
        "Return TSV: start<tab>end<tab>reason.\n"
    )
    prompt = textwrap.shorten(text, max_chars, placeholder="[…]")
    body = {"model": MISTRAL_MODEL, "system": system_msg, "prompt": prompt, "stream": False, "options": {"temperature":0.1}}

    for i in range(retries):
        resp = requests.post(oLLAMA_URL, json=body, timeout=timeout)
        data = resp.json().get("response","").lstrip()
        lines = data.splitlines()
        if lines and '\t' in lines[0]:
            clips = []
            for row in csv.reader(lines, delimiter='\t'):
                if row and re.match(r"^\d{2}:\d{2}:\d{2}$", row[0]):
                    clips.append({"start":row[0],"end":row[1],"reason":row[2]})
            if clips:
                return clips
        body["system"] += "\nEnsure TSV format and logical clips."
        time.sleep(1)
    return []

# ── MAIN EXECUTION ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    for url in URLS:
        vid = re.search(r"(?:v=|be/)([A-Za-z0-9_-]{11})", url).group(1)
        audio_file = TMP_DIR / f"{vid}.m4a"
        transcript_file = TRANSCRIPT_DIR / f"{vid}.srt"
        video_file = TMP_DIR / f"{vid}.mp4"

        logger.info(f"Processing {vid}")

        # Download audio + video
        if not audio_file.exists():
            download_audio(url, audio_file)
        if not video_file.exists():
            run_quiet([YT_DLP, "-f", "mp4", "-o", str(video_file), url])

        # Handle transcript
        if transcript_file.exists():
            resp = input(f"Use existing transcript for {vid}? [y/N]: ")
            if resp.lower().startswith('y'):
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript_text = f.read()
            else:
                segments = whisper_transcribe(audio_file)
                write_srt(segments, transcript_file)
                transcript_text = " ".join(s['text'] for s in segments)
        else:
            segments = whisper_transcribe(audio_file)
            write_srt(segments, transcript_file)
            transcript_text = " ".join(s['text'] for s in segments)

        # Ask Mistral for clips
        clips = ask_mistral_complete(transcript_text)
        if not clips:
            logger.warning(f"No clips returned for {vid}")
            continue

        # Trim and save clips
        duration = int(segments[-1]['end']) + 1
        for idx, clip in enumerate(clips, start=1):
            start = fix_timestamp(clip['start'], duration)
            end = fix_timestamp(clip['end'], duration)
            dst = CLIP_DIR / f"{vid}_{idx}_{start.replace(':','-')}_{end.replace(':','-')}.mp4"
            trim_clip(video_file, dst, start, end)
            logger.info(f"Saved clip: {dst}")

    logger.info("All done.")
