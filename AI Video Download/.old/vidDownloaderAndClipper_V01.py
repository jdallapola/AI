# vidDownloaderAndClipper.py
#
# 1.  Get YouTube captions (or Whisper fallback)
# 2.  Ask phi3 (Ollama) for up‑to‑two 30–60 s clip suggestions (JSON)
# 3.  Trim those segments with ffmpeg
# 4.  Write MP4s to ./clips/

import os, subprocess, json, re, pathlib, textwrap, shutil, requests
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from faster_whisper import WhisperModel

# ── USER LIST ────────────────────────────────────────────────────────────────
URLS = [
    "https://www.youtube.com/watch?v=6_m6a0zeLq0",
    "https://www.youtube.com/watch?v=UHZDGvTXsLw",
]

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent
YT_DLP   = shutil.which("yt-dlp")  or r"C:\Users\nunes\AppData\Roaming\Python\Python313\Scripts\yt-dlp.exe"
FFMPEG   = shutil.which("ffmpeg")  or r"C:\tools\ffmpeg\bin\ffmpeg.exe"
WHISPER_MODEL = r"C:\tools\whisper\ggml-base.bin"   # whisper.cpp model
OLLAMA_URL   = "http://localhost:11434/api/generate"
MODEL_NAME   = "phi3"

TMP_DIR  = BASE_DIR / "temp_audio"
OUT_DIR  = BASE_DIR / "clips"
TMP_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ── QUIET SUBPROCESS (avoid WinError 6) ───────────────────────────────────────
CREATE_NO_WINDOW = 0x08000000  # hide console windows on Windows

def run_quiet(cmd):
    return subprocess.run(
        cmd,
        stdin = subprocess.DEVNULL,
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
        creationflags = CREATE_NO_WINDOW if os.name == "nt" else 0,
        check = False
    ).returncode == 0

# ── UTILS ─────────────────────────────────────────────────────────────────────
def video_id(url):
    m = re.search(r"(?:v=|be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None

def fetch_auto_captions(v_id, langs=("pt", "en")):
    try:
        return YouTubeTranscriptApi.get_transcript(v_id, languages=list(langs))
    except (TranscriptsDisabled, NoTranscriptFound):
        return None

def download_audio(url, dest):
    return run_quiet([YT_DLP, "-f", "bestaudio", "-o", str(dest), url])

def whisper_transcribe(audio_path):
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segs, _ = model.transcribe(str(audio_path), language="pt")
    return [{"start": s.start, "end": s.end, "text": s.text} for s in segs]

def trim_clip(src, dst, start, end):
    run_quiet([FFMPEG, "-hide_banner", "-loglevel", "error",
               "-ss", start, "-to", end, "-i", str(src),
               "-c", "copy", str(dst)])

# fix weird timestamps (HH may actually be MM)
def fix_timestamp(ts, max_sec):
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 2:                   # MM:SS
        parts = [0] + parts
    h, m, s = parts
    sec = h*3600 + m*60 + s
    if h > 0 and sec > max_sec:           # treat HH as minutes
        sec = (h*60 + m)*60 + s
    sec = min(sec, max_sec-1)
    return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"

# ── ASK PHI3 ──────────────────────────────────────────────────────────────────
def ask_phi3(transcript, max_chars=1500, timeout=180):
    system_msg = (
        "You are a social‑media editor. "
        "Choose up to TWO 30‑60‑second segments (HH:MM:SS format, HH=00) that will be most engaging on TikTok."
    )
    example = '{"clips":[{"start":"00:48:00","end":"01:15:00","reason":"Vivid overview"}]}'
    prompt_body = textwrap.shorten(transcript, max_chars, placeholder=" …[truncated]")

    body = {
        "model": MODEL_NAME,
        "system": system_msg,
        "prompt": (
            "Return ONLY valid JSON following the example below.\n"
            f"EXAMPLE:\n{example}\n\nTRANSCRIPT:\n{prompt_body}"
        ),
        "temperature": 0.4,
        "stream": False,
    }

    try:
        r = requests.post(OLLAMA_URL, json=body, timeout=timeout)
        r.raise_for_status()
        raw = r.json()["response"]
        print(f"--- {MODEL_NAME} raw ---\n", raw[:250], "…")
        match = re.search(r"\{.*\}", raw, re.S)
        return json.loads(match.group(0)).get("clips", []) if match else []
    except Exception as e:
        print(f"⚠️  {MODEL_NAME} error ({e}); no clips.")
        return []

# ── MAIN PIPELINE ────────────────────────────────────────────────────────────
for idx, url in enumerate(URLS, 1):
    vid = video_id(url)
    if not vid:
        print(f"❌ Invalid URL: {url}")
        continue

    caps = fetch_auto_captions(vid)
    if caps:
        print(f"✔ Captions found for {vid}")
    else:
        audio = TMP_DIR / f"{vid}.m4a"
        if not audio.exists() and not download_audio(url, audio):
            print(f"❌ Audio download failed for {vid}")
            continue
        print(f"ℹ Whisper transcribing {vid} …")
        caps = whisper_transcribe(audio)

    if not caps:
        print(f"❌ No transcript for {vid}")
        continue

    transcript = " ".join(c["text"] for c in caps)
    clips = ask_phi3(transcript)
    if not clips:
        print(f"❌ {MODEL_NAME} returned no clips for {vid}")
        continue

    video_file = TMP_DIR / f"{vid}.mp4"
    if not video_file.exists():
        run_quiet([YT_DLP, "-f", "mp4", "-o", str(video_file), url])

    video_len = int(caps[-1]["end"]) + 1

    for n, clip in enumerate(clips, 1):
        start = fix_timestamp(clip["start"], video_len)
        end   = fix_timestamp(clip["end"],   video_len)
        if start >= end:
            print(f"⚠️ Skip malformed clip {start}->{end}")
            continue

        dst = OUT_DIR / f"{idx:02d}-{n}_{vid}_{start.replace(':','-')}-{end.replace(':','-')}.mp4"
        trim_clip(video_file, dst, start, end)
        print(f"[SAVED] {dst}")

print("\nDONE ✔")
