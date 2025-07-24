import json
import pathlib
import logging
import subprocess
import os
import re
import csv
import pandas as pd
from pathlib import Path
from faster_whisper import WhisperModel
from openai import OpenAI



# ── CONFIG ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
WHISPER_MODEL = "medium"

TOGETHER_API_KEY = "a53d1d0a1b7ed91fd49d026edf5dd81fb4b52ed4267342748e1f9a577fda861c"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

client = OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz/v1")

def get_youtube_links_sorted_by_length(query, max_results=100):
    search_query = f"ytsearchdate{max_results}:{query}"
    command = [
        "yt-dlp",
        "--dump-json",
        search_query
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout.strip().split("\n")
        videos = []

        for line in output:
            video_data = json.loads(line)
            # Skip if duration is missing (e.g., livestreams)
            if 'duration' in video_data and video_data['duration'] is not None:
                videos.append({
                    'url': video_data['webpage_url'],
                    'title': video_data['title'],
                    'duration': video_data['duration']  # in seconds
                })

        # Sort by video length (ascending)
        videos_sorted = sorted(videos, key=lambda x: x['duration'])

        return videos_sorted

    except subprocess.CalledProcessError as e:
        print("Error running yt-dlp:", e)
        return []

video_urls_file = Path("video_urls.csv")
if video_urls_file.exists():
    logger.info("✅ 'video_urls.csv already exists. Skipping video scraping step.")
    videos = pd.read_csv("video_urls.csv", encoding="utf-8-sig")
    urls = videos["url"].dropna().tolist()
else:
    query = "rubens figueiredo cientista político"
    videos = get_youtube_links_sorted_by_length(query, max_results=100)

    import csv
    
    csv_file = "video_urls.csv"
    
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "title", "duration"])
        writer.writeheader()
        writer.writerows(videos)
    
    print(f"✅ Saved {len(videos)} videos to {csv_file} with UTF-8 BOM.")


    vidURLs = [video['url'] for video in videos]
   
YTDLP_CMD = [
    "yt-dlp",
    "-f", "bestvideo+bestaudio",
    "-o", "test/full.%(ext)s",
]

# ── FUNCTIONS ────────────────────────────────────────────────────────────────
def whisper_transcribe_word_map(audio_path: pathlib.Path):
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(audio_path),
        language="pt",
        beam_size=5,
        word_timestamps=True
    )
    word_map = [
        {"word": w.word, "start": w.start, "end": w.end}
        for seg in segments for w in seg.words
    ]
    out_dir = pathlib.Path.cwd() / "wordMaps"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{audio_path.stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(word_map, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved word map to: {out_file}")
    return word_map

def build_and_save_clean_transcript(audio_path: pathlib.Path):
    word_map_file = pathlib.Path.cwd() / "wordMaps" / f"{audio_path.stem}.json"
    out_dir = pathlib.Path.cwd() / "cleanTranscripts"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{audio_path.stem}.txt"

    with open(word_map_file, "r", encoding="utf-8") as f:
        word_map = json.load(f)

    transcript = " ".join(item["word"] for item in word_map)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(transcript)

    logger.info(f"Saved clean transcript to: {out_file}")
    return transcript

def clean_text(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")
  
def extract_segment_texts(ai_response: str) -> list[str]:
    """
    Tries multiple patterns to robustly extract segment texts from AI response.
    Handles variations like:
    - With or without '###'
    - With or without bold or smart quotes
    """
    patterns = [
        r'### Segment \d+:\s*\n\s*"([^"]+)"',
        r'\*\*Segment \d:\*\*\s*\n\s*"([^"]+)"',
        r'Segment \d+:.*?\n\s*["“](.+?)["”]'
    ]
    for pat in patterns:
        found = re.findall(pat, ai_response)
        if found:
            return found
    logging.warning("⚠️ No segments matched any known pattern.")
    return []


def ask_ai_for_segments(transcript: str) -> str:
    prompt = f"""
    Analyze the transcript and extract up to 3 highly engaging shareable blocks of analysis, suitable for use in standalone video excerpts online.
    
    Rules:
    - Each chosen segment must start and end naturally, beginning at the start of a sentence and ending at the end of a sentence.
    - Chosen segments should have a rationale - a start, middle, and end to their ideas. 
    - Each segment must cover a full argument, idea, or explanation.
    - Avoid "punchlines" or isolated comments. A good segment includes: A setup or background, A key argument or insight, (Ideally) a reflection or implication.
    - Each segment should ideally correspond to approximately 60–120 seconds when spoken (around 70–380 words).
    - Clearly explain why each selected segment would resonate strongly as a social media clip. Your reasoning should highlight engagement value, impact, insightfulness, clarity, or potential controversy.
    - **Do NOT choose short excerpts**. Prioritize completeness and depth. A segment that develops a full argument or idea is preferred over a punchy sentence.
    - Each clip should have clear qualitative value: be insightful, impactful, or potentially controversial.
    - Avoid any repetition across segments — choose distinct, non-overlapping ideas.
    - Provide your response strictly in the following structured format, numbering segments:
    
    Segment 1: "Full text of the chosen segment"
    Reason: One concise sentence justifying its choice.
    
    Segment 2: "Full text of the chosen segment"
    Reason: One concise sentence justifying its choice.
    
    Segment 3: "Full text of the chosen segment"
    Reason: One concise sentence justifying its choice.
    
    Transcript:
    \"\"\"
    {transcript}
    \"\"\"
    """
    
    print("\n Prompt enviado ao DeepSeek:\n" + "="*80)
    print(prompt)
    print("="*80 + "\n")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=3500,
    )
    
    return response.choices[0].message.content

def match_segments_to_timestamps(segments: list[str], word_map: list[dict]) -> list[dict]:
    """
    For each segment, extract the first 3 and last 3 words.
    Search the word_map for those sequences and get the start and end timestamps.
    """
    def normalize(text):
        return re.sub(r"[^\w\s]", "", text).strip().lower()

    # Clean word_map words just once for efficiency
    cleaned_word_map = [normalize(w["word"]) for w in word_map]

    results = []

    for seg in segments:
        words = normalize(seg).split()
        if len(words) < 6:
            logging.warning(f"❗ Segment too short to extract key: '{seg[:50]}…'")
            results.append({"start": None, "end": None, "text": seg})
            continue

        key_start = words[:3]
        key_end = words[-3:]

        # Find start index
        def find_sequence(target_words):
            for i in range(len(cleaned_word_map) - len(target_words) + 1):
                if cleaned_word_map[i:i + len(target_words)] == target_words:
                    return i
            return None

        start_idx = find_sequence(key_start)
        end_idx = find_sequence(key_end)

        if start_idx is not None and end_idx is not None and end_idx + 2 < len(word_map):
            start_ts = word_map[start_idx]["start"]
            end_ts = word_map[end_idx + 2]["end"]  # +2 because key_end has 3 words
            results.append({
                "start": start_ts,
                "end": end_ts,
                "text": seg
            })
        else:
            logging.warning(f"❗ Could not match segment:\n→ {seg[:80]}…")
            results.append({
                "start": None,
                "end": None,
                "text": seg
            })

    return results


def trim_clip(src: Path, dst: Path, start: float, end: float):
    """
    Use ffmpeg to trim a clip from the source video.
    """
    duration = end - start
    cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-i", str(src),
        "-t", str(duration),
        "-c", "copy",
        "-y",  # overwrite if exists
        str(dst)
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode == 0:
        logger.info(f"[SAVED] {dst.name}")
    else:
        logger.warning(f"⚠️ Failed to cut: {dst.name}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_dir = pathlib.Path.cwd() / "test"
    clip_dir = pathlib.Path.cwd() / "clips"
    word_map_dir = pathlib.Path.cwd() / "wordMaps"
    transcript_dir = pathlib.Path.cwd() / "cleanTranscripts"
    ai_output_dir = pathlib.Path.cwd() / "aiResponses"

    # Ensure all output dirs exist
    for d in [test_dir, clip_dir, word_map_dir, transcript_dir, ai_output_dir]:
        d.mkdir(exist_ok=True)

    # Read all video URLs
    with open("video_urls.csv", "r", encoding="utf-8-sig") as f:
          reader = csv.DictReader(f)
          urls = [row["url"].strip() for row in reader if row["url"].strip()]

    for idx, url in enumerate(urls, 1):
        vid_slug = f"video_{idx:03d}"
        video_path = test_dir / f"{vid_slug}.mp4"

        # Download the video
        logger.info(f"📥 Downloading: {url}")
        result = subprocess.run([
            "yt-dlp",
            "-f", "bestvideo+bestaudio",
            "--merge-output-format", "mp4",
            "-o", str(video_path),
            url
        ])
        if result.returncode != 0:
            logger.warning(f"❌ Failed to download {url}")
            continue

        # Transcribe
        logger.info(f"📝 Transcribing: {video_path.name}")
        word_map = whisper_transcribe_word_map(video_path)
        txt = build_and_save_clean_transcript(video_path)

        # Clean and send to LLM
        logger.info(f"🤖 Sending transcript to Together.ai...")
        cleaned_txt = clean_text(txt)
        ai_response = ask_ai_for_segments(cleaned_txt)

        # Save AI output
        ai_output_file = ai_output_dir / f"{vid_slug}_ai.txt"
        with open(ai_output_file, "w", encoding="utf-8") as f:
            f.write(ai_response)

        # Match segments and cut clips
        segment_texts = extract_segment_texts(ai_response)
        matched = match_segments_to_timestamps(segment_texts, word_map)

        for i, clip in enumerate(matched, 1):
            if clip["start"] is None or clip["end"] is None:
                continue
            start = clip["start"]
            end = clip["end"]
            slug = re.sub(r"[^\w\-]+", "_", clip["text"][:50].strip())[:30]
            out_file = clip_dir / f"{vid_slug}_{i:02d}_{slug}.mp4"
            trim_clip(video_path, out_file, start, end)

        # Cleanup
        logger.info(f"🗑️ Removing original video: {video_path}")
        video_path.unlink(missing_ok=True)

        logger.info(f"✅ Finished processing {vid_slug}\n" + "-"*50)


