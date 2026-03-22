"""
process.py — YouTube Shorts pipeline with lip-movement speaker tracking
Usage:
    python process.py video.mp4 --clips "1:20-2:10, 5:44-6:30"

Requirements:
    pip install faster-whisper pillow moviepy numpy opencv-python mediapipe==0.10.9
    FFmpeg must be installed and in PATH
    Download Bangers-Regular.ttf from Google Fonts and place in same folder
"""

import argparse
import os
import subprocess
import sys
import re
import collections

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy import VideoFileClip, CompositeVideoClip, ImageClip
from faster_whisper import WhisperModel


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OUTPUT_DIR       = "output_clips"
FONT_PATH        = "Bangers-Regular.ttf"
FONT_SIZE        = 90
OUTPUT_WIDTH     = 1080
OUTPUT_HEIGHT    = 1920
WORDS_PER_SUB    = 3
WHISPER_MODEL    = "base"
SUBTITLE_Y_RATIO = 0.82

# Camera tracking
ZOOM_FACTOR      = 1.00   # zoom into speaker (1.0 = no zoom)
SLIDE_SMOOTHING  = 0.04   # pan smoothness — lower = slower (0.0–1.0)
FACE_SAMPLE_FPS  = 6      # frames per second to sample (higher = more accurate)

# Lip movement detection
LIP_OPEN_THRESHOLD  = 0.018  # min lip ratio to count as "open"
LIP_HISTORY_FRAMES  = 6      # how many frames to average lip movement over
LIP_MOVEMENT_MIN    = 0.004  # min variance to count as "talking"


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def parse_timestamps(clips_str):
    results = []
    for entry in [e.strip() for e in clips_str.split(",")]:
        match = re.match(r"(\d+:\d+(?::\d+)?)-(\d+:\d+(?::\d+)?)", entry)
        if not match:
            print(f"[WARN] Skipping invalid timestamp: {entry}")
            continue
        results.append((ts_to_sec(match.group(1)), ts_to_sec(match.group(2))))
    return results


def ts_to_sec(ts):
    parts = list(map(int, ts.split(":")))
    if len(parts) == 2: return parts[0] * 60 + parts[1]
    if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0


def sec_to_ts(s):
    h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
    return f"{h:02}:{m:02}:{sec:06.3f}" if h > 0 else f"{m:02}:{sec:06.3f}"


def get_video_info(path):
    cap    = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, w, h, frames


def is_vertical(w, h):
    return h > w


# ─────────────────────────────────────────────
# STEP 1 — CUT
# ─────────────────────────────────────────────

def cut_clip(input_path, start, end, output_path):
    print(f"  ✂  Cutting {sec_to_ts(start)} → {sec_to_ts(end)}")
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start), "-to", str(end),
        "-i", input_path,
        "-c:v", "libx264", "-c:a", "aac",
        "-avoid_negative_ts", "make_zero", output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ─────────────────────────────────────────────
# STEP 2 — LIP MOVEMENT SPEAKER TRACKING
# ─────────────────────────────────────────────

# MediaPipe FaceMesh lip landmark indices
# Upper lip top: 13, Lower lip bottom: 14 (inner), outer: 0, 17
UPPER_LIP = 13
LOWER_LIP = 14
FACE_TOP   = 10   # forehead landmark for face height reference
FACE_BOT   = 152  # chin landmark

def get_lip_openness(landmarks, w, h):
    """
    Returns lip openness ratio = lip gap / face height.
    Normalized so it's scale-independent.
    """
    upper = landmarks[UPPER_LIP]
    lower = landmarks[LOWER_LIP]
    top   = landmarks[FACE_TOP]
    bot   = landmarks[FACE_BOT]

    lip_gap   = abs(lower.y - upper.y) * h
    face_h    = abs(bot.y - top.y) * h
    if face_h < 1: return 0.0
    return lip_gap / face_h


def get_face_center(landmarks, w, h):
    """Returns (cx, cy) of face in pixel coordinates."""
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return np.mean(xs), np.mean(ys)


def detect_speaker_over_time(video_path, vid_w, vid_h, fps):
    """
    For each sampled frame, detect all faces + their lip openness.
    Returns dict: { time: (cx, cy) } of the most active speaking face.
    Falls back to largest face if no lip movement detected.
    """
    cap          = cv2.VideoCapture(video_path)
    sample_every = max(1, int(fps / FACE_SAMPLE_FPS))

    mp_mesh  = mp.solutions.face_mesh
    speaker_data = {}

    # Per-face lip history: track lip openness over recent frames
    # Key = face_id (approximate cluster), value = deque of openness values
    face_histories = {}  # face_idx → deque

    with mp_mesh.FaceMesh(
        max_num_faces=6,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as mesh:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            if frame_idx % sample_every == 0:
                t   = frame_idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = mesh.process(rgb)

                if res.multi_face_landmarks:
                    best_cx, best_cy = vid_w / 2, vid_h / 2
                    best_score = -1

                    for fi, face_lms in enumerate(res.multi_face_landmarks):
                        lms      = face_lms.landmark
                        openness = get_lip_openness(lms, vid_w, vid_h)
                        cx, cy   = get_face_center(lms, vid_w, vid_h)

                        # Track history per face slot
                        if fi not in face_histories:
                            face_histories[fi] = collections.deque(
                                maxlen=LIP_HISTORY_FRAMES)
                        face_histories[fi].append(openness)

                        # Score = variance of lip openness (moving lips = talking)
                        history = face_histories[fi]
                        if len(history) >= 2:
                            score = np.var(list(history))
                        else:
                            score = openness  # fallback

                        if score > best_score:
                            best_score = score
                            best_cx, best_cy = cx, cy

                    speaker_data[t] = (best_cx, best_cy)

            frame_idx += 1

    cap.release()
    return speaker_data


def build_camera_path(speaker_data, total_frames, fps, default_cx, default_cy):
    """Smooth lerp camera path toward active speaker for every frame."""
    times = sorted(speaker_data.keys())

    def speaker_at(t):
        if not times: return default_cx, default_cy
        closest = min(times, key=lambda x: abs(x - t))
        return speaker_data[closest] if abs(closest - t) < 1.5 else (default_cx, default_cy)
        
    path = []
    cx, cy = default_cx, default_cy
    for i in range(total_frames):
        tcx, tcy = speaker_at(i / fps)
        target_cx = (tcx + default_cx) / 2
        target_cy = (tcy + default_cy) / 2
        cx = cx + (target_cx - cx) * SLIDE_SMOOTHING
        cy = cy + (target_cy - cy) * SLIDE_SMOOTHING
        # cx = cx + (tcx - cx) * SLIDE_SMOOTHING
        # cy = cy + (tcy - cy) * SLIDE_SMOOTHING
        path.append((cx, cy))
    return path


def crop_frame_landscape(frame, cx, cy, vid_w, vid_h):
    """16:9 → zoom + crop to 9:16 centered on speaker."""
    crop_w = int(vid_w / ZOOM_FACTOR)
    crop_h = int(crop_w * OUTPUT_HEIGHT / OUTPUT_WIDTH)
    if crop_h > vid_h:
        crop_h = vid_h
        crop_w = int(crop_h * OUTPUT_WIDTH / OUTPUT_HEIGHT)
    x1 = max(0, min(int(cx - crop_w / 2), vid_w - crop_w))
    y1 = max(0, min(int(cy - crop_h / 2), vid_h - crop_h))
    cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
    return cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                      interpolation=cv2.INTER_LINEAR)


def crop_frame_vertical(frame, cx, cy, vid_w, vid_h):
    """9:16 → zoom in toward speaker, no aspect ratio change."""
    crop_w = int(vid_w / ZOOM_FACTOR)
    crop_h = int(vid_h / ZOOM_FACTOR)
    x1 = max(0, min(int(cx - crop_w / 2), vid_w - crop_w))
    y1 = max(0, min(int(cy - crop_h / 2), vid_h - crop_h))
    cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
    return cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                      interpolation=cv2.INTER_LINEAR)


def apply_camera_tracking(input_path, output_path):
    """Render every frame with smooth lip-aware speaker tracking."""
    fps, vid_w, vid_h, total_frames = get_video_info(input_path)
    vertical = is_vertical(vid_w, vid_h)

    print(f"  🎥 Source: {vid_w}×{vid_h} ({'vertical' if vertical else 'landscape'})")
    print(f"  👄 Detecting lip movement to find active speaker...")
    speaker_data = detect_speaker_over_time(input_path, vid_w, vid_h, fps)
    print(f"  ✔  Tracked speaker at {len(speaker_data)} sample points")

    camera_path = build_camera_path(
        speaker_data, total_frames, fps, vid_w / 2, vid_h / 2)

    print(f"  📐 Rendering {total_frames} frames...")
    cap    = cv2.VideoCapture(input_path)
    tmp    = output_path.replace(".mp4", "_noaudio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp, fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cx, cy = camera_path[min(idx, len(camera_path) - 1)]
        if vertical:
            out = crop_frame_vertical(frame, cx, cy, vid_w, vid_h)
        else:
            out = crop_frame_landscape(frame, cx, cy, vid_w, vid_h)
        writer.write(out)
        idx += 1

    cap.release()
    writer.release()

    subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp, "-i", input_path,
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(tmp)


# ─────────────────────────────────────────────
# STEP 3 — TRANSCRIBE
# ─────────────────────────────────────────────

def transcribe(video_path, model):
    print(f"  🎙  Transcribing with Whisper...")
    segments, _ = model.transcribe(video_path, word_timestamps=True)
    words = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append({"word": w.word.strip().upper(),
                               "start": w.start, "end": w.end})
    return words


# ─────────────────────────────────────────────
# STEP 4 — KARAOKE SUBTITLES
# ─────────────────────────────────────────────

HIGHLIGHT_COLOR = (0, 255, 0, 255)
NORMAL_COLOR    = (255, 255, 255, 255)
STROKE_COLOR    = (0, 0, 0, 255)
STROKE_WIDTH    = 10
SHADOW_OFFSET   = 5
SHADOW_BLUR     = 8
WORD_SPACING    = 18


def measure_word(draw, word, font):
    bb = draw.textbbox((0, 0), word, font=font)
    return bb[2] - bb[0], bb[3] - bb[1]


def draw_word_styled(img, word, x, y, font, fill_color):
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.text((x + SHADOW_OFFSET, y + SHADOW_OFFSET), word, font=font,
            fill=(0, 0, 0, 130))
    shadow = shadow.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))
    img = Image.alpha_composite(img, shadow)
    draw = ImageDraw.Draw(img)
    for dx in range(-STROKE_WIDTH, STROKE_WIDTH + 1, 2):
        for dy in range(-STROKE_WIDTH, STROKE_WIDTH + 1, 2):
            if dx == 0 and dy == 0: continue
            draw.text((x + dx, y + dy), word, font=font, fill=STROKE_COLOR)
    draw.text((x, y), word, font=font, fill=fill_color)
    return img


def draw_karaoke_frame(word_list, highlight_idx, frame_size, font):
    img   = Image.new("RGBA", frame_size, (0, 0, 0, 0))
    dummy = ImageDraw.Draw(img)
    sizes = [measure_word(dummy, w, font) for w in word_list]
    total_w = sum(s[0] for s in sizes) + WORD_SPACING * (len(word_list) - 1)
    max_h   = max(s[1] for s in sizes)
    x = (frame_size[0] - total_w) // 2
    y = int(frame_size[1] * SUBTITLE_Y_RATIO) - max_h // 2
    for i, word in enumerate(word_list):
        color = HIGHLIGHT_COLOR if i == highlight_idx else NORMAL_COLOR
        img = draw_word_styled(img, word, x, y, font, color)
        x += sizes[i][0] + WORD_SPACING
    return img


def build_karaoke_events(words, n=WORDS_PER_SUB):
    events = []
    for i in range(0, len(words), n):
        chunk = words[i:i+n]
        texts = [w["word"] for w in chunk]
        for j, w in enumerate(chunk):
            start = w["start"]
            end   = chunk[j+1]["start"] if j+1 < len(chunk) else w["end"]
            if end - start < 0.05: end = start + 0.1
            events.append({"words": texts, "highlight_idx": j,
                            "start": start, "end": end})
    return events


def burn_subtitles(video_path, words, output_path):
    print(f"  💬 Burning subtitles...")
    if not words:
        import shutil; shutil.copy(video_path, output_path); return

    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except IOError:
        print(f"  [WARN] Font not found, using default.")
        font = ImageFont.load_default()

    clip   = VideoFileClip(video_path)
    W, H   = clip.size
    events = build_karaoke_events(words)

    sub_clips = []
    for ev in events:
        sub_img  = draw_karaoke_frame(ev["words"], ev["highlight_idx"], (W, H), font)
        sub_arr  = np.array(sub_img)
        duration = ev["end"] - ev["start"]
        if duration <= 0: continue
        sub_clips.append(
            ImageClip(sub_arr)
            .with_start(ev["start"])
            .with_duration(duration)
        )

    final = CompositeVideoClip([clip] + sub_clips)
    final.write_videofile(output_path, codec="libx264",
                          audio_codec="aac", fps=clip.fps, logger=None)
    clip.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YouTube Shorts pipeline")
    parser.add_argument("video", help="Input video file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--clips",
                       help='Manual timestamps e.g. "1:20-2:10, 5:44-6:30"')
    group.add_argument("--json",
                       help="Gemini highlights JSON e.g. highlights.json")
    args = parser.parse_args()
 
    if not os.path.exists(args.video):
        print(f"[ERROR] File not found: {args.video}"); sys.exit(1)
 
    if args.json:
        if not os.path.exists(args.json):
            print(f"[ERROR] JSON not found: {args.json}"); sys.exit(1)
        clips = load_json_clips(args.json)
        print(f"📋 Loaded {len(clips)} clips from {args.json}")
    else:
        timestamps = parse_timestamps(args.clips)
        if not timestamps:
            print("[ERROR] No valid timestamps."); sys.exit(1)
        clips = [{"start": s, "end": e, "title": f"clip_{i:02d}"}
                 for i, (s, e) in enumerate(timestamps, 1)]
 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    print(f"\n🎬 Loading Whisper model ({WHISPER_MODEL})...")
    whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
 
    for i, clip in enumerate(clips, 1):
        start = clip["start"]
        end   = clip["end"]
        title = clip["title"]
 
        print(f"\n── Clip {i}/{len(clips)}: {title} ──────────────────")
 
        raw_path     = os.path.join(OUTPUT_DIR, f"{i:02d}_{title}_raw.mp4")
        tracked_path = os.path.join(OUTPUT_DIR, f"{i:02d}_{title}_tracked.mp4")
        final_path   = os.path.join(OUTPUT_DIR, f"{i:02d}_{title}.mp4")
 
        cut_clip(args.video, start, end, raw_path)
        apply_camera_tracking(raw_path, tracked_path)
        words = transcribe(tracked_path, whisper_model)
        burn_subtitles(tracked_path, words, final_path)
 
        os.remove(raw_path)
        os.remove(tracked_path)
        print(f"  ✅ Saved: {final_path}")
 
    print(f"\n🎉 Done! {len(clips)} clip(s) saved to '{OUTPUT_DIR}/'")
 
#   ss
if __name__ == "__main__":
    main()
 