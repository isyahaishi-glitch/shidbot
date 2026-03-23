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
try:
    from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
except ImportError:
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
FORCE_916        = False  # True = force 9:16 output, False = keep original aspect ratio
WORDS_PER_SUB    = 3
WHISPER_MODEL    = "base"
SUBTITLE_Y_RATIO = 0.65

# Camera tracking
ZOOM_X           = 1.25   # horizontal zoom (>1 = zoom in, 1.0 = no zoom, 0.0 = letterbox)
ZOOM_Y           = 1.25   # vertical zoom   (>1 = zoom in, 1.0 = no zoom, 0.0 = letterbox)
SLIDE_SMOOTHING  = 0.06   # pan smoothness — lower = slower (0.0–1.0)
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
    # Handle formats: 1:20, 1:20.5, 00:01:20, 00:01:20.500
    parts = ts.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 2: return int(parts[0]) * 60 + parts[1]
    if len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + parts[2]
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
# STEP 1.5 — CONVERT TO 9:16 VIA FFMPEG
# ─────────────────────────────────────────────

# FFmpeg transform settings — edit these to match your DaVinci settings
FFMPEG_SCALE  = "1876:-2"       # scale width (height auto)
FFMPEG_CROP_W = "1080"          # crop width
FFMPEG_CROP_H = "1055"          # crop height
FFMPEG_PAD_W  = "1080"          # final width
FFMPEG_PAD_H  = "1920"          # final height
FFMPEG_PAD_X  = "0"             # pad X offset
FFMPEG_PAD_Y  = "432"           # pad Y offset


def convert_to_916(input_path, output_path):
    """Apply scale+crop+pad to convert source to 9:16 using FFmpeg."""
    print(f"  📐 Converting to 9:16 via FFmpeg...")
    vf = f"scale={FFMPEG_SCALE},crop={FFMPEG_CROP_W}:{FFMPEG_CROP_H},pad={FFMPEG_PAD_W}:{FFMPEG_PAD_H}:{FFMPEG_PAD_X}:{FFMPEG_PAD_Y}"
    r = subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-c:a", "aac", "-ar", "48000", "-b:a", "192k",
        output_path
    ], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("FFmpeg 9:16 convert failed: " + r.stderr)
    print(f"  ✔  Converted ({os.path.getsize(output_path)//1024}KB)")


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
    Supports both old (solutions) and new (tasks) MediaPipe APIs.
    Returns dict: { time: (cx, cy) } of the most active speaking face.
    """
    cap          = cv2.VideoCapture(video_path)
    sample_every = max(1, int(fps / FACE_SAMPLE_FPS))
    speaker_data = {}
    face_histories = {}

    # ── Try new Tasks API first (mediapipe >= 0.10.13) ──────────────
    try:
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        import urllib.request, tempfile, os as _os

        # Download FaceLandmarker model if not cached
        model_path = _os.path.join(tempfile.gettempdir(), "face_landmarker.task")
        if not _os.path.exists(model_path):
            print("  📥 Downloading FaceLandmarker model (~30MB, once only)...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)

        base_opts = mp_tasks.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            num_faces=6,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % sample_every == 0:
                t   = frame_idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)

                if result.face_landmarks:
                    best_cx, best_cy = vid_w / 2, vid_h / 2
                    best_score = -1
                    for fi, face_lms in enumerate(result.face_landmarks):
                        lms      = face_lms
                        openness = get_lip_openness(lms, vid_w, vid_h)
                        cx, cy   = get_face_center(lms, vid_w, vid_h)
                        if fi not in face_histories:
                            face_histories[fi] = collections.deque(maxlen=LIP_HISTORY_FRAMES)
                        face_histories[fi].append(openness)
                        history = face_histories[fi]
                        score = np.var(list(history)) if len(history) >= 2 else openness
                        if score > best_score:
                            best_score = score
                            best_cx, best_cy = cx, cy
                    speaker_data[t] = (best_cx, best_cy)
            frame_idx += 1

        landmarker.close()
        cap.release()
        return speaker_data

    # ── Fall back to old solutions API (mediapipe <= 0.10.9) ────────
    except Exception as e:
        print(f"  [INFO] Tasks API unavailable ({e}), trying solutions API...")
        cap.release()
        cap = cv2.VideoCapture(video_path)

    try:
        mp_mesh = mp.solutions.face_mesh
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
                            if fi not in face_histories:
                                face_histories[fi] = collections.deque(maxlen=LIP_HISTORY_FRAMES)
                            face_histories[fi].append(openness)
                            history = face_histories[fi]
                            score = np.var(list(history)) if len(history) >= 2 else openness
                            if score > best_score:
                                best_score = score
                                best_cx, best_cy = cx, cy
                        speaker_data[t] = (best_cx, best_cy)
                frame_idx += 1
    except Exception as e2:
        print(f"  [WARN] Face detection failed: {e2}. Camera will stay centered.")

    cap.release()
    return speaker_data


# Set to True to lock vertical axis (camera only pans left/right)
LOCK_Y_AXIS = True


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
        cx = cx + (tcx - cx) * SLIDE_SMOOTHING
        # Line 301 — Y axis locked if LOCK_Y_AXIS=True, stays at center
        if not LOCK_Y_AXIS:
            cy = cy + (tcy - cy) * SLIDE_SMOOTHING
        path.append((cx, cy))
    return path


def letterbox_frame(frame, vid_w, vid_h):
    """Fit full frame into 9:16 with black bars (no cropping)."""
    out = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
    scale = min(OUTPUT_WIDTH / vid_w, OUTPUT_HEIGHT / vid_h)
    new_w = int(vid_w * scale)
    new_h = int(vid_h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x_off = (OUTPUT_WIDTH - new_w) // 2
    y_off = (OUTPUT_HEIGHT - new_h) // 2
    out[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return out


def crop_frame_landscape(frame, cx, cy, vid_w, vid_h):
    """Zoom + pan. FORCE_916=False keeps original aspect ratio."""
    out_w = OUTPUT_WIDTH if FORCE_916 else vid_w
    out_h = OUTPUT_HEIGHT if FORCE_916 else vid_h
    if ZOOM_X <= 0.0 and ZOOM_Y <= 0.0:
        return letterbox_frame(frame, vid_w, vid_h)
    crop_w = int(vid_w / ZOOM_X)
    crop_h = int(vid_h / ZOOM_Y)
    if crop_h > vid_h: crop_h = vid_h
    if crop_w > vid_w: crop_w = vid_w
    x1 = max(0, min(int(cx - crop_w / 2), vid_w - crop_w))
    y1 = max(0, min(int(cy - crop_h / 2), vid_h - crop_h))
    cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def crop_frame_vertical(frame, cx, cy, vid_w, vid_h):
    """9:16 zoom with independent X/Y. Set both to 0.0 for letterbox."""
    out_w = OUTPUT_WIDTH if FORCE_916 else vid_w
    out_h = OUTPUT_HEIGHT if FORCE_916 else vid_h
    if ZOOM_X <= 0.0 and ZOOM_Y <= 0.0:
        return letterbox_frame(frame, vid_w, vid_h)
    crop_w = int(vid_w / ZOOM_X)
    crop_h = int(vid_h / ZOOM_Y)
    x1 = max(0, min(int(cx - crop_w / 2), vid_w - crop_w))
    y1 = max(0, min(int(cy - crop_h / 2), vid_h - crop_h))
    cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
    return cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def apply_camera_tracking(input_path, output_path):
    """Render every frame with smooth lip-aware speaker tracking."""
    fps, vid_w, vid_h, total_frames = get_video_info(input_path)
    vertical = is_vertical(vid_w, vid_h)

    print(f"  🎥 Source: {vid_w}×{vid_h} ({'vertical' if vertical else 'landscape'})")
    print(f"  👄 Detecting lip movement to find active speaker...")
    try:
        speaker_data = detect_speaker_over_time(input_path, vid_w, vid_h, fps)
        print(f"  ✔  Tracked speaker at {len(speaker_data)} sample points")
    except Exception as e:
        print(f"  [WARN] Face tracking failed ({e}), using center crop fallback")
        speaker_data = {}

    camera_path = build_camera_path(
        speaker_data, total_frames, fps, vid_w / 2, vid_h / 2)

    print(f"  📐 Rendering {total_frames} frames...")
    cap     = cv2.VideoCapture(input_path)
    tmp_avi = output_path.replace(".mp4", "_noaudio.avi")
    tmp     = output_path.replace(".mp4", "_noaudio.mp4")

    # Write frames using MJPG into AVI (most compatible on Windows)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out_w = OUTPUT_WIDTH if FORCE_916 else vid_w
    out_h = OUTPUT_HEIGHT if FORCE_916 else vid_h
    writer = cv2.VideoWriter(tmp_avi, fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("OpenCV VideoWriter failed to open — check codec support.")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cx, cy = camera_path[min(idx, len(camera_path) - 1)]
        out = crop_frame_vertical(frame, cx, cy, vid_w, vid_h) if vertical else crop_frame_landscape(frame, cx, cy, vid_w, vid_h)
        writer.write(out)
        idx += 1

    cap.release()
    writer.release()

    if not os.path.exists(tmp_avi) or os.path.getsize(tmp_avi) == 0:
        raise RuntimeError("AVI write failed, file missing or empty: " + tmp_avi)
    print(f"  ✔  AVI written ({os.path.getsize(tmp_avi)//1024}KB)")

    # Convert AVI → MP4
    r1 = subprocess.run([
        "ffmpeg", "-y", "-i", tmp_avi,
        "-c:v", "libx264", "-preset", "fast", tmp
    ], capture_output=True, text=True)
    if r1.returncode != 0:
        raise RuntimeError("FFmpeg AVI to MP4 failed: " + r1.stderr)
    os.remove(tmp_avi)

    if not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
        raise RuntimeError("MP4 conversion failed, file missing: " + tmp)

    # Merge audio — use -map 1:a? (trailing ? = optional, skip if no audio)
    r2 = subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp, "-i", input_path,
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest", output_path
    ], capture_output=True, text=True)
    if r2.returncode != 0:
        # Fallback: just copy video without audio
        r2b = subprocess.run([
            "ffmpeg", "-y", "-i", tmp,
            "-c:v", "copy", output_path
        ], capture_output=True, text=True)
        if r2b.returncode != 0:
            raise RuntimeError("FFmpeg merge failed: " + r2b.stderr)
        print("  [WARN] No audio stream found, output has no audio.")
    os.remove(tmp)

    if not os.path.exists(output_path):
        raise RuntimeError("Final output missing: " + output_path)
    print(f"  ✔  Tracked video ready ({os.path.getsize(output_path)//1024}KB)")


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
            .set_start(ev["start"])
            .set_duration(duration)
        )

    final = CompositeVideoClip([clip] + sub_clips)
    final.write_videofile(output_path, codec="libx264",
                          audio_codec="aac", fps=clip.fps, logger=None)
    clip.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def sanitize_filename(name):
    """Convert title to safe filename."""
    import re as _re
    name = name.lower().strip()
    name = _re.sub(r"[^\w\s-]", "", name)
    name = _re.sub(r"[\s]+", "_", name)
    name = _re.sub(r"_+", "_", name)
    return name[:60]


def load_json_clips(json_path):
    """
    Load Gemini highlights JSON.
    Supports both:
      { "timestamp": "0:00-0:18", "title": "..." }
      { "start": "0:00", "end": "0:18", "title": "..." }
    """
    import json as _json
    with open(json_path, "r", encoding="utf-8") as f:
        data = _json.load(f)
    clips = []
    for item in data:
        if "timestamp" in item:
            # Handle "0:00-0:18" or "0:00.000-0:18.000"
            # Split on "-" but only the last occurrence to avoid negative numbers
            ts = item["timestamp"]
            dash_idx = ts.rfind("-")
            start = ts_to_sec(ts[:dash_idx].strip())
            end   = ts_to_sec(ts[dash_idx+1:].strip())
        else:
            start = ts_to_sec(item["start"])
            end   = ts_to_sec(item["end"])
        title = sanitize_filename(item.get("title", "clip"))
        clips.append({"start": start, "end": end, "title": title})
    return clips


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

        raw_path       = os.path.join(OUTPUT_DIR, f"{i:02d}_{title}_raw.mp4")
        converted_path = os.path.join(OUTPUT_DIR, f"{i:02d}_{title}_916.mp4")
        tracked_path   = os.path.join(OUTPUT_DIR, f"{i:02d}_{title}_tracked.mp4")
        final_path     = os.path.join(OUTPUT_DIR, f"{i:02d}_{title}.mp4")

        cut_clip(args.video, start, end, raw_path)         # Step 1
        convert_to_916(raw_path, converted_path)            # Step 1.5
        apply_camera_tracking(converted_path, tracked_path) # Step 2
        words = transcribe(raw_path, whisper_model)         # Step 3 (raw has audio)
        burn_subtitles(tracked_path, words, final_path)     # Step 4

        os.remove(raw_path)
        os.remove(converted_path)
        os.remove(tracked_path)
        print(f"  ✅ Saved: {final_path}")

    print(f"\n🎉 Done! {len(clips)} clip(s) saved to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
    # python main.py  2.mp4 --json n.json