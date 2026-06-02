from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


@dataclass(frozen=True)
class FrameExtractionResult:
    frames: list[Path]
    video_info: VideoInfo
    sample_fps: float
    frame_interval: int
    backend: str

    def to_dict(self) -> dict:
        return {
            "frames": [str(path) for path in self.frames],
            "num_frames": len(self.frames),
            "video_info": asdict(self.video_info),
            "sample_fps": self.sample_fps,
            "frame_interval": self.frame_interval,
            "backend": self.backend,
        }


def inspect_video(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    cap.release()

    return VideoInfo(
        path=str(video_path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration,
    )


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    sample_fps: float = 1.0,
    max_frames: int | None = None,
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> FrameExtractionResult:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    video_info = inspect_video(video_path)

    if overwrite and frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = sorted(frames_dir.glob("*.jpg"))
    if existing_frames and not overwrite:
        fps = video_info.fps if video_info.fps > 0 else sample_fps
        return FrameExtractionResult(
            frames=existing_frames,
            video_info=video_info,
            sample_fps=sample_fps,
            frame_interval=max(int(round(fps / sample_fps)), 1),
            backend="existing",
        )

    fps = video_info.fps if video_info.fps > 0 else sample_fps
    frame_interval = max(int(round(fps / sample_fps)), 1)
    if max_frames is not None and video_info.frame_count > 0:
        frame_interval = max(frame_interval, int(math.ceil(video_info.frame_count / max_frames)))

    if shutil.which("ffmpeg"):
        saved = _extract_frames_with_ffmpeg(
            video_path=video_path,
            frames_dir=frames_dir,
            sample_fps=sample_fps,
            max_frames=max_frames,
        )
        return FrameExtractionResult(
            frames=saved,
            video_info=video_info,
            sample_fps=sample_fps,
            frame_interval=frame_interval,
            backend="ffmpeg",
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved: list[Path] = []
    frame_idx = 0
    saved_idx = 0
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval == 0:
            frame_path = frames_dir / f"{saved_idx:06d}.jpg"
            if not cv2.imwrite(str(frame_path), frame, encode_params):
                raise RuntimeError(f"Failed to write extracted frame: {frame_path}")
            saved.append(frame_path)
            saved_idx += 1

        frame_idx += 1

    cap.release()

    if not saved:
        raise RuntimeError(f"No frames were extracted from {video_path}")

    return FrameExtractionResult(
        frames=saved,
        video_info=video_info,
        sample_fps=sample_fps,
        frame_interval=frame_interval,
        backend="opencv",
    )


def _extract_frames_with_ffmpeg(
    video_path: Path,
    frames_dir: Path,
    sample_fps: float,
    max_frames: int | None,
) -> list[Path]:
    output_pattern = frames_dir / "%06d.jpg"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={sample_fps}",
        "-q:v",
        "2",
        "-start_number",
        "0",
    ]
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.append(str(output_pattern))

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg frame extraction failed:\n"
            f"command: {' '.join(command)}\n"
            f"stderr: {result.stderr}"
        )

    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"ffmpeg did not extract any frames from {video_path}")
    return frames


def write_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    max_images: int = 24,
    columns: int = 6,
    thumb_width: int = 240,
) -> Path:
    if not image_paths:
        raise ValueError("image_paths cannot be empty")

    selected = _evenly_spaced(image_paths, max_images)
    thumbs = []
    for image_path in selected:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        scale = thumb_width / max(image.shape[1], 1)
        thumb_size = (thumb_width, max(1, int(round(image.shape[0] * scale))))
        thumbs.append(cv2.resize(image, thumb_size, interpolation=cv2.INTER_AREA))

    if not thumbs:
        raise RuntimeError("Could not read any images for contact sheet")

    columns = max(1, columns)
    rows = int(math.ceil(len(thumbs) / columns))
    thumb_height = max(thumb.shape[0] for thumb in thumbs)
    sheet = np.full((rows * thumb_height, columns * thumb_width, 3), 245, dtype=np.uint8)

    for idx, thumb in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        y0 = row * thumb_height
        x0 = col * thumb_width
        sheet[y0 : y0 + thumb.shape[0], x0 : x0 + thumb.shape[1]] = thumb

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f"Failed to write contact sheet: {output_path}")
    return output_path


def _evenly_spaced(items: list[Path], max_items: int) -> list[Path]:
    if len(items) <= max_items:
        return items
    indices = np.linspace(0, len(items) - 1, max_items).round().astype(int)
    return [items[int(idx)] for idx in indices]
