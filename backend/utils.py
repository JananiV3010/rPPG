import cv2
import numpy as np


def extract_frames(video_path: str) -> list[np.ndarray]:
    """
    Read a video file and return all frames as a list of RGB numpy arrays.

    OpenCV decodes frames in BGR by default, so each frame is converted to
    RGB before being appended — downstream MediaPipe and model code both
    expect RGB input.

    NOTE — memory: all frames are loaded into memory at once. This is
    intentional and acceptable for the expected input constraints of this
    app: clips up to 60 seconds recorded at standard webcam resolutions
    (typically 640x480 or 1280x720 at 30 fps). A 60-second 720p clip
    produces ~1800 frames, which fits comfortably in RAM (~1.4 GB worst
    case). Do not use this function for longer recordings or high-resolution
    video without switching to a generator-based approach.

    Args:
        video_path: Absolute or relative path to the video file.
            Expected: max 60 seconds, webcam resolution (up to 1280x720),
            30 fps typical.

    Returns:
        List of frames, each shaped (H, W, 3) in RGB uint8.

    Raises:
        ValueError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()

    return frames


def get_video_fps(video_path: str) -> float:
    """
    Return the frame rate of a video file.

    Args:
        video_path: Absolute or relative path to the video file.

    Returns:
        Frames per second as a float (e.g. 30.0).

    Raises:
        ValueError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()

    return fps
