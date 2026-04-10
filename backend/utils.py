import cv2
import numpy as np


def extract_frames(video_path: str) -> list[np.ndarray]:
    """
    Read a video file and return all frames as a list of RGB numpy arrays.

    OpenCV decodes frames in BGR by default, so each frame is converted to
    RGB before being appended — downstream MediaPipe and model code both
    expect RGB input.

    Args:
        video_path: Absolute or relative path to the video file.

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
