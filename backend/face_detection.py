import cv2
import mediapipe as mp
import numpy as np

# FaceMesh is initialised once at module load time. Creating it inside the
# function would reload the underlying TFLite model on every call — once per
# frame — which is prohibitively slow for a 60-second clip.
_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,   # treat each frame independently (no tracking)
    max_num_faces=1,
    refine_landmarks=False,   # forehead landmarks don't need iris refinement
)

# MediaPipe FaceMesh landmark indices that outline the forehead region.
# These are the upper-face boundary points from the canonical 468-point mesh.
FOREHEAD_LANDMARK_IDS = [10, 338, 297, 332, 284]


def extract_forehead_roi(frame: np.ndarray) -> np.ndarray | None:
    """
    Return the mean RGB value of the forehead ROI in a single frame.

    Runs MediaPipe FaceMesh on the frame, locates the forehead polygon
    defined by FOREHEAD_LANDMARK_IDS, builds a pixel mask for that region,
    and returns the mean colour across all pixels inside it.

    Args:
        frame: RGB image as a numpy array of shape (H, W, 3) uint8.

    Returns:
        Mean RGB value as a numpy array of shape (3,) and dtype float64,
        or None if no face is detected in the frame.
    """
    h, w = frame.shape[:2]

    results = _face_mesh.process(frame)

    if not results.multi_face_landmarks:
        return None

    # Take the first (and only) detected face
    landmarks = results.multi_face_landmarks[0].landmark

    # Landmark coordinates are normalised to [0, 1] — convert to pixels
    points = np.array(
        [
            [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
            for idx in FOREHEAD_LANDMARK_IDS
        ],
        dtype=np.int32,
    )

    # Fill the forehead polygon on a single-channel mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points], color=255)

    # Gather all pixels inside the mask
    roi_pixels = frame[mask == 255]  # shape (N, 3)

    if roi_pixels.size == 0:
        # Polygon was too small to cover any pixels (e.g. very low resolution)
        return None

    return roi_pixels.mean(axis=0)  # shape (3,)
