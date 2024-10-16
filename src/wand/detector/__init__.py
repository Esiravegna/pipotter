from wand.detector.detector_blobs import WandDetector as BlobsDetector
from wand.detector.detector_circles import WandDetector as CirclesDetector
from wand.detector.detector_fast import WandDetector as FastDetector
from wand.detector.detector_circles_optimized import (
    WandDetector as CirclesDetectorOptimized,
)
from wand.detector.detector_optical_flow import WandDetector as OpticalFlowDetector

AVAILABLE_DETECTORS = {
    "blobs": BlobsDetector,
    "circles": CirclesDetector,
    "fast": FastDetector,
    "circopt": CirclesDetectorOptimized,
    "optical_flow": OpticalFlowDetector,
}


def get_detector(detector_type, video, **kwargs):
    """
    Factory function to get the desired detector.

    Args:
        detector_type: Type of the detector to instantiate ('default', 'advanced', etc.).
        video: Video source object.
        **kwargs: Additional parameters specific to the detector type.

    Returns:
        An instance of the desired detector class.
    """
    if detector_type not in AVAILABLE_DETECTORS.keys():
        raise ValueError(f"Unknown detector type: {detector_type}")
    return AVAILABLE_DETECTORS.get(detector_type)(video, **kwargs)
