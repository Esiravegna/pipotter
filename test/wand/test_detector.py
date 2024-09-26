import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from wand.detector import WandDetector, WandError


# Custom mock class to simulate video input
class MockVideoCapture:
    def __init__(self, frames):
        self.frames = frames
        self.index = 0

    def read(self):
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame
        return False, None

    def isOpened(self):
        return True

    def release(self):
        pass


@pytest.fixture
def mock_video():
    # Create mock frames (grayscale images)
    frame1 = np.zeros((100, 100), dtype=np.uint8)
    frame2 = np.zeros((100, 100), dtype=np.uint8)
    return MockVideoCapture([frame1, frame2])


@pytest.fixture
def detector(mock_video):
    # Initialize the WandDetector with the mock video capture object
    return WandDetector(video=mock_video)


def test_initialization(detector):
    assert detector.video.isOpened()
    assert detector.prev_frame_gray is None
    assert detector.prev_circles is None
    assert detector.spells_container is not None


@patch.object(
    WandDetector, "_process_frame", return_value=np.zeros((100, 100), dtype=np.uint8)
)
@patch.object(WandDetector, "_find_circles", return_value=np.array([[50, 50]]))
def test_find_wand(mock_find_circles, mock_process_frame, detector):
    detector.find_wand()
    assert detector.prev_frame_gray is not None
    assert detector.prev_circles is not None
    assert mock_find_circles.called


def test_read_wand_initializes_if_no_previous_frame(detector):
    frame = np.zeros((100, 100), dtype=np.uint8)
    with patch.object(detector, "find_wand", return_value=None):
        spell = detector.read_wand(frame)
        assert spell.size == 0
        assert detector.find_wand.called
