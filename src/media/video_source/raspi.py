from queue import Queue, Full
import threading
import logging
import cv2
from picamera2 import Picamera2
from media.video_source.video_source import VideoSource

logger = logging.getLogger(__name__)


class PiCameraCV(VideoSource):
    """
    An OpenCV VideoCapture wrapper for PiCamera2 for headless systems, using grayscale images.
    """

    def __init__(self, resolution=(320, 240), flip=False, queue_size=2):
        """
        Initialize the PiCamera with a frame queue for concurrent access.
        :param resolution: tuple of (width, height) to set the camera resolution
        :param flip: whether to flip the captured images
        :param queue_size: max number of frames to buffer in the queue
        """
        self.camera = Picamera2()
        config = self.camera.create_video_configuration(main={"size": resolution})
        self.camera.configure(config)
        self.camera.set_controls({"Saturation": 0})
        self.camera.set_controls(
            {
                "AeEnable": True,
                "AwbEnable": False,  # Disable auto white balance
                "ColourGains": (1.5, 1.0),  # Adjust gains for IR sensitivity
                "FrameDurationLimits": (
                    30000,
                    30000,
                ),  # Set frame duration to control framerate
            }
        )
        self.camera.start()

        self.flip = flip
        self.frame_queue = Queue(maxsize=queue_size)
        self.stopped = False

        self.capture_thread = threading.Thread(target=self.update, args=())
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def update(self):
        """Continuously capture frames, convert to grayscale, and store them into the queue."""
        while not self.stopped:
            frame = self.camera.capture_array()

            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply flip if needed
            if self.flip:
                gray_frame = cv2.flip(gray_frame, 1)
            try:
                self.frame_queue.put_nowait(gray_frame)
            except Full:
                pass

    def read(self):
        """
        Retrieve the latest frame from the queue.
        :return: (boolean, frame) where boolean indicates success, and frame is the latest captured frame.
        """
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                return True, frame
            else:
                logger.warning("No frames available in the queue.")
                return False, None
        except Exception as e:
            logger.error(f"Error reading frame from PiCamera: {e}")
            return False, None

    def end(self):
        """Stop the camera and the frame capture thread."""
        self.stopped = True
        self.camera.stop()
