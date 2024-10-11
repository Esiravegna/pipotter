from queue import Queue, Full
import threading
import logging
import cv2
from picamera2 import Picamera2
from media.video_source.video_source import VideoSource

logger = logging.getLogger(__name__)


class PiCameraCV(VideoSource):
    """
    An OpenCV VideoCapture wrapper for PiCamera2 for headless systems, optimized for detecting an IR reflective wand.
    """

    def __init__(
        self, resolution=(320, 240), flip=False, queue_size=2, low_light_mode=True
    ):
        """
        Initialize the PiCamera with a frame queue for concurrent access, optimizing for IR detection.
        :param resolution: tuple of (width, height) to set the camera resolution
        :param flip: whether to flip the captured images
        :param queue_size: max number of frames to buffer in the queue
        :param low_light_mode: whether to enable settings optimized for low light
        """
        self.camera = Picamera2()
        config = self.camera.create_video_configuration(main={"size": resolution})
        self.camera.configure(config)

        # Controls for color gains and exposure optimization
        ir_sensitivity_gains = (
            2.0,
            1.0,
        )  # Adjust gains for higher IR sensitivity, boosting red channel
        frame_duration = (
            (10000, 30000) if low_light_mode else (30000, 30000)
        )  # Lower exposure time in low light

        self.camera.set_controls(
            {
                "Saturation": 0,  # Grayscale-like output
                "AeEnable": True,
                "AwbEnable": False,  # Disable auto white balance for more stable IR readings
                "ColourGains": ir_sensitivity_gains,  # IR sensitivity tuning (higher red)
                "FrameDurationLimits": frame_duration,  # Frame duration affects exposure time and framerate
            }
        )

        # Start camera
        self.camera.start()

        self.flip = flip
        self.frame_queue = Queue(maxsize=queue_size)
        self.stopped = False

        # Start capture thread
        self.capture_thread = threading.Thread(target=self.update, args=())
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def update(self):
        """Continuously capture frames, convert to grayscale, and store them into the queue."""
        while not self.stopped:
            frame = self.camera.capture_array()

            # Convert frame to grayscale to prioritize IR light detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply flip if needed
            if self.flip:
                gray_frame = cv2.flip(gray_frame, 1)

            # Put the frame into the queue (non-blocking)
            try:
                self.frame_queue.put_nowait(gray_frame)
            except Full:
                # Ignore if the queue is full
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
