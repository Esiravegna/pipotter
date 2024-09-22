import logging
import cv2
from picamera2 import Picamera2
from threading import Thread

logger = logging.getLogger(__name__)

class PiCameraCV(object):
    """
    An OpenCV VideoCapture wrapper for PiCamera2 for headless systems, using grayscale images.
    """

    def __init__(self, resolution=(640, 480), flip=False):
        """
        The constructor
        :param resolution: tuple of (width, height) to set the camera resolution
        :param framerate: int, the frame rate of the camera
        :param flip: list of indices to run the cv2 flip command. Empty to not run anything
        """
        self.camera = Picamera2()
        config = self.camera.create_still_configuration(
            main={"size": resolution}
        )
        self.camera.configure(config)
        self.camera.set_controls({'Saturation': 0})
        self.camera.start()
        self.flip = flip
        self.resolution = resolution
        # Initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False
        Thread(target=self.update, args=()).start()

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while not self.stopped:
            # Capture an image
            self.frame = self.camera.capture_array()
            logger.debug(f"Captured frame shape: {self.frame.shape}")
    def read(self):
        """
        Read a frame from the camera.
        :return: (boolean, cv2Image) captured from the camera, or False, None on error as per the read cv2 VideoCapture command
        """
        frame, ret =  None, False
        if self.frame is None:
            logger.error("self.frame is None, waiting for camera update.")
        else:
            try:
                frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                # Let's flip the images as needed
                if self.flip:
                    frame = cv2.flip(frame, 1)
                ret = True
            except Exception as e:
                logger.error("Unable to read from PiCamera due to {} reading {}".format(e, frame))    
        return ret, frame

    def end(self):
        self.stopped = True
        self.camera.stop()
