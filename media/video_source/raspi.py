import logging
import cv2
from picamera2 import Picamera2
from threading import Thread

logger = logging.getLogger(__name__)

class PiCameraCV(object):
    """
    An OpenCV VideoCapture wrapper for PiCamera2 for headless systems, using grayscale images.
    """

    def __init__(self, resolution=(640, 480), flip=[]):
        """
        The constructor
        :param resolution: tuple of (width, height) to set the camera resolution
        :param framerate: int, the frame rate of the camera
        :param flip: list of indices to run the cv2 flip command. Empty to not run anything
        """
        self.camera = Picamera2()
        self.camera.configure(self.camera.create_still_configuration(
            main={"format": 'YUV420', "size": resolution}
        ))
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
            frame = self.camera.capture_array()
            # as per https://github.com/raspberrypi/picamera2/issues/698
            self.frame = frame[:self.resolution[1], :self.resolution[0]]

    def read(self):
        """
        Read a frame from the camera.
        :return: (boolean, cv2Image) captured from the camera, or False, None on error as per the read cv2 VideoCapture command
        """
        frame, ret = None, False
        try:
            frame = self.frame
            # Let's flip the images as needed
            for a_flip in self.flip:
                cv2.flip(frame, a_flip, frame)
            ret = True
        except Exception as e:
            logger.error("Unable to read from PiCamera due to {}".format(e))
        finally:
            return ret, frame

    def end(self):
        self.stopped = True
        self.camera.stop()
