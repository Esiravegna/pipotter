import io
import logging
import imutils
import cv2
import numpy as np
from media.video_source.video_source import VideoSource

logger = logging.getLogger(__name__)


class PiCameraCV(object):
    """
    An OpenCV VideoCapture wrapper for PiCamera
    """

    def __init__(self, camera, flip=[], max_width=800):
        """
        The constructor
        :param camera: a valid, set raspberry pi camera object as per piraspicam
        :param flip: tuple of index to run the cv2 flip command. Empty to not running anything
        :param max_width: int, max width to resize image
        """
        self.camera = camera
        self.stream = io.BytesIO()
        self.flip = flip
        self.max_width = max_width

    def read(self):
        """
        given the proper initialized camera, reads a frame from it
        :return: (boolean, cv2Image) captured from the camera, or False, None on error  as per the read cv2 VideoCapture command
        """

        frame, ret = None, False
        try:
            self.camera.capture(self.tream, format='jpeg')
            frame = cv2.imdecode(np.fromstring(self.stream.getvalue(), dtype=np.uint8), 1)
            # Let's flip the images as needed
            for a_flip in self.flip:
                cv2.flip(frame, a_flip, frame)
            ret = True
            frame = imutils.resize(frame, width=self.max_width)
        except Exception as e:
            logger.error("Unable to read from RaspiCame due to {}".format(e))
        finally:
            return ret, frame

    def end(self):
        self.camera.close()
