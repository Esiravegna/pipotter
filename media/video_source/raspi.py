import io
import logging
import imutils
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
from threading import Thread

from media.video_source.video_source import VideoSource

logger = logging.getLogger(__name__)


class PiCameraCV(object):
    """
    An OpenCV VideoCapture wrapper for PiCamera
    """

    def __init__(self, camera, flip=[], resolution=(640, 480), framerate=32):
        """
        The constructor
        :param camera: a valid, set raspberry pi camera object as per piraspicam
        :param flip: tuple of index to run the cv2 flip command. Empty to not running anything
        :param max_width: int, max width to resize image
        """
        self.camera = camera
        self.camera.start_preview()
        self.camera.resolution = resolution
        self.flip = flip
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
            format="bgr", use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False
        Thread(target=self.update, args=()).start()
        

    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)
 
            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return


    def read(self):
        """
        given the proper initialized camera, reads a frame from it
        :return: (boolean, cv2Image) captured from the camera, or False, None on error  as per the read cv2 VideoCapture command
        """
        frame, ret = None, False
        try:
            frame = self.frame
            # Let's flip the images as needed
            for a_flip in self.flip:
                cv2.flip(frame, a_flip, frame)
            ret = True
        except Exception as e:
            logger.error("Unable to read from RaspiCame due to {}".format(e))
        finally:
            return ret, frame

    def end(self):
        self.stopped = True
