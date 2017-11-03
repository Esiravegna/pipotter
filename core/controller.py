import logging
import time

import cv2

from media.video_source import VALID_SOURCES, looper, picamera
from wand.detector import WandDetector

logger = logging.getLogger(__name__)
from core.config import settings

END_KEY = settings['PIPOTTER_END_LOOP']
SECONDS_TO_DRAW = settings['PIPOTTER_SECONDS_TO_DRAW']


class PiPotterController(object):
    """
    PiPotter controller. Initializes all the objects and 
    """

    def __init__(self, video_source_name, draw_windows=False, **kwargs):
        """
        The Controller
        :param video_source_name: (string) picamera | looper, any of the controllers to extract the images
        :param draw_windows: Boolean, should the windows being drawn
        :param args: extra args
        :param kwargs: extra kwargs
        """
        # Let's the validation begin
        logger.debug("initializing controller")
        flip = settings['PIPOTTER_FLIP_VIDEO']
        if video_source_name not in VALID_SOURCES:
            raise Exception("Invalid controller name :{}. Must be either of : {}".format(VALID_SOURCES))
        if video_source_name == 'picamera':
            try:
                camera = kwargs['picamera']
                video = picamera(camera, flip=flip)
            except KeyError:
                raise Exception(
                    "For use a picamera source, a picamera parameter with a valid camera should be provided")
        elif video_source_name == 'looper':
            try:
                video_file = kwargs['video_file']
                video = looper(video_file, flip=flip)
            except KeyError:
                raise Exception(
                    "For use a video loop source, a video file parameter with a valid camera should be provided")
        self.wand_detector = WandDetector(video=video, draw_windows=draw_windows)
        self.draw_windows = draw_windows

    def _terminate(self):
        """
        Internal. Closes all the things
        :return: 
        """
        self.video.end()
        if self.draw_windows:
            cv2.destroyAllWindows()

    def run(self):
        """
        runs the controller
        """
        logger.info("Starting PiPotter...")
        self.wand_detector.find_wand()
        logger.debug("Found a wand, starting loop")
        while True:
            # Main Loop
            sigil = self.wand_detector.read_wand()
            t_end = time.time() + SECONDS_TO_DRAW
            while time.time() < t_end:
                # for the next seconds, build a sigil.
                self.wand_detector.read_wand()
                key = cv2.waitKey(1) & 0xFF
                if key == ord(END_KEY):
                    break
            logger.debug("read finished, processing")
            key = cv2.waitKey(1) & 0xFF
            if key == ord(END_KEY):
                logger.info("Terminating PiPotter...")
                break
