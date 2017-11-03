
import logging
from os.path import exists
import cv2
import imutils
from core.error import MediaError
from media.video_source.video_source import VideoSource
logger = logging.getLogger(__name__)


class VideoLooper(VideoSource):
    """
    Loops from a video using opencv
    """
    def __init__(self, videofile, flip=[], max_width=800):
        """
        The constructor
        :param videofile: (string), path to the video file being used
        :param flip: tuple of index to run the cv2 flip command. Empty to not running anything
        :param max_width: int, max width to resize image
        """
        if not exists(videofile):
            raise MediaError("Unable to find {}".format(videofile))
        self.video = cv2.VideoCapture(videofile)
        self.source = videofile
        self.frame_counter = 0
        self.total_frames = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        self.flip = flip
        self.max_width = max_width

    def read(self):
        """
        given the proper initialized video, reads a frame from it
        :return: (boolean, cv2Image) captured from the videofile, or False, None on error  as per the read cv2 VideoCapture command
        """
        ret, frame = self.video.read()
        self.frame_counter += 1
        # did we reached the end of the video? If so, reset the counter and loop over again
        if self.frame_counter == self.total_frames:
            self.video  = cv2.VideoCapture(self.source)
            self.frame_counter = 0
        if ret:
            for a_flip in self.flip:
                cv2.flip(frame, a_flip, frame)
                frame = imutils.resize(frame, width=self.max_width)
                cv2.imshow("looper live", frame)
        return ret, frame

    def end(self):
        """
        Gracefully terminates the thing
        """
        self.video.release()