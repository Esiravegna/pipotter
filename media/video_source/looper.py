import logging
from os.path import exists
import cv2
from threading import Thread

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
        logger.debug(
            "Looping {}.. Getting number of frames, please wait".format(videofile)
        )
        self.video = cv2.VideoCapture(videofile)
        self.source = videofile
        self.frame_counter = 0
        self.total_frames = self._count_frames_manual(cv2.VideoCapture(videofile))
        self.flip = flip
        self.max_width = max_width
        logger.debug("{} ready with {} frames".format(videofile, self.total_frames))

    def _count_frames_manual(self, video):
        """
        Frames counting is really buggy, got to do this old fashioned.
        :param: video, a valid opencv2 video
        :return: int, the number of frames
        """
        total = 0
        # loop over the frames of the video
        while True:
            # grab the current frame
            (grabbed, frame) = video.read()
            # check to see if we have reached the end of the
            # video
            if not grabbed:
                break
            # increment the total number of frames read
            total += 1
        return total

    def _read(self):
        """
        reads a single frame
        """
        self.frame_counter += 1
        return self.video.read()

    def read(self):
        """
        given the proper initialized video, reads a frame from it
        :return: (boolean, cv2Image) captured from the videofile, or False, None on error  as per the read cv2 VideoCapture command
        """
        ret, frame = self._read()
        # did we reached the end of the video? If so, reset the counter and loop over again
        if self.frame_counter >= self.total_frames:
            logger.debug("Looping back {} to zero".format(self.source))
            self.video.release()
            self.video = cv2.VideoCapture(self.source)
            self.frame_counter = 0
            ret, frame = self._read()
        if ret:
            for a_flip in self.flip:
                cv2.flip(frame, a_flip, frame)
                frame = imutils.resize(frame, width=self.max_width)
        return ret, frame

    def end(self):
        """
        Gracefully terminates the thing
        """
        self.video.release()
