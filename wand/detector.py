import logging
from math import hypot
import cv2
import numpy as np
from core.error import WandError
from wand.spellscontainer import SpellsContainer

logger = logging.getLogger(__name__)

class WandDetector(object):
    def __init__(self, video, dilation_params=(5, 5), windows_size=(15, 15), max_level=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                 circles_dp=3, circles_mindist=100, circles_minradius=2, circles_maxradius=8,
                 circles_threshold=5, movement_threshold=80):
        """
        Initialize WandDetector.
        """
        self.video = video
        self.lk_params = dict(winSize=windows_size, maxLevel=max_level, criteria=criteria)
        self.dilation_params = dilation_params
        self.movement_threshold = movement_threshold

        # Circle detection parameters
        self.circles_dp = circles_dp
        self.circles_mindist = circles_mindist
        self.circles_minradius = circles_minradius
        self.circles_maxradius = circles_maxradius
        self.circles_threshold = circles_threshold

        self.prev_frame_gray = None
        self.prev_circles = []
        self.sigil_mask = None
        self.maybe_a_spell = np.array([])

        self.spells_container = SpellsContainer()
        self.sigil_color = [255] * 3

        # Used to store frames for streaming purposes
        self.latest_wand_frame = None
        self.latest_sigil_frame = None

    def find_wand(self):
        """
        Detects wands in the video stream.
        """
        circles = None
        gray = None
        try:
            ret = False
            while circles is None:
                while not ret:
                    ret, frame = self.video.read()
                gray = self._to_gray(frame)
                circles = self._find_circles(gray)

            self.prev_circles = circles
            self.prev_frame_gray = gray
            self.sigil_mask = np.zeros_like(frame)
            self.spells_container.reset()
        except Exception as e:
            logger.error(f"Error detecting a wand: {e}")

        return gray, circles

    def read_wand(self, frame):
        """
        Reads wand movement and processes it.
        """
        if self.prev_frame_gray is None:
            raise WandError("No previous frame found, terminating")

        try:
            gray = self._to_gray(frame)
            new_circles, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame_gray, gray,
                                                            self.prev_circles, None, **self.lk_params)
            new_valid_points = new_circles[st == 1]
            old_valid_points = self.prev_circles[st == 1]

            for i, (new, old) in enumerate(zip(new_valid_points, old_valid_points)):
                a, b = new.ravel()
                c, d = old.ravel()

                if i < self.circles_threshold:
                    dist = hypot(a - c, b - d)
                    if dist < self.movement_threshold:
                        self.spells_container[i] = [a, b, c, d]
                        cv2.line(self.sigil_mask, (a, b), (c, d), self.sigil_color, 3)
                        cv2.circle(frame, (a, b), 5, self.sigil_color, -1)
                        cv2.putText(frame, str(i), (a, b), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

            left, top, right, bottom = self.spells_container.get_box()
            self.maybe_a_spell = self.sigil_mask[top:bottom, left:right].copy()

            img = cv2.add(frame, self.sigil_mask)
            self.latest_wand_frame = img.copy()
            self.latest_sigil_frame = self.maybe_a_spell.copy()

            self.prev_frame_gray = gray.copy()
            self.prev_circles = new_valid_points.reshape(-1, 1, 2)
        except (TypeError, ValueError, cv2.error) as e:
            logger.error(f"Error reading the wand: {e}")

        return self.maybe_a_spell

    def _to_gray(self, frame):
        """Converts a frame to grayscale."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _find_circles(self, gray_frame):
        """Detects circles (wands) in the frame."""
        try:
            circles = cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, dp=self.circles_dp,
                                       minDist=self.circles_mindist, param1=240, param2=8,
                                       minRadius=self.circles_minradius, maxRadius=self.circles_maxradius)
            if circles is not None:
                circles.shape = (circles.shape[1], 1, circles.shape[2])
                return circles[:, :, 0:2]
        except Exception as e:
            logger.error(f"Unable to detect circles: {e}")
        return None
