import logging
from math import hypot

import cv2
import numpy as np

from core.config import settings
from core.error import WandError

logger = logging.getLogger(__name__)
END_KEY = settings['PIPOTTER_END_LOOP']


class CropRectangle(object):
    """
    A very simple object in charge to update and keep a rectangle for numpy slicing afterwards
    """

    def __init__(self):
        """
        Just the constructor
        """
        self.top = False
        self.bottom = False
        self.left = False
        self.right = False

    def update(self, top, left, right, bottom):
        """
        given a box, updates the parameters should it be needed.
        The rationale is to expand the box containing the coordinates, and never decrease it
        :param top: (number) top of the box
        :param left: (number) left of the same thing
        :param right: (number) right of the box)
        :param bottom: (numner) bottom of the box
        """
        # So, the left and top, in order to be updated, must be smaller than the previus values and at minimun, zero.
        self.left = int(left) if (left < self.left or not self.left and left >= 0) else self.left
        self.top = int(top) if (top < self.top or not self.top and top >= 0) else self.top
        # For the right and bottm, must be bigger, without upper cap. Handle with care.
        self.right = int(right) if (right > self.right or not self.right) else self.right
        self.bottom = int(bottom) if (bottom > self.bottom or not self.bottom) else self.bottom

    def crop(self, mask, inplace=False):
        """
        Given a numpy mask, crops it and returns a copy
        :param mask: numpy array like, sliceable and implementing copy
        :param inplace: boolean, whether to return a copy of the array (default) or not.
        :return: crop (copy) of the same array
        """
        if inplace:
            result = mask[self.top:self.bottom, self.left:self.right]
        else:
            result = mask[self.top:self.bottom, self.left:self.right].copy()
        return result


class WandDetector(object):
    """
    Detects and reads a wand movement from a given CV2 streamable object
    baased on 
    https://github.com/sean-obrien/rpotter
    """

    def __init__(self, video, dilation_params=(5, 5), windows_size=(15, 15), max_level=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                 circles_dp=3,
                 circles_mindist=100,
                 circles_minradius=4,
                 circles_maxradius=14,
                 circles_threshold=5,
                 movement_threshold=80,
                 draw_windows=False):
        """
        The constructor
        :param video: A proper cv2 video renderer or the raspberry pi OpenCV object. MUST IMPLEMENT THE read() method.
        :param dilation_params:(tuple) a convoluted Opencv Dilation kernel
        :param windows_size: (tuple) Window size for motion tracking
        :param max_level: max depth level for motion detection
        :param criteria: criteria to use for the KNN clustering for motion tracking
        :param circles_dp: (float) dp parameter for the HughCircles method for CV2.
        :param circles_mindist: (int) minimum distance for the aforementioned method
        :param circles_minradius: (int) minum radius of the detected circles, aforementioned method
        :param circles_maxradius: (int) max radius for the same method.
        :param circles_threshold: (int) threshold of circles to read, aka, the top circles_threshold circles detected.
        :param movement_threshold: (int) threshold of movement after which an Optical Flow is computed.
        :param draw_windows: boolean: Debugging purposes, uses cv.imshow and creates the proper masks.
        """
        self.video = video
        self.draw_windows = draw_windows
        # Image manipulation params
        self.lk_params = dict(winSize=windows_size, maxLevel=max_level, criteria=criteria)
        self.dilation_params = dilation_params
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.movement_threshold = movement_threshold
        # objects for circle detection
        self.circles_dp = circles_dp
        self.circles_mindist = circles_mindist
        self.circles_minradius = circles_minradius
        self.circles_maxradius = circles_maxradius
        self.circles_threshold = circles_threshold
        # rectangle in which a potential spell is contained
        self.cropbox = CropRectangle()
        # color to trace the sigils, white in this case
        self.sigil_color = [255] * 3
        # Internal state of tracked objects
        self.prev_frame_gray = None  # The previous frame, grayscaled, in order to track movement
        self.prev_circles = []  # The previous detected circles.
        self.sigil_mask = None  # Where the gestures are drawn
        # The most important element: the mask in which a gesture is stored
        self.maybe_a_spell = None

    def find_wand(self):
        """
        Tries to detect all the wands in a scene. Will keep getting frames until we got a proper list of circles
        :return tuple (first frame to grayscale, list of circles)
        """
        circles = None
        gray = None
        logger.debug("Reading frame")
        try:
            ret = False
            circles = None
            # let's loop until we detect a wand:
            while circles is None:
                # while we did got not a valid frame...
                while not ret:
                    ret, frame = self.video.read()
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(END_KEY):
                        break
                # now, to gray
                gray = self._to_gray(frame)
                # Let's keep looping until we get a wand detected
                logger.debug("Starting to detect wands")
                circles = self._find_circles(gray)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(END_KEY):
                    break
            # Now let's update the internal states
            self.prev_circles = circles
            self.prev_frame_gray = gray
            # it's a good moment now to initialize the sigil mask
            self.sigil_mask = np.zeros_like(frame)
            logger.debug(circles)

        except Exception as e:
            logger.error("Error detecting a wand: {}".format(e))
        return gray, circles

    def read_wand(self):
        """
        Now that we already got a wand detected, let's try to capture a spell
        
        The rationale is: each call of the method will update the self.maybe_a_spell mask
        The controller is responsible for running this as long as it see fit.
        :return: a masked spell
        """
        logger.debug("Reading a wand")
        if self.prev_frame_gray is None:
            raise WandError("No previous frame found, terminating")

        # Grab the current frame:
        try:
            ret = False
            logger.debug("Trying to get a valid frame")
            # while we did got not a valid frame...
            while not ret:
                ret, frame = self.video.read()
            # now, to gray
            gray = self._to_gray(frame)

            # Let's do a bit of Optical Flow
            new_circles, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame_gray,
                                                            gray,
                                                            self.prev_frame_gray,
                                                            None, **self.lk_params)
            # Select good points
            new_valid_points = new_circles[st == 1]
            old_valid_points = self.prev_circles[st == 1]
            logger.debug("Got a valid frame!")
            for i, (new, old) in enumerate(zip(new_valid_points, old_valid_points)):
                logger.debug("Reading points")
                a, b = new.ravel()
                c, d = old.ravel()
                self.cropbox.update(a, b, c, d)
                # only try to detect gesture on highly-rated points
                if i < self.circles_threshold:
                    dist = hypot(a - c, b - d)
                    if dist < self.movment_threshold:
                        cv2.line(self.sigil_mask, (a, b), (c, d), self.sigil_color, 3)
                        if self.draw_windows:
                            cv2.circle(frame, (a, b), 5, self.sigil_color, -1)
                            cv2.putText(frame, str(i), (a, b), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            # There, let's crop the sigil mask to get the maybeaspell thingie
            self.maybe_a_spell = self.cropbox.crop(self.sigil_mask)
            img = cv2.add(frame, self.sigil_mask)

            # Let's update the global objects so the next iteration considers the current object as the previous.
            self.prev_frame_gray = gray.copy()
            self.prev_circles = new_valid_points.reshape(-1, 1, 2)
            # If we want to show the content,so be it.
            if self.draw_windows:
                cv2.putText(img, "Press {} to close.".format(END_KEY), (5, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                cv2.imshow("Raspberry Potter previous frame", self.prev_frame_gray)
                cv2.imshow("Raspberry Potter debug", img)
                cv2.imshow("Raspberry potter read sigil", self.maybe_a_spell)
        except Exception as e:
            logger.error("Error reading the wand: {}".format(e))
        return self.maybe_a_spell

    def _to_gray(self, a_frame, dilate_iterations=1):
        """
        Internal use. Turns a frame into a grayscaled, blurred image used for circle detection
        :param a_frame: (np array representing image) of a a valid frame
        :param dilate_iterations: (int) iterations for the dilate frame
        :return: a grayscaled, single channel, numpy array representing the frame
        """
        gray = cv2.cvtColor(a_frame, cv2.COLOR_BGR2GRAY)
        dilate_kernel = np.ones(self.dilation_params, np.uint8)
        gray = cv2.dilate(gray, dilate_kernel, iterations=dilate_iterations)
        return self.clahe.apply(gray)

    def _find_circles(self, a_frame):
        logging.debug("Detecting circles in the scene, aka, Wands")
        detected_circles = None
        try:
            detected_circles = cv2.HoughCircles(a_frame, cv2.HOUGH_GRADIENT,
                                                dp=self.circles_dp,
                                                minDist=self.circles_mindist,
                                                # These seems to been hard discovered by the owner of the original code,
                                                # not touching it.
                                                param1=240, param2=8,
                                                minRadius=self.circles_minradius,
                                                maxRadius=self.circles_maxradius)
        except Exception as e:
            logger.error("Unable to detect circles due to {}".format(e))
        if detected_circles is not None:
            # So, we detected something, reshaping
            logging.debug("Wands detected")
            detected_circles.shape = (detected_circles.shape[1], 1, detected_circles.shape[2])
            detected_circles = detected_circles[:, :, 0:2]
        return detected_circles
