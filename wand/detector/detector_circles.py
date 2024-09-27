import cv2
import logging
import numpy as np
from math import hypot
from core.error import WandError
from wand.spellscontainer import SpellsContainer
from wand.detector.base import BaseDetector

logger = logging.getLogger(__name__)


class WandDetector(BaseDetector):
    def __init__(
        self,
        video,
        dilation_params=(5, 5),
        windows_size=(15, 15),
        max_level=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        circles_dp=3,
        circles_mindist=100,
        circles_minradius=1,
        circles_maxradius=2,
        circles_threshold=5,
        movement_threshold=80,
        clahe_clip_limit=2.0,  # Default CLAHE clip limit
        hough_param1=200,  # Default HoughCircles param1
        hough_param2=5,  # Default HoughCircles param2
    ):
        """
        Initialize WandDetector.
        """
        self.video = video
        self.lk_params = dict(
            winSize=windows_size, maxLevel=max_level, criteria=criteria
        )
        self.dilation_params = dilation_params
        self.movement_threshold = movement_threshold

        # Circle detection parameters
        self.circles_dp = circles_dp
        self.circles_mindist = circles_mindist
        self.circles_minradius = circles_minradius
        self.circles_maxradius = circles_maxradius
        self.circles_threshold = circles_threshold

        # Additional parameters for adaptive processing
        self.clahe_clip_limit = clahe_clip_limit
        self.hough_param1 = hough_param1
        self.hough_param2 = hough_param2

        # Initialize tracking variables
        self.prev_frame_gray = None
        self.prev_circles = None  # Set to None initially
        self.sigil_mask = None
        self.maybe_a_spell = np.array([])

        # Spell storage container
        self.spells_container = SpellsContainer()
        self.sigil_color = [255] * 3

        # Store the latest frames for external use
        self.latest_wand_frame = None
        self.latest_sigil_frame = None
        self.latest_debug_frame = None
        self.latest_circles_frame = None

        # Background subtractor
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )

        # CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=(8, 8)
        )

        # Morphological kernel
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def find_wand(self):
        """
        Detects the wand (or reflective tip) in the video stream.
        Uses circle detection on preprocessed (grayscale) frames.
        """
        circles = None
        try:
            while circles is None:
                ret, frame = self.video.read()
                if not ret or frame is None:
                    raise WandError("Unable to read frame from video stream.")
                gray = frame  # Frame is already in grayscale

                # Apply image processing enhancements
                processed_frame = self._process_frame(gray)

                # Detect circles
                circles = self._find_circles(processed_frame)

                # Save frame with detected circles
                debug_frame = frame.copy()
                self.latest_circles_frame = processed_frame.copy()
                if circles is not None:
                    logger.debug(f"circles {circles}")
                    for circle in circles:
                        x, y = circle
                        cv2.circle(debug_frame, (int(x), int(y)), 5, (0, 0, 255), 1)
                        cv2.circle(
                            self.latest_circles_frame,
                            (int(x), int(y)),
                            5,
                            (255, 255, 255),
                            2,
                        )
                    self.latest_debug_frame = debug_frame
            circles = circles.astype(np.float32)
            # Store debug frame for streaming
            self.prev_circles = circles
            self.prev_frame_gray = gray
            self.sigil_mask = np.zeros_like(frame)
            self.spells_container.reset()
        except Exception as e:
            logger.error(f"Error detecting wand: {e}")
            raise WandError("Wand detection failed.")

    def read_wand(self, frame):
        """
        Reads wand movement and processes it.
        Expects a preprocessed (grayscale) frame.
        """

        # Convert frame to grayscale if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply image processing enhancements
        processed_frame = self._process_frame(frame)

        if self.spells_container.auto_clear() or self.prev_frame_gray is None:
            logger.info(
                "Spell list expired or no previous frame. Initializing wand detection."
            )
            self.find_wand()  # Call the method to initialize internal state
            if self.prev_frame_gray is None:
                logger.error("Unable to initialize wand detection.")
                self.latest_wand_frame = frame.copy()  # Always update with raw frame
                return self.maybe_a_spell  # Return empty array if initialization fails

        # Validate self.prev_circles before using it in calcOpticalFlowPyrLK
        if self.prev_circles is None or len(self.prev_circles) == 0:
            logger.error("No previous circles found to track. Skipping frame.")
            self.latest_wand_frame = frame.copy()  # Always update with raw frame
            return self.maybe_a_spell

        try:
            # Ensure correct shape for prev_circles before calculating optical flow
            self.prev_circles = self.prev_circles.reshape(-1, 1, 2).astype(np.float32)

            # Calculate optical flow to track movement
            new_circles, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame_gray,
                processed_frame,
                self.prev_circles,
                None,
                **self.lk_params,
            )

            # Flatten the status array for proper boolean indexing
            status = status.flatten()  # Now status will have shape (n,)

            # Reshape new_circles and prev_circles to (-1, 2) for compatibility with status
            new_circles = new_circles.reshape(-1, 2)
            self.prev_circles = self.prev_circles.reshape(-1, 2)

            # Ensure that status and new_circles have the same number of points
            if new_circles.shape[0] != status.shape[0]:
                raise ValueError("Mismatch between new_circles and status shape!")

            # Filter valid points based on the status returned by OpticalFlowPyrLK
            valid_new = new_circles[status == 1]
            valid_old = self.prev_circles[status == 1]

            # Check if there are valid points left after filtering
            if len(valid_new) == 0:
                logger.info("No valid points found after filtering. Skipping frame.")
                self.latest_wand_frame = frame.copy()  # Always update with raw frame
                return self.maybe_a_spell  # Return empty array if no valid points

            # Filter out static circles (optional based on your previous logic)
            moving_circles = self._filter_static_circles(valid_new, valid_old)
            if len(moving_circles) == 0:
                logger.info("No moving circles found after filtering static ones.")
                self.latest_wand_frame = frame.copy()  # Always update with raw frame
                return self.maybe_a_spell  # Return empty array if no moving circles

            # Update internal state and debug frame
            self._process_valid_points(frame, moving_circles, valid_old)

            # Get bounding box for the detected sigil
            left, top, right, bottom = self.spells_container.get_box()
            self.maybe_a_spell = self.sigil_mask[top:bottom, left:right].copy()

            # Add the sigil mask to the frame and update the latest wand frame
            self.latest_wand_frame = cv2.add(frame, self.sigil_mask)
            self.latest_sigil_frame = self.maybe_a_spell

            # Update the previous frame and circles for next iteration
            self.prev_frame_gray = processed_frame.copy()
            self.prev_circles = moving_circles.reshape(-1, 1, 2).astype(np.float32)

        except (cv2.error, ValueError, Exception) as e:
            logger.error(f"Error reading wand: {e}")
            self.latest_wand_frame = (
                frame.copy()
            )  # Always update with raw frame on failure
            return self.maybe_a_spell  # Return empty array on failure

        return self.maybe_a_spell

    def get_a_spell_maybe(self):
        """
        Return the trace as a NumPy array of points.
        """
        return self.maybe_a_spell  # No change needed, already a NumPy array of points

    def reset(self):
        """
        Resets the internal state so the cycle can begin anew
        """
        self.latest_wand_frame = None
        self.latest_sigil_frame = None
        self.latest_debug_frame = None
        return self.spells_container.reset()

    def _process_frame(self, frame):
        """
        Apply preprocessing steps to enhance wand detection and binarize
        the image to focus on white elements.
        """
        # Step 1: Apply CLAHE to enhance local contrast
        frame = self.clahe.apply(frame)

        # Step 2: Apply background subtraction to remove the background
        fg_mask = self.backSub.apply(frame)

        # Step 3: Apply morphological operations to clean up the mask
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)

        # Step 4: Apply thresholding to binarize the frame
        # Set a threshold value to keep only bright (white) elements
        # _, binarized_frame = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)

        # Optional Step: Apply additional morphological operations if needed
        binarized_frame = cv2.dilate(frame, self.kernel, iterations=1)
        binarized_frame = cv2.erode(frame, self.kernel, iterations=1)

        return binarized_frame

    def _find_circles(self, gray_frame):
        """
        Detects circles (wands) in the preprocessed grayscale frame using HoughCircles.
        """
        result = None
        try:
            circles = cv2.HoughCircles(
                gray_frame,
                cv2.HOUGH_GRADIENT,
                dp=self.circles_dp,
                minDist=self.circles_mindist,
                param1=self.hough_param1,  # Lower param1 for less sensitivity to noise
                param2=self.hough_param2,  # Increase param2 for higher detection threshold
                minRadius=self.circles_minradius,
                maxRadius=self.circles_maxradius,
            )
            if circles is not None:
                circles = np.uint16(np.around(circles))
                result = circles[:, :, :2].reshape(-1, 2)
        except cv2.error as e:
            logger.error(f"Error detecting circles: {e}")
        return result

    def _filter_static_circles(self, new_circles, old_circles):
        """
        Filters out circles that do not move significantly over time.
        """
        moving_circles = []
        for new, old in zip(new_circles, old_circles):
            dist = hypot(new[0] - old[0], new[1] - old[1])
            if dist > self.movement_threshold / 2:
                moving_circles.append(new)
        return np.array(moving_circles).reshape(-1, 1, 2)

    def _process_valid_points(self, frame, new_points, old_points):
        """
        Processes valid optical flow points, draws the sigil, and updates the spells container.
        """
        for i, (new, old) in enumerate(zip(new_points, old_points)):
            a, b = new.ravel()  # New circle positions (floats)
            c, d = old.ravel()  # Old circle positions (floats)

            # Only draw valid movements within the distance threshold
            dist = hypot(a - c, b - d)
            if dist < self.movement_threshold:
                self.spells_container[i] = [a, b, c, d]

                pt1 = (int(a), int(b))
                pt2 = (int(c), int(d))

                if self._valid_point(pt1) and self._valid_point(pt2):
                    # Draw the sigil line on the mask
                    cv2.line(self.sigil_mask, pt1, pt2, self.sigil_color, 3)
                    cv2.circle(frame, pt1, 5, self.sigil_color, -1)
                    cv2.putText(
                        frame, str(i), pt1, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255)
                    )
                else:
                    logger.error(f"Points out of bounds: pt1={pt1}, pt2={pt2}")

    def _valid_point(self, point):
        """
        Validates if the given point is within the frame boundaries.
        """
        x, y = point
        return 0 <= x < self.sigil_mask.shape[1] and 0 <= y < self.sigil_mask.shape[0]
