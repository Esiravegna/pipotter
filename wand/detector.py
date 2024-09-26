import cv2
import logging
import numpy as np
from math import hypot
from core.error import WandError
from wand.spellscontainer import SpellsContainer

logger = logging.getLogger(__name__)


class WandDetector:
    def __init__(
        self,
        video,
        movement_threshold=300,
    ):
        """
        Initialize WandDetector.
        """
        self.video = video
        self.movement_threshold = movement_threshold

        # Initialize tracking variables
        self.prev_keypoints = None
        self.sigil_mask = None
        self.maybe_a_spell = np.array([])

        # Spell storage container
        self.spells_container = SpellsContainer(expiration_time=15)
        self.sigil_color = [255] * 3

        # Store the latest frames for external use
        self.latest_wand_frame = None
        self.latest_sigil_frame = None
        self.latest_debug_frame = None
        self.latest_keypoints_frame = None

        # Blob detector parameters
        self.detector = self._create_blob_detector()

    def _create_blob_detector(self):
        """
        Create a SimpleBlobDetector with specified parameters.
        """
        params = cv2.SimpleBlobDetector_Params()

        # Set thresholds
        params.minThreshold = 150
        params.maxThreshold = 250

        # Filter by color (looking for bright objects)
        params.filterByColor = True
        params.blobColor = 255

        # Filter by circularity (looking for round objects)
        params.filterByCircularity = True
        params.minCircularity = 0.68

        # Filter by area
        params.filterByArea = True
        params.minArea = 30

        # Create and return the detector
        return cv2.SimpleBlobDetector_create(params)

    def detect_wand(self, frame):
        """
        Detects and processes the wand movement in the given frame.
        """
        if self.spells_container.auto_clear():
            logger.debug("Spell list expired or no valid points. Resetting.")
            self.reset()

        # Convert frame to grayscale if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect keypoints using blob detection
        keypoints = self.detector.detect(frame)

        # Save the frame with detected keypoints for debugging
        self.latest_keypoints_frame = cv2.drawKeypoints(
            frame,
            keypoints,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        # Validate keypoints before proceeding
        if not keypoints:
            logger.debug("No keypoints detected. Skipping frame.")
            self.latest_wand_frame = frame.copy()  # Always update with raw frame
            return self.maybe_a_spell

        try:
            # Convert keypoints to a format for processing
            new_keypoints = [kp.pt for kp in keypoints]
            new_keypoints = np.array(new_keypoints, dtype=np.float32).reshape(-1, 1, 2)

            # Check if previous keypoints are None before filtering
            if self.prev_keypoints is None:
                self.prev_keypoints = new_keypoints
                logger.info("Initialized previous keypoints.")
                self.sigil_mask = np.zeros_like(frame)
                return self.maybe_a_spell

            # Calculate distances between old and new keypoints to filter out static ones
            moving_keypoints = self._filter_static_keypoints(
                new_keypoints, self.prev_keypoints
            )
            if moving_keypoints is None or len(moving_keypoints) == 0:
                logger.info("No moving keypoints found after filtering static ones.")
                self.latest_wand_frame = frame.copy()  # Always update with raw frame
                return self.maybe_a_spell  # Return empty array if no moving keypoints

            # Update internal state and debug frame
            self._process_valid_points(frame, moving_keypoints, self.prev_keypoints)

            # Get bounding box for the detected sigil
            left, top, right, bottom = self.spells_container.get_box()
            self.maybe_a_spell = self.sigil_mask[top:bottom, left:right].copy()

            # Add the sigil mask to the frame and update the latest wand frame
            self.latest_wand_frame = cv2.add(frame, self.sigil_mask)
            self.latest_sigil_frame = self.maybe_a_spell

            # Update the previous frame and keypoints for next iteration
            self.prev_keypoints = moving_keypoints.reshape(-1, 1, 2).astype(np.float32)

        except (cv2.error, ValueError, Exception) as e:
            logger.error(f"Error detecting wand: {e}")
            self.latest_wand_frame = (
                frame.copy()
            )  # Always update with raw frame on failure
            return self.maybe_a_spell  # Return empty array on failure

        return self.maybe_a_spell

    def reset(self):
        """
        Resets the internal state so the cycle can begin anew.
        """
        self.latest_sigil_frame = None
        self.latest_debug_frame = None
        self.maybe_a_spell = np.array([])
        self.spells_container.reset()

    def _filter_static_keypoints(self, new_keypoints, old_keypoints):
        """
        Filters out keypoints that do not move significantly over time.
        """
        # Check if either new_keypoints or old_keypoints is None before iterating
        if new_keypoints is None or old_keypoints is None:
            logger.error("new_keypoints or old_keypoints is None.")
            return None

        moving_keypoints = []
        for new, old in zip(new_keypoints, old_keypoints):
            dist = hypot(new[0][0] - old[0][0], new[0][1] - old[0][1])
            if dist > self.movement_threshold / 2:
                moving_keypoints.append(new)

        return np.array(moving_keypoints).reshape(-1, 1, 2)

    def _process_valid_points(self, frame, new_points, old_points):
        """
        Processes valid points, draws the sigil, and updates the spells container.
        """
        # Check if new_points or old_points is None before iterating
        if new_points is None or old_points is None:
            logger.error("new_points or old_points is None.")
            return

        for i, (new, old) in enumerate(zip(new_points, old_points)):
            a, b = new.ravel()  # New point positions (floats)
            c, d = old.ravel()  # Old point positions (floats)

            # Only draw valid movements within the distance threshold
            dist = hypot(a - c, b - d)
            logger.info(f"Distance {dist}")
            if dist < self.movement_threshold:
                logger.info(f"adding {[a, b, c, d]} to the spells contaner")
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
