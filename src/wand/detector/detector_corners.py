import cv2
import logging
import numpy as np
import time
import math
from wand.detector.base import BaseDetector
from core.utils import resize_with_aspect_ratio

logger = logging.getLogger(__name__)


class WandDetector(BaseDetector):
    POINTS_BUFFER_SIZE = 20
    TRACE_THICKNESS = 3
    MAX_TRACE_SPEED = 250  # pixels/second
    CROPPED_IMG_MARGIN = 10  # pixels
    OUTPUT_SIZE = 224  # Output image size for the spell representation

    def __init__(self, video):
        self.video = video
        self.tracePoints = []
        self.cameraFrame = None
        self.prev_cameraFrame = None  # Store the previous frame for optical flow
        self.last_keypoint_int_time = None

        # Feature parameters for goodFeaturesToTrack (from Shi-Tomasi corner detection)
        self.feature_params = dict(
            maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )

        # Parameters for Lucas-Kanade Optical Flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self.bgsub = cv2.createBackgroundSubtractorMOG2()  # Background subtractor

    def detect_wand(self, frame):
        """
        Detects and tracks the wand movement in the given frame using Optical Flow.

        Args:
            frame: The current grayscale video frame to be processed.
        """
        self.cameraFrame = frame
        speed = 0

        # Apply background subtraction to isolate moving objects
        fgmask = self.bgsub.apply(self.cameraFrame)
        bgSubbedCameraFrame = cv2.bitwise_and(
            self.cameraFrame, self.cameraFrame, mask=fgmask
        )

        if self.tracePoints is None or len(self.tracePoints) == 0:
            # Initialize points using Shi-Tomasi corner detection (goodFeaturesToTrack)
            self.tracePoints = cv2.goodFeaturesToTrack(
                bgSubbedCameraFrame, mask=None, **self.feature_params
            )
        else:
            # Use Optical Flow to track the points in the next frame
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_cameraFrame,
                self.cameraFrame,
                self.tracePoints,
                None,
                **self.lk_params
            )

            # Filter out the good points based on the status output
            good_new = next_points[status == 1]
            good_old = self.tracePoints[status == 1]

            # Update points if tracking is successful
            if len(good_new) > 0:
                currentKeypointTime = time.time()
                if len(self.tracePoints) > 0:
                    # Calculate speed by measuring the distance between the last and current point
                    elapsed = currentKeypointTime - self.last_keypoint_int_time
                    pt1 = good_old[-1]
                    pt2 = good_new[-1]
                    distance = self._distance(pt1, pt2)
                    speed = distance / elapsed

                    # Append the current point if it is moving below the maximum trace speed
                    if speed < self.MAX_TRACE_SPEED:
                        self.tracePoints = good_new.reshape(-1, 1, 2)
                        self.last_keypoint_int_time = currentKeypointTime

                        # Optionally, draw trace (line) between points
                        for i, (new, old) in enumerate(zip(good_new, good_old)):
                            a, b = new.ravel()
                            c, d = old.ravel()
                            cv2.line(
                                self.cameraFrame,
                                (int(a), int(b)),
                                (int(c), int(d)),
                                (255, 0, 0),
                                thickness=self.TRACE_THICKNESS,
                            )
                else:
                    # First keypoint initialization after optical flow
                    self.tracePoints = good_new.reshape(-1, 1, 2)
                    self.last_keypoint_int_time = currentKeypointTime

            else:
                # If tracking fails, reinitialize points using Shi-Tomasi
                self.tracePoints = cv2.goodFeaturesToTrack(
                    bgSubbedCameraFrame, mask=None, **self.feature_params
                )

        # Update the previous frame for the next call
        self.prev_cameraFrame = self.cameraFrame.copy()

    def _distance(self, pt1, pt2):
        """Compute Euclidean distance between two points."""
        return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def reset(self):
        """Reset the trace points and start fresh."""
        self.tracePoints = []
        self.prev_cameraFrame = None  # Clear the previous frame

    def check_trace_validity(self):
        """
        Checks if the current trace is valid based on specific criteria (such as length, speed, etc.).
        """
        return len(self.tracePoints) > self.POINTS_BUFFER_SIZE

    def superimpose_trace_and_debug_info(self, frame):
        """
        Draws the trace and debug information on the frame for visualization.
        """
        for i in range(1, len(self.tracePoints)):
            pt1 = self.tracePoints[i - 1]
            pt2 = self.tracePoints[i]
            pt1_coords = (int(pt1[0]), int(pt1[1]))
            pt2_coords = (int(pt2[0]), int(pt2[1]))
            cv2.line(
                frame,
                pt1_coords,
                pt2_coords,
                (0, 255, 0),
                thickness=self.TRACE_THICKNESS,
            )

    def get_a_spell_maybe(self):
        """
        Placeholder method to determine if the trace forms a spell. This could involve analyzing the pattern.
        """
        result = np.zeros((224, 224), np.uint8)
        if self.check_trace_validity():
            result = self._crop_save_trace()
        return result

    def _update_trace_boundaries(self):
        """
        Update the bounding box around the trace points for cropping and saving the trace.
        """
        if len(self.tracePoints) > 0:
            xs = [pt[0] for pt in self.tracePoints]
            ys = [pt[1] for pt in self.tracePoints]
            x_min = min(xs) - self.CROPPED_IMG_MARGIN
            x_max = max(xs) + self.CROPPED_IMG_MARGIN
            y_min = min(ys) - self.CROPPED_IMG_MARGIN
            y_max = max(ys) + self.CROPPED_IMG_MARGIN
            return (x_min, y_min, x_max, y_max)
        return None

    def _crop_save_trace(self, frame):
        """
        Crop and save the trace image to be used as input for further spell recognition.
        """
        bounds = self._update_trace_boundaries()
        if bounds:
            x_min, y_min, x_max, y_max = bounds
            cropped_frame = frame[y_min:y_max, x_min:x_max]
            resized_cropped_frame = resize_with_aspect_ratio(
                cropped_frame, (self.OUTPUT_SIZE, self.OUTPUT_SIZE)
            )
            return resized_cropped_frame
        return None
