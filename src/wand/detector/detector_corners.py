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
        self.last_keypoint_int_time = None

        # Feature parameters for goodFeaturesToTrack (from Shi-Tomasi corner detection)
        self.feature_params = dict(
            maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )

        self.bgsub = cv2.createBackgroundSubtractorMOG2()  # Background subtractor

    def detect_wand(self, frame):
        """
        Detects the wand movement in the given frame and updates the trace.

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

        # Detect points using goodFeaturesToTrack (from Shi-Tomasi)
        points = cv2.goodFeaturesToTrack(
            bgSubbedCameraFrame, mask=None, **self.feature_params
        )

        if points is not None:
            currentKeypointTime = time.time()
            if len(self.tracePoints) > 0:
                # Calculate speed by measuring the distance between the last keypoint and the current one
                elapsed = currentKeypointTime - self.last_keypoint_int_time
                pt1 = self.tracePoints[-1]
                pt2 = points[0]
                distance = self._distance(pt1, pt2)
                speed = distance / elapsed

                # Append the current point if it is moving below the maximum trace speed
                if speed < self.MAX_TRACE_SPEED:
                    self.tracePoints.append(pt2)
                    self.last_keypoint_int_time = currentKeypointTime

                    # Optionally, draw trace (line) between points
                    pt1_coords = (int(pt1[0]), int(pt1[1]))
                    pt2_coords = (int(pt2[0]), int(pt2[1]))
                    cv2.line(
                        self.cameraFrame,
                        pt1_coords,
                        pt2_coords,
                        (255, 0, 0),
                        thickness=self.TRACE_THICKNESS,
                    )
            else:
                # First keypoint initialization
                self.tracePoints.append(points[0])
                self.last_keypoint_int_time = currentKeypointTime

    def _distance(self, pt1, pt2):
        """Compute Euclidean distance between two points."""
        return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def reset(self):
        """Reset the trace points and start fresh."""
        self.tracePoints = []

    def check_trace_validity(self):
        """
        Checks if the current trace is valid based on specific criteria (such as length, speed, etc.).
        """
        result = False
        if len(self.tracePoints) > self.POINTS_BUFFER_SIZE:
            result = True
        return result

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
