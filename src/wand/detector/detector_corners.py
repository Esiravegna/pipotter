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
        frame = self.get_valid_frame()
        self.frame_height, self.frame_width = frame.shape  # Get frame shape
        self.tracePoints = []
        self.cameraFrame = None
        self.prev_cameraFrame = None  # Store the previous frame for optical flow
        self.last_keypoint_int_time = None
        self.latest_wand_frame = np.zeros((frame.shape), np.uint8)
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

    def get_valid_frame(self):
        """
        Continuously read frames from the video source until a valid frame is obtained.

        Returns:
            frame: A valid frame from the video source.
        """
        while True:
            ret, frame = self.video.read()
            if ret:
                return frame
            else:
                logger.warning("Failed to get a valid frame. Retrying...")
                time.sleep(0.1)  # Optional: short delay before retrying

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
                **self.lk_params,
            )

            # Filter out the good points based on the status output
            good_new = next_points[status == 1]
            good_old = self.tracePoints[status == 1]

            # Update points if tracking is successful
            if len(good_new) > 0:
                currentKeypointTime = time.time()

                if self.last_keypoint_int_time is None:
                    # On the first frame, just initialize the keypoint time
                    self.last_keypoint_int_time = currentKeypointTime
                else:
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

                # If this is the first initialization of the points
                if len(self.tracePoints) == 0:
                    self.tracePoints = good_new.reshape(-1, 1, 2)
                    self.last_keypoint_int_time = currentKeypointTime

            else:
                # If tracking fails, reinitialize points using Shi-Tomasi
                self.tracePoints = cv2.goodFeaturesToTrack(
                    bgSubbedCameraFrame, mask=None, **self.feature_params
                )

        # Update the previous frame for the next call
        self.prev_cameraFrame = self.cameraFrame.copy()

        # Generate the latest wand frame with tracing, keypoints, speed, and debug info
        self.latest_wand_frame = self.superimpose_trace_and_debug_info(
            self.cameraFrame, bgSubbedCameraFrame, self.tracePoints, speed
        )

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

    def superimpose_trace_and_debug_info(
        self, video_frame, tracing_frame, keypoints, speed, alpha=0.6
    ):
        """
        Superimpose the wand trace, detected keypoints, and debug information onto the current video frame.

        Args:
            video_frame: The original video frame to display.
            tracing_frame: The frame containing the wand trace.
            keypoints: The list of detected keypoints.
            speed: The speed of the wand movement.
            alpha: Transparency factor for the overlay.

        Returns:
            The combined frame with trace, keypoints, and debug information.
        """
        # Ensure both frames have the same dimensions
        if tracing_frame.shape[:2] != video_frame.shape[:2]:
            tracing_frame = cv2.resize(
                tracing_frame, (video_frame.shape[1], video_frame.shape[0])
            )

        # Convert tracing frame to a 3-channel BGR image if it's not already
        if len(tracing_frame.shape) == 2:  # Single channel image
            tracing_frame_bgr = cv2.cvtColor(tracing_frame, cv2.COLOR_GRAY2BGR)
        elif len(tracing_frame.shape) == 3 and tracing_frame.shape[2] == 1:
            tracing_frame_bgr = cv2.cvtColor(tracing_frame[:, :, 0], cv2.COLOR_GRAY2BGR)
        else:
            tracing_frame_bgr = tracing_frame

        # Convert video frame to BGR if it's a single channel image
        if len(video_frame.shape) == 2 or video_frame.shape[2] == 1:
            video_frame_bgr = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2BGR)
        else:
            video_frame_bgr = video_frame

        # Create colored trace overlay
        colored_trace = np.zeros_like(tracing_frame_bgr)
        colored_trace[:, :, 2] = tracing_frame_bgr[:, :, 0]  # Red channel for trace

        # Add keypoints to colored_trace
        for keypoint in keypoints:
            pt = (int(keypoint[0][0]), int(keypoint[0][1]))
            cv2.circle(
                colored_trace, pt, 5, (0, 255, 0), 3
            )  # Green circles for keypoints

        # Combine the video frame with the trace and keypoints
        combined_frame = cv2.addWeighted(video_frame_bgr, 1.0, colored_trace, alpha, 0)

        # Add debug information
        if speed != -1:
            cv2.putText(
                combined_frame,
                f"Speed: {speed:.2f} px/s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            cv2.putText(
                combined_frame,
                f"Points: {len(self.tracePoints)}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 250),
                1,
            )
            cv2.putText(
                combined_frame,
                f"Time: {time.strftime('%H:%M:%S')}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 250),
                1,
            )

        return combined_frame

    def get_a_spell_maybe(self):
        """
        Placeholder method to determine if the trace forms a spell. This could involve analyzing the pattern.
        """
        result = np.zeros((224, 224), np.uint8)
        if self.check_trace_validity():
            result = self._crop_save_trace(self.cameraFrame)
        return result

    def _update_trace_boundaries(self):
        """
        Update the bounding box around the trace points for cropping and saving the trace.
        """
        if len(self.tracePoints) > 0:
            # Since self.tracePoints is likely a NumPy array of shape (N, 1, 2), adjust the access
            xs = [pt[0][0] for pt in self.tracePoints]
            ys = [pt[0][1] for pt in self.tracePoints]
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
