import cv2
import numpy as np
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


class WandDetector:
    def __init__(
        self,
        video,
        brightness_threshold=150,
        max_trace_speed=550,
        trace_buffer_size=40,
        max_trace_duration=2.0,
    ):
        """
        Initialize the WandDetector with a video source.

        Args:
            video: Video source object with a `read()` method that returns (ret, frame).
            brightness_threshold: Minimum pixel intensity to consider as wand tip (IR light reflection).
            max_trace_speed: Maximum speed (in pixels/second) to consider valid wand movement.
            trace_buffer_size: Maximum number of points in the trace buffer.
            max_trace_duration: Maximum duration (in seconds) to keep points in the buffer.
        """
        self.video = video
        self.brightness_threshold = brightness_threshold
        self.max_trace_speed = max_trace_speed
        self.trace_buffer_size = trace_buffer_size
        self.max_trace_duration = max_trace_duration

        # Get initial frame to determine frame dimensions
        frame = self.get_valid_frame()
        self.frame_height, self.frame_width = frame.shape  # Get frame shape

        # Frames for different purposes
        self.latest_wand_frame = None  # Raw frame being processed
        self.latest_debug_frame = None  # Frame with debug annotations
        self.latest_keypoints_frame = None  # Frame highlighting keypoints
        self.wand_move_tracing_frame = np.zeros(
            (self.frame_height, self.frame_width), np.uint8
        )  # Initialize trace frame

        self.previous_wand_tip = (
            None  # Store previous wand tip position to track movement
        )
        self.wand_tip_movement = deque(
            maxlen=trace_buffer_size
        )  # Deque for tracking wand tip movements

        # Initialize blob detector
        self.blob_detector = self._get_blob_detector()

        # Timestamp for tracking speed
        self.last_keypoint_time = time.time()
        self.last_detection_time = time.time()  # Track last time a wand was detected

    def _get_blob_detector(self):
        """
        Create and configure the blob detector used to detect the wand tip with refined parameters.

        Returns:
            A configured blob detector.
        """
        # Set up SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()

        # Configure the parameters for the blob detector
        params.filterByColor = True
        params.blobColor = (
            255  # Detect white blobs (assumes wand tip is bright/reflective)
        )

        # Refined area parameters to filter out noise and irrelevant blobs
        params.filterByArea = True
        params.minArea = (
            20  # Increased minimum area to filter out smaller noise artifacts
        )
        params.maxArea = 2000  # Decreased maximum area to ignore larger blobs

        # Additional filters to refine detection
        params.filterByCircularity = True
        params.minCircularity = (
            0.7  # Set a minimum circularity to filter out irregular shapes
        )

        params.filterByConvexity = True
        params.minConvexity = (
            0.8  # Set a minimum convexity to filter out non-convex shapes
        )

        params.filterByInertia = True
        params.minInertiaRatio = (
            0.4  # Set a minimum inertia ratio to filter out elongated shapes
        )

        # Create a blob detector with the given parameters
        blob_detector = cv2.SimpleBlobDetector_create(params)

        return blob_detector

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
        Detect wand tip movement in the given frame and update internal state.

        Args:
            frame: The current grayscale video frame to be processed.
        """
        # Detect blobs in the frame
        keypoints = self.blob_detector.detect(frame)

        # Create a copy of the frame for processing and superimposing the trace
        self.latest_wand_frame = frame.copy()  # Store raw frame
        self.latest_debug_frame = frame.copy()  # Base frame for debug annotations
        self.latest_keypoints_frame = (
            frame.copy()
        )  # Base frame for keypoints visualization
        speed = -1
        current_time = time.time()

        if keypoints:
            # Update last detection time
            self.last_detection_time = current_time

            # Process detected keypoints
            if self.previous_wand_tip:
                elapsed_time = current_time - self.last_keypoint_time
                distance = self._distance(self.previous_wand_tip.pt, keypoints[0].pt)
                speed = distance / elapsed_time
                if speed < self.max_trace_speed:
                    # Update trace if speed is within limit
                    self._update_trace(keypoints[0], current_time)
                else:
                    logger.warning(
                        f"Excessive wand speed detected: {speed:.2f} pixels/second."
                    )
            else:
                # Initial detection, no speed check
                self._update_trace(keypoints[0], current_time)

            # Update previous wand tip and timestamp
            self.previous_wand_tip = keypoints[0]
            self.last_keypoint_time = current_time

            # Update maybe_a_spell with current wand tip trajectory
            self._update_debug_frame(keypoints[0], speed)
            logger.info(f"Wand detected at {keypoints[0].pt}.")
        else:
            # No keypoints detected, check for timeout
            if (current_time - self.last_detection_time) > self.max_trace_duration:
                logger.warning(
                    "Wand not detected for a long time. Resetting the trace."
                )
                self._update_debug_frame_no_detection(
                    current_time - self.last_detection_time
                )
                self.reset()  # Automatically reset if no wand detected for max_trace_duration
            else:
                self._update_debug_frame_no_detection(
                    current_time - self.last_detection_time
                )

        # Superimpose the trace on the latest_wand_frame if it is available
        if self.latest_wand_frame is not None:
            # Superimpose the trace on the latest_wand_frame
            self.latest_wand_frame = self.superimpose_trace_on_video(
                self.latest_wand_frame, self.wand_move_tracing_frame
            )

    def _update_trace(self, keypoint, current_time):
        """
        Update the wand trace with the new keypoint.

        Args:
            frame: The current frame to draw the trace on.
            keypoint: The detected wand tip position as a cv2.KeyPoint object.
            current_time: The current timestamp of the detection.
        """
        pt1 = (
            (int(self.previous_wand_tip.pt[0]), int(self.previous_wand_tip.pt[1]))
            if self.previous_wand_tip
            else None
        )
        pt2 = (int(keypoint.pt[0]), int(keypoint.pt[1]))

        # Add new point with timestamp to the deque
        self._update_trace_buffer(keypoint, current_time)

        # Draw the trace line on the tracing frame if there's a previous point
        if pt1:
            cv2.line(self.wand_move_tracing_frame, pt1, pt2, 255, 2)

        # Update keypoints frame with current wand tip (visualization)
        # Optional: You can skip this visualization if it's not needed
        cv2.circle(self.latest_keypoints_frame, pt2, 3, (0, 0, 255), 1)

    def _update_trace_buffer(self, keypoint, current_time):
        """
        Update the trace buffer by adding a new keypoint and timestamp, and removing old points if needed.

        Args:
            keypoint: The detected wand tip position as a cv2.KeyPoint object.
            current_time: The current timestamp of the detection.
        """
        # Add new keypoint with timestamp to the deque
        self.wand_tip_movement.append((keypoint, current_time))

        # Remove points that are older than max_trace_duration
        self._remove_old_points(current_time)

    def _remove_old_points(self, current_time):
        """
        Remove points from the trace buffer that are older than the max_trace_duration.

        Args:
            current_time: The current timestamp to compare against.
        """
        while (
            self.wand_tip_movement
            and (current_time - self.wand_tip_movement[0][1]) > self.max_trace_duration
        ):
            self.wand_tip_movement.popleft()  # Remove the oldest point if it's too old

    def _update_debug_frame(self, keypoint, speed):
        """
        Update the debug frame with annotations for the detected wand tip.

        Args:
            keypoint: The detected wand tip position as a cv2.KeyPoint object.
            speed: The speed of the wand movement.
        """
        pt = (int(keypoint.pt[0]), int(keypoint.pt[1]))
        cv2.circle(self.latest_debug_frame, pt, 5, (255, 0, 0), -1)
        cv2.putText(
            self.latest_debug_frame,
            f"Wand Tip: {pt}",
            (pt[0] + 10, pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
        cv2.putText(
            self.latest_debug_frame,
            f"Speed: {speed:.2f} px/s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
        cv2.putText(
            self.latest_debug_frame,
            f"Points: {len(self.wand_tip_movement)}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
        cv2.putText(
            self.latest_debug_frame,
            f"Time: {time.strftime('%H:%M:%S')}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    def _update_debug_frame_no_detection(self, time_since_last_detection):
        """
        Update the debug frame with a message when the wand has not been detected for a while.

        Args:
            time_since_last_detection: Time elapsed since the last wand detection.
        """
        cv2.putText(
            self.latest_debug_frame,
            "Wand Not Detected!",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            self.latest_debug_frame,
            f"Time Elapsed: {time_since_last_detection:.2f} s",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    def _distance(self, pt1, pt2):
        """
        Calculate the Euclidean distance between two points.

        Args:
            pt1: A cv2.KeyPoint representing the first point.
            pt2: A cv2.KeyPoint representing the second point.

        Returns:
            The Euclidean distance between the two points.
        """
        point1 = np.array(pt1)
        point2 = np.array(pt2)
        return np.linalg.norm(point1 - point2)

    def check_trace_validity(self):
        """
        Check if the wand trace is valid based on trace length and time criteria.

        Returns:
            True if the trace is valid and ready for further processing.
        """
        result = False
        if len(self.wand_tip_movement) > self.trace_buffer_size - 5:
            result = True

        return result

    def _crop_and_redraw_trace(self):
        """
        Generate a new clean image with the wand trace drawn, cropped to the bounding box of the trace.
        This avoids thick lines due to multiple drawings on the same canvas.

        Returns:
            result: The newly drawn and cropped trace image.
        """
        cropped_trace = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        # Ensure there are enough points in the trace buffer
        if self.check_trace_validity():
            # Get the bounding box for the trace based on the stored points
            trace_x = [int(k.pt[0]) for k, _ in self.wand_tip_movement]
            trace_y = [int(k.pt[1]) for k, _ in self.wand_tip_movement]
            upper_left = (max(min(trace_x) - 10, 0), max(min(trace_y) - 10, 0))
            lower_right = (
                min(max(trace_x) + 10, self.frame_width),
                min(max(trace_y) + 10, self.frame_height),
            )

            # Create a blank canvas for the cropped area
            crop_width = lower_right[0] - upper_left[0]
            crop_height = lower_right[1] - upper_left[1]
            cropped_trace = np.zeros((crop_height, crop_width), dtype=np.uint8)

            # Draw the trace lines based on the buffered points on this new canvas
            for i in range(1, len(self.wand_tip_movement)):
                pt1 = (
                    int(self.wand_tip_movement[i - 1][0].pt[0]) - upper_left[0],
                    int(self.wand_tip_movement[i - 1][0].pt[1]) - upper_left[1],
                )
                pt2 = (
                    int(self.wand_tip_movement[i][0].pt[0]) - upper_left[0],
                    int(self.wand_tip_movement[i][0].pt[1]) - upper_left[1],
                )

                # Draw a thin line on the clean canvas
                cv2.line(cropped_trace, pt1, pt2, 255, 2)
        return cropped_trace

    def _crop_trace(self):
        """
        Crop the wand trace image to the bounding box of the trace.

        Returns:
            Cropped image of the trace.
        """
        result = np.zeros((self.frame_width, self.frame_height), np.uint8)
        if self.wand_tip_movement:
            # Get the bounding box for the trace
            trace_x = [int(k.pt[0]) for k, _ in self.wand_tip_movement]
            trace_y = [int(k.pt[1]) for k, _ in self.wand_tip_movement]
            upper_left = (max(min(trace_x) - 10, 0), max(min(trace_y) - 10, 0))
            lower_right = (
                min(max(trace_x) + 10, self.frame_width),
                min(max(trace_y) + 10, self.frame_height),
            )

            # Crop the trace without resizing
            cropped_trace = self.wand_move_tracing_frame[
                upper_left[1] : lower_right[1], upper_left[0] : lower_right[0]
            ]
            result = cropped_trace.astype(np.uint8)
        return result

    def get_a_spell_maybe(self):
        """
        Convert the spell points to a visual image.
        """
        return self._crop_trace()

    def reset(self):
        """
        Reset the internal state after a spell is detected and processed.
        """
        self.wand_tip_movement.clear()  # Clear movement trajectory
        self.previous_wand_tip = None
        self.wand_move_tracing_frame.fill(0)  # Clear tracing frame
        self.last_detection_time = time.time()  # Reset last detection time
        logger.debug("WandDetector state has been reset.")

    # Function to superimpose the trace on the current video frame
    def superimpose_trace_on_video(self, video_frame, tracing_frame, alpha=0.6):
        """
        Superimpose the wand trace onto the current video frame.
        """
        # Print shapes for debugging
        logger.debug(f"Video frame shape: {video_frame.shape}")
        logger.debug(f"Tracing frame shape: {tracing_frame.shape}")

        # Ensure both frames have the same dimensions
        if tracing_frame.shape[:2] != video_frame.shape[:2]:
            tracing_frame = cv2.resize(
                tracing_frame, (video_frame.shape[1], video_frame.shape[0])
            )

        if len(video_frame.shape) == 2 or video_frame.shape[2] == 1:
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2BGR)

        tracing_frame_bgr = cv2.cvtColor(tracing_frame, cv2.COLOR_GRAY2BGR)

        colored_trace = np.zeros_like(tracing_frame_bgr)
        colored_trace[:, :, 2] = tracing_frame

        superimposed_frame = cv2.addWeighted(video_frame, 1.0, colored_trace, alpha, 0)

        return superimposed_frame
