import cv2
import logging
import numpy as np
import time
import math
from wand.detector.base import BaseDetector

logger = logging.getLogger(__name__)


class WandDetector(BaseDetector):
    POINTS_BUFFER_SIZE = 20
    TRACE_THICKNESS = 3
    MAX_TRACE_SPEED = 400  # pixels/second
    CROPPED_IMG_MARGIN = 10  # pixels
    OUTPUT_SIZE = 224  # Output image size for the spell representation

    def __init__(self, video):
        """
        Constructor that accepts a video object compatible with the raspicam.

        Args:
            video: Video capture object with a read() method that returns a frame.
        """
        self.video = video

        # Get initial frame to determine frame dimensions
        frame = self.get_valid_frame()
        self.frame_height, self.frame_width = frame.shape

        self.bgsub = cv2.createBackgroundSubtractorMOG2(200)
        self.wand_move_tracing_frame = np.zeros(
            (self.frame_height, self.frame_width, 1), np.uint8
        )
        self.tracePoints = []
        self.blobKeypoints = []
        self.last_keypoint_int_time = time.time()
        self._traceUpperCorner = (self.frame_width, self.frame_height)
        self._traceLowerCorner = (0, 0)
        self._blobDetector = self.get_blob_detector()

        self.latest_wand_frame = np.zeros((frame.shape), np.uint8)

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
                time.sleep(0.1)

    def get_blob_detector(self):
        """
        Create and configure the blob detector used to detect the wand tip with refined parameters.

        Returns:
            A configured blob detector.
        """
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255  # Detect white blobs (wand tip is bright/reflective)
        params.filterByArea = True
        params.minArea = 7  # Adjust min area based on wand tip size. 50 for 640x480
        params.maxArea = (
            300  # Adjust max area for filtering out larger noise 800 for 640x480
        )
        params.filterByCircularity = True
        params.minCircularity = 0.6  # Target circular features like the wand tip
        params.filterByConvexity = True
        params.minConvexity = 0.8
        params.filterByInertia = True
        params.minInertiaRatio = 0.4
        return cv2.SimpleBlobDetector_create(params)

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

        # Preprocessing: apply Gaussian smoothing to reduce noise in the IR image
        smoothed_frame = cv2.GaussianBlur(fgmask, (5, 5), 0)

        # Adaptive thresholding to segment the bright wand tip from the background
        _, thresholded_frame = cv2.threshold(
            smoothed_frame, 200, 255, cv2.THRESH_BINARY
        )

        # Clean up noise with morphological operations (remove small artifacts)
        kernel = np.ones((2, 2), np.uint8)
        clean_frame = cv2.morphologyEx(thresholded_frame, cv2.MORPH_OPEN, kernel)

        # Detect blobs in the cleaned frame
        self.blobKeypoints = list(self._blobDetector.detect(clean_frame))

        # Detect circles using Hough Circle Transform for additional robustness
        circles = cv2.HoughCircles(
            clean_frame,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=30,
            param2=20,
            minRadius=5,
            maxRadius=15,
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for x, y, r in circles:
                keypoint = cv2.KeyPoint(float(x), float(y), float(r * 2))
                self.blobKeypoints.append(keypoint)

        if len(self.blobKeypoints) > 0:
            currentKeypointTime = time.time()
            if len(self.tracePoints) > 0:
                elapsed = currentKeypointTime - self.last_keypoint_int_time
                pt1 = self.tracePoints[-1]
                pt2 = self.blobKeypoints[0]
                distance = self._distance(pt1, pt2)
                speed = distance / elapsed

                if speed < self.MAX_TRACE_SPEED:
                    # Append the current keypoint
                    self.tracePoints.append(pt2)

                    # Smooth the keypoints over the last few frames to reduce jitter
                    if len(self.tracePoints) > 5:
                        smoothed_x = np.mean([pt.pt[0] for pt in self.tracePoints[-5:]])
                        smoothed_y = np.mean([pt.pt[1] for pt in self.tracePoints[-5:]])
                        pt2 = cv2.KeyPoint(smoothed_x, smoothed_y, pt2.size)

                    # Convert keypoint coordinates to integers
                    pt1_coords = (int(pt1.pt[0]), int(pt1.pt[1]))
                    pt2_coords = (int(pt2.pt[0]), int(pt2.pt[1]))

                    # Draw line on tracing frame
                    cv2.line(
                        self.wand_move_tracing_frame,
                        pt1_coords,
                        pt2_coords,
                        255,
                        self.TRACE_THICKNESS,
                        lineType=cv2.LINE_AA,
                    )
                    self.last_keypoint_int_time = currentKeypointTime
            else:
                self.last_keypoint_int_time = currentKeypointTime
                self.tracePoints.append(self.blobKeypoints[0])

        # Superimpose the trace and debug info onto the latest wand frame
        self.latest_wand_frame = self.superimpose_trace_and_debug_info(
            clean_frame.copy(),
            self.wand_move_tracing_frame,
            self.blobKeypoints,
            speed,
        )

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
        if len(tracing_frame.shape) == 2:  # Single channel image (grayscale)
            tracing_frame_bgr = cv2.cvtColor(tracing_frame, cv2.COLOR_GRAY2BGR)
        elif (
            len(tracing_frame.shape) == 3 and tracing_frame.shape[2] == 1
        ):  # Grayscale but has an extra dimension
            tracing_frame_bgr = cv2.cvtColor(tracing_frame[:, :, 0], cv2.COLOR_GRAY2BGR)
        else:
            tracing_frame_bgr = tracing_frame

        # Convert video frame to BGR if it's a single channel image
        if len(video_frame.shape) == 2 or video_frame.shape[2] == 1:
            video_frame_bgr = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2BGR)
        else:
            video_frame_bgr = video_frame

        # Ensure both tracing_frame_bgr and colored_trace have the same shape
        if tracing_frame_bgr.shape[:2] != video_frame_bgr.shape[:2]:
            tracing_frame_bgr = cv2.resize(
                tracing_frame_bgr, (video_frame_bgr.shape[1], video_frame_bgr.shape[0])
            )

        # Create colored trace overlay with the same shape as the video frame
        colored_trace = np.zeros_like(video_frame_bgr)

        # Assign the trace to the red channel
        colored_trace[:, :, 2] = tracing_frame_bgr[:, :, 0]  # Red channel for trace

        # Add keypoints to colored_trace
        for keypoint in keypoints:
            pt = (int(keypoint.pt[0]), int(keypoint.pt[1]))
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

    def reset(self):
        self.wand_move_tracing_frame = np.zeros(
            (self.frame_height, self.frame_width, 1), np.uint8
        )
        self.tracePoints = []
        self.latest_wand_frame = self.superimpose_trace_and_debug_info(
            self.get_valid_frame(), self.wand_move_tracing_frame, self.blobKeypoints, 0
        )

    def check_trace_validity(self, max_time=2):
        result = False
        if len(self.blobKeypoints) == 0:
            current_keypoint_int_time = time.time()
            trace_length = len(self.tracePoints)
            elapsed = current_keypoint_int_time - self.last_keypoint_int_time

            logger.debug(
                f"Checking trace validity: elapsed={elapsed:.2f}s, trace_length={trace_length}, max_time={max_time}"
            )

            if trace_length >= self.POINTS_BUFFER_SIZE - 5:
                result = True

            elif elapsed < max_time and trace_length > (self.POINTS_BUFFER_SIZE // 3):
                result = True
            else:
                self.reset()

            if result:
                self._update_trace_boundaries()

        return result

    def get_a_spell_maybe(self):
        result = np.zeros((224, 224), np.uint8)
        if self.check_trace_validity():
            result = self._crop_save_trace()
        return result

    def _update_trace_boundaries(self):
        self._traceUpperCorner = (self.frame_width, self.frame_height)
        self._traceLowerCorner = (0, 0)
        for point in self.tracePoints:
            pt = (point.pt[0], point.pt[1])
            if pt[0] < self._traceUpperCorner[0]:
                self._traceUpperCorner = (pt[0], self._traceUpperCorner[1])
            if pt[0] > self._traceLowerCorner[0]:
                self._traceLowerCorner = (pt[0], self._traceLowerCorner[1])
            if pt[1] < self._traceUpperCorner[1]:
                self._traceUpperCorner = (self._traceUpperCorner[0], pt[1])
            if pt[1] > self._traceLowerCorner[1]:
                self._traceLowerCorner = (self._traceLowerCorner[0], pt[1])

    def _crop_save_trace(self):
        upper_x = int(max(self._traceUpperCorner[0] - self.CROPPED_IMG_MARGIN, 0))
        upper_y = int(max(self._traceUpperCorner[1] - self.CROPPED_IMG_MARGIN, 0))
        lower_x = int(
            min(self._traceLowerCorner[0] + self.CROPPED_IMG_MARGIN, self.frame_width)
        )
        lower_y = int(
            min(self._traceLowerCorner[1] + self.CROPPED_IMG_MARGIN, self.frame_height)
        )

        cropped_trace = self.wand_move_tracing_frame[upper_y:lower_y, upper_x:lower_x]
        trace_height, trace_width = cropped_trace.shape[:2]

        scale = min(224 / trace_width, 224 / trace_height)
        new_width = int(trace_width * scale)
        new_height = int(trace_height * scale)
        resized_cropped_trace = cv2.resize(
            cropped_trace, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        final_trace_cell = np.zeros((224, 224), np.uint8)
        pad_x = (224 - new_width) // 2
        pad_y = (224 - new_height) // 2
        final_trace_cell[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = (
            resized_cropped_trace
        )

        return final_trace_cell

    def _distance(self, pt1, pt2):
        return math.sqrt((pt1.pt[0] - pt2.pt[0]) ** 2 + (pt1.pt[1] - pt2.pt[1]) ** 2)
