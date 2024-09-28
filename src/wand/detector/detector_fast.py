import cv2
import logging
import numpy as np
import time
import math
from wand.detector.base import BaseDetector

logger = logging.getLogger(__name__)


class WandDetector(BaseDetector):
    POINTS_BUFFER_SIZE = 40
    TRACE_THICKNESS = 3
    MAX_TRACE_SPEED = 150  # pixels/second
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
        self.frame_height, self.frame_width = frame.shape  # Get frame shape

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

        self.latest_wand_frame = np.zeros(
            (frame.shape), np.uint8
        )  # Raw frame being processed

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

    def get_blob_detector(self):
        """
        Create and configure the blob detector used to detect the wand tip with refined parameters.

        Returns:
            A configured blob detector.
        """
        # params = cv2.SimpleBlobDetector_Params()
        # params.minThreshold = 150
        # params.maxThreshold = 255
        # params.filterByColor = True
        # params.blobColor = 255
        # params.filterByArea = True
        # params.minArea = 10
        # params.maxArea = 40
        # params.filterByCircularity = True
        # params.minCircularity = 0.5
        # params.filterByConvexity = True
        # params.minConvexity = 0.5
        # params.filterByInertia = False

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
        params.maxArea = 1000  # Decreased maximum area to ignore larger blobs

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
        # return cv2.SimpleBlobDetector_create(params)

    def detect_wand(self, frame):
        """
        Detects the wand movement in the given frame and updates the trace.

        Args:
            frame: The current grayscale video frame to be processed.
        """
        self.cameraFrame = frame
        self.latest_wand_frame = frame.copy()  # Store raw frame
        speed = 0

        # Apply histogram equalization for better contrast
        equalized_frame = cv2.equalizeHist(self.cameraFrame)

        # Adaptive thresholding for better wand isolation
        adaptive_frame = cv2.adaptiveThreshold(
            equalized_frame,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        # Apply background subtraction
        fgmask = self.bgsub.apply(adaptive_frame)
        bgSubbedCameraFrame = cv2.bitwise_and(adaptive_frame, fgmask)

        # Apply Gaussian Blur to reduce noise
        blurred_frame = cv2.GaussianBlur(bgSubbedCameraFrame, (5, 5), 0)

        # Detect blobs
        self.blobKeypoints = list(self._blobDetector.detect(blurred_frame))

        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred_frame,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=25,  # Lowering to detect more circles
            minRadius=5,
            maxRadius=20,  # Adjusting for potential wand tip size
        )

        # If circles are detected, add them as keypoints
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for x, y, r in circles:
                try:
                    keypoint = cv2.KeyPoint(float(x), float(y), float(r * 2))
                    self.blobKeypoints.append(keypoint)
                except Exception as e:
                    logger.error(f"Error creating or appending KeyPoint: {e}")

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
                    )
                    self.last_keypoint_int_time = currentKeypointTime
            else:
                self.last_keypoint_int_time = currentKeypointTime
                self.tracePoints.append(self.blobKeypoints[0])

        # Superimpose the trace and debug info onto the latest wand frame
        self.latest_wand_frame = self.superimpose_trace_and_debug_info(
            self.latest_wand_frame,
            self.wand_move_tracing_frame,
            self.blobKeypoints,
            speed,
        )

    def detect_wand_orig(self, frame):
        """
        Detects the wand movement in the given frame and updates the trace.

        Args:
            frame: The current grayscale video frame to be processed.
        """
        self.cameraFrame = frame
        self.latest_wand_frame = frame.copy()  # Store raw frame
        speed = 0
        fgmask = self.bgsub.apply(self.cameraFrame)
        bgSubbedCameraFrame = cv2.bitwise_and(self.cameraFrame, fgmask)

        # Detect blobs
        self.blobKeypoints = self._blobDetector.detect(bgSubbedCameraFrame)

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
                    )
                    self.last_keypoint_int_time = currentKeypointTime
            else:
                self.last_keypoint_int_time = currentKeypointTime
                self.tracePoints.append(self.blobKeypoints[0])

        # Superimpose the trace and debug info onto the latest wand frame
        self.latest_wand_frame = self.superimpose_trace_and_debug_info(
            self.latest_wand_frame,
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
            pt = (int(keypoint.pt[0]), int(keypoint.pt[1]))
            cv2.circle(
                colored_trace, pt, 5, (0, 255, 0), -1
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
                (0, 255, 0),
                1,
            )
            cv2.putText(
                combined_frame,
                f"Points: {len(self.tracePoints)}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                combined_frame,
                f"Time: {time.strftime('%H:%M:%S')}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        return combined_frame

    def reset(self):
        """
        Resets the internal state, clearing the current wand trace.
        """
        self.wand_move_tracing_frame = np.zeros(
            (self.frame_height, self.frame_width, 1), np.uint8
        )
        self.tracePoints = []
        self.latest_wand_frame = None  # Raw frame being processed

    def check_trace_validity(self, max_time=2):
        """
        Checks if the wand trace is valid based on trace length and time criteria.

        Args:
            max_time (int): Maximum allowed time (in seconds) since the last detected keypoint
                            to consider the trace valid. Default is 2 seconds.

        Returns:
            bool: True if the trace is valid and ready for further processing, False otherwise.
        """
        result = False

        if len(self.blobKeypoints) == 0:
            current_keypoint_int_time = time.time()
            trace_length = len(self.tracePoints)
            elapsed = current_keypoint_int_time - self.last_keypoint_int_time
            # Log key variables for debugging
            logger.debug(
                f"Checking trace validity: elapsed={elapsed:.2f}s, trace_length={trace_length}, max_time={max_time}"
            )
            # Early exit if trace length is sufficient regardless of time
            if trace_length >= self.POINTS_BUFFER_SIZE - 5:
                result = True
            # Check if elapsed time is less than max_time and trace length is sufficient
            elif elapsed < max_time and trace_length > (self.POINTS_BUFFER_SIZE // 3):
                result = True
            else:
                self.reset()
            if result:
                self._update_trace_boundaries()
        return result

    def get_a_spell_maybe(self):
        """
        Returns a cropped black and white image of the potential spell trace.

        Returns:
            A 224x224 cropped black and white image of the wand trace.
        """
        result = np.zeros((224, 244), np.uint8)
        if self.check_trace_validity():
            result = self._crop_save_trace()
        return result

    def _update_trace_boundaries(self):
        """
        Updates the bounding box boundaries of the trace based on the detected points.
        """
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
        """
        Crops and resizes the detected wand trace to a standard size.

        Returns:
            A 224x224 numpy array representing the cropped and resized wand trace.
        """
        # Ensure all slice indices are integers
        upper_x = int(max(self._traceUpperCorner[0] - self.CROPPED_IMG_MARGIN, 0))
        upper_y = int(max(self._traceUpperCorner[1] - self.CROPPED_IMG_MARGIN, 0))
        lower_x = int(
            min(self._traceLowerCorner[0] + self.CROPPED_IMG_MARGIN, self.frame_width)
        )
        lower_y = int(
            min(self._traceLowerCorner[1] + self.CROPPED_IMG_MARGIN, self.frame_height)
        )

        trace_width = int(lower_x - upper_x)
        trace_height = int(lower_y - upper_y)

        size_width = size_height = 224  # Output size of the cropped image

        if trace_height > trace_width:
            size_height = 224
            size_width = int(trace_width * 224 / trace_height)
        else:
            size_width = 224
            size_height = int(trace_height * 224 / trace_width)

        clone = self.wand_move_tracing_frame.copy()
        crop = clone[upper_y:lower_y, upper_x:lower_x]
        resizedCroppedTrace = cv2.resize(crop, (size_width, size_height))
        finalTraceCell = np.zeros((224, 224), np.uint8)  # 224x224 black and white image
        finalTraceCell[
            : resizedCroppedTrace.shape[0], : resizedCroppedTrace.shape[1]
        ] = resizedCroppedTrace

        return finalTraceCell

    def _distance(self, pt1, pt2):
        """
        Calculates the Euclidean distance between two keypoints.

        Args:
            pt1: The first keypoint.
            pt2: The second keypoint.

        Returns:
            The Euclidean distance between the two points.
        """
        return math.sqrt((pt1.pt[0] - pt2.pt[0]) ** 2 + (pt1.pt[1] - pt2.pt[1]) ** 2)


# Usage example:
# video = cv2.VideoCapture(0)  # Example video object from a webcam
# wand_detector = WandDetector(video)
# frame = wand_detector.get_valid_frame()
# wand_detector.detect_wand(frame)
# if wand_detector.check_trace_validity():
#     trace_image = wand_detector.get_me_a_spell_maybe()
#     # Do something with trace_image
# wand_detector.reset()
