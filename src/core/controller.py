import cv2
import logging
import numpy as np
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import threading
import time

from media.video_source import VALID_SOURCES, looper, picamera
from wand.detector import get_detector
from wand.spell_net2.model import SpellNet
from sfx.factory import EffectFactory
from core.config import settings
from core.utils import pad_to_square

logger = logging.getLogger(__name__)
BASE_IMAGES_DIR = "media/base_images"
# Create FastAPI app instance
app = FastAPI()
app.mount("/media", StaticFiles(directory="media"), name="media")

# Set up the Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

# Placeholder for a PiPotter controller instance
PiPotter = None


# Function to set the PiPotter instance
def set_pipotter(controller_instance):
    global PiPotter
    PiPotter = controller_instance


class PiPotterController:
    def __init__(self, video_source_name, configuration_file, **kwargs):
        """
        Initialize PiPotter controller.
        """
        logger.info("Initializing PiPotterController")
        flip = settings["PIPOTTER_FLIP_VIDEO"]
        self.spell_frame = np.zeros((224, 224), np.uint8)
        self.latest_predictions = {}

        if video_source_name not in VALID_SOURCES:
            raise ValueError(
                f"Invalid controller name: {video_source_name}. Must be one of: {VALID_SOURCES}"
            )

        # Set up video source
        if video_source_name == "picamera":
            camera = kwargs.get("camera")
            if not camera:
                raise ValueError(
                    "For picamera source, a valid 'camera' parameter should be provided"
                )
            self.video = picamera(flip=flip)
        elif video_source_name == "looper":
            video_file = kwargs.get("video_file")
            if not video_file:
                raise ValueError(
                    "For looper source, a valid 'video_file' parameter should be provided"
                )
            self.video = looper(video_file, flip=flip)

        self.save_images_directory = kwargs.get("save_images_directory", None)

        logger.debug("Initializing WandDetector")
        self.wand_detector = get_detector(detector_type="circles", video=self.video)

        logger.debug("Initializing SpellNet")
        self.spell_net = SpellNet()
        self.spell_threshold = settings["PIPOTTER_THRESHOLD_TRIGGER"]

        logger.debug("Creating effects container")
        self.effects = EffectFactory(config_file=configuration_file)

        logger.info("Initialization complete. Ready to go!")
        self.effects[settings["PIPOTTER_READY_SFX"]].run()

        # Thread Control
        self.stop_event = threading.Event()

        # Start the main video processing in a separate thread
        self.video_thread = threading.Thread(target=self.run)
        self.video_thread.daemon = True
        self.video_thread.start()

    def run(self):
        """Runs the PiPotter controller."""
        logger.info("Starting PiPotter...")
        try:
            while not self.stop_event.is_set():
                ret, frame = self.video.read()
                if ret and frame is not None:
                    logger.debug("Got a new frame")
                    self.wand_detector.detect_wand(frame)
                    maybe_a_spell = self.wand_detector.get_a_spell_maybe()
                    if (
                        maybe_a_spell is not None
                        and np.count_nonzero(maybe_a_spell) > 0
                    ):  # A sigil is detected
                        spell_name, self.preprocessed_frame = self._process_sigil(
                            maybe_a_spell
                        )
                        logger.debug(f"Updated preprocessed_frame at {time.time()}")
                        self._accio_spell(spell_name)
                else:
                    logger.warning("Failed to read frame from video source.")
                time.sleep(0.05)  # Small delay to avoid busy-waiting, adjust as needed
        except KeyboardInterrupt:
            logger.info("Shutting down PiPotter...")
            self.stop_event.set()
        finally:
            self._terminate()

    def _process_sigil(self, a_sigil):
        """Process a sigil by feeding it through the SpellNet model."""

        # Step 1: Pad the sigil to square
        squared_sigil = pad_to_square(a_sigil)
        self.spell_frame = squared_sigil

        # Step 2: Classify using the SpellNet model
        predictions = self.spell_net.classify(squared_sigil)
        self.latest_predictions = predictions
        logger.info(f"predict {predictions}")

        # Step 3: Save the raw sigil
        self._save_file(a_sigil, preffix="RAW")

        # Step 4: Filter predictions based on a threshold
        possible = {k: v for k, v in predictions.items() if v >= self.spell_threshold}
        result = (
            max(possible, key=possible.get)
            if possible
            else settings["PIPOTTER_NO_SPELL_LABEL"]
        )

        # Step 5: Save the image with the result label as prefix
        self._save_file(squared_sigil, preffix=result)

        return result, squared_sigil

    def _terminate(self):
        """Terminate the video stream and close resources."""
        self.video.end()

    def _save_file(self, img, suffix="", preffix=""):
        """Saves the image if save_images_directory is set."""
        if self.save_images_directory:
            filename = f"{preffix}{time.time()}{suffix}.png"
            full_filename = join(self.save_images_directory, filename)
            logger.debug(f"Saving image as {full_filename}")
            cv2.imwrite(full_filename, img)

    def _accio_spell(self, spell_name):
        """Runs the effect for the given spell."""
        if spell_name != settings["PIPOTTER_NO_SPELL_LABEL"]:
            logger.info(f"Running effect for spell {spell_name}")
            self.effects[spell_name].run()
        else:
            logger.debug("No spell detected")
        self.wand_detector.reset()


### FastAPI methods for debuf streaming
def convert_numpy_types(data):
    """
    Convert numpy data types to native Python types.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy array to list
    elif isinstance(data, np.generic):  # Check for numpy scalar types like np.float32
        return data.item()  # Convert numpy scalar to native Python type
    elif isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    else:
        return data


@app.get("/video_feed")
async def video_feed():
    if PiPotter is None:
        raise HTTPException(
            status_code=500, detail="PiPotter controller not initialized."
        )
    return StreamingResponse(
        video_frame_generator(PiPotter),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/spell_feed")
async def preprocessed_feed():
    if PiPotter is None:
        raise HTTPException(
            status_code=500, detail="PiPotter controller not initialized."
        )
    return StreamingResponse(
        spell_frame_generator(PiPotter),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/predictions", response_class=JSONResponse)
async def get_predictions():
    if PiPotter is None:
        raise HTTPException(
            status_code=500, detail="PiPotter controller not initialized."
        )
    # Return the latest predictions stored in PiPotter
    result = JSONResponse(content=convert_numpy_types(PiPotter.latest_predictions))
    return result


@app.get("/", response_class=HTMLResponse)
async def all_feeds(request: Request):
    """Serve a webpage displaying all the feeds and additional data."""
    # List all images in the base_images directory
    base_images = [
        {"filename": filename, "title": os.path.splitext(filename)[0]}
        for filename in os.listdir(BASE_IMAGES_DIR)
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    return templates.TemplateResponse(
        "index.html", {"request": request, "base_images": base_images}
    )


### Frame Generators for Streaming
def video_frame_generator(controller):
    """Generate frames for the video feed."""
    while not controller.stop_event.is_set():
        if controller.wand_detector.latest_wand_frame is not None:
            logger.debug("Streaming latest_wand_frame to client.")
            ret, jpeg = cv2.imencode(".jpg", controller.wand_detector.latest_wand_frame)
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
                )
            else:
                logger.warning("Failed to encode latest_wand_frame to JPEG.")
        time.sleep(0.05)


def spell_frame_generator(controller):
    """Generate frames for the preprocessed frame feed."""
    while not controller.stop_event.is_set():
        if controller.spell_frame is not None:
            logger.debug("Streaming preprocessed_frame to client.")
            ret, jpeg = cv2.imencode(".jpg", controller.spell_frame)
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
                )
            else:
                logger.warning("Failed to encode spell_frame to JPEG.")
        else:
            logger.warning("No spell_frame available for streaming.")
        time.sleep(0.05)
