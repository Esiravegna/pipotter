import logging
import time
import cv2
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
import threading
from media.video_source import VALID_SOURCES, looper, picamera
from wand.detector import WandDetector
from wand.spell_net2.model import SpellNet
from sfx.factory import EffectFactory
from core.config import settings
from core.utils import pad_to_square

logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI()
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
        self.spell_frame = None
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
        self.wand_detector = WandDetector(video=self.video)

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
                    if self.wand_detector.maybe_a_spell.size > 0:  # A sigil is detected
                        spell_name, self.preprocessed_frame = self._process_sigil(
                            self.wand_detector.maybe_a_spell
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
        squared_sigil = pad_to_square(a_sigil)
        self.spell_frame = squared_sigil
        predictions = self.spell_net.classify(squared_sigil)
        logger.info(f"predict {predictions}")
        self._save_file(a_sigil, preffix="RAW")

        possible = {k: v for k, v in predictions.items() if v >= self.spell_threshold}
        result = (
            max(possible, key=possible.get)
            if possible
            else settings["PIPOTTER_NO_SPELL_LABEL"]
        )

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
            self.wand_detector.reset()
            logger.info(f"Running effect for spell {spell_name}")
            self.effects[spell_name].run()
        else:
            logger.debug("No spell detected")


### FastAPI endpoints for streaming


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


@app.get("/sigil_feed")
async def sigil_feed():
    if PiPotter is None:
        raise HTTPException(
            status_code=500, detail="PiPotter controller not initialized."
        )
    return StreamingResponse(
        sigil_frame_generator(PiPotter),
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


@app.get("/debug_feed")
async def debug_feed():
    """Stream the debug feed showing detected circles."""
    if PiPotter is None:
        raise HTTPException(
            status_code=500, detail="PiPotter controller not initialized."
        )
    return StreamingResponse(
        debug_frame_generator(PiPotter),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/keypoints_feed")
async def debug_feed():
    """Stream the debug feed showing detected preprocessed circles."""
    if PiPotter is None:
        raise HTTPException(
            status_code=500, detail="PiPotter controller not initialized."
        )
    return StreamingResponse(
        keypoints_frame_generator(PiPotter),
        media_type="multipart/x-mixed-replace; boundary=frame",
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
        else:
            logger.warning("No latest_wand_frame available for streaming.")
        time.sleep(0.05)


def sigil_frame_generator(controller):
    """Generate frames for the sigil detection feed."""
    while not controller.stop_event.is_set():
        if controller.wand_detector.latest_sigil_frame is not None:
            logger.debug("Streaming latest_sigil_frame to client.")
            ret, jpeg = cv2.imencode(
                ".jpg", controller.wand_detector.latest_sigil_frame
            )
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
                )
            else:
                logger.warning("Failed to encode latest_sigil_frame to JPEG.")
        else:
            logger.warning("No latest_sigil_frame available for streaming.")
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


def debug_frame_generator(controller):
    """Generate frames for the debug feed showing detected circles."""
    while not controller.stop_event.is_set():
        frame = controller.wand_detector.latest_debug_frame
        if frame is not None:
            ret, jpeg = cv2.imencode(
                ".jpg", controller.wand_detector.latest_debug_frame
            )
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
                )
        time.sleep(0.05)


def keypoints_frame_generator(controller):
    """Generate frames for the debug feed showing detected circles preprocessed."""
    while not controller.stop_event.is_set():
        frame = controller.wand_detector.latest_keypoints_frame
        if frame is not None:
            ret, jpeg = cv2.imencode(
                ".jpg", controller.wand_detector.latest_keypoints_frame
            )
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n"
                )
        time.sleep(0.05)
