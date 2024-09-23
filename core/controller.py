import logging
import time
import queue
import cv2
from flask import Flask, Response
from os.path import join
import threading
from media.video_source import VALID_SOURCES, looper, picamera
from wand.detector import WandDetector
from wand.spell_net2.model import SpellNet
from sfx.factory import EffectFactory
from core.config import settings
from core.utils import pad_to_square

logger = logging.getLogger(__name__)

SECONDS_TO_DRAW = settings["PIPOTTER_SECONDS_TO_DRAW"]

# Flask app for real-time monitoring
app = Flask(__name__)


class PiPotterController(object):
    def __init__(self, video_source_name, configuration_file, **kwargs):
        """
        Initialize PiPotter controller
        """
        logger.info("Initializing PiPotterController")
        flip = settings["PIPOTTER_FLIP_VIDEO"]

        if video_source_name not in VALID_SOURCES:
            raise Exception(
                f"Invalid controller name: {video_source_name}. Must be one of: {VALID_SOURCES}"
            )

        if video_source_name == "picamera":
            try:
                camera = kwargs["camera"]
                self.video = picamera(flip=flip)
            except KeyError:
                raise Exception(
                    "For picamera source, a valid 'camera' parameter should be provided"
                )
        elif video_source_name == "looper":
            try:
                video_file = kwargs["video_file"]
                self.video = looper(video_file, flip=flip)
            except KeyError:
                raise Exception(
                    "For looper source, a valid 'video_file' parameter should be provided"
                )

        self.save_images_directory = kwargs.get("save_images_directory", None)

        logger.debug("Initializing wand detector")
        self.wand_detector = WandDetector(video=self.video)

        logger.debug("Initializing SpellNet")
        self.spell_net = SpellNet()
        self.spell_threshold = settings["PIPOTTER_THRESHOLD_TRIGGER"]

        logger.debug("Creating effects container")
        self.effects = EffectFactory(config_file=configuration_file)

        logger.info("Initialization complete. Ready to go!")
        self.effects[settings["PIPOTTER_READY_SFX"]].run()

        # Queue for video frames
        self.frame_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()
        self.preprocessed_frame = None

        # Bind routes to instance methods using add_url_rule
        app.add_url_rule("/video_feed", "video_feed", self.video_feed)
        app.add_url_rule("/sigil_feed", "sigil_feed", self.sigil_feed)
        app.add_url_rule(
            "/preprocessed_feed", "preprocessed_feed", self.preprocessed_feed
        )

        # Start Flask server in a thread
        self.flask_thread = threading.Thread(target=self._start_flask_server)
        self.flask_thread.daemon = True
        self.flask_thread.start()

    def run(self):
        """Runs the PiPotter controller."""
        logger.info("Starting PiPotter...")
        try:
            while True:
                ret, frame = self.video.read()
                if ret and frame is not None:
                    self.wand_detector.latest_wand_frame = frame  # Store for streaming
                    self.frame_queue.put(frame)  # Put the frame in the queue
                    # Perform wand detection and effect execution
                    self.wand_detection()
                time.sleep(0.1)  # Small delay to prevent busy-waiting
        except KeyboardInterrupt:
            logger.info("Shutting down PiPotter...")
            self.stop_event.set()
        finally:
            self._terminate()

    def wand_detection(self):
        """Process video frames and classify spells synchronously."""
        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            self.wand_detector.read_wand(frame)
            if self.wand_detector.maybe_a_spell.shape[0]:
                spell_name, self.preprocessed_frame = self._process_sigil(
                    self.wand_detector.maybe_a_spell
                )
                self._accio_spell(spell_name)

    def _start_flask_server(self):
        """Start Flask server for real-time monitoring."""
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

    def video_feed(self):
        """Stream the raw video feed."""
        return Response(
            self._frame_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def sigil_feed(self):
        """Stream the sigil detection feed."""
        return Response(
            self._sigil_frame_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def preprocessed_feed(self):
        """Stream the preprocessed frames."""
        return Response(
            self._preprocessed_frame_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def _frame_generator(self):
        """Generate frames for the camera feed."""
        while not self.stop_event.is_set():
            if self.wand_detector.latest_wand_frame is not None:
                ret, jpeg = cv2.imencode(".jpg", self.wand_detector.latest_wand_frame)
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg.tobytes()
                        + b"\r\n\r\n"
                    )
            time.sleep(0.1)

    def _sigil_frame_generator(self):
        """Generate frames for the sigil detection feed."""
        while not self.stop_event.is_set():
            if self.wand_detector.latest_sigil_frame is not None:
                ret, jpeg = cv2.imencode(".jpg", self.wand_detector.latest_sigil_frame)
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg.tobytes()
                        + b"\r\n\r\n"
                    )
            time.sleep(0.1)

    def _preprocessed_frame_generator(self):
        """Generate frames for the preprocessed image feed."""
        while not self.stop_event.is_set():
            if self.preprocessed_frame is not None:
                ret, jpeg = cv2.imencode(".jpg", self.preprocessed_frame)
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg.tobytes()
                        + b"\r\n\r\n"
                    )
            time.sleep(0.1)

    def _terminate(self):
        """Internal. Closes all the things"""
        self.video.end()

    def _process_sigil(self, a_sigil):
        """Process a sigil by feeding it through the SpellNet model"""
        squared = pad_to_square(a_sigil)
        predictions = self.spell_net.classify(squared)
        self._save_file(a_sigil, preffix="RAW")
        possible = {k: v for k, v in predictions.items() if v >= self.spell_threshold}
        result = (
            max(possible, key=possible.get)
            if possible
            else settings["PIPOTTER_NO_SPELL_LABEL"]
        )
        self._save_file(squared, preffix=result)
        return result, squared

    def _save_file(self, img, suffix="", preffix=""):
        """Saves the image if save_images_directory is set"""
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
            logger.info("No spell detected")
