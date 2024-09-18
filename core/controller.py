import logging
import threading
import time
import queue
from os.path import join
import cv2

from media.video_source import VALID_SOURCES, looper, picamera
from wand.detector import WandDetector
from wand.spell_net.model import SpellNet
from sfx.factory import EffectFactory

logger = logging.getLogger(__name__)
from core.config import settings
from core.utils import pad_to_square

SECONDS_TO_DRAW = settings['PIPOTTER_SECONDS_TO_DRAW']

class PiPotterController(object):
    """
    PiPotter controller. Initializes all the objects and runs the core
    """

    def __init__(self, video_source_name, configuration_file, draw_windows=False, **kwargs):
        """
        The Controller
        :param video_source_name: (string) picamera | looper, any of the controllers to extract the images
        :param configuration_file: (string) the json configuration file for the EffectsFactory
        :param draw_windows: Boolean, should the windows be drawn
        :param args: extra args
        :param kwargs: extra kwargs
        """
        logger.info("Initializing PiPotterController")

        flip = settings['PIPOTTER_FLIP_VIDEO']
        if video_source_name not in VALID_SOURCES:
            raise Exception(f"Invalid controller name: {video_source_name}. Must be one of: {VALID_SOURCES}")

        if video_source_name == 'picamera':
            try:
                camera = kwargs['camera']
                logger.debug("Using PiCamera")
                self.video = picamera(camera, flip=flip)
            except KeyError:
                raise Exception("For picamera source, a valid 'camera' parameter should be provided")
        elif video_source_name == 'looper':
            try:
                video_file = kwargs['video_file']
                logger.debug(f"Using video file located at {video_file}")
                self.video = looper(video_file, flip=flip)
            except KeyError:
                raise Exception("For looper source, a valid 'video_file' parameter should be provided")

        logger.debug("Initializing wand detector")
        self.wand_detector = WandDetector(video=self.video, draw_windows=draw_windows)
        self.draw_windows = draw_windows
        self.save_images_directory = kwargs.get('save_images_directory', None)

        logger.debug("Initializing SpellNet")
        self.spell_net = SpellNet()

        self.spell_threshold = settings['PIPOTTER_THRESHOLD_TRIGGER']
        logger.debug("Creating the effects container")
        self.effects = EffectFactory(config_file=configuration_file)

        logger.info("Initialization complete. Ready to go!")
        self.effects[settings['PIPOTTER_READY_SFX']].run()

        # Threading setup
        self.frame_queue = queue.Queue(maxsize=5)  # Queue to hold video frames for processing
        self.spell_queue = queue.Queue()           # Queue to hold classified spells for effects
        self.stop_event = threading.Event()        # Event to signal threads to stop

    def _terminate(self):
        """Internal. Closes all the things"""
        self.video.end()
        if self.draw_windows:
            cv2.destroyAllWindows()

    def _save_file(self, img, suffix="", preffix=""):
        """Saves the image if save_images_directory is set"""
        if self.save_images_directory:
            if suffix:
                suffix = "_" + suffix
            if preffix:
                preffix += "_"
            filename = f"{preffix}{time.time()}{suffix}.png"
            full_filename = join(self.save_images_directory, filename)
            logger.debug(f"Saving image as {full_filename}")
            cv2.imwrite(full_filename, img)

    def _process_sigil(self, a_sigil):
        """Process a sigil by feeding it through the SpellNet model"""
        logger.debug(f'Processing sigil of shape {a_sigil.shape}')
        result = settings['PIPOTTER_NO_SPELL_LABEL']
        squared = pad_to_square(a_sigil)
        logger.debug('Feeding sigil through the network')
        predictions = self.spell_net.classify(squared)
        logger.debug(f"SpellNet results: {predictions}")
        self._save_file(a_sigil, preffix="RAW")

        possible = {k: v for k, v in predictions.items() if v >= self.spell_threshold}
        if possible:
            logger.debug(f"SpellNet candidates: {possible}")
            result = max(possible, key=possible.get)
            self._save_file(squared, preffix=f"{result}")

        return result

    def _accio_spell(self, spellname):
        """Runs the effect for the given spell"""
        if spellname != settings['PIPOTTER_NO_SPELL_LABEL']:
            logger.info(f"Running effect for spell {spellname}")
            self.effects[spellname].run()
        else:
            logger.info('No spell detected')

    def video_capture_thread(self):
        """Thread to capture video frames continuously"""
        while not self.stop_event.is_set():
            frame = self.video.read()  # Capture a frame
            if frame is not None:
                try:
                    self.frame_queue.put(frame, timeout=1)  # Place frame in queue for processing
                except queue.Full:
                    pass  # Drop frame if queue is full

    def wand_detection_thread(self):
        """Thread to process video frames and classify spells"""
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)  # Get a frame from the queue
                self.wand_detector.read_wand(frame)

                if self.wand_detector.maybe_a_spell.shape[0]:  # If a sigil is detected
                    maybe_a_spellname = self._process_sigil(self.wand_detector.maybe_a_spell)
                    self.spell_queue.put(maybe_a_spellname)  # Put the spell in queue for effects
            except queue.Empty:
                pass

    def effect_execution_thread(self):
        """Thread to execute effects for classified spells"""
        while not self.stop_event.is_set():
            try:
                spellname = self.spell_queue.get(timeout=1)  # Get the classified spell
                self._accio_spell(spellname)  # Execute the corresponding effect
            except queue.Empty:
                pass

    def run(self):
        """Runs the PiPotter controller"""
        logger.info("Starting PiPotter...")

        # Start video capture, wand detection, and effect execution threads
        video_thread = threading.Thread(target=self.video_capture_thread)
        wand_thread = threading.Thread(target=self.wand_detection_thread)
        effect_thread = threading.Thread(target=self.effect_execution_thread)

        video_thread.start()
        wand_thread.start()
        effect_thread.start()

        try:
            while True:
                time.sleep(1)  # Keep the main loop alive
        except KeyboardInterrupt:
            logger.info("Shutting down PiPotter...")
            self.stop_event.set()  # Signal threads to stop
            video_thread.join()
            wand_thread.join()
            effect_thread.join()
        finally:
            self._terminate()  # Cleanup and close all resources
