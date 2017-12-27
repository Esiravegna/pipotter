import logging

import time
from os.path import join

import cv2

from media.video_source import VALID_SOURCES, looper, picamera
from wand.detector import WandDetector
from wand.spell_net.model import SpellNet
from sfx.factory import EffectFactory

logger = logging.getLogger(__name__)
from core.config import settings
from core.utils import pad_to_square

END_KEY = settings['PIPOTTER_END_LOOP']
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
        :param draw_windows: Boolean, should the windows being drawn
        :param args: extra args
        :param kwargs: extra kwargs
        """
        # Let's the validation begin
        logger.info("initializing controller")
        flip = settings['PIPOTTER_FLIP_VIDEO']
        if video_source_name not in VALID_SOURCES:
            raise Exception("Invalid controller name :{}. Must be either of : {}".format(VALID_SOURCES))
        if video_source_name == 'picamera':
            try:
                camera = kwargs['camera']
                logger.debug("Using PiCamera")
                self.video = picamera(camera, flip=flip)
            except KeyError:
                raise Exception(
                    "For use a picamera source, a picamera parameter with a valid camera should be provided")
        elif video_source_name == 'looper':
            try:
                video_file = kwargs['video_file']
                logger.debug("Using a video file located in {}".format(video_file))
                self.video = looper(video_file, flip=flip)
            except KeyError:
                raise Exception(
                    "For use a video loop source, a video file parameter with a valid camera should be provided")
        logger.debug("Initializing wand detector")
        self.wand_detector = WandDetector(video=self.video, draw_windows=draw_windows)
        self.draw_windows = draw_windows
        # should we receive a directory to save each image, do it.
        self.save_images_directory = kwargs.get('save_images_directory', None)
        # let's initialize the model
        self.spell_net = SpellNet()
        # this threshold will be used to determine if we got a spell (above the thresholld) or noise (below)
        self.spell_threshold = settings['PIPOTTER_THRESHOLD_TRIGGER']
        logger.debug("Creating the effects container")
        self.effects = EffectFactory(config_file=configuration_file)

    def _terminate(self):
        """
        Internal. Closes all the things
        """
        self.video.end()
        if self.draw_windows:
            cv2.destroyAllWindows()

    def _save_file(self, img, suffix="", preffix=""):
        """
        Given an img file and if save_images_directory is set, saves the img file with preffix_time_suffix.png name
        :param img: a cv2 readable image array
        :param suffix: str, a string to append to the filename
        :param preffix: str, a string to prepend to the filename
        """
        if self.save_images_directory:
            if suffix:
                suffix = "_" + suffix
            if preffix:
                preffix += "_"
            filename = "{}{}{}.png".format(preffix, time.time(), suffix)
            full_filename = join(self.save_images_directory, filename)
            logger.debug("Saving  the sigil into {}".format(full_filename))
            cv2.imwrite(full_filename, img)

    def _process_sigil(self, a_sigil):
        """
        Process a sigil: gets the image, pads2square it, runs it trough the network
        :param a_sigil: numpy array containing an image of maybe a spell
        :return: a detected sigil out of the classess
        """
        logger.debug('processing sigil {}'.format(a_sigil.shape))
        # by default, we don't know
        result = settings['PIPOTTER_NO_SPELL_LABEL']
        squared = pad_to_square(a_sigil)
        logger.debug('Feeding trough the network ')
        predictions = self.spell_net.classify(squared)
        logger.debug("SpellNet got these results {}".format(predictions))
        self._save_file(a_sigil, preffix="RAW")
        # let's filter them by the threshold
        possible = dict((k, v) for k, v in predictions.items() if v >= self.spell_threshold)
        if possible:
            logger.debug("SpellNet got these candidates {}".format(possible))
            # so we got a possible spell. Let's get the highest ranked
            result = max(possible, key=possible.get)
            self._save_file(squared, preffix="{}".format(result))
        return result

    def _accio_spell(self, spellname):
        """
        Given a spellname, runs the associated effect
        :param spellname: a string containing any valid spell or background if none
        """
        if spellname != settings['PIPOTTER_NO_SPELL_LABEL']:
            logger.info("Running sequence for {}".format(spellname))
            self.effects[spellname].run()
        else:
            logger.info('no spell detected')

    def run(self):
        """
        runs the controller
        """
        logger.info("Starting PiPotter...")
        self.wand_detector.find_wand()
        logger.debug("Found a wand, starting loop")
        while True:
            # Main Loop
            t_end = time.time() + SECONDS_TO_DRAW
            while time.time() < t_end:
                self.wand_detector.read_wand()
                # for the next seconds, build a sigil.
                key = cv2.waitKey(1) & 0xFF
                if key == ord(END_KEY):
                    break
            maybe_a_spellname = self._process_sigil(self.wand_detector.maybe_a_spell)
            self._accio_spell(maybe_a_spellname)
            logger.debug("read finished, waiting for the next wand movement")
            self.wand_detector.find_wand()
            key = cv2.waitKey(1) & 0xFF
            if key == ord(END_KEY):
                logger.info("Terminating PiPotter...")
                break
