import base64
import logging
from json import loads

import numpy as np
import requests
from keras.models import load_model
from keras.utils import get_file

from core.config import settings

logger = logging.getLogger(__name__)
H5 = "spell_net.h5"
CLASSES = "classes.json"


class SpellNet(object):
    """
    Just an object that encapsulates running an image trough the network
    """

    def __init__(self, model_path=settings['PIPOTTER_MODEL_DIRECTORY'],
                 remote_server=settings['PIPOTTER_REMOTE_SPELLNET_SERVER'],
                 remote_location='https://s3.amazonaws.com/pipotter/spell_net'):
        """
        The constructor
        :param model_path: Where to save the files, defaults to settings['PIPOTTER_MODEL_DIRECTORY'] 
        :param model_path: Where to save the files, defaults to settings['PIPOTTER_MODEL_DIRECTORY'] 
        :param remote_server: a valid URL to use spellNet if the server is remote
        :param remote_location: S3 bucket to look the files on.
        """
        logger.info("Initializing SpellNet")
        self.remote_server = remote_server
        if not remote_server:
            self._local_model(remote_location, model_path)
            logger.info("Local SpellNet up and running!")
        else:
            logger.info("remote SpellNet expected at {}".format(remote_server))

    def _local_model(self, remote_location, model_path):
        """
        Builds a local model
        :param remote_location: where to get the h5 file
        :param model_path: Where to save the files, defaults to settings['PIPOTTER_MODEL_DIRECTORY'] 
        """
        logger.debug("Loading model file")
        model_file = get_file(fname=H5, origin="{}/{}".format(remote_location, H5), cache_dir=model_path)
        self.model = load_model(model_file)
        class_file = get_file(fname=CLASSES, origin="{}/{}".format(remote_location, CLASSES), cache_dir=model_path)
        logger.debug("Loading class file")
        with open(class_file, 'r') as jfile:
            self.classes = loads(jfile.read())

    def classify(self, image):
        """
        Given a valid image returns a prediction
        :param image: numpy.array of shape (32, 32, 3)
        :return: dictionary of {class1: prob1, ..., classN:ProbN}
        """
        results = False
        if not self.remote_server:
            logger.debug("Using local server")
            predictions = self.model.predict(np.array([image / 255]), verbose=1)
            results = {self.classes[str(k)]: v for k, v in enumerate(predictions[0])}
        else:
            logger.debug("Using remote server")
            payload = {'image': base64.b64encode(np.array([image / 255]))}
            try:
                r = requests.post(self.remote_server, data=payload)
                results = r.json()
            except IOError as e:
                logger.error("requests to {} failed due to {}".format(self.remote_server, e))
        return results
