import logging
from json import loads

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
                 remote_location='https://s3.amazonaws.com/pipotter/spell_net'):
        """
        The constructor
        :param model_path: Where to save the files, defaults to settings['PIPOTTER_MODEL_DIRECTORY'] 
        :param remote_location: S3 bucket to look the files on.
        """
        logger.info("Initializing SpellNet")
        logger.debug("Loading model file")
        model_file = get_file(fname=H5, origin="{}/{}".format(remote_location, H5), cache_dir=model_path)
        self.model = load_model(model_file)
        class_file = get_file(fname=CLASSES, origin="{}/{}".format(remote_location, CLASSES), cache_dir=model_path)
        logger.debug("Loading class file")
        with open(class_file, 'r') as jfile:
            self.classes = loads(jfile.read())
        logger.info("SpellNet up and running")

    def classify(self, image):
        """
        Given a valid image returns a prediction
        :param image: numpy.array of shape (32, 32, 3)
        :return: dictionary of {class1: prob1, ..., classN:ProbN}
        """
        predictions = self.model.predict(image, verbose=1)
        return {self.classes[str(k)]: v for k, v in enumerate(predictions[0])}
