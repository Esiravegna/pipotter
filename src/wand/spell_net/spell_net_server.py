"""
A very simpe keras serving model
"""

import base64
import logging
from json import loads

from flask import Flask, request
import io
from keras.models import load_model
from keras.utils import get_file
import numpy as np
from PIL import Image

from core.config import settings

logger = logging.getLogger(__name__)
H5 = "spell_net.h5"
CLASSES = "classes.json"


def load_model(
    model_path=settings["PIPOTTER_MODEL_DIRECTORY"],
    remote_location="https://s3.amazonaws.com/pipotter/spell_net",
):
    """
    Loads the model in memory
    :param model_path: where to store the model, defaults to config
    :param remote_location: where to get the models, defaults to config
    :return: keras model, classes
    """
    logger.debug(
        "Loading model file from {} into {}".format(remote_location, model_path)
    )
    model_file = get_file(
        fname=H5, origin="{}/{}".format(remote_location, H5), cache_dir=model_path
    )
    model = load_model(model_file)
    class_file = get_file(
        fname=CLASSES,
        origin="{}/{}".format(remote_location, CLASSES),
        cache_dir=model_path,
    )
    logger.debug("Loading class file")
    with open(class_file, "r") as jfile:
        classes = loads(jfile.read())
    return model, classes


logging.info("Initializing app")
app = Flask(__name__)

global model, classes
model, classes = load_model()


@app.route("/predict/", methods=["POST"])
def predict():
    payload = request.json()
    image_encoded = payload["image"]
    image = np.asarray(Image.open(io.BytesIO(base64.b64decode(image_encoded))))
    predictions = model.predict(np.array([image / 255]), verbose=1)
    response = {classes[str(k)]: v for k, v in enumerate(predictions[0])}
    return response


if __name__ == "__main__":
    port = int(settings["PIPOTTER_REMOTE_SPELLNET_SERVER_PORT"])
    logger.info("Starting PiPotter Server")
    app.run(host="0.0.0.0", port=port, debug=True)
