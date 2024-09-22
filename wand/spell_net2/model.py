import logging
import numpy as np
import tflite_runtime.interpreter as tflite  # Use tflite_runtime for embedded systems
import time
from core.config import settings
from core.utils import pad_to_square

logger = logging.getLogger(__name__)
TFLITE_MODEL = "model.tflite"
LABELS = "labels.txt"

class SpellNet(object):
    """
    Object that encapsulates running an image through the TFLite model.
    """

    def __init__(self, model_path=settings['PIPOTTER_MODEL_DIRECTORY']):
        """
        The constructor initializes the local TFLite model and labels.
        :param model_path: Directory where the model and labels files are stored. Defaults to current directory.
        """
        logger.info("Initializing SpellNet")
        self._local_model(model_path)
        logger.info("Local SpellNet up and running!")

    def _local_model(self, model_path):
        """
        Loads the TFLite model and label file from the local directory.
        :param model_path: Directory where the model and labels files are stored.
        """
        logger.debug("Loading TFLite model file")
        
        # Load the TFLite model from the local directory
        model_file = f"{model_path}/{TFLITE_MODEL}"
        self.interpreter = tflite.Interpreter(model_path=model_file)
        self.interpreter.allocate_tensors()

        # Load the labels from the labels.txt file in the local directory
        labels_file = f"{model_path}/{LABELS}"
        logger.debug("Loading labels file")
        with open(labels_file, 'r') as file:
            self.labels = [line.strip() for line in file.readlines()]

        # Get input and output details for the TFLite interpreter
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Extract the expected input shape from the TFLite model
        self.input_shape = self.input_details[0]['shape'][1:3]  # Usually (height, width)

    def _resize_image(self, image):
        """
        Resized the image for the proper size
        """
        result = image
        if not np.array_equal(image.shape,self.input_shape):
            result = pad_to_square(image)
        return result
    
    def _preprocess_image(self, image):
        """
        Preprocesses the image for the EfficientNet B0 TFLite model, including resizing and normalization.
        :param image: numpy.array representing the image.
        :return: Preprocessed image ready for inference.
        """
        logger.debug(f"Resizing image to {self.input_shape}")
        
        # Convert the image to PIL format for easy resizing
        # Resize the image to match the input shape of the model (e.g., 224x224 for EfficientNet B0)
        working_image = self._resize_image(image)
        # Convert back to numpy array
        image_array = np.array(working_image)
        # EfficientNet B0 normalization: Scale to [0, 1] and then to [-1, 1]
        image_array = image_array / 255.0  # Rescale to [0, 1]
        image_array = image_array * 2.0 - 1.0  # Normalize to [-1, 1]

        return image_array

    def classify(self, image):
        """
        Given a valid image, returns a prediction using the TFLite model.
        :param image: numpy.array of shape (HxWx3)
        :return: dictionary of {class1: prob1, ..., classN: probN}
        """
        logger.debug("Using local TFLite model")
        start_time = time.time()
        # Preprocess the image: resize and normalize for EfficientNet B0
        input_data = self._preprocess_image(image)
        end_time = time.time()
        logger.info(f"Preprocessing time: {(end_time - start_time) * 1000:.2f} ms") 
        logger.debug(f"got an image of shape {input_data.shape}")
        # Add batch dimension and ensure the data type matches the model's input type
        start_time = time.time()
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        logger.debug(f"Expected shape {self.input_details[0]['shape']}, got {input_data.shape}")
        # Set the input tensor for the interpreter
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference
        self.interpreter.invoke()

        # Get the output tensor (predictions)
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        end_time = time.time()
        logger.info(f"Inference time: {(end_time - start_time) * 1000:.2f} ms") 
      
        # Create the results dictionary with class names and probabilities
        results = {self.labels[i]: output_data[i] for i in range(len(self.labels))}
        logger.info(f"got this prediction: {results}")
        return results
