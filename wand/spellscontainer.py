import numpy as np
import time


class SpellSigil(object):
    """
    A graph object containing a mumpy array of the lines being drawn
    """

    def __init__(self, pixels_margin=10):
        self.history = []
        self.margin = pixels_margin

    def add(self, left, top, right, bottom):
        """
        Appends a line defined by left, top, right, bottom coordinate set
        :param: left (int) the x1 value
        :param: top (int) the y1 value
        :param: right (int) the x2 value
        :param: bottom (int) the y2 value
        """
        f = lambda x: max(
            0, x
        )  # no negatives allowed. Somehow and for unknown reasons opencv2 sometimes produce some
        new_point = [f(left), f(top), f(right), f(bottom)]
        if self.history:
            # We will not add the same point, useful to filter static points
            if np.array_equal(self.history[-1], new_point):
                return False
        self.history.append([f(left), f(top), f(right), f(bottom)])

    def __len__(self):
        return len(self.history)

    def get_box(self):
        """
        returns the left, top, right, bottom bounding box
        """
        if not len(self.history):
            raise ValueError(
                "The bounding boxes cannot be extracted from an empty history"
            )
        matrix = np.array(self.history)
        left, top, _, _ = matrix.min(axis=0)
        _, _, right, bottom = matrix.max(axis=0)

        # It may happen when there is a single line that the end values are higher. Thus:
        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top
        top -= self.margin if top - self.margin >= 0 else 0
        left -= self.margin if top - self.margin >= 0 else 0
        bottom += self.margin
        right += self.margin
        return left, top, right, bottom


class SpellsContainer(object):
    """
    A container for the lines drawn
    """

    def __init__(self, expiration_time=60):
        """
        Initializes an empty dictionary and sets expiration time in seconds
        :param expiration_time: (int) Time in seconds after which the container is considered stale
        """
        self.sigils = {}
        self.last_update_time = None
        self.expiration_time = expiration_time

    def __setitem__(self, key, item):
        """
        Adds an item to the sigils dict. If does not exist, creates a SpellSigil object.
        Updates the last update time.
        :param key: (string or int) Key to use
        :param item: (tuple) left, top, right, bottom tuple
        """
        try:
            left, top, right, bottom = item
        except (ValueError, TypeError):
            raise ValueError(f"{item} is not a (left, top, right, bottom) tuple")

        if key not in self.sigils.keys():
            self.sigils[key] = SpellSigil()

        self.sigils[key].add(int(left), int(top), int(right), int(bottom))
        self.last_update_time = time.time()  # Update the last update time

    def __len__(self):
        return len(self.sigils)

    def __delitem__(self, key):
        del self.sigils[key]

    def reset(self):
        """
        Simply erases the history and resets the last update time
        :return: the previous history
        """
        previous = self.sigils
        self.sigils = {}
        self.last_update_time = None
        return previous

    def get_box(self):
        """
        Returns the bounding boxes of the longest sigil, that should be the one being drawn.
        The rationale here is that a wand being moved will produce a more complex, longer line than
        false positives circles detected.
        Returns the left, top, right, bottom bounding box.
        """
        if not len(self.sigils):
            raise ValueError("Cannot proceed with an empty sigils history")
        return self._longest_vector(self.sigils).get_box()

    def _longest_vector(self, vector):
        """
        Internal. From a list of lists, returns the longest.
        :param vector: the vector of lists
        :return: iterable, the longest one or the first should all of them be of the same length
        """
        return vector[max(vector, key=lambda k: len(vector[k]))]

    def is_stale(self):
        """
        Checks if the container is stale based on the expiration time.
        :return: (bool) True if stale, False otherwise
        """
        if self.last_update_time is None:
            return True  # Consider stale if no updates have been made
        return (time.time() - self.last_update_time) > self.expiration_time

    def auto_clear(self):
        """
        Clears the container if it is stale.
        :return: (bool) True if the container was cleared, False otherwise
        """
        result = False
        if self.is_stale():
            self.reset()
            result = True
        return result
