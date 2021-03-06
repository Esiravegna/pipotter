import logging
from concurrent.futures import ThreadPoolExecutor
from sfx.effect import Effect
logger = logging.getLogger(__name__)


class EffectContainer:
    """
    A container for Effects objects. 
    Essentially holds a list of objects and runs in sequence.
    """

    def __init__(self):
        """
        Just creates an empty container
        """
        self.queue = []
        self.tasks = ThreadPoolExecutor(max_workers=2)

    def run(self):
        """
        Runs all the effects in the sequence
        :return: True on all the effects completed, False otherwise
        """
        failed = 0
        logger.info("Running all the stored effects")
        if self.queue:
            #Concurrency babe!
            for an_effect in self.queue:
                try:
                   f = self.tasks.submit(an_effect.run).result()
                except (ValueError, NotImplemented, IOError) as e:
                    logger.error("unable to run {} due to {} by {}".format(an_effect, e, r))
                    failed += 1
        logger.debug("All run with {} failed".format(failed))
        return not failed

    def __len__(self):
        return len(self.queue)

    def __delitem__(self, ii):
        """Delete an item"""
        del self.queue[ii]

    def __setitem__(self, ii, val):
        """
        Adds an item
        :param ii: numeric index
        :param val: an Effect type
        """
        if not isinstance(val, Effect):
            raise ValueError("{} is not an Effect type".format(val))
        self.queue[ii] = val

    def append(self, val):
        """
        Just the list.append method
        :param val: an Effect type
        """
        if not isinstance(val, Effect):
            raise ValueError("{} is not an Effect type".format(val))
        self.queue.append(val)
