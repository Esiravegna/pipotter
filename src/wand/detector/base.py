from abc import ABC, abstractmethod
import subprocess
from sfx.audio.effect import AudioEffect


class BaseDetector(ABC):
    @abstractmethod
    def detect_wand(self, frame):
        """
        Detect wand tip movement in the given frame and update internal state.

        Args:
            frame: The current grayscale video frame to be processed.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the internal state after a spell is detected and processed.
        """
        pass

    @abstractmethod
    def check_trace_validity(self):
        """
        Check if the wand trace is valid based on trace length and time criteria.

        Returns:
            True if the trace is valid and ready for further processing.
        """
        pass

    @abstractmethod
    def get_a_spell_maybe(self):
        """
        Return the current spell trace representation, either as a set of points or an image.
        """
        pass

    def play_wand_detected_sound(
        self, effect=AudioEffect(the_filename="./media/audio/protego.mp3")
    ):
        """
        When a wand is detected
        """
        effect.run()

    def play_wand_lost_sound(
        self, effect=AudioEffect(the_filename="./media/audio/whoosh.mp3")
    ):
        """
        When a wand is lost
        """
        effect.run()
