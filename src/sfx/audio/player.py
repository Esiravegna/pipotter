import logging
from os.path import exists
from errno import ENOENT
import subprocess
from core.error import SFXError
from core.config import settings

log = logging.getLogger(__name__)


class Mediaplayer(object):
    """
    Just a media player using ffplay
    """

    player = None

    def play(self, file_to_play):
        """
        Given a file to play, calls an ffplay instance.
        :param file_to_play: <str> path to the file
        """
        log.debug(f"About to play {file_to_play}")

        if not exists(file_to_play):
            raise SFXError(f"Unable to reproduce {file_to_play}")

        # Retrieve extra audio commands from settings, such as volume
        extra_audio_commands = sum(
            [cmd.split() for cmd in settings.get("PIPOTTER_EXTRA_AUDIO_COMMANDS", [])],
            [],
        )

        # Construct the ffplay command
        command = ["ffplay", "-nodisp", "-autoexit"] + extra_audio_commands
        command.append(file_to_play)
        log.info(f"Final ffplay command: {' '.join(command)}")

        try:
            log.info("Calling ffplay")
            # Start the ffplay process
            self.player = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as e:
            if e.errno == ENOENT:
                raise SFXError(f"Check that ffplay is installed: {e}")
            else:
                raise SFXError(f"Error when calling ffplay: {e}")

    def stop(self):
        """
        Stops the player should it be active
        """
        if self.player is not None and self.player.poll() is None:
            log.info("Stopping ffplay")
            # Terminate the ffplay process
            self.player.terminate()
            self.player = None
