import time
import json
import logging
from core.error import SFXError
from sfx.effect import Effect

logger = logging.getLogger(__name__)


class UtilEffect(Effect):
    """
    A utility effect class that adds useful methods for animations, such as waiting,
    and manages them via a JSON-based configuration.

    Example of JSON configuration:

    {
        "commands": [
            {"command": "wait", "payload": {"seconds": 2}},
            {"command": "cleanup"}
        ]
    }
    """

    def __init__(self, jsonable_string: str):
        """
        Initialize the UtilEffect class from a JSON string.
        :param jsonable_string: A JSON string with the commands.
        """
        super().__init__()
        self.commands = []
        self.name = "UtilEffect"

        logger.info("Initializing UtilEffect")
        self._read_json(jsonable_string)

    def _read_json(self, jsonable_string: str):
        """
        Parse the JSON configuration and initialize the utility effect.
        """
        logger.debug("Parsing UtilEffect JSON configuration")
        try:
            config = json.loads(jsonable_string)
        except ValueError as e:
            logger.error(f"Invalid JSON configuration: {e}")
            raise SFXError(f"Invalid JSON configuration: {e}")

        try:
            # Parse commands
            logger.info("Parsing UtilEffect commands")
            for command in config.get("commands", []):
                cmd_type = command.get("command")
                payload = command.get("payload", {})

                if cmd_type == "wait":
                    seconds = payload.get("seconds", 0)
                    if seconds < 0:
                        logger.error(f"Invalid wait time: {seconds}")
                        raise SFXError(f"Invalid wait time: {seconds}")
                    self.commands.append(("wait", seconds))
                    logger.debug(f"Command added: Wait for {seconds} seconds")

                elif cmd_type == "cleanup":
                    self.commands.append(("cleanup",))
                    logger.debug("Command added: Cleanup resources")

                else:
                    logger.warning(f"Unknown command: {cmd_type}")

        except KeyError as e:
            logger.error(f"Missing required configuration parameter: {e}")
            raise SFXError(f"Missing required configuration parameter: {e}")

    def wait(self, seconds: int):
        """Wait for the specified number of seconds."""
        logger.info(f"Waiting for {seconds} seconds")
        try:
            time.sleep(seconds)
            logger.info(f"Waited for {seconds} seconds")
        except Exception as e:
            logger.error(f"Error during wait: {e}")
            raise SFXError(f"Wait failed: {e}")

    def run(self):
        """Run the utility effect commands."""
        logger.info("Running UtilEffect commands")
        for command in self.commands:
            action = command[0]

            if action == "wait":
                seconds = command[1]
                self.wait(seconds)

            elif action == "cleanup":
                logger.info("Running cleanup command")
                self.cleanup()

    def cleanup(self):
        """Cleanup resources if needed."""
        logger.info("Cleaning up resources for UtilEffect")

    def __enter__(self):
        """Enter method for use in a context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit method for use in a context manager, ensuring cleanup is called."""
        self.cleanup()
