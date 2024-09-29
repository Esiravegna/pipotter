import json
import logging
import time
import RPi.GPIO as GPIO
from core.error import SFXError
from sfx.effect import Effect

logger = logging.getLogger(__name__)


class ServoMotor(Effect):
    """
    A class to control a servo motor using GPIO and PWM, initialized from JSON configuration.
    """

    def __init__(self, jsonable_string):
        """
        Initialize the ServoMotor class from a JSON string.
        :param jsonable_string: A JSON string with servo configuration and commands.
        """
        super().__init__()
        self.gpio_pin = None
        self.frequency = 50
        self.pwm = None
        self.commands = []
        self.name = "ServomotorEffect"

        logger.info("Initializing ServoMotor")
        self._read_json(jsonable_string)

    def _read_json(self, jsonable_string):
        """
        Parse the JSON configuration and store the servo motor commands.
        """
        logger.debug("Parsing ServoMotor JSON configuration")
        try:
            config = json.loads(jsonable_string)
        except ValueError as e:
            logger.error(f"Invalid JSON configuration: {e}")
            raise SFXError(f"Invalid JSON configuration: {e}")

        try:
            self.gpio_pin = config["gpio_pin"]
            self.frequency = config.get("frequency", 50)

            # Store commands without initializing GPIO and PWM yet
            logger.info("Storing ServoMotor commands")
            for command in config.get("commands", []):
                cmd_type = command.get("command")
                payload = command.get("payload", {})

                if cmd_type == "set_angle":
                    angle = payload.get("angle", 0)
                    self.commands.append(("set_angle", angle))
                    logger.debug(f"Command added: Set angle to {angle} degrees")

                elif cmd_type == "stop":
                    self.commands.append(("stop",))
                    logger.debug("Command added: Stop ServoMotor")

                else:
                    logger.warning(f"Unknown command: {cmd_type}")

        except KeyError as e:
            logger.error(f"Missing required configuration parameter: {e}")
            raise SFXError(f"Missing required configuration parameter: {e}")

    def _initialize_pwm(self):
        """
        Initialize GPIO and PWM. This is called when the effect is run, not during object creation.
        """
        logger.info(f"Setting up GPIO pin {self.gpio_pin} for ServoMotor")
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.gpio_pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.gpio_pin, self.frequency)
        self.pwm.start(0)
        logger.info(
            f"ServoMotor initialized on pin {self.gpio_pin} with frequency {self.frequency}Hz"
        )

    def set_angle(self, angle):
        """
        Set the servo angle.
        """
        logger.info(f"Setting servo angle to {angle} degrees")
        duty_cycle = (0.05 * self.frequency) + (angle / 18.0)
        self.pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)
        self.pwm.ChangeDutyCycle(0)

    def stop(self):
        """Stop the servo motor and cleanup GPIO."""
        logger.info("Stopping ServoMotor")
        if self.pwm:
            self.pwm.stop()
            GPIO.cleanup(self.gpio_pin)

    def run(self):
        """
        Run the commands for controlling the servo motor. Initialize GPIO/PWM if necessary.
        """
        # Initialize GPIO and PWM when the effect is actually run
        if self.pwm is None:
            self._initialize_pwm()

        logger.info("Running ServoMotor commands")
        for command in self.commands:
            action = command[0]

            if action == "set_angle":
                angle = command[1]
                logger.info(f"Executing command: Set angle to {angle} degrees")
                self.set_angle(angle)

            elif action == "stop":
                logger.info("Executing command: Stop ServoMotor")
                self.stop()

    def __del__(self):
        """Ensure cleanup when the object is deleted."""
        self.stop()
