import logging
import click
import uvicorn
from sys import exit
from core.log import configure_log
from core.config import settings
from core.controller import PiPotterController, set_pipotter


# Configure logging
configure_log(settings["PIPOTTER_LOGLEVEL"])
logger = logging.getLogger(__name__)

BANNER = """
\n
 ███████████   ███  ███████████            █████     █████                          █████ █████
░░███░░░░░███ ░░░  ░░███░░░░░███          ░░███     ░░███                          ░░███ ░░███ 
 ░███    ░███ ████  ░███    ░███  ██████  ███████   ███████    ██████  ████████     ░███  ░███ 
 ░██████████ ░░███  ░██████████  ███░░███░░░███░   ░░░███░    ███░░███░░███░░███    ░███  ░███ 
 ░███░░░░░░   ░███  ░███░░░░░░  ░███ ░███  ░███      ░███    ░███████  ░███ ░░░     ░███  ░███ 
 ░███         ░███  ░███        ░███ ░███  ░███ ███  ░███ ███░███░░░   ░███         ░███  ░███ 
 █████        █████ █████       ░░██████   ░░█████   ░░█████ ░░██████  █████        █████ █████
░░░░░        ░░░░░ ░░░░░         ░░░░░░     ░░░░░     ░░░░░   ░░░░░░  ░░░░░        ░░░░░ ░░░░░ 
                                                                                               
                                                                                               
\n                                                                                               
"""

print(BANNER)


@click.command()
@click.option(
    "--video-source",
    required=True,
    type=click.Choice(["looper", "picamera"]),
    help="Video source to use : picamera or loop",
    default="picamera",
)
@click.option("--video-file", help="Video file to loop")
@click.option(
    "--save_images_directory",
    help="Store the detected movements as images in the passed directory",
)
@click.option(
    "--config-file",
    help="Where to read the configuration file. Defaults to ./config.json",
    default="./config.json",
)
def run_command(video_source, video_file, save_images_directory, config_file):
    """
    The main method.
    Initializes the PiPotterController and runs the FastAPI app.
    """
    # Collect video source arguments
    arguments = {}
    if video_source == "looper":
        arguments["video_file"] = video_file
    elif video_source == "picamera":
        arguments["camera"] = True

    if save_images_directory:
        arguments["save_images_directory"] = save_images_directory

    # Initialize the PiPotterController
    global PiPotter
    PiPotter = PiPotterController(
        video_source_name=video_source, configuration_file=config_file, **arguments
    )
    # Horrid hack
    set_pipotter(PiPotter)

    # Start FastAPI app using Uvicorn
    uvicorn.run(
        "core.controller:app",
        host="0.0.0.0",
        port=8000,
        log_level=settings["PIPOTTER_LOGLEVEL"].lower(),
    )


if __name__ == "__main__":
    try:
        run_command()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        exit(0)
