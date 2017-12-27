import logging
from picamera import PiCamera
cam = PiCamera()
import click

from core.config import settings
from core.controller import PiPotterController
from core.log import configure_log

configure_log(settings['PIPOTTER_LOGLEVEL'])
logger = logging.getLogger(__name__)


@click.command()
@click.option('--video-source', required=True, type=click.Choice(['looper', 'picamera']),
              help='Video source to use : picamera or loop')
@click.option('--video-file', help='Video file to loop')
@click.option('--video-file', help='Video file to loop')
@click.option('--draw-windows/--no-draw-windows', default=False, help='Draw windows from CV2')
@click.option('--save_images_directory', help='Store the detected movements as images in the passed directory')
@click.option('--config-file', help='Where to read the configuration file. Defaults to ./config.json',
              default='./config.json')
def run_command(video_source, video_file, draw_windows, save_images_directory, config_file):
    """
    The main method
    """
    # Let's create the proper video source first
    if video_source == 'looper':
        arguments = {'video_file': video_file}
    elif video_source == 'picamera':
        from media.video_source import picamera
        cam.resolution = (640, 480)
        cam.framerate = 24
        arguments = {'camera': cam}
    if save_images_directory:
        arguments['save_images_directory'] = save_images_directory
    controller = PiPotterController(video_source_name=video_source, draw_windows=draw_windows,
                                    configuration_file=config_file, **arguments)
    controller.run()


if __name__ == '__main__':
    run_command()
