import logging

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
def run_command(video_source, video_file, draw_windows):
    """
    The main method
    """
    # Let's create the proper video source first
    if video_source == 'looper':
        arguments = {'video_file': video_file}
    elif video_source == 'picamera':
        from media.video_source import picamera
        import picamera
        cam = picamera.PiCamera()
        cam.resolution = (640, 480)
        cam.framerate = 24
        arguments = {'video_file': video_file}
    controller = PiPotterController(video_source_name=video_source, draw_windows=draw_windows, **arguments)
    controller.run()


if __name__ == '__main__':
    run_command()
