class PiPotterError(Exception):
    """
    The base exception
    """

    pass


class SFXError(PiPotterError):
    """
    Raised when an error arises in the SFX module
    """

    pass


class MediaError(PiPotterError):
    """
    raised when a media/video error is raised
    """


class WandError(PiPotterError):
    """
    Raised when a Wand error is detected
    """
