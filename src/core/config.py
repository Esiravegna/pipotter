from __future__ import division
from pathlib import Path
from os import getenv

project_root = Path(__file__).parent

settings = {
    # Loglevel read from the env
    "PIPOTTER_LOGLEVEL": getenv("PIPOTTER_LOGLEVEL", "INFO"),
    # Extra audios should these be needed. For instance, the default allows a USB soundcard attached to ALSA.
    "PIPOTTER_EXTRA_AUDIO_COMMANDS": ["-volume", "15"],
    # Flip list, as per all the video_source objects
    "PIPOTTER_FLIP_VIDEO": getenv("PIPOTTER_FLIP_VIDEO", [2]),
    # How many seconds will be used to draw, aka, how long the scanner will track a wand after calling it done
    "PIPOTTER_SECONDS_TO_DRAW": getenv("PIPOTTER_SECONDS_TO_DRAW", 3),
    # Where to retrieve/store the SpellNet file
    "PIPOTTER_MODEL_DIRECTORY": getenv(
        "PIPOTTER_MODEL_DIRECTORY", "src/wand/spell_net2"
    ),
    # Dimensions of the to use for SpellNet2. Defaults to 224, as SpellNet was trained with that size
    "PIPOTTER_SIDE_SPELL_NET": 224,
    # Threshold to trigger a positive detection. If it is lower than this, we will not trigger an effect.
    # This should be at least 1/the number of possible spells. In this case, we've six. please take a look at
    # the media/spell_net2/spell_net.ipynb file
    "PIPOTTER_THRESHOLD_TRIGGER": 0.49,
    # Lenght of buffer size"
    "PIPOTTER_SPELL_HISTORY_SIZE": 2,
    "PIPOTTER_TRIWIZARD_SPELL_NAME": "triwizard_cup",
    # How do we cann a label for no spell detected? defaults to background. It cannot be any of the spells already existing
    "PIPOTTER_NO_SPELL_LABEL": getenv("PIPOTTER_NO_SPELL_LABEL", "background"),
    # Optionally, we will use a remote SpellNET server?
    "PIPOTTER_REMOTE_SPELLNET_SERVER": getenv("PIPOTTER_REMOTE_SPELLNET_SERVER", False),
    "PIPOTTER_REMOTE_SPELLNET_SERVER_PORT": getenv(
        "PIPOTTER_REMOTE_SPELLNET_SERVER_PORT", 4242
    ),
    # Effect to indicate we're set
    "PIPOTTER_READY_SFX": getenv("PIPOTTER_READY_SFX", "pipotter_ready"),
}
