import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))
from sfx.factory import EffectFactory

logger = logging.getLogger(__name__)
logging.info("Reading effects...")
SFXs = EffectFactory(config_file="./config/config.json")
logger.info(f"Effects ready: {[a_spell for a_spell in SFXs]}")
print([a_spell for a_spell in SFXs])
for a_spell in SFXs:
    print(f"Running {a_spell}")
    SFXs[a_spell].run()
