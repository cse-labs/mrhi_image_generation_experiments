import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import directories
from directories import (
    generated_dir,
)
from generate_event_handler import run_onpack_process

load_dotenv()
input_file_name_relative = "pack/pack_image.png"
mrhi_image_loc_relative = "mrhi/mrhi.jpeg"
bottom_text = "12 PACKS"
scan_dir = False

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logging.getLogger("saicinpainting.training.trainers.base").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.info("Setting PYTORCH_ENABLE_MPS_FALLBACK=1 for mac machines to fallback to CPU")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

if __name__ == "__main__":
    base_image_location = os.path.join(
        directories.images_dir, f"{input_file_name_relative}"
    )
    # Clean up generated mask and generated directories
    for file in os.listdir(generated_dir):
        os.remove(os.path.join(generated_dir, file))
    for file in os.listdir(directories.generated_mask_dir):
        os.remove(os.path.join(directories.generated_mask_dir, file))
    generated_image_bytes, validation_results, evaluation_results = run_onpack_process(
        original_file_path=Path(base_image_location),
        generated_dir=Path(generated_dir),
        mask_dir=Path(directories.generated_mask_dir),
        mrhi_dir=Path(directories.big_lama_model_dir),
        original_mrhi_image=Path(
            os.path.join(
                directories.images_dir,
                f"{mrhi_image_loc_relative}",
            )
        ),
        text_input=bottom_text,
        should_validate=False,
        should_evaluate=False,
    )
