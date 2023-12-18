import logging
import os
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import cornerlozenges
import directories
import inpaint_lama
from directories import fonts_dir
from inpaint_lama import LamaInpainter
from mask import create_mask_and_write, get_predictions, create_masks
load_dotenv()
logger = logging.getLogger(__name__)

def create_mask_and_write(original_image: str, mask_dir: Path, generated_dir: Path):
    '''"""
This function creates a mask for an original image and writes it to a specified directory.

Args:
    original_image (str): The path to the original image.
    mask_dir (Path): The directory where the mask will be written.
    generated_dir (Path): The directory where the generated image will be written.

Raises:
    FileNotFoundError: If the original image does not exist.

Note:
    This function uses the `get_predictions` function to generate predictions for the original image.
    It then uses the `create_masks` function to create a mask based on these predictions.
    The mask is then written to the `mask_dir`.
"""'''
    predictions = get_predictions(Path(original_image))
    create_masks(original_image, predictions, mask_dir)
    print('Waiting for mask to be created')

def generate_onpack(orignal_file: Path, temp_mask_dir: Path, temp_generated_dir: Path, bottom_text: str, final_mask_dir: str=directories.generated_mask_dir, final_output_dir: str=directories.generated_dir) -> str:
    '''"""
This function generates an onpack image by creating a mask and writing it to a temporary directory. It then copies the mask files to a final directory for future debugging. If the environment variable "ONLY_MASK" is set to "false", it uses either the LamaInpainter or ZitsInpainter to inpaint the image. If a bottom text is provided, it is processed and added to the image. The function raises an exception if more than one file is generated. The final image is then copied to the output directory.

Args:
    orignal_file (Path): The path of the original file.
    temp_mask_dir (Path): The path of the temporary mask directory.
    temp_generated_dir (Path): The path of the temporary generated directory.
    bottom_text (str): The text to be added at the bottom of the image.
    final_mask_dir (str, optional): The path of the final mask directory. Defaults to directories.generated_mask_dir.
    final_output_dir (str, optional): The path of the final output directory. Defaults to directories.generated_dir.

Returns:
    str: The path of the copied file location if "ONLY_MASK" is set to "false", else the path of the mask file.
"""'''
    create_mask_and_write(str(orignal_file), Path(temp_mask_dir), Path(temp_generated_dir))
    for ff in os.listdir(temp_mask_dir):
        if ff.endswith('.png'):
            if ff.endswith('_mask.png'):
                mask_file = os.path.join(final_mask_dir, ff)
            try:
                shutil.copy2(os.path.join(temp_mask_dir, ff), final_mask_dir)
            except shutil.SameFileError as sfe:
                pass
            logger.debug('Copied files from {} to {}'.format(temp_mask_dir, final_mask_dir))
    if os.getenv('ONLY_MASK', 'false').lower() == 'false':
        inpainter = LamaInpainter(str(directories.big_lama_model_dir), str(temp_mask_dir), str(temp_generated_dir), '.png')
        inpaint_ouput: list[str] = inpainter.inpaint()
        for image_file in inpaint_ouput:
            if bottom_text != '' and bottom_text is not None:
                modified_image = cornerlozenges.process(image_file, font_dir=fonts_dir, text=bottom_text)
                modified_image.save(image_file)
        if len(inpaint_ouput) != 1:
            raise Exception(f'Was expecting exactly one file, but got {len(inpaint_ouput)}')
        logger.info('Copying file {} to output directory'.format(inpaint_ouput[0]))
        try:
            src = os.path.join(temp_generated_dir, inpaint_ouput[0])
            copied_file_location = shutil.copy2(src, os.path.join(final_output_dir, f'output_{os.path.basename(src)}'))
        except shutil.SameFileError as sfe:
            logger.error(f'Ignoring SameFileError. This must be runnning with main.py. The message is {str(sfe)}')
            copied_file_location = os.path.join(temp_generated_dir, inpaint_ouput[0])
        return copied_file_location
    else:
        return mask_file

def generate_mrhi_onpack(original_image: Path, mask_image: Path, output_image: Path, bottom_text: str, big_lama_model_dir: Path=directories.big_lama_model_dir):
    '''"""
This function generates a modified image with a mask and text overlay, using the Big LAMA model for inpainting.

Args:
    original_image (Path): The path to the original image.
    mask_image (Path): The path to the mask image.
    output_image (Path): The path where the output image will be saved.
    bottom_text (str): The text to be overlaid at the bottom of the image.
    big_lama_model_dir (Path, optional): The directory of the Big LAMA model. Defaults to directories.big_lama_model_dir.

Returns:
    dict: A dictionary mapping the original image path to the path of the generated image.

Raises:
    FileNotFoundError: If the original image or mask image does not exist.
    Exception: If there is an error during the inpainting or image processing steps.
"""'''
    with tempfile.TemporaryDirectory() as tmpdir:
        mask_dir = os.path.join(tmpdir, 'mask')
        generated_dir = os.path.join(tmpdir, 'generated')
        mrhi_dir = os.path.join(tmpdir, 'mrhi')
        original_dir = os.path.join(tmpdir, 'original')
        os.makedirs(mask_dir)
        os.makedirs(generated_dir)
        os.makedirs(mrhi_dir)
        os.makedirs(original_dir)
        orignal_file = shutil.copy2(original_image, original_dir)
        create_mask_and_write(orignal_file, Path(mask_dir), Path(generated_dir))
        inpaint_ouput: list[str] = inpaint_lama.inpaint_lama(big_lama_model_dir, mask_dir, generated_dir, '.png')
        to_be_evaluated = []
        for image_file in inpaint_ouput:
            modified_image = cornerlozenges.process(image_file, font_dir=fonts_dir, text=bottom_text)
            modified_image.save(image_file)
        source_generated_map = {}
        for file in os.listdir(generated_dir):
            logger.info('Copying file {} to output directory'.format(file))
            copied_file_location = shutil.copy2(os.path.join(generated_dir, file), directories.generated_dir)
            source_generated_map[original_image] = os.path.join(directories.generated_dir, copied_file_location)
        return source_generated_map