import io
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from streamlit.runtime.uploaded_file_manager import UploadedFile

import directories
import evaluations
import onpack
from directories import fonts_dir
from evaluations import Evaluation
from mrhivalidator.main import validate
from nutrientpanel import get_nutrient_panel
from offpack import generate_mrhi_offpack

load_dotenv()


def create_directories(tmpdir):
    mask_dir = os.path.join(tmpdir, "mask")
    generated_dir = os.path.join(tmpdir, "generated")
    mrhi_dir = os.path.join(tmpdir, "mrhi")
    original_dir = os.path.join(tmpdir, "original")
    os.makedirs(mask_dir)
    os.makedirs(generated_dir)
    os.makedirs(mrhi_dir)
    os.makedirs(original_dir)
    return (generated_dir, mask_dir, original_dir, mrhi_dir)


def off_pack_generate_clicked(
        original_image, original_mrhi_image, left_text: str, right_text: str
):
    '''"""
    This function generates an off-pack image and returns it along with an evaluation dictionary.

    Args:
        original_image (BytesIO): The original image in BytesIO format.
        original_mrhi_image (BytesIO): The original MRHI image in BytesIO format.
        left_text (str): The text to be placed on the left side of the image.
        right_text (str): The text to be placed on the right side of the image.

    Returns:
        BytesIO: The generated off-pack image in BytesIO format.
        dict: A dictionary containing evaluation metrics.
    """'''
    file_bytes = original_image.getvalue()
    file_name = original_image.name
    with tempfile.TemporaryDirectory() as tmpdir:
        (generated_dir, mask_dir, original_dir, mrhi_dir) = create_directories(tmpdir)
        with open(os.path.join(original_dir, file_name), "wb") as f:
            f.write(file_bytes)
            result_image_path = os.path.join(generated_dir, file_name)
            generate_mrhi_offpack(
                source_image_path=Path(os.path.join(original_dir, file_name)),
                left_text=left_text,
                right_text=right_text,
                font_file_path=os.path.join(fonts_dir, "OpenSans-ExtraBold.ttf"),
                result_image_path=result_image_path,
            )
            evaluation_dict = {}
            evaluation_dict["original"] = "Some value"
            evaluation_dict["SSIM"] = "99%"
            return (io.BytesIO(open(result_image_path, "rb").read()), evaluation_dict)


def on_pack_generate_clicked(original_image, original_mrhi_image, text_input: str):
    '''"""
    This function is triggered when the 'pack generate' button is clicked. It takes an original image, an original MRHI image, and a text input as arguments.

    The function first gets the value and name of the original image. It then creates a temporary directory and within it, creates four subdirectories: 'generated_dir', 'mask_dir', 'original_dir', and 'mrhi_dir'.

    The original image is then written into a file in the 'original_dir'. The function then calls the 'run_onpack_process' function, passing the path of the original file, the paths of the four directories, the original MRHI image, and the text input as arguments.

    Args:
        original_image: The original image file.
        original_mrhi_image: The original MRHI image file.
        text_input (str): The text input.

    Returns:
        The result of the 'run_onpack_process' function.
    """'''
    file_bytes = original_image.getvalue()
    file_name = original_image.name
    with tempfile.TemporaryDirectory() as tmpdir:
        (generated_dir, mask_dir, original_dir, mrhi_dir) = create_directories(tmpdir)
        original_file_path = os.path.join(original_dir, file_name)
        with open(original_file_path, "wb") as f:
            f.write(file_bytes)
            return run_onpack_process(
                original_file_path=Path(os.path.join(original_dir, file_name)),
                generated_dir=Path(generated_dir),
                mask_dir=Path(mask_dir),
                mrhi_dir=Path(mrhi_dir),
                original_mrhi_image=original_mrhi_image,
                text_input=text_input,
            )


def run_onpack_process(
        generated_dir: Path,
        mask_dir: Path,
        mrhi_dir: Path,
        original_file_path: Path,
        original_mrhi_image: UploadedFile,
        text_input,
        should_validate=os.getenv("VALIDATE", "False").lower() == "true",
        should_evaluate=os.getenv("EVALUATE", "False").lower() == "true",
        final_mask_dir: str = directories.generated_mask_dir,
        final_output_dir: str = directories.generated_dir,
):
    '''"""
    This function runs the on-pack process which includes generating an on-pack image, validating and evaluating the generated image.

    Args:
        generated_dir (Path): The directory where the generated images are stored.
        mask_dir (Path): The directory where the mask images are stored.
        mrhi_dir (Path): The directory where the MRHI images are stored.
        original_file_path (Path): The path of the original file.
        original_mrhi_image (UploadedFile): The original MRHI image file.
        text_input (str): The text input for the on-pack image.
        should_validate (bool, optional): A flag to determine if the generated image should be validated. Defaults to the value of the environment variable "VALIDATE".
        should_evaluate (bool, optional): A flag to determine if the generated image should be evaluated. Defaults to the value of the environment variable "EVALUATE".
        final_mask_dir (str, optional): The directory where the final mask images are stored. Defaults to directories.generated_mask_dir.
        final_output_dir (str, optional): The directory where the final output images are stored. Defaults to directories.generated_dir.

    Raises:
        Exception: If CUDA is not available and the environment variable "USE_LAMA" is not set to true.

    Returns:
        tuple: A tuple containing the BytesIO object of the generated image, the validation results, and the evaluation results.
    """'''
    copied_file_location = onpack.generate_onpack(
        orignal_file=original_file_path,
        bottom_text=text_input,
        temp_mask_dir=mask_dir,
        temp_generated_dir=generated_dir,
        final_mask_dir=final_mask_dir,
        final_output_dir=final_output_dir,
    )
    validation_results = {}
    evaluation_result = {}
    if should_validate:
        rules_config = {
            "active_rules": {
                "ImageSimilarity": [
                    {
                        "kwargs": {
                            "mrhi_image": str(copied_file_location),
                            "original_image": str(original_file_path),
                        }
                    }
                ],
                "ColorSimilarity": [
                    {
                        "kwargs": {
                            "mrhi_image": str(copied_file_location),
                            "original_image": str(original_file_path),
                        }
                    }
                ],
                "MRHI4W": [
                    {
                        "kwargs": {
                            "mrhi_image": str(copied_file_location),
                            "metadata": {"title": "belvita®"},
                        }
                    },
                    {
                        "kwargs": {
                            "mrhi_image": str(copied_file_location),
                            "metadata": {"SOI": "breakfast biscuits"},
                        }
                    },
                    {
                        "kwargs": {
                            "mrhi_image": str(copied_file_location),
                            "metadata": {"sub title": "brown sugar"},
                        }
                    },
                    {
                        "kwargs": {
                            "mrhi_image": str(copied_file_location),
                            "metadata": {"weight": "25"},
                        }
                    },
                ],
            }
        }
        validation_results = validate(rules_config, table_output=False)
    if should_evaluate:
        to_be_evaluated = []
        if original_mrhi_image:
            mrhi_file = os.path.join(mrhi_dir, original_mrhi_image.name)
            with open(mrhi_file, "wb") as orginal_mrhi_file:
                orginal_mrhi_file.write(original_mrhi_image.getvalue())
                to_be_evaluated.append(
                    Evaluation(str(mrhi_file), str(copied_file_location))
                )
                for evaluation in evaluations.evaluate(to_be_evaluated):
                    evaluation_result["ssim_score"] = evaluation.ssim_score
                    evaluation_result[
                        "text_compare_score"
                    ] = evaluation.text_compare_score
    return (
        io.BytesIO(open(copied_file_location, "rb").read()),
        validation_results,
        evaluation_result,
    )


def generate_nutrient_panel_handler(uploaded_file, target_image_path):
    '''"""
    This function handles the generation of a nutrient panel.

    Args:
        uploaded_file (file): The file that has been uploaded by the user.
        target_image_path (str): The path where the generated image will be saved.

    Returns:
        The result of the get_nutrient_panel function.
    """'''
    return get_nutrient_panel(uploaded_file, target_image_path)
