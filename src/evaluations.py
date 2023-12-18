import logging
import os
from image_similarity_measures.evaluate import evaluation
from directories import mrhi_dir, generated_dir
from evaluation.text import evaluate_text_on_images

logger = logging.getLogger(__name__)


class Evaluation:
    def __init__(
        self, generated_mrhi, existing_mrhi, ssim_score=0, text_compare_score=0
    ):
        '''"""
        Initialize the instance variables of the class.

        Args:
            generated_mrhi: The generated MRHI.
            existing_mrhi: The existing MRHI.
            ssim_score (float, optional): The SSIM score. Defaults to 0.
            text_compare_score (float, optional): The text comparison score. Defaults to 0.
        """'''
        self.ssim_score = ssim_score
        self.existing_mrhi = existing_mrhi
        self.text_compare_score = text_compare_score
        self.generated_mrhi = generated_mrhi

    def __str__(self):
        '''"""
        This method is a special method in Python, known as a "dunder" method for its double underscores. It is used to return a string representation of an object. In this case, it returns a formatted string that includes the existing_mrhi, generated_mrhi, ssim_score, and text_compare_score attributes of the object.
        """'''
        return f"existing_mrhi: {self.existing_mrhi}, generated_mrhi: {self.generated_mrhi}, ssim_score: {self.ssim_score}, text_compare_score: {self.text_compare_score}"

    def __repr__(self):
        '''"""
        This method returns a string representation of the object. It calls the __str__ method of the object for this purpose.
        """'''
        return self.__str__()


def evaluate(to_be_evaluated: list[Evaluation]) -> list[Evaluation]:
    '''"""
    This function evaluates a list of Evaluation objects. For each Evaluation object, it calculates the SSIM score and text comparison score between the existing and generated MRHI images.

    Args:
        to_be_evaluated (list[Evaluation]): A list of Evaluation objects to be evaluated.

    Returns:
        list[Evaluation]: The input list of Evaluation objects, updated with the calculated SSIM and text comparison scores.
    """'''
    logger.info(f"Running Evaluations for {len(to_be_evaluated)} images")
    for evalu in to_be_evaluated:
        logger.info(f"Evaluating {evalu.existing_mrhi} and {evalu.generated_mrhi}")
        score = evaluation(
            org_img_path=evalu.existing_mrhi,
            pred_img_path=evalu.generated_mrhi,
            metrics=["ssim"],
        )
        evalu.ssim_score = score["ssim"]
        evalu.text_compare_score = evaluate_text_on_images(
            evalu.existing_mrhi, evalu.generated_mrhi
        )
    return to_be_evaluated


def run_evaluations(generated_images: list[str]):
    '''"""
    This function runs evaluations on a list of generated images. It logs the process of adding each image to the evaluation.
    It replaces the '_mask' in the file name with '_mrhi' and checks if the original file exists in the 'mrhi_dir'.
    If it does, it adds the file to the 'to_be_evaluated' list. If it doesn't, it adds the file with a '.jpeg' extension instead.
    Finally, it runs the 'evaluate' function on the 'to_be_evaluated' list and logs the evaluations found.

    Args:
        generated_images (list[str]): A list of file paths to the generated images to be evaluated.

    Raises:
        FileNotFoundError: If the original file does not exist in the 'mrhi_dir'.

    Returns:
        evaluations: The result of the 'evaluate' function on the 'to_be_evaluated' list.
    """'''
    to_be_evaluated = []
    logger.info("Running Evaluations")
    for file in generated_images:
        file = os.path.basename(file)
        logger.info("Adding file {} to evaluation".format(file))
        org_file_name = file.replace("_mask", "_mrhi")
        org_file_name = org_file_name.replace(".png", "")
        print(f"org_file_name: {org_file_name}")
        if os.path.exists(os.path.join(mrhi_dir, f"{org_file_name}.png")):
            to_be_evaluated.append(
                Evaluation(
                    os.path.join(generated_dir, file),
                    os.path.join(mrhi_dir, f"{org_file_name}.png"),
                )
            )
        else:
            to_be_evaluated.append(
                Evaluation(
                    os.path.join(generated_dir, file),
                    os.path.join(mrhi_dir, f"{org_file_name}.jpeg"),
                )
            )
    evaluations = evaluate(to_be_evaluated)
    logger.info(f"found evaluations: {evaluations}")
