import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from io import BytesIO
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from azure.cognitiveservices.vision.customvision.prediction.models import (
    ImagePrediction,
)
from dotenv import load_dotenv
from msrest.authentication import ApiKeyCredentials
from ImageCompression import convert_to_monochrome

load_dotenv()
prediction_key = os.environ["CUSTOM_VISION_KEY"]
endpoint = os.environ["CUSTOM_VISION_ENDPOINT"]
prediction_credentials = ApiKeyCredentials(
    in_headers={"Prediction-key": prediction_key}
)
predictor = CustomVisionPredictionClient(endpoint, prediction_credentials)


def perform_canny_edge_detection(img):
    """
    Perform Canny edge detection on an image.

    This function converts the input image to grayscale, applies a Gaussian blur, and then performs Canny edge detection.

    Parameters:
    img (numpy.ndarray): The input image on which edge detection is to be performed.

    Returns:
    edges (numpy.ndarray): The output image after applying Canny edge detection.
    """

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=100)
    return edges


def is_contour_inside_roi(contour, roi_coordinates):
    """
    Check if a contour is inside a given region of interest (ROI).

    Parameters
    ----------
    contour : array
        The contour to check, typically obtained from cv2.findContours.
    roi_coordinates : tuple
        The coordinates of the ROI in the format (x1, y1, x2, y2).

    Returns
    -------
    bool
        True if the contour is inside the ROI, False otherwise.
    """
    (x, y, w, h) = cv2.boundingRect(contour)
    return (
        roi_coordinates[0] <= x <= roi_coordinates[2]
        and roi_coordinates[1] <= y <= roi_coordinates[3]
        and (x + w <= roi_coordinates[2])
        and (y + h <= roi_coordinates[3])
    )


def get_edge_contours(img):
    """
    Function to get the contours of the edges in an image.

    Parameters:
    img (numpy.ndarray): Input image.

    Returns:
    list: A list of contours of the edges in the image.
    """
    edges = perform_canny_edge_detection(img)
    (edge_contours, _) = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return edge_contours


def detect_contours_in_rect(img, edge_contours, roi_coordinates):
    '''"""
    This function detects contours within a specified region of interest (ROI) in an image.

    Args:
        img (numpy.ndarray): The input image.
        edge_contours (list): A list of contours detected in the image.
        roi_coordinates (tuple): A tuple specifying the coordinates of the ROI.

    Returns:
        list: A list of contours that are located within the ROI.

    Example:
        >>> img = cv2.imread('image.png', 0)
        >>> edge_contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        >>> roi_coordinates = (10, 10, 50, 50)
        >>> contours_in_roi = detect_contours_in_rect(img, edge_contours, roi_coordinates)
    """'''
    contours_inside_roi = []
    for i, contour in enumerate(edge_contours):
        if is_contour_inside_roi(contour, roi_coordinates):
            contours_inside_roi.append(contour)
    return contours_inside_roi


def create_masks(original_image, prediction_result, mask_dir):
    '''"""
    This function creates masks from the prediction results of an original image and saves them in a specified directory.

    Args:
        original_image (str): The path to the original image.
        prediction_result (list): The list of prediction results.
        mask_dir (str): The directory where the masks will be saved.

    Raises:
        shutil.SameFileError: If the mask directory and input file directory are the same.
    """'''
    filename_with_extension = os.path.basename(original_image)
    (input_file_name, input_file_ext) = os.path.splitext(filename_with_extension)
    print(f"Creating masks... from {original_image}")
    img = cv2.imread(original_image)
    mask = np.zeros(img.shape[:2], np.uint8)
    white_color = (255, 255, 255)
    edge_contours = get_edge_contours(img)
    for i, prediction in enumerate(prediction_result):
        y = int(prediction.bounding_box.top * img.shape[0]) - 2
        h = int(prediction.bounding_box.height * img.shape[0]) + 10
        x = int(prediction.bounding_box.left * img.shape[1]) - 2
        w = int(prediction.bounding_box.width * img.shape[1]) + 10
        mask[y : y + h, x : x + w] = 255
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)
    if input_file_ext != ".png":
        cv2.imwrite(
            os.path.join(mask_dir, f"{input_file_name}.png"), cv2.imread(original_image)
        )
    else:
        try:
            shutil.copyfile(
                original_image, os.path.join(mask_dir, f"{input_file_name}.png")
            )
        except shutil.SameFileError as sfe:
            print("Looks like the mask dir and input file dir are same")
    mask = cv2.GaussianBlur(
        mask, (0, 0), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_DEFAULT
    )
    cv2.imwrite(os.path.join(mask_dir, f"{input_file_name}_mask.png"), mask)
    img = cv2.imread(original_image)
    img[mask == 0] = 255
    cv2.imwrite(os.path.join(mask_dir, f"{input_file_name}_dummy.png"), img)


def get_predictions(file_path: Path):
    '''"""
    This function takes a file path as input, converts the image at the file path to monochrome, and then uses a predictor to detect images.

    The predictor uses the project ID and published name from the environment variables "CUSTOM_VISION_PROJECT_ID" and "CUSTOM_VISION_ITERATION_NAME" respectively.

    The image data is read in bytes. The function then returns a list of predictions with a probability greater than 0.8.

    If no predictions are found, the function prints a message and returns None.

    Args:
        file_path (Path): The path of the image file to be processed.

    Returns:
        list: A list of predictions with a probability greater than 0.8. Returns None if no predictions are found.
    """'''
    monochrome_image = convert_to_monochrome(file_path)
    with BytesIO() as byte_io:
        monochrome_image.save(byte_io, format="PNG")
        byte_io.seek(0)
        image_bytes = byte_io.read()
    results = predictor.detect_image(
        project_id=os.environ["CUSTOM_VISION_PROJECT_ID"],
        published_name=os.environ["CUSTOM_VISION_ITERATION_NAME"],
        image_data=image_bytes,
    )
    prediction_result = [
        prediction for prediction in results.predictions if prediction.probability > 0.8
    ]
    if len(prediction_result) == 0:
        print("No predictions found for the image")
        return None
    return prediction_result


def create_mask_and_write(base_image_location: Path, pack_dir, generated_mask_dir):
    '''"""
    This function creates a mask based on predictions and writes it to a specified directory.

    Args:
        base_image_location (Path): The location of the base image.
        pack_dir (str): The directory where the pack is located.
        generated_mask_dir (str): The directory where the generated mask will be written.

    Returns:
        None
    """'''
    predictions = get_predictions(pack_dir)
    create_masks(base_image_location, predictions, generated_mask_dir)
