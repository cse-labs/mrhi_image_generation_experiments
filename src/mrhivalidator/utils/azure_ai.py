from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import os
import time

ocr_map = {}

def get_azure_ocr_results(image_path) -> dict[str, list]:
    """
    Perform OCR on an image using Azure OCR service and return results.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Dict[str, list]: A dictionary containing text and coordinates (integer).
    """
    subscription_key = os.environ["VISION_KEY"]
    endpoint = os.environ["VISION_ENDPOINT"]

    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    # Open the image
    read_image = open(image_path, "rb")

    # Call API with image and raw response (allows you to get the operation location)
    read_response = computervision_client.read_in_stream(read_image, raw=True)
    # Get the operation location (URL with ID as last appendage)
    read_operation_location = read_response.headers["Operation-Location"]
    # Take the ID off and use to get results
    operation_id = read_operation_location.split("/")[-1]

    # Call the "GET" API and wait for the retrieval of the results
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status.lower () not in ['notstarted', 'running']:
            break
        time.sleep(10)

    # Print results, line by line
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                ocr_map[line.text.lower()] = line.bounding_box
                # optimization to break it in words, to effectively query user's text
                words = line.text.split()
                 # Iterate through words and add to the ocr_map
                for word in words:
                    ocr_map[word.lower()] = line.bounding_box
                    
    return ocr_map
