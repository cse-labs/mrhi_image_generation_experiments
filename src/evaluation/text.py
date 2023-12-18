from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials


from Levenshtein import ratio
import numpy as np
from array import array
import os
from PIL import Image
import sys
import time
from dotenv import load_dotenv

load_dotenv()
load_dotenv("..")


def get_ocr(image_path):
    subscription_key = os.environ["VISION_KEY"]
    endpoint = os.environ["VISION_ENDPOINT"]

    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

    # Get an image with text
    with open(image_path, "rb") as filestream:
        read_response = computervision_client.read_in_stream(filestream,  raw=True)
    # Call API with URL and raw response (allows you to get the operation location)

    # Get the operation location (URL with an ID at the end) from the response
    read_operation_location = read_response.headers["Operation-Location"]
    # Grab the ID from the URL
    operation_id = read_operation_location.split("/")[-1]

    # Call the "GET" API and wait for it to retrieve the results 
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    # Print the detected text, line by line
    if read_result.status == OperationStatusCodes.succeeded:
        text_results = read_result.analyze_result.read_results    
        return text_results
    else:
        raise Exception("Reading OCR results failed.")


def ocr_match_score(word_obj1, word_obj2, bbox_dist_weight=0.5):
    # check if words match    
    score = ratio(word_obj1.text, word_obj2.text)
    # calc bounding box distance
    bbox_dist = ocr_bbox_distance(
        word_obj1.bounding_box, word_obj2.bounding_box)
    score -= bbox_dist * bbox_dist_weight
    # score = (1 - min(bbox_dist, 1.0) + score) / 2
    return max(score, 0.0)
    # return score if score >= score_thresh else 0.0
    

def ocr_bbox_distance(bbox1, bbox2):
    dist = 0
    for v1, v2 in zip(bbox1, bbox2):
        dist += abs(v1/v2 - 1)
    return min(dist/8, 1.0)


def evaluate_text_on_images(target_image, generated_image, lines_weight=0.25):
    ocr_results_target = get_ocr(target_image)
    ocr_results_output = get_ocr(generated_image)

    lines_num_target = len(ocr_results_target[0].lines)
    lines_num_output = len(ocr_results_output[0].lines)

    ocr_match_matrix = np.zeros((lines_num_target, lines_num_output))
    for i, ocr_target in enumerate(ocr_results_target[0].lines):
        for j, ocr_output in enumerate(ocr_results_output[0].lines):
            ocr_match_matrix[i][j] = ocr_match_score(ocr_target, ocr_output)
    # print("===")
    # print(ocr_match_matrix)
    # print("===")

    avg_ocr_match_score = np.average(np.max(ocr_match_matrix, axis=-1))
    ocr_lines_score = min(abs(lines_num_target - lines_num_output) / lines_num_target, 1.0)
    # print(lines_num_target, lines_num_output, avg_ocr_match_score, ocr_lines_score)

    return avg_ocr_match_score - (ocr_lines_score * lines_weight)



# from collections import Counter

# final_scores = []
# idxs = np.argmax(ocr_match_matrix, axis=-1)
# counter = Counter(idxs)
# for i, j in enumerate(idxs):
#     if counter[j] == 1:
#         final_scores.append(ocr_match_matrix[i][j])
#     else:
#         final_scores.append(
