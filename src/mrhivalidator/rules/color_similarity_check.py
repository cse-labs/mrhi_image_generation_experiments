# color_similarity_check.py

import cv2
from mrhivalidator.utils.criteria import CheckResult
from mrhivalidator.utils.base_check import BaseCheck
from mrhivalidator.models.evaluation_result import EvaluationResult

RULE_NAME = "ColorSimilarity"
class ColorSimilarity(BaseCheck):

    
    def evaluate(self,**kwargs) -> EvaluationResult:
        
        # Read the images
        mrhi_image_path = cv2.imread(kwargs['mrhi_image'])
        original_image_path = cv2.imread(kwargs['original_image'])

        # Convert both images to HSV
        mrhi_image_path_hsv = cv2.cvtColor(mrhi_image_path, cv2.COLOR_BGR2HSV)
        original_image_path_hsv = cv2.cvtColor(original_image_path, cv2.COLOR_BGR2HSV)

        # Compute histograms
        hist1 = cv2.calcHist([mrhi_image_path_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([original_image_path_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])

        # Normalize histograms
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Compute Bhattacharyya distance
        color_difference = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

        # Compute percentage similarity
        percentage_similarity = (1 - color_difference) * 100

        print(f"Color Similarity: {percentage_similarity:.2f}%")
        
        result = CheckResult.PASS if percentage_similarity > 80 else CheckResult.FAIL

        return EvaluationResult(result=result, score=percentage_similarity, range=self.RANGE)
