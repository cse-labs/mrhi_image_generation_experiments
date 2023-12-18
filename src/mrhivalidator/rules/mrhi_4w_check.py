import mrhivalidator.utils.image_processing as image_processing
import mrhivalidator.utils.azure_ai as azure_ai
from mrhivalidator.utils.criteria import CheckResult
from mrhivalidator.utils.base_check import BaseCheck
from mrhivalidator.models.evaluation_result import EvaluationResult
import cv2

RULE_NAME = "MRHI4W"

class MRHI4W(BaseCheck):

    def evaluate(self,**kwargs) -> EvaluationResult:
        #if image path and metadata is empty, return 0
        if not kwargs['mrhi_image'] and kwargs['metadata']:
            return 0
        metadata = kwargs['metadata']
        metadata_type = next(iter(metadata.keys()))  # Get the first metadata type key
        metadata_value = metadata.get(metadata_type)
        #print(f"Metadata Type: {metadata_type}")
        #print(f"Metadata Value: {metadata_value}")
        
        # Get the OCR results from Azure
        ocr_map = azure_ai.get_azure_ocr_results(kwargs['mrhi_image'])
        #print(ocr_map)
        # Calculate the area of the full image
        image_area = self.calculate_image_area(kwargs['mrhi_image'])
        #print(f"Area of Full Image: {image_area} square pixels")
        
        word_bounding_box = self.get_bounding_box_for_word(ocr_map,metadata_value)
        print(f"Word's Bounding Box: {word_bounding_box}")
        if not word_bounding_box:
            return EvaluationResult(result=CheckResult.FAIL, score=0, range=self.RANGE, explanation="Word not found in the image")
            
        # Calculate the area of the word's bounding box
        word_area = self.calculate_quadrilateral_area(word_bounding_box)
    
        print(f"Area of Word's Bounding Box: {word_area} square pixels")
        
        # Find the difference between the image area and word area
        percentage_difference = (word_area / image_area) * 100
        print(f"percentage difference is {percentage_difference:.2f}%")
        explanation = "The difference between the full image and {type}-{value} is {difference:.2f}%".format(type=metadata_type,value=metadata_value,difference=percentage_difference )
        # If percentage differences of the smallest ratio to the biggest one is greater than 60%, then return 1
        result = CheckResult.FAIL if percentage_difference <= 0.2 else CheckResult.PASS
        return EvaluationResult(result=result, score=percentage_difference, range=self.RANGE, explanation=explanation)
    
    def get_bounding_box_for_word(self, ocr_map, word):
        # Convert the word to lowercase before checking in the ocr_map
        word = word.lower()
        if word in ocr_map:
            bounding_box = tuple(ocr_map[word])  # Convert the list to a tuple
            return bounding_box
        else:
            return None  # Return None if the word is not found in the ocr_map

    # Function to calculate the area of a bounding box
    def calculate_quadrilateral_area(self,bounding_box):
        x1, y1, x2, y2, x3, y3, x4, y4 = bounding_box
        return 0.5 * abs((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

    
    def calculate_image_area(self,image_path):
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        return height * width
    
    
    