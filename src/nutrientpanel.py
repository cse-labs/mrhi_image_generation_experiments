import json
import os
import requests
from PIL import Image, ImageDraw
from dotenv import load_dotenv
load_dotenv()

def extract_bounding_box_image(image, left, top, width, height):
    '''"""
This function extracts a bounding box image from the given image.

Parameters:
    image (PIL.Image): The source image from which to extract the bounding box.
    left (float): The left coordinate of the bounding box, as a fraction of the image width.
    top (float): The top coordinate of the bounding box, as a fraction of the image height.
    width (float): The width of the bounding box, as a fraction of the image width.
    height (float): The height of the bounding box, as a fraction of the image height.

Returns:
    PIL.Image: The extracted bounding box image.
"""'''
    left = left * image.width
    top = top * image.height
    right = left + width * image.width
    bottom = top + height * image.height
    bounding_box_image = image.crop((left, top, right, bottom))
    return bounding_box_image

def get_nutrient_panel(uploaded_file, target_image_path):
    '''"""
This function takes an uploaded file and a target image path as input, and returns an image with bounding boxes around the nutrient panel. 

The function first reads the file content and prepares it for a POST request. It then sets the headers for the request, including the content type and prediction key. 

The function then performs the POST request and loads the response into a JSON object. 

Next, the function filters the predictions in the response, keeping only those with a probability above 0.75 and a tag name of "Nutrient-Panel". 

The filtered predictions are then sorted by probability in descending order. 

The function then loads the image and displays it with bounding boxes. It does this by opening the image, creating a canvas to draw on, and then drawing bounding boxes on top of the image. 

Finally, the function extracts the bounded box image and saves it as "bounding_box_image.jpg", and returns the extracted bounded box image.

Parameters:
uploaded_file (file): The file to be processed.
target_image_path (str): The path where the image will be saved.

Returns:
extracted_bounded_box (Image): The image with bounding boxes around the nutrient panel.
"""'''
    file_data = uploaded_file.read()
    headers = {'Content-Type': 'application/octet-stream', 'prediction-key': os.environ['NUTRIENT_COMPUTER_VISION_KEY']}
    response = requests.post(os.environ['NUTRIENT_COMPUTER_VISION_ENDPOINT'], data=file_data, headers=headers)
    responseJSON = json.loads(response.text)
    filtered_predictions = [prediction for prediction in responseJSON['predictions'] if prediction['probability'] > 0.75 and prediction['tagName'] == 'Nutrient-Panel']
    sorted_predictions = sorted(filtered_predictions, key=lambda x: x['probability'], reverse=True)
    top_prediction = sorted_predictions[0]
    image = uploaded_file
    with Image.open(uploaded_file) as im:
        draw = ImageDraw.Draw(im)
        extracted_bounded_box = extract_bounding_box_image(im, top_prediction['boundingBox']['left'], top_prediction['boundingBox']['top'], top_prediction['boundingBox']['width'], top_prediction['boundingBox']['height'])
        extracted_bounded_box.save('bounding_box_image.jpg')
        return extracted_bounded_box