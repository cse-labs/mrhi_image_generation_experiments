import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def largest_rectangle_in_image(image):
    '''"""
This function takes an image as input and returns the coordinates and dimensions of the largest rectangle found in the image.

Parameters:
    image (numpy.ndarray): The input image.

Returns:
    tuple: A tuple containing the x and y coordinates of the top left corner of the rectangle, and the width and height of the rectangle.

Steps:
1. The function first converts the image to grayscale.
2. It then applies a binary inverse threshold to segment the background and foreground.
3. The function finds contours in the segmented image.
4. It filters out small contours (noise) and keeps the larger ones.
5. The function then finds the largest bounding rectangle among the filtered contours.
6. Finally, it returns the coordinates and dimensions of the largest rectangle.
"""'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (_, thresholded) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    (contours, _) = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]
    largest_rectangle = None
    largest_area = 0
    for contour in filtered_contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_rectangle = (x, y, w, h)
    (x, y, w, h) = largest_rectangle
    return (x, y, w, h)

def create_rounded_square_with_shadow(image, x1, y1, size, border_thickness, radius, box_color):
    '''"""
This function creates a rounded square with a shadow effect on an image.

Parameters:
    image (numpy.ndarray): The input image on which the square is to be drawn.
    x1, y1 (int): The coordinates of the top left corner of the square.
    size (int): The size of the square.
    border_thickness (int): The thickness of the border of the square.
    radius (int): The radius of the corners of the square.
    box_color (tuple): The color of the square in RGB format.

Returns:
    cv2_image (numpy.ndarray): The output image with the rounded square and shadow effect.

Example:
    >>> img = np.zeros((500, 500, 3), dtype=np.uint8)
    >>> output_img = create_rounded_square_with_shadow(img, 50, 50, 200, 10, 20, (255, 0, 0))
"""'''
    print(f'in create rounded rectangle ,  box_color : {box_color}')
    pillow_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pillow_image)
    (x2, y2) = (x1 + size, y1 + size)
    outer_border_thickness = border_thickness / 7
    outer_x1 = x1 - outer_border_thickness
    outer_y1 = y1 - outer_border_thickness
    outer_x2 = x2 + outer_border_thickness
    outer_y2 = y2 + outer_border_thickness
    shadow_color = (180, 180, 180)
    draw.rounded_rectangle([(outer_x1, outer_y1), (outer_x2, outer_y2)], radius, fill=shadow_color, outline=shadow_color, width=outer_border_thickness)
    draw.rounded_rectangle([(x1, y1), (x2, y2)], radius, fill='white', outline='white', width=border_thickness)
    inner_x1 = x1 + 0.5 * border_thickness
    inner_y1 = y1 + 0.5 * border_thickness
    inner_x2 = x2 - 0.5 * border_thickness
    inner_y2 = y2 - 0.5 * border_thickness
    draw.rounded_rectangle([(inner_x1, inner_y1), (inner_x2, inner_y2)], radius, fill=box_color, outline='white')
    cv2_image = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)
    return cv2_image

def get_font_sizes(square_size):
    '''"""
This function calculates and returns the font sizes based on the given square size.

Parameters:
    square_size (int): The size of the square.

Returns:
    tuple: A tuple containing two integers representing the font sizes.
"""'''
    print(f'square_size: {square_size}')
    font1 = int(square_size * 0.5027)
    font2 = int(font1 * 0.44)
    return (font1, font2)

def add_text(font_dir, pillow_image, text, box_x, box_y, box_size):
    '''"""
This function adds text to a given image using the Pillow library. The text is split into two parts and each part is drawn separately on the image.

Args:
    font_dir (str): The directory where the font file is located.
    pillow_image (PIL.Image): The image to which the text will be added.
    text (str): The text to be added to the image. It should contain two words.
    box_x (int): The x-coordinate of the box where the text will be added.
    box_y (int): The y-coordinate of the box where the text will be added.
    box_size (int): The size of the box where the text will be added.

Returns:
    PIL.Image: The image with the added text.
"""'''
    draw = ImageDraw.Draw(pillow_image)
    font_location = os.path.join(font_dir, 'OpenSans-ExtraBold.ttf')
    (font_size1, font_size2) = get_font_sizes(box_size)
    print(f'font_size1 : {font_size1}')
    print(f'font_size2 : {font_size2}')
    font1 = ImageFont.truetype(font_location, font_size1, encoding='unic')
    font2 = ImageFont.truetype(font_location, font_size2, encoding='unic')
    words = text.split()
    text1 = words[0]
    (text_width1, text_height1) = draw.textsize(text1, font=font1)
    text2 = words[1]
    (text_width2, text_height2) = draw.textsize(text2, font=font2)
    x_centered_text1 = box_x + text_width1 // 3
    y_centered_text1 = box_y + text_height1 // 16
    x_centered_text2 = box_x + text_width2 // 5
    y_centered_text2 = y_centered_text1 + text_height1 + text_height1 // 12
    draw.text((x_centered_text1, y_centered_text1), text1, (255, 255, 255), font=font1)
    draw.text((x_centered_text2, y_centered_text2), text2, (255, 255, 255), font=font2)
    return pillow_image

def increase_canvas_size(cv_image, box_size):
    '''"""
def increase_canvas_size(cv_image, box_size):
    """
    This function increases the canvas size of a given image.

    Parameters:
    cv_image (numpy.ndarray): The input image in OpenCV format.
    box_size (int): The size of the box to be subtracted from the original image dimensions.

    Returns:
    numpy.ndarray: The image with increased canvas size.
    """
    test_img = cv_image.copy()
    # Scale down to box_size
    print(test_img.shape)
    # print(final_shape)
    new_image_Scaled = cv2.resize(
        test_img,
        (test_img.shape[0] - box_size, test_img.shape[1] - box_size),
        interpolation=cv2.INTER_LINEAR,
    )
    image_with_increases_canvas = np.ones_like(test_img) * 255
    image_with_increases_canvas[
        0 : new_image_Scaled.shape[0], 0 : new_image_Scaled.shape[1]
    ] = new_image_Scaled
    return image_with_increases_canvas
"""'''
    test_img = cv_image.copy()
    print(test_img.shape)
    new_image_Scaled = cv2.resize(test_img, (test_img.shape[0] - box_size, test_img.shape[1] - box_size), interpolation=cv2.INTER_LINEAR)
    image_with_increases_canvas = np.ones_like(test_img) * 255
    image_with_increases_canvas[0:new_image_Scaled.shape[0], 0:new_image_Scaled.shape[1]] = new_image_Scaled
    return image_with_increases_canvas

def get_box_size(image):
    '''"""
This function calculates the size of a box based on the largest rectangle in an image.

Parameters:
image (numpy array): The input image.

Returns:
box_size (int): The size of the box. If the calculated size is less than 200, it returns 200.
"""'''
    (x, y, w, h) = largest_rectangle_in_image(image)
    box_size = int(w * 0.176)
    if box_size < 200:
        box_size = 200
    return box_size

def process(image_path, font_dir, text):
    '''"""
This function processes an image by increasing its canvas size, finding the largest rectangle in the image, creating a square box within that rectangle, and adding text to the image.

Args:
    image_path (str): The path to the image file.
    font_dir (str): The directory where the font file is located.
    text (str): The text to be added to the image.

Returns:
    pil_image_with_text (PIL.Image.Image): The processed image with added text.

Raises:
    FileNotFoundError: If the image file or font file does not exist.
    ValueError: If the text is empty or None.
"""'''
    image = cv2.imread(image_path)
    image = increase_canvas_size(image, get_box_size(image))
    (x, y, w, h) = largest_rectangle_in_image(image)
    print(f'x:{x} , y: {y} , w:{w} , h: {h}')
    box_border_thickness = 10
    box_size = get_box_size(image)
    print(f'box_size: {box_size}')
    bottom_right_x = x + w + 50
    bottom_right_y = y + h - 100
    box_x = bottom_right_x - box_size
    box_y = bottom_right_y - 2 * box_border_thickness
    box_color = (0, 86, 184)
    box_radius = box_size * 0.086
    square_shadow_image = create_rounded_square_with_shadow(image, box_x, box_y, box_size, box_border_thickness, box_radius, box_color)
    img = cv2.cvtColor(square_shadow_image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    pil_image_with_text = add_text(font_dir, im_pil, text, box_x, box_y, box_size)
    return pil_image_with_text
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print('Usage: python corner_lozenges.py expects image name & text as argument')
        sys.exit(1)
    source_image_path = sys.argv[1]
    text = sys.argv[2]
    font_dir = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'fonts')
    pil_image_with_text = process(source_image_path, font_dir, text)
    pil_image_with_text.save('images\\output\\lozenges\\corner_lozenges.jpg')