import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans


def get_dominant_colors(image_path, num_colors=3, min_distance_to_white=30):
    '''"""
    This function extracts the dominant colors from an image.

    Parameters:
        image_path (str): The path to the image file.
        num_colors (int, optional): The number of dominant colors to extract. Defaults to 3.
        min_distance_to_white (int, optional): The minimum Euclidean distance from white a color must be to be considered dominant. Defaults to 30.

    Returns:
        list: A list of the dominant colors in the image, represented as [R, G, B] arrays. Colors close to white are filtered out.
    """'''
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    filtered_colors = []
    for color in dominant_colors:
        distance_to_white = np.linalg.norm(color - [255, 255, 255])
        if distance_to_white >= min_distance_to_white:
            filtered_colors.append(color)
    return filtered_colors


def wrap_text(text, font, max_width, draw):
    '''"""
    This function wraps the given text into multiple lines based on the maximum width provided. Currently it supports only a double digit number and text separated by space.

    Parameters:
        text (str): The text to be wrapped.
        font (ImageFont.FreeTypeFont): The font to be used for the text.
        max_width (int): The maximum width of the text line.
        draw (ImageDraw.Draw): The draw instance.

    Returns:
        list: A list of strings where each string is a line of text that fits within the max_width.
    """'''
    lines = []
    current_line = ""
    for word in text.split():
        test_line = current_line + " " + word if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        test_width = bbox[2] - bbox[0]
        if test_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines


def draw_wrapped_text_on_image(
    image_path, text, rectangle_coords, font_path, font_color=(255, 255, 255)
):
    '''"""
    This function draws wrapped text on an image within a specified rectangle.

    Args:
        image_path (str): The path to the image file.
        text (str): The text to be drawn on the image.
        rectangle_coords (tuple): A tuple of four integers specifying the rectangle within which the text should be drawn. The tuple should be in the format (x, y, width, height), where (x, y) is the top-left corner of the rectangle.
        font_path (str): The path to the .ttf font file to be used.
        font_color (tuple, optional): A tuple of three integers specifying the RGB color of the font. Defaults to white (255, 255, 255).

    Returns:
        None. The function directly modifies the image file specified by image_path.

    Raises:
        FileNotFoundError: If the specified image file or font file does not exist.
        ValueError: If rectangle_coords is not a tuple of four integers.
    """'''
    image = cv2.imread(image_path)
    pillow_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pillow_image)
    font_size = 1
    last_valid_font_size = None
    font = ImageFont.truetype(font_path, font_size)
    while True:
        font = ImageFont.truetype(font_path, font_size)
        text_lines = wrap_text(text, font, rectangle_coords[2], draw)
        total_height = len(text_lines) * font.getsize("Ag")[1]
        if total_height <= rectangle_coords[3] and font_size < 100:
            last_valid_font_size = font_size
            font_size += 1
        else:
            font_size = last_valid_font_size
            font = ImageFont.truetype(font_path, font_size)
            text_lines = wrap_text(text, font, rectangle_coords[2], draw)
            total_height = len(text_lines) * font.getsize("Ag")[1]
            break
    print(f"text_lines : {text_lines}")
    print(f"font_size : {font_size}")
    print(f"total_height : {total_height}")
    x_center = rectangle_coords[0] + rectangle_coords[2] // 2
    y_center = rectangle_coords[1] + rectangle_coords[3] // 2
    x_start = x_center - rectangle_coords[2] // 2
    y_start = y_center - total_height // 2
    print(f"x_center: {x_center}")
    print(f"y_center: {y_center}")
    print(f"x_start: {x_start}")
    print(f"y_start: {y_start}")
    y_draw = y_start
    for line in text_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        draw_x = x_start + (rectangle_coords[2] - line_width) // 2
        draw.text((draw_x, y_draw), line, fill=font_color, font=font)
        y_draw += bbox[3] - bbox[1]
    image = np.array(pillow_image)
    cv2.imwrite(image_path, image)


def rotate_rectange(
    rectangle_height,
    rectangle_width,
    demarcation_x,
    rotation_angle,
    result_image_path,
    image,
    left_rectangle_color,
    right_rectangle_color,
    left_part,
    right_part,
):
    '''"""
    This function rotates a rectangle in an image and saves the result.

    Parameters:
    rectangle_height (int): The height of the rectangle.
    rectangle_width (int): The width of the rectangle.
    demarcation_x (int): The x-coordinate of the demarcation line.
    rotation_angle (float): The angle of rotation in degrees.
    result_image_path (str): The path where the resulting image will be saved.
    image (array): The image array.
    left_rectangle_color (tuple): The color of the left part of the rectangle in BGR format.
    right_rectangle_color (tuple): The color of the right part of the rectangle in BGR format.
    left_part (array): The left part of the rectangle.
    right_part (array): The right part of the rectangle.

    Returns:
    None. The function saves the resulting image to the specified path.
    """'''
    rotation_center = (0, rectangle_height)
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_angle, 1.0)
    rotated_right_part = cv2.warpAffine(
        right_part, rotation_matrix, (rectangle_width - demarcation_x, rectangle_height)
    )
    mask = np.zeros(rotated_right_part.shape[:2], dtype=np.uint8)
    cv2.rectangle(
        mask,
        (0, 0),
        (rotated_right_part.shape[1], rotated_right_part.shape[0]),
        255,
        thickness=-1,
    )
    rotated_mask = cv2.warpAffine(
        mask, rotation_matrix, (rectangle_width - demarcation_x, rectangle_height)
    )
    right_part[rotated_mask == 255] = left_rectangle_color
    cv2.imwrite(result_image_path, image)


def create_bottom_lozenges(
    image, lozenges_rectangle_scale_factor, demarcation_factor, source_image_path
):
    '''"""
    This function creates bottom lozenges on an image by dividing the bottom part of the image into two rectangles and coloring them with the dominant colors of the image.

    Args:
        image (numpy.ndarray): The input image.
        lozenges_rectangle_scale_factor (float): The scale factor to calculate the height of the bottom lozenges rectangle.
        demarcation_factor (float): The factor to calculate the demarcation point that divides the bottom lozenges rectangle into two parts.
        source_image_path (str): The path of the source image to get the dominant colors.

    Returns:
        tuple: A tuple containing the modified image, the position and dimensions of the bottom lozenges rectangle, the demarcation point, the left and right parts of the rectangle, and their respective colors.
    """'''
    (image_height, image_width, _) = image.shape
    print(f"image (w,h): {(image_width, image_height)}")
    rectangle_height = int(image_height * lozenges_rectangle_scale_factor)
    rectangle_width = image_width
    print(f"bottom_rectangle (w,h): {(rectangle_width, rectangle_height)}")
    rectangle_x = 0
    rectangle_y = image_height - rectangle_height
    print(f"bottom_rectangle (x,y): {(rectangle_x, rectangle_y)}")
    demarcation_x = int(rectangle_width * demarcation_factor)
    print(f"demarcation_x : {demarcation_x}")
    left_part = image[
        rectangle_y : rectangle_y + rectangle_height, rectangle_x:demarcation_x
    ]
    right_part = image[
        rectangle_y : rectangle_y + rectangle_height,
        demarcation_x : rectangle_x + rectangle_width,
    ]
    dominant_colors = get_dominant_colors(source_image_path)
    left_rectangle_color = (
        int(dominant_colors[0][2]),
        int(dominant_colors[0][1]),
        int(dominant_colors[0][0]),
    )
    right_rectangle_color = (
        int(dominant_colors[1][2]),
        int(dominant_colors[1][1]),
        int(dominant_colors[1][0]),
    )
    left_part[:, :] = left_rectangle_color
    right_part[:, :] = right_rectangle_color
    return (
        image,
        rectangle_x,
        rectangle_y,
        rectangle_width,
        rectangle_height,
        demarcation_x,
        left_part,
        right_part,
        left_rectangle_color,
        right_rectangle_color,
    )


def process_bottom_lozenges(
    source_image_path, left_text, right_text, font_file_path, result_image_path
):
    '''"""
    This function processes the bottom lozenges of an image. It reads an image from a source path, creates two lozenge-shaped rectangles at the bottom of the image, rotates the right rectangle, and writes text on both rectangles.

    Args:
        source_image_path (str): The path to the source image.
        left_text (str): The text to be written on the left rectangle.
        right_text (str): The text to be written on the right rectangle.
        font_file_path (str): The path to the font file to be used for the text.
        result_image_path (str): The path where the resulting image will be saved.

    Returns:
        None. The resulting image is saved at the path specified by result_image_path.
    """'''
    lozenges_rectangle_scale_factor = 0.12
    rotation_angle = 75
    demarcation_factor = 0.75
    image = cv2.imread(source_image_path)
    (
        image,
        rectangle_x,
        rectangle_y,
        rectangle_width,
        rectangle_height,
        demarcation_x,
        left_part,
        right_part,
        left_rectangle_color,
        right_rectangle_color,
    ) = create_bottom_lozenges(
        image, lozenges_rectangle_scale_factor, demarcation_factor, source_image_path
    )
    rotate_rectange(
        rectangle_height,
        rectangle_width,
        demarcation_x,
        rotation_angle,
        result_image_path,
        image,
        left_rectangle_color,
        right_rectangle_color,
        left_part,
        right_part,
    )
    left_rectangle_coordinates = (
        rectangle_x,
        rectangle_y,
        left_part.shape[1],
        left_part.shape[0],
    )
    print(
        f"left_rectangle_coordinates: xy ({(rectangle_x, rectangle_y)}), width : {left_part.shape[1]} , height: {left_part.shape[0]}"
    )
    draw_wrapped_text_on_image(
        result_image_path, left_text, left_rectangle_coordinates, font_file_path
    )
    right_rectangle_coordinates = (
        demarcation_x,
        rectangle_y,
        right_part.shape[1],
        right_part.shape[0],
    )
    print(
        f"right_rectangle_coordinates: xy ({(demarcation_x, rectangle_y)}), width : {right_part.shape[1]} , height: {right_part.shape[0]}"
    )
    draw_wrapped_text_on_image(
        result_image_path, right_text, right_rectangle_coordinates, font_file_path
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python bottom_lozenges.py expects image name , left text and right text as arguments"
        )
        sys.exit(1)
    source_image_path = sys.argv[1]
    left_text = sys.argv[2]
    right_text = sys.argv[3]
    font_file_path = "fonts\\OpenSans-ExtraBold.ttf"
    result_image_path = "images\\output\\lozenges\\bottom_lozenges.jpg"
    process_bottom_lozenges(
        source_image_path, left_text, right_text, font_file_path, result_image_path
    )
