from PIL import Image


def convert_to_monochrome(image_path):
    '''
    Convert a given image to monochrome (grayscale) to reduce size

    Parameters:
    image_path (str): The path to the image file that needs to be converted.

    Returns:
    Image: A monochrome version of the original image.
    '''
    image = Image.open(image_path)
    monochrome_image = image.convert("L")
    return monochrome_image
