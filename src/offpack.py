from bottomlozenges import process_bottom_lozenges

def generate_mrhi_offpack(source_image_path, left_text, right_text, font_file_path, result_image_path):
    '''"""
This function generates a modified image with text overlays on the bottom lozenges.

Args:
    source_image_path (str): The path to the source image.
    left_text (str): The text to be overlaid on the left bottom lozenge.
    right_text (str): The text to be overlaid on the right bottom lozenge.
    font_file_path (str): The path to the font file to be used for the text overlays.
    result_image_path (str): The path where the resulting image will be saved.

Returns:
    None. The resulting image is saved at the location specified by result_image_path.
"""'''
    process_bottom_lozenges(str(source_image_path), left_text, right_text, font_file_path, str(result_image_path))