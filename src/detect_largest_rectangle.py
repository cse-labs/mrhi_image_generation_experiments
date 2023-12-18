import cv2
import os

# Input and output folder paths
input_folder = "images/box"  # Replace with the path to your input image folder
output_folder = "images/output/box_result"  # Replace with the path to your output image folder

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get a list of image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

# Process each image
for image_file in image_files:

    # Load the image
    img = cv2.imread(os.path.join(input_folder, image_file))

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply canny edge detection
    edged = cv2.Canny(gray, 50, 150)

    threshold = cv2.adaptiveThreshold(edged, 255, 1, 1, 11, 2)

    # Find the contours
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize variables to keep track of the largest rectangle
    largest_area = 0
    largest_rectangle = None 

    # Process the contours and find the largest rectangle
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > largest_area:
            largest_area = area
            x, y, w, h = cv2.boundingRect(cnt)
            largest_rectangle = (x, y, x + w, y + h)

    # Draw and save the largest rectangle on the original image
    if largest_rectangle is not None:
        x1, y1, x2, y2 = largest_rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the output image
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, img)


cv2.destroyAllWindows()