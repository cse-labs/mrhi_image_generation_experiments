import cv2

def get_image_largest_rectangle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a binary mask
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (noise) and keep the larger ones
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

    # Find the largest bounding rectangle
    largest_rectangle = None
    largest_area = 0
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draws a green rectangle around the contour
        area = w * h
        if area > largest_area:
            largest_area = area
            largest_rectangle = (x, y, w, h)
    #print(f"largest rectangle is {largest_rectangle}")
    #print(f"image area is {largest_area}")
    
    #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draws a green rectangle around the contour
    
    return largest_rectangle