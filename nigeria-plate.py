import cv2
import pytesseract

# Set the path for Tesseract if it's not in your PATH environment variable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_number_plate(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply edge detection
    edged_image = cv2.Canny(blurred_image, 100, 200)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edged_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    number_plate_contour = None

    for contour in contours:
        # Approximate the contour and check if it is a rectangle
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

        if len(approx) == 4:  # Found a rectangle
            number_plate_contour = approx
            break

    if number_plate_contour is not None:
        # Create a mask for the number plate
        mask = cv2.zeros(gray_image.shape, dtype="uint8")
        cv2.drawContours(mask, [number_plate_contour], -1, 255, -1)

        # Extract the number plate region
        (x, y, w, h) = cv2.boundingRect(number_plate_contour)
        number_plate_image = gray_image[y:y + h, x:x + w]

        # Use Tesseract to extract text
        custom_config = r'--oem 3 --psm 8'
        number_plate_text = pytesseract.image_to_string(number_plate_image, config=custom_config)

        return number_plate_text.strip()
    else:
        return "No number plate detected"

# Example usage
if __name__ == "__main__":
    image_path = 'path_to_your_number_plate_image.jpg'
    detected_plate = detect_number_plate(image_path)
    print("Detected Number Plate:", detected_plate)
