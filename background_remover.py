from rembg import remove
import cv2
import numpy as np

def remove_bg(input_path, output_path):
    # Read the input image
    image = cv2.imread(input_path)
    
    # Convert image to RGB (rembg requires RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Remove background
    output = remove(image_rgb)
    
    # Convert back to BGR (OpenCV format)
    output_bgr = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
    
    # Save the result
    cv2.imwrite(output_path, output_bgr)
    print(f"Processed image saved at: {output_path}")

# Example usage
# remove_bg("input.jpg", "output.png")
