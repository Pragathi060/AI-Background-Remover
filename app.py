from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)

# Create an uploads folder if not exists
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Ensure file is uploaded
        if "file" not in request.files:
            return "No file uploaded!", 400
        
        file = request.files["file"]
        if file.filename == "":
            return "No file selected!", 400

        # Read the uploaded image
        img_array = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

        if image is None:
            return "Invalid image file", 400

        # Convert to BGRA (Ensure transparency support)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # Remove white background (simple thresholding)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        b, g, r, _ = cv2.split(image)
        final_image = cv2.merge([b, g, r, alpha])

        # Save processed image
        output_path = os.path.join(OUTPUT_FOLDER, "output.png")
        cv2.imwrite(output_path, final_image)

        return send_file(output_path, as_attachment=True)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
