from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import os
from rembg import remove  # Ensure you have installed rembg: `pip install rembg`

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        output_path = os.path.join(OUTPUT_FOLDER, f"no_bg_{file.filename}")

        file.save(filepath)  # Save original file

        # Process image to remove background
        try:
            input_image = cv2.imread(filepath)  # Read image
            if input_image is None:
                return "Error: Could not read the image", 400

            input_bytes = cv2.imencode(".png", input_image)[1].tobytes()
            output_bytes = remove(input_bytes)  # Remove background
            output_image = cv2.imdecode(np.frombuffer(output_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

            # Save processed image
            cv2.imwrite(output_path, output_image)

            return render_template("index.html", processed_image=output_path)

        except Exception as e:
            return f"Error processing image: {str(e)}", 500

    return render_template("index.html", processed_image=None)

if __name__ == "__main__":
    app.run(debug=True)
