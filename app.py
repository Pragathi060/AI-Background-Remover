from flask import Flask, render_template, request, send_file
from rembg import remove
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            output_path = os.path.join(RESULT_FOLDER, f"removed_{file.filename}")

            file.save(input_path)
            
            # Read image
            image = cv2.imread(input_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output = remove(image_rgb)
            output_bgr = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)

            # Save processed image
            cv2.imwrite(output_path, output_bgr)

            return send_file(output_path, as_attachment=True)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
