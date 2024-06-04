from flask import Flask, request, render_template
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from PIL import Image
import io

app = Flask(__name__, template_folder='template')

model = load_model('numtect/Model/Model.keras')

def preprocess_image(image):
    # Convert the image to grayscale and resize to the required size
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension if needed
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html')
    if request.method == "POST":
        if 'image' not in request.files:
            return "No file part"
        file = request.files['image']
        if file.filename == '':
            return "No selected file"
        if file:
            image = Image.open(io.BytesIO(file.read()))
            input_data = preprocess_image(image)
            prediction = model.predict(input_data)
            return render_template('index.html', prediction=prediction[0])
        
if __name__ == '__main__':
    app.run(debug=True)