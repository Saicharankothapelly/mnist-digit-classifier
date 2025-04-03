from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load the trained MNIST model
model = tf.keras.models.load_model("cnn_model.h5", compile = False)

def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to MNIST format
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    return image

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(io.BytesIO(file.read()))
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            digit = np.argmax(prediction)
            return render_template('index.html', prediction=digit)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
