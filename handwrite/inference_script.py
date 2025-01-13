import sys
import os
from pathlib import Path
from tensorflow import keras
from PIL import Image
import numpy as np

# Constants
LABEL_MAP = {
    0: 48, 1: 49, 2: 50, 3: 51, 4: 52, 5: 53, 6: 54, 7: 55, 8: 56, 9: 57,
    10: 65, 11: 66, 12: 67, 13: 68, 14: 69, 15: 70, 16: 71, 17: 72, 18: 73,
    19: 74, 20: 75, 21: 76, 22: 77, 23: 78, 24: 79, 25: 80, 26: 81, 27: 82,
    28: 83, 29: 84, 30: 85, 31: 86, 32: 87, 33: 88, 34: 89, 35: 90
}

# Get the absolute path of the script's directory
SCRIPT_DIR = Path(__file__).resolve().parent

# Construct the model path relative to the script's directory
MODEL_PATH = SCRIPT_DIR / "../model/cnn_model.h5"

# Load the model once at the start
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Error: Model file not found at {MODEL_PATH}")
    return keras.models.load_model(MODEL_PATH)

# Load model at the beginning
model = load_model()

# Preprocess the image (Grayscale, Resize, and Normalize)
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to match model input
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        image_array = 1 - image_array  # Invert colors (if necessary for your model)
        return image_array.reshape((1, 28, 28))  # Reshape to the correct input shape
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Perform inference and return the corresponding label
def perform_inference(image_path):
    image_input = preprocess_image(image_path)
    if image_input is not None:
        predicted_class = np.argmax(model.predict(image_input, verbose=0), axis=1)
        return LABEL_MAP.get(predicted_class[0], "Unknown")  # Return label or 'Unknown' if class is missing
    return None

# Main logic for processing a directory of images
def process_directory(directory):
    directory_path = Path(directory)
    if not directory_path.is_dir():
        print(f"Error: {directory} is not a valid directory.")
        return

    # Find all image files in the directory
    image_files = [f for f in directory_path.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}]

    if not image_files:
        print("No image files found in the specified directory.")
        return

    # Process each image and output the result
    for image_file in image_files:
        predicted_label = perform_inference(image_file)
        if predicted_label:
            formatted_label = f"{predicted_label:03}"
            print(f"{formatted_label}, {image_file.resolve()}")

# Main entry point for the script
def main():
    if len(sys.argv) != 2:
        print("Usage: python inference_script.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    process_directory(directory_path)

if __name__ == '__main__':
    main()
