from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from handwrite.data_processing import (
    display_sample_images, rotate_image, visualize_label_distribution,
    calculate_mean_images, plot_mean_images, calculate_distances_from_mean,
    plot_distances, compute_bounds, filter_outliers, preprocess_dataset
)
from handwrite.model import CNNModel


ROOT_DIR = Path(__file__).parent.parent

TRAIN = ROOT_DIR / "data" / "emnist-balanced-train.csv"
LABEL = ROOT_DIR / "data" / "emnist-balanced-mapping.txt"

def main():
    """Main function to load data, preprocess it, and train the model."""
    # Load and preprocess dataset
    train_dataset = pd.read_csv(TRAIN)
    # Create column names and assign to the DataFrame
    columns = ['label'] + list(range(784))
    train_dataset.columns = columns

    # Display the first few rows and shape of the training dataset
    print(train_dataset.head(10))
    print('Training Data Shape:', train_dataset.shape)

    # Preprocess dataset
    train_pixels, train_labels, num_classes = preprocess_dataset(train_dataset)

    # Load label dictionary
    label_dictionary = {}
    with open(LABEL, "r") as file:
        for line in file:
            index, label = map(int, line.strip().split())
            label_dictionary[index] = chr(label)

    print(label_dictionary)

    # Display sample images from the training set
    display_sample_images(train_pixels, train_labels, label_dictionary)

    # Rotate the training and test images
    train_pixels = rotate_image(train_pixels)

    # Display sample images from the training set after rotation
    display_sample_images(train_pixels, train_labels, label_dictionary)

    # Visualize the distribution of labels
    visualize_label_distribution(train_labels, label_dictionary)

    # Calculate and plot mean images
    mean_images = calculate_mean_images(train_pixels, train_labels, num_classes)
    plot_mean_images(mean_images, label_dictionary, num_classes)

    # Calculate distances from mean and plot distances
    distances = calculate_distances_from_mean(train_pixels, train_labels, mean_images, num_classes)
    plot_distances(distances, label_dictionary)

    # Filter out outliers and update the dataset
    bounds = compute_bounds(distances)
    train_pixels_filtered, train_labels_filtered = filter_outliers(
        train_pixels, train_labels, distances, bounds
    )

    # Update the training data
    train_pixels = train_pixels_filtered
    train_labels = train_labels_filtered

    # Recalculate and visualize new distances
    distances = calculate_distances_from_mean(train_pixels, train_labels, mean_images, num_classes)
    plot_distances(distances, label_dictionary)

    # Split the data into training, validation and test sets
    train_pixels, valid_pixels, train_labels, valid_labels = train_test_split(train_pixels, train_labels, test_size=0.3)
    valid_pixels, test_pixels, valid_labels, test_labels = train_test_split(valid_pixels, valid_labels, test_size=0.2)

    # Print shapes of the resulting datasets
    print('Training data shape:', train_pixels.shape)
    print('Training labels shape:', train_labels.shape)
    print('Validation data shape:', valid_pixels.shape)
    print('Validation labels shape:', valid_labels.shape)

    # Initialize the CNN model
    model = CNNModel(input_shape=(28, 28, 1), num_classes=num_classes)

    # Compile the model
    model.compile()

    # Train the model
    model.train(train_pixels, train_labels, valid_pixels, valid_labels, epochs=10)

    # Save the trained model
    model.save_model()

    # Plot accuracy and loss curves
    model.plot_accuracy_loss()

    # Print model metrics on validation and test datasets
    print("Validation set metrics:")
    val_loss, val_accuracy = model.evaluate(valid_pixels, valid_labels)
    print(f"Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    print("Test set metrics:")
    test_loss, test_accuracy = model.evaluate(test_pixels, test_labels)
    print(f"Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

    # Make predictions on the test data
    test_predictions = model.predict(test_pixels)
    test_pred_labels = np.argmax(test_predictions, axis=1)

    # Plot confusion matrix for test data
    print("Confusion Matrix (Test Data):")
    model.plot_confusion_matrix(test_labels, test_pred_labels, label_dictionary)

if __name__ == "__main__":
    main()
