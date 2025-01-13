from pathlib import Path

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class CNNModel:
    def __init__(self, input_shape, num_classes):
        """
        Initialize the CNN model.

        Parameters
        ----------
        input_shape : tuple
            Shape of input images (height, width, channels)
        num_classes : int
            Number of classification categories
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        self.history = None

    def _build_model(self):
        """
        Build the CNN model architecture.

        Returns
        -------
        tf.keras.Model
            Constructed CNN model with defined layers
        """
        model = tf.keras.models.Sequential([
            # Convolutional layers
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation='relu',
                input_shape=self.input_shape
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation='relu'
            ),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            # Fully connected layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(units=self.num_classes, activation='softmax')
        ])
        return model

    def compile(self, learning_rate=0.001):
        """
        Compile the model with specified optimizer and loss function.

        Parameters
        ----------
        learning_rate : float, optional
            Learning rate for the Adam optimizer (default=0.001)
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(
            self,
            train_x,
            train_y,
            valid_x,
            valid_y,
            epochs=10,
            batch_size=32
    ):
        """
        Train the model on the provided dataset.

        Parameters
        ----------
        train_x : np.ndarray
            Training data features
        train_y : np.ndarray
            Training data labels
        valid_x : np.ndarray
            Validation data features
        valid_y : np.ndarray
            Validation data labels
        epochs : int, optional
            Number of training epochs (default=10)
        batch_size : int, optional
            Batch size for training (default=32)
        """
        self.history = self.model.fit(
            train_x,
            train_y,
            validation_data=(valid_x, valid_y),
            epochs=epochs,
            batch_size=batch_size
        )

    def evaluate(self, data, labels):
        """
        Evaluate the model on a dataset.

        Parameters
        ----------
        data : np.ndarray
            Input data for evaluation
        labels : np.ndarray
            True labels for the input data

        Returns
        -------
        Tuple[float, float]
            Model loss and accuracy
        """
        return self.model.evaluate(data, labels, verbose=0)

    def plot_accuracy_loss(self):
        """
        Plot training and validation metrics over epochs.

        Displays two plots:
        1. Training and validation accuracy
        2. Training and validation loss
        """
        if self.history is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")

        history = self.history.history
        epochs = range(len(history['accuracy']))

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot accuracy
        ax1.plot(epochs, history['accuracy'], 'r', label='Training')
        ax1.plot(epochs, history['val_accuracy'], 'b', label='Validation')
        ax1.set_title('Training and Validation Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)

        # Plot loss
        ax2.plot(epochs, history['loss'], 'r', label='Training')
        ax2.plot(epochs, history['val_loss'], 'b', label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(
            self,
            true_labels,
            predicted_labels,
            label_dict
    ):
        """
        Plot confusion matrix for model predictions.

        Parameters
        ----------
        true_labels : np.ndarray
            Ground truth labels
        predicted_labels : np.ndarray
            Model predictions
        label_dict : Dict[int, str]
            Mapping of label indices to class names
        """
        conf_matrix = confusion_matrix(true_labels, predicted_labels)

        plt.figure(figsize=(20, 12))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False
        )

        # Customize plot
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")

        # Set ticks and labels
        ticks = np.arange(len(label_dict)) + 0.5
        labels = [label_dict[i] for i in range(len(label_dict))]
        plt.xticks(ticks, labels, rotation=90)
        plt.yticks(ticks, labels, rotation=0)

        plt.tight_layout()
        plt.show()

    def predict(self, data):
        """
        Make predictions on input data.

        Parameters
        ----------
        data : np.ndarray
            Input data for prediction

        Returns
        -------
        np.ndarray
            Model predictions
        """
        return self.model.predict(data)

    def save_model(self, model_name="cnn_model.h5"):
        """
        Save the trained model to disk.

        Parameters
        ----------
        model_name : str, optional
            Name of the saved model file (default="cnn_model.h5")
        """
        model_folder = Path(__file__).resolve().parent.parent / "model"
        model_folder.mkdir(parents=True, exist_ok=True)

        model_path = model_folder / model_name
        self.model.save(str(model_path))
        print(f"Model saved to {model_path}")