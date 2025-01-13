import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import pairwise_distances

def preprocess_dataset(train_dataset):
    """Preprocesses the dataset by checking for missing values, reshaping pixel data, and scaling it."""
    print("Missing values in train dataset:", train_dataset.isna().any().any())
    train_dataset = train_dataset.dropna()
    print('Training Data Shape after dropping missing values:', train_dataset.shape)

    # Separate class labels and pixel values
    train_pixels = train_dataset.drop('label', axis=1)
    train_labels = train_dataset['label'].astype('int').to_numpy()
    train_pixels = np.reshape(train_pixels.values, (train_pixels.shape[0], 28, 28)) / 255

    print("Train pixels shape:", train_pixels.shape)
    num_classes = len(np.unique(train_labels))
    print("Number of classes:", num_classes)

    return train_pixels, train_labels, num_classes

def display_sample_images(pixels, labels, label_dict):
    """
    Display a 5x5 grid of sample images from the dataset with their corresponding labels.

    Parameters
    ----------
    pixels : array-like
        Array of image data with shape (n_samples, height, width)
    labels : array-like
        Array of label indices corresponding to the images
    label_dict : dict
        Dictionary mapping label indices to their text descriptions

    Returns
    -------
    None
        Displays the plot using matplotlib
    """
    plt.figure(figsize=(10, 8))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(pixels[i], cmap='gray')
        plt.title(f'Label: {label_dict[labels[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def rotate_image(image):
    """
    Rotate and flip an image by performing horizontal flip and 90-degree rotation.

    Parameters
    ----------
    image : numpy.ndarray
        Input image array with shape (channels, height, width)

    Returns
    -------
    numpy.ndarray
        Transformed image array with same shape as input

    Notes
    -----
    The transformation consists of:
    1. Horizontal flip along the width axis
    2. 90-degree rotation in the height-width plane
    """
    image = np.flip(image, axis=2)  # Flip horizontally
    image = np.rot90(image, axes=(1, 2))  # Rotate 90 degrees
    return image



def visualize_label_distribution(labels, label_dict):
    """
    Create a horizontal bar chart showing the distribution of labels in the dataset.

    Parameters
    ----------
    labels : array-like
        Array of label indices representing the class of each sample
    label_dict : dict
        Dictionary mapping label indices to their text descriptions

    Returns
    -------
    None
        Displays the plot using matplotlib
    """
    # Calculate label frequencies
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Create visualization
    plt.figure(figsize=(10, 12))
    plt.barh(unique_labels, label_counts, color=sns.color_palette('viridis', len(unique_labels)))
    plt.title('Distribution of Labels')
    plt.ylabel('Labels', fontsize=12)
    plt.xlabel('Count', fontsize=12)
    plt.grid(True)

    # Set custom labels
    names_labels = [label_dict[index] for index in unique_labels]
    plt.yticks(unique_labels, names_labels)
    plt.show()


def calculate_mean_images(images, labels, num_classes):
    """
    Calculate the mean image for each class in the dataset.

    Parameters
    ----------
    images : numpy.ndarray
        Array of image data with shape (n_samples, height, width)
    labels : numpy.ndarray
        Array of label indices with shape (n_samples,)
    num_classes : int
        Number of unique classes in the dataset

    Returns
    -------
    numpy.ndarray
        Array of mean images with shape (num_classes, height, width)
        Each element represents the average image for a class
    """
    mean_images = np.zeros((num_classes, 28, 28), dtype=np.float32)
    for label in range(num_classes):
        sample_image_idx = np.where(labels == label)[0]
        mean_images[label] = np.mean(images[sample_image_idx], axis=0)
    return mean_images


def plot_mean_images(mean_images, label_dict, num_classes):
    """
    Create a grid plot of mean images for each class with their corresponding labels.

    Parameters
    ----------
    mean_images : numpy.ndarray
        Array of mean images with shape (num_classes, height, width)
    label_dict : dict
        Dictionary mapping class indices to their text descriptions
    num_classes : int
        Number of unique classes to plot

    Returns
    -------
    None
        Displays the plot using matplotlib
    """
    plt.figure(figsize=(15, 48))
    for i in range(num_classes):
        plt.subplot(16, 4, i + 1)
        plt.title(f'Label: {label_dict[i]}')
        plt.imshow(mean_images[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def calculate_distances_from_mean(images, labels, mean_images, num_classes):
    """
    Calculate Euclidean distances between each image and its class mean.

    Parameters
    ----------
    images : numpy.ndarray
        Array of image data with shape (n_samples, height, width)
    labels : numpy.ndarray
        Array of label indices with shape (n_samples,)
    mean_images : numpy.ndarray
        Array of mean images for each class with shape (num_classes, height, width)
    num_classes : int
        Number of unique classes in the dataset

    Returns
    -------
    list
        List of numpy arrays containing distances for each class.
        Each array contains the Euclidean distances between the class images
        and their respective mean image.
    """
    distances = []
    for label in range(num_classes):
        sample_image_idx = np.where(labels == label)[0]
        class_images = images[sample_image_idx]
        class_mean = mean_images[label].reshape(1, -1)
        class_distances = pairwise_distances(class_images.reshape(class_images.shape[0], -1),
                                             class_mean,
                                             metric='euclidean')
        distances.append(class_distances.flatten())
    return distances


def plot_distances(distances, label_dict):
    """
    Create a boxplot showing the distribution of Euclidean distances for each class.

    Parameters
    ----------
    distances : list
        List of numpy arrays containing distances for each class,
        as returned by calculate_distances_from_mean()
    label_dict : dict
        Dictionary mapping class indices to their text descriptions

    Returns
    -------
    None
        Displays the plot using matplotlib
    """
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=distances)
    plt.title('Euclidean Distance of Training Images from Class Mean')
    plt.xlabel('Class Label')
    plt.ylabel('Euclidean Distance')
    plt.xticks(ticks=range(len(label_dict)),
               labels=[label_dict[i] for i in range(len(label_dict))])
    plt.grid(True)
    plt.show()


def compute_bounds(distances):
    """
    Compute the upper and lower bounds for outlier detection using the boxplot method.

    Parameters
    ----------
    distances : list
        List of numpy arrays containing distances for each class,
        as returned by calculate_distances_from_mean()

    Returns
    -------
    list of tuple
        List of (lower_bound, upper_bound) tuples for each class.
        Bounds are calculated using the interquartile range (IQR) method:
        - lower_bound = Q1 - 1.5 * IQR
        - upper_bound = Q3 + 1.5 * IQR

    Notes
    -----
    The function uses the standard boxplot outlier detection method:
    1. Calculates Q1 (25th percentile) and Q3 (75th percentile)
    2. Computes IQR = Q3 - Q1
    3. Sets bounds at Q1 - 1.5*IQR and Q3 + 1.5*IQR
    """
    bounds = []
    for d in distances:
        q1 = np.percentile(d, 25)
        q3 = np.percentile(d, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        bounds.append((lower_bound, upper_bound))
    return bounds


def filter_outliers(images, labels, distances, bounds):
    """
    Filter out outlier images based on their distances from class means.

    Parameters
    ----------
    images : numpy.ndarray
        Array of image data with shape (n_samples, height, width)
    labels : numpy.ndarray
        Array of label indices with shape (n_samples,)
    distances : list
        List of numpy arrays containing distances for each class
    bounds : list of tuple
        List of (lower_bound, upper_bound) tuples for each class,
        as returned by compute_bounds()

    Returns
    -------
    tuple
        (filtered_images, filtered_labels) where:
        - filtered_images: numpy.ndarray of images with outliers removed
        - filtered_labels: numpy.ndarray of corresponding labels

    Notes
    -----
    For each class:
    1. Identifies images within the computed bounds
    2. Keeps only those images and their labels
    3. Concatenates filtered data from all classes
    """
    filtered_images = []
    filtered_labels = []
    for label in range(len(bounds)):
        sample_image_idx = np.where(labels == label)[0]
        lower_bound, upper_bound = bounds[label]
        inlier_idx = np.where(
            (distances[label] >= lower_bound) &
            (distances[label] <= upper_bound)
        )[0]
        filtered_images.append(images[sample_image_idx][inlier_idx])
        filtered_labels.append(labels[sample_image_idx][inlier_idx])

    return np.concatenate(filtered_images, axis=0), np.concatenate(filtered_labels, axis=0)





