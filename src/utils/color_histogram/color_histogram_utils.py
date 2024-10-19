import cv2
import numpy as np
from matplotlib import pyplot as plt


def example_grayscale():
    img_path = '../../../data/icf_dass_faces/00008/0000005.jpg'
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    plt.plot(histogram, color='k')
    plt.show()


def example_colored():
    img_path = '../../../data/icf_dass_faces/00008/0000005.jpg'
    image = cv2.imread(img_path)
    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.show()


def build_histogram(image, bins=256):
    """
    manually building histograms...
    Args:
        image (): cv2 BGR image, np.ndarray...
        bins ():
    """
    # convert from BGR to RGB
    rgb_image = np.flip(image, 2)
    # show the image
    plt.imshow(rgb_image)
    # convert to a vector
    image_vector = rgb_image.reshape(1, -1, 3)
    # break into given number of bins
    div = 256 / bins
    bins_vector = (image_vector / div).astype(int)
    # get the red, green, and blue channels
    red = bins_vector[0, :, 0]
    green = bins_vector[0, :, 1]
    blue = bins_vector[0, :, 2]
    # build the histograms and display
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    axs[0].hist(red, bins=bins, color='r')
    axs[1].hist(green, bins=bins, color='g')
    axs[2].hist(blue, bins=bins, color='b')
    plt.show()


def get_histogram_vector(image, bins=32):
    """

    Args:
        image (): cv2 BGR image, np.ndarray...
        bins ():

    Returns:

    """
    red = cv2.calcHist(
        [image], [2], None, [bins], [0, 256]
    )
    green = cv2.calcHist(
        [image], [1], None, [bins], [0, 256]
    )
    blue = cv2.calcHist(
        [image], [0], None, [bins], [0, 256]
    )
    vector = np.concatenate([red, green, blue], axis=0)
    vector = vector.reshape(-1)
    return vector


def get_histogram_vector_batched(images, bins=32):
    """

    Args:
        images (): B, H, W, C
        bins ():

    Returns:

    """
    vectors = []
    for image in images:
        vector = get_histogram_vector(image, bins)
        vectors.append(vector)
    return np.array(vectors)


def euclidean(a, b):
    return np.linalg.norm(a - b)


def cosine(a, b):
    """
    Using our cosine function we can calculate the similarity
    which varies from 0 (highly dissimilar) to 1 (identical).
    Args:
        a ():
        b ():

    Returns:

    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if __name__ == '__main__':
    # example_grayscale()
    # example_colored()
    img_path = '../../../data/icf_dass_faces/00008/0000005.jpg'
    img_path_same = '../../../data/icf_dass_faces/00008/0000011.jpg'
    img_path_different = '../../../data/icf_dass_faces/00048/0000016.jpg'
    # reads in BGR
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image_same = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image_different = cv2.imread(img_path_different, cv2.IMREAD_COLOR)
    # build_histogram(image, bins=64)

    vec = get_histogram_vector(image)
    vec_same = get_histogram_vector(image_same)
    vec_different = get_histogram_vector(image_different)

    cosine_distance_same = cosine(vec, vec_same)
    cosine_distance_different = cosine(vec, vec_different)

    print(cosine_distance_same, 'same img distance', cosine_distance_different, 'diff. img distance')
