import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

from utils.constants import natural_images_dir, figures_dir
from utils.general import get_doc_filenames


def entropy(array: np.ndarray) -> float:
    if array.sum():
        probs = array / array.sum()
        ent = -np.nansum(probs * np.log(probs))
    else:
        ent = 0.0
    return ent


def image_hist_entropy(image: np.ndarray) -> float:
    hist_image = np.bincount(image.ravel(), minlength=256)
    return entropy(hist_image)


def calculate_entropy_threshold(
    images_path: str, figures_path: str, plot_stats: bool = False
) -> float:

    natural_images_files = get_doc_filenames(images_path, extension=".ppm")
    figures_files = get_doc_filenames(
        figures_dir, extension=".ppm"
    ) + get_doc_filenames(figures_path, extension=".pbm")

    images = [
        cv.imread(image_file, cv.IMREAD_GRAYSCALE)
        for image_file in natural_images_files
    ]
    figures = [
        cv.imread(figure_file, cv.IMREAD_GRAYSCALE) for figure_file in figures_files
    ]

    image_entropies = [image_hist_entropy(image) for image in images]
    figure_entropies = [image_hist_entropy(figure) for figure in figures]

    y_true = np.concatenate(
        (np.ones(len(image_entropies)), np.zeros(len(figure_entropies)))
    ).astype(int)
    x = np.array(image_entropies + figure_entropies)
    thresholds = np.linspace(min(figure_entropies), max(image_entropies), 1000)
    f_scores = np.zeros(thresholds.size)
    precisions = np.zeros(thresholds.size)
    recalls = np.zeros(thresholds.size)
    for i, threshold in enumerate(thresholds):
        y_pred = (x > threshold).astype(int)
        precisions[i], recalls[i], f_scores[i], _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary"
        )

    if plot_stats:
        plt.plot(thresholds, f_scores)
        plt.plot(thresholds, precisions)
        plt.plot(thresholds, recalls)
        plt.legend(["F1", "precision", "recall"])
        plt.show()

    opt_threshold = thresholds[f_scores.argmax()]
    return opt_threshold


if __name__ == "__main__":
    print(
        "Optimal threshold: {}".format(
            calculate_entropy_threshold(natural_images_dir, figures_dir)
        )
    )
