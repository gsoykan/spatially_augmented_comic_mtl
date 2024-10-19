import random
from typing import List, Optional

import cv2
import numpy as np
from matplotlib import pyplot as plt

import albumentations as A

BOX_COLOR = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128]
]
TEXT_COLOR = (255, 255, 255)  # White


# TODO: @gsoykan decide box and text color based on class
def visualize_bbox(img, bbox, class_name, color, thickness=2, scale_bb: bool = False):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    if scale_bb:
        img_h, img_w, img_c = img.shape
        x_min, x_max, y_min, y_max = x_min * img_w, x_max * img_w, y_min * img_h, y_max * img_h

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize_all_bboxes(image,
                         bboxes,
                         category_ids,
                         category_id_to_name,
                         scale_bb: bool = False,
                         save_path: Optional[str] = None):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name, color=BOX_COLOR[int(category_id)], thickness=2, scale_bb=scale_bb)
    plt.figure(figsize=(12, 12))
    plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path)

    plt.imshow(img)


def visualize_segmentation_polygon(img: np.ndarray,
                                   polygons: List,
                                   color=(0, 255, 0)) -> np.ndarray:
    """
     visualizes segmentation polygons especially useful for speech balloons
    @param img: np array of image
    @param polygons: list of polygons, polygons are nd array so reshaping happens here
    @param color: color of polygon border
    """
    for polygon in polygons:
        a_poly = np.array(polygon.reshape((-1, 2)), np.int32)
        cv2.polylines(img, [a_poly], True, color, 3)
    return img
