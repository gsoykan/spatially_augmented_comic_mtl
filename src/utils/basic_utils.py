import itertools
import json
import os
import random
from enum import Enum
from typing import List, Optional, Callable, Tuple, Dict

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from numba import jit
from tqdm import tqdm
from shapely.geometry import box

from src.datamodules.components.batch_element.batch_element import BatchElement
from src.datamodules.components.batch_element.panel_in_batch_element import PanelInBatchElement
from src.utils.pickle_helper import PickleHelper


def search_files(extension='.ttf',
                 folder='H:\\',
                 filename_condition: Optional[Callable[[str], bool]] = None,
                 limit: Optional[int] = None,
                 enable_tqdm: bool = False):
    if limit:
        files = []
        for r, d, f in (tqdm(os.walk(folder), desc='walking in folders...') if enable_tqdm else os.walk(folder)):
            if limit is not None and len(files) >= limit:
                break
            for file in f:
                if file.endswith(extension):
                    filename = r + "/" + file
                    if filename_condition is not None:
                        if filename_condition(filename):
                            files.append(filename)
                    else:
                        files.append(filename)
        return files
    else:
        # TODO: @gsoykan - add filename condition to alt-search files 2 ... it causes bugs
        return alternative_search_files_2(extension, folder)


def scandir_walk(top):
    for entry in os.scandir(top):
        if entry.is_dir(follow_symlinks=False):
            yield from scandir_walk(entry.path)
        else:
            yield entry.path


# this is way faster...
def alternative_search_files_2(extension=".ttf", folder="H:\\") -> List[str]:
    return [
        file for file in tqdm(scandir_walk(folder), desc="walking in folder..")
        if file.endswith(extension)
    ]


@jit
def extract_batch_elements_from_json_files(data_dir,
                                           json_files,
                                           limit_size: Optional[int] = None,
                                           max_context_panel_count: Optional[int] = None) -> List[BatchElement]:
    batch_elements = []
    for json_file_name in json_files:
        comic_no = int(json_file_name.split('.')[0].split('_')[0])
        json_file = open(os.path.join(data_dir, json_file_name))
        data = json.load(json_file)
        series_batch_elements = data['batch_elements']
        for b_e in series_batch_elements:
            batch_elements.append(BatchElement.init_from_custom_json(custom_json=b_e,
                                                                     book_id=str(comic_no),
                                                                     max_context_panel_count=max_context_panel_count))
        json_file.close()
    if limit_size:
        batch_elements = random.sample(batch_elements, limit_size)
    return batch_elements


def extract_ordered_text_from_panel(panel: PanelInBatchElement,
                                    panel_transformation: Optional[
                                        Callable[[PanelInBatchElement], PanelInBatchElement]] = None,
                                    text_transformation: Optional[Callable[[str], Tuple[List[int], int]]] = None) -> \
        Tuple[List[str], List[int]]:
    if panel_transformation:
        panel = panel_transformation(panel)
    sorted_texts = list(map(lambda x: text_transformation(x.text) if text_transformation else x.text,
                            panel.texts))
    # TODO: @gsoykan this can be faster and combined with upper lambda
    texts = list(map(lambda x: x[0], sorted_texts))
    text_lengths = list(map(lambda x: x[1], sorted_texts))
    return texts, text_lengths


def extract_ordered_text_from_multiple_panels(panels: List[PanelInBatchElement],
                                              panel_transformation: Optional[
                                                  Callable[[PanelInBatchElement], PanelInBatchElement]] = None,
                                              text_transformation: Optional[
                                                  Callable[[str], Tuple[List[int], int]]] = None) -> \
        Tuple[List[str], List[int]]:
    results = list(
        map(lambda p: extract_ordered_text_from_panel(p,
                                                      text_transformation=text_transformation,
                                                      panel_transformation=panel_transformation),
            panels))
    result_texts = list(map(lambda x: x[0], results))
    result_text_lengths = list(map(lambda x: x[1], results))
    return result_texts, result_text_lengths


def extract_image_from_panel(panel: PanelInBatchElement,
                             panel_dir: str,
                             vision_transform: Optional[A.Compose],
                             mask_speech_bubbles: bool = True):
    img_path = panel.generate_img_path(panel_dir)
    try:
        if mask_speech_bubbles:
            img = read_or_get_image_masked(img_path,
                                           masks=panel.get_speech_bubble_bounding_boxes(),
                                           return_pil_image=False)
        else:
            img = cv2.imread(img_path)  # # H x W x C
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert img is not None
    except Exception as e:
        print('error while reading the image: ', img_path)
        raise e
    if vision_transform:
        img = vision_transform(image=img)['image']
    return img


def load_image_embedding_from_panel(panel: PanelInBatchElement,
                                    embedding_dir: str) -> np.ndarray:
    return PickleHelper.load_object(panel.generate_img_embedding_path(embedding_dir, extension='pkl'))


def read_or_get_image(img,
                      read_rgb: bool = False):
    img_str = ""
    if not isinstance(img, (np.ndarray, str)):
        raise AssertionError('Images must be strings or numpy arrays')

    if isinstance(img, str):
        img_str = img
        img = cv2.imread(img)

    if img is None:
        raise AssertionError('Image could not be read: ' + img_str)

    if read_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def read_or_get_image_masked(img,
                             read_rgb: bool = True,
                             masks: List = [],
                             mask_fill_value=0,
                             return_pil_image: bool = True):
    read_image = read_or_get_image(img, read_rgb=read_rgb)
    for bb in masks:
        read_image[int(bb[1]): int(bb[3]), int(bb[0]):int(bb[2])] = mask_fill_value
    return Image.fromarray(read_image) if return_pil_image else read_image


# source: https://stackoverflow.com/a/42728126/8265079 | https://stackoverflow.com/questions/42727586/nest-level-of-a-list
def nest_level(obj):
    # Not a list? So the nest level will always be 0:
    if type(obj) != list:
        return 0
    # Now we're dealing only with list objects:
    max_level = 0
    for item in obj:
        # Getting recursively the level for each item in the list,
        # then updating the max found level:
        max_level = max(max_level, nest_level(item))
    # Adding 1, because 'obj' is a list (here is the recursion magic):
    return max_level + 1


def read_ad_pages(ad_page_path: str = '../../data/ad_pages_original.txt'):
    with open(ad_page_path) as f:
        lines = f.readlines()
    ad_pages = []
    for line in lines:
        comic_no, page_no = line.strip().split('---')
        ad_pages.append((int(comic_no), int(page_no)))
    return ad_pages


def flatten_list(l: List):
    return list(itertools.chain(*l))


def all_equal(lst):
    return all(element == lst[0] for element in lst)


def map_values(obj: Dict, fn):
    return dict((k, fn(v)) for k, v in obj.items())


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def inverse_normalize(tensor,
                      mean=(0.485, 0.456, 0.406),
                      std=(0.229, 0.224, 0.225),
                      in_place: bool = True):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    if not in_place:
        tensor = tensor.clone()
    tensor.mul_(std).add_(mean)
    return tensor


class OCRFileKey(str, Enum):
    COMIC_NO = 'comic_no'
    PAGE_NO = 'page_no'
    PANEL_NO = 'panel_no'
    TEXTBOX_NO = 'textbox_no'
    DIALOG_OR_NARRATION = 'dialog_or_narration'
    TEXT = 'text'
    x1 = 'x1'
    y1 = 'y1'
    x2 = 'x2'
    y2 = 'y2'


def merge_pt_boxes(box1: torch.Tensor, box2) -> torch.Tensor:
    xmin = torch.min(box1[0], box2[0])
    ymin = torch.min(box1[1], box2[1])
    xmax = torch.max(box1[2], box2[2])
    ymax = torch.max(box1[3], box2[3])
    return torch.tensor([xmin, ymin, xmax, ymax], dtype=box1.dtype, device=box1.device)


def box_intersection_rate(source_bb: List, target_bb: List) -> float:
    box_1, box_2 = box(*source_bb), box(*target_bb)
    if box_1.area == 0:
        return 0
    return box_1.intersection(box_2).area / box_1.area


def box_to_box_center_distance(box1: torch.Tensor,
                               box2: torch.Tensor):
    # Calculate the center coordinates of each box
    center1_x = (box1[0] + box1[2]) / 2  # Scalar
    center1_y = (box1[1] + box1[3]) / 2  # Scalar
    center2_x = (box2[0] + box2[2]) / 2  # Scalar
    center2_y = (box2[1] + box2[3]) / 2  # Scalar

    # Calculate the absolute center-to-center distance between the boxes
    center_distance = torch.sqrt(torch.pow(center1_x - center2_x, 2) + torch.pow(center1_y - center2_y, 2))  # Scalar
    return center_distance


def sort_elements_by_z_order(box_list: List[torch.Tensor],
                             grid_count: float = 4.0,
                             normalization_box: Optional[torch.Tensor] = None):
    if len(box_list) == 0:
        return []

    boxes = torch.stack(box_list, dim=0)

    # Calculate the center coordinates of each bounding box
    x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
    y_centers = (boxes[:, 1] + boxes[:, 3]) / 2

    x_centers_min = x_centers.min()
    x_centers_max = x_centers.max()
    y_centers_min = y_centers.min()
    y_centers_max = y_centers.max()

    if normalization_box is not None:
        x_centers_min = normalization_box[[0, 2]].min()
        x_centers_max = normalization_box[[0, 2]].max()
        y_centers_min = y_centers.min()  # normalization_box[[1, 3]].min() - this makes top bubbles to snap together
        y_centers_max = normalization_box[[1, 3]].max()

    # Normalize the coordinates to the range [0, 1]
    x_centers_normalized = (x_centers - x_centers_min) / (x_centers_max - x_centers_min)
    y_centers_normalized = (y_centers - y_centers_min) / (y_centers_max - y_centers_min)

    x_centers_snapped = torch.round(x_centers_normalized * grid_count) / grid_count
    y_centers_snapped = torch.round(y_centers_normalized * grid_count) / grid_count

    ind = np.lexsort((x_centers_snapped.cpu().numpy(), y_centers_snapped.cpu().numpy()))

    return ind


def extract_bounding_box_from_mask(mask,
                                   alternative_mask: Optional[np.ndarray] = None) -> Optional[List]:
    # Find rows and columns with non-zero values
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if (not any(rows) or not any(cols)) and alternative_mask is not None:
        # then there must be alternative mask
        rows = np.any(alternative_mask, axis=1)
        cols = np.any(alternative_mask, axis=0)

    # if all false then return None
    if not any(rows) or not any(cols):
        return None

    # Find the first and last row and column indices with non-zero values
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Return the bounding box coordinates as [xmin, ymin, xmax, ymax]
    return [xmin, ymin, xmax, ymax]


def smooth_edges(mask):
    # Convert the mask to binary image format (0 and 255)
    mask = mask.astype(np.uint8) * 255

    # Create a structuring element for the morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Adjust the kernel size as needed

    # Apply a closing operation to close small gaps in the mask
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply an opening operation to smooth out the edges
    smoothed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    return smoothed / 255


def bbox_area(bbox: torch.Tensor):
    width = bbox[2].item() - bbox[0].item()
    height = bbox[3].item() - bbox[1].item()
    area = width * height
    return area


def cat_df(dfs: List[Optional[pd.DataFrame]]) -> Optional[pd.DataFrame]:
    dfs = [elem for elem in dfs if elem is not None]
    if len(dfs) == 0:
        return None
    elif len(dfs) == 1:
        return dfs[0]

    return pd.concat(dfs, ignore_index=True)


if __name__ == '__main__':
    # read_ad_pages()
    extract_batch_elements_from_json_files(
        data_dir='/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/data/text_cloze_jsons',
        json_files=['0.json']
    )
