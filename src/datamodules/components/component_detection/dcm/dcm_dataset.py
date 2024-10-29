import os
from enum import Enum
from typing import List, Dict, Optional, Tuple

import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import numpy as np
from src.utils.detection.yolox.boxes import single_xyxy2cxcywh
from torch.utils.data import Dataset


class DCMDatasetMode(str, Enum):
    train = 'train'
    test = 'test'
    val = 'val'


class Point:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)


class Polygon:
    def __init__(self, label: str, points: List[Point]):
        self.points = points
        self.label = label


class BoundingBox:
    def __init__(self,
                 label: str,
                 points: List[Point],
                 id: Optional[str] = None):
        self.label = label
        self.points = points
        self.id = id

    def check_if_valid(self):
        x_s = list(map(lambda point: point.x, self.points))
        y_s = list(map(lambda point: point.y, self.points))
        x_s.sort()
        y_s.sort()
        return x_s[0] != x_s[-1] and y_s[0] != y_s[-1]

    def to_xyxy(self):
        x_s = list(map(lambda point: point.x, self.points))
        y_s = list(map(lambda point: point.y, self.points))
        x_s.sort()
        y_s.sort()
        assert x_s[0] != x_s[-1] and y_s[0] != y_s[-1]
        return [x_s[0], y_s[0], x_s[-1], y_s[-1]]

    def to_mxmywh(self):
        x_0, y_0, x_1, y_1 = self.to_xyxy()
        return [x_0, y_0, x_1 - x_0, y_1 - y_0]

    def to_cxcywh(self, width, height):
        return single_xyxy2cxcywh(self.to_xyxy(), width, height)

    @staticmethod
    def category_id_to_name():
        return {
            0: DCMSVGBoundingBoxClass.character,
            1: DCMSVGBoundingBoxClass.face,
            2: DCMSVGBoundingBoxClass.gtpanel,
            3: DCMSVGBoundingBoxClass.gtballoon,
            4: DCMSVGBoundingBoxClass.gtNarative
        }

    def generate_instance_mask(self, img_height, img_width, bypass: bool = False):
        whole_mask = np.zeros((img_height, img_width))
        if bypass:
            # TODO: @gsoykan - transpose channels
            return whole_mask.astype(int)

        polygon_points = list(map(lambda point: (point.x, point.y), self.points))
        mask = Image.new("1", (img_width, img_height))
        draw = ImageDraw.Draw(mask)
        draw.polygon(polygon_points, fill=1)
        whole_mask += np.array(mask)
        return whole_mask.astype(int)

    def get_class_by_label(self, start_from: int = 1) -> int:
        label = self.label
        if label == DCMSVGBoundingBoxClass.character:
            return start_from + 0
        elif label == DCMSVGBoundingBoxClass.face:
            return start_from + 1
        elif label == DCMSVGBoundingBoxClass.gtpanel:
            return start_from + 2
        elif label == DCMSVGBoundingBoxClass.gtballoon:
            return start_from + 3
        elif label == DCMSVGBoundingBoxClass.gtNarative:
            return start_from + 4
        else:
            raise Exception("unknown label")


class DCMSVGBoundingBoxClass(str, Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    gtpanel = 'gtpanel'
    gtNarative = 'gtNarative'
    gtballoon = 'gtballoon'
    inconsistent = 'inconsistent'
    character = 'character'
    face = 'face'


class DCMMultiTaskLearningData:
    def __init__(self,
                 img_path,
                 img_width,
                 img_height,
                 bounding_boxes: List[BoundingBox],
                 polygons: List[Polygon]):
        self.img_path = img_path
        self.img_width = img_width
        self.img_height = img_height
        self.bounding_box_data = DCMBoundingBoxData(img_path,
                                                    bounding_boxes=bounding_boxes,
                                                    img_width=int(img_width),
                                                    img_height=int(img_height))
        self.dcm_segmentation_data = DCMSegmentationData(img_path,
                                                         img_width=int(img_width),
                                                         img_height=int(img_height),
                                                         polygons=polygons)


class DCMSegmentationData:
    def __init__(self,
                 img_path: str,
                 img_width: int,
                 img_height: int,
                 polygons: List[Polygon]):
        self.img_path = img_path
        self.polygons = polygons
        self.img_width = img_width
        self.img_height = img_height
        self.num_segmentation_classes = len(DCMSVGAnnotationClass.list()) - 1

    def generate_mask(self):
        whole_mask = np.zeros((self.img_height, self.img_width, self.num_segmentation_classes))
        for polygon in self.polygons:
            mask_index = self.get_mask_index_by_label(polygon.label)
            polygon_points = list(map(lambda point: (point.x, point.y), polygon.points))
            mask = Image.new("1", (self.img_width, self.img_height))
            draw = ImageDraw.Draw(mask)
            draw.polygon(polygon_points, fill=1)
            whole_mask[:, :, mask_index] += np.array(mask)
        whole_mask = np.clip(whole_mask, a_min=0, a_max=1)
        return whole_mask.astype(int)

    # class_dict => {'class_name': mask_index}
    def generate_custom_mask(self, class_dict: Dict):
        whole_mask = np.zeros((self.img_height, self.img_width, len(class_dict)))
        for polygon in self.polygons:
            mask_index = class_dict.get(polygon.label)
            if mask_index is None:
                continue
            polygon_points = list(map(lambda point: (point.x, point.y), polygon.points))
            mask = Image.new("1", (self.img_width, self.img_height))
            draw = ImageDraw.Draw(mask)
            draw.polygon(polygon_points, fill=1)
            whole_mask[:, :, mask_index] += np.array(mask)
        whole_mask = np.clip(whole_mask, a_min=0, a_max=1)
        return whole_mask.astype(int)

    @staticmethod
    def get_mask_index_by_label(label) -> int:
        if label == DCMSVGAnnotationClass.gtpanel:
            return 0
        elif label == DCMSVGAnnotationClass.gtNarative:
            return 1
        elif label == DCMSVGAnnotationClass.gtballoon:
            return 2
        else:
            raise Exception("unknown label")


class DCMSVGAnnotationClass(str, Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    gtpanel = 'gtpanel'
    gtNarative = 'gtNarative'
    gtballoon = "gtballoon"
    inconsistent = "inconsistent"


class DCMBoundingBoxData:
    def __init__(self,
                 img_path,
                 img_width,
                 img_height,
                 bounding_boxes: List[BoundingBox],
                 balloon_links: List[Tuple[str, str]],
                 negative_balloon_links: List[Tuple[str, str]],
                 face_to_char_links: List[Tuple[str, str]],):
        self.img_path = img_path
        self.bounding_boxes = bounding_boxes
        self.img_width = img_width
        self.img_height = img_height
        self.balloon_links = balloon_links
        self.negative_balloon_links = negative_balloon_links
        self.face_to_char_links = face_to_char_links

    # in pascal voc format
    def generate_annotations(self, format: str = 'pascal_voc'):
        if format == 'pascal_voc':
            bounding_boxes = list(map(lambda bounding_box: bounding_box.to_xyxy(),
                                      self.bounding_boxes))
        elif format == 'yolo':
            # this is scaled between 0 - 1
            bounding_boxes = list(map(lambda bounding_box: bounding_box.to_cxcywh(self.img_width, self.img_height),
                                      self.bounding_boxes))
        elif format == 'mx_my_w_h':
            # this is unscaled
            bounding_boxes = list(map(lambda bounding_box: bounding_box.to_mxmywh(),
                                      self.bounding_boxes))
        else:
            raise Exception("unknown bounding box annotation format")
        category_ids = list(map(lambda bounding_box: bounding_box.get_class_by_label(),
                                self.bounding_boxes))
        bb_ids = list(map(lambda bounding_box: bounding_box.id, self.bounding_boxes))
        return bounding_boxes, category_ids, bb_ids

    def generate_instance_masks(self):
        return list(map(lambda bounding_box: bounding_box.generate_instance_mask(self.img_height, self.img_width),
                        self.bounding_boxes))


def searchfiles(extension='.ttf', folder='H:\\'):
    files = []
    for r, d, f in os.walk(folder):
        for file in f:
            if file.endswith(extension):
                files.append(r + "/" + file)
    return files


class DCMDataset(Dataset):
    def __init__(self, dataset_file_path):
        self.dataset_file_path = dataset_file_path

    @staticmethod
    def get_annotation_svg_paths(svg_dir: str) -> List[str]:
        return searchfiles('.svg', svg_dir)

    @staticmethod
    def get_image_file_paths(img_dir: str) -> Dict[str, Dict[str, str]]:
        file_paths = searchfiles('.jpg', img_dir)
        file_path_dict = {}
        for file_path in file_paths:
            path_head, path_tail = os.path.split(file_path)
            page_title = ".".join(path_tail.split(".")[:-1])
            comic_name = path_head.split(os.sep)[-1]
            if file_path_dict.get(comic_name, None) is None:
                file_path_dict[comic_name] = {}
            file_path_dict[comic_name].__setitem__(page_title, file_path)
        return file_path_dict

    def _save_dataset(self, dataset):
        import pickle
        with open(self.dataset_file_path, 'wb') as dataset_file:
            pickle.dump(dataset, dataset_file)

    def load_dataset(self, data_dir: Optional[str] = None, override_img_paths: bool = True):
        import pickle
        if not os.path.isfile(self.dataset_file_path):
            return None
        with open(self.dataset_file_path, 'rb') as dataset_file:
            loaded_dataset = pickle.load(dataset_file)
            if override_img_paths:
                assert data_dir is not None, "data_dir should be provided for overriding img paths..."
                for instance in loaded_dataset:
                    prev_path = instance.img_path
                    new_path = os.path.join(data_dir, prev_path.split('/data/')[1])
                    instance.img_path = new_path
        return loaded_dataset
