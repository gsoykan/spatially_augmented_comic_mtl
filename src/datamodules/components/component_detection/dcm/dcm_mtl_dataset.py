import os
import urllib
from typing import List, Tuple
from xml.dom import minidom

import albumentations as A
import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.datamodules.components.component_detection.dcm.dcm_dataset import DCMDataset, \
    Point, \
    DCMSVGBoundingBoxClass, \
    BoundingBox, \
    Polygon, DCMMultiTaskLearningData
from src.utils.detection.bounding_box_utils import visualize_all_bboxes


class DCMMultiTaskLearningDataset(DCMDataset):
    def __init__(self,
                 # data_dir: ${work_dir}/data/
                 data_dir: str,
                 element_slice: Tuple[int, int] = None,
                 transform: A.Compose = None,
                 max_labels: int = 100):
        dataset_file_path = os.path.join(data_dir, 'DCM772/mtl_dataset.obj')
        super().__init__(dataset_file_path)
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'DCM772/images')
        self.svg_dir = os.path.join(data_dir, 'DCM772/output')
        self.max_labels = max_labels
        self.svg_paths = self.get_annotation_svg_paths(self.svg_dir)
        self.img_paths = self.get_image_file_paths(self.image_dir)
        self.dataset = self.prepare_dataset()
        if element_slice is not None:
            self.dataset = self.dataset[element_slice[0]: element_slice[1]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.dataset[idx].img_path
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.dataset[idx] \
            .dcm_segmentation_data \
            .generate_custom_mask(class_dict={DCMSVGBoundingBoxClass.gtballoon: 0})
        bounding_boxes, category_ids = self.dataset[idx] \
            .bounding_box_data \
            .generate_annotations(format='yolo')

        category_ids = np.expand_dims(np.array(category_ids), 1)
        bounding_boxes = np.array(bounding_boxes)
        padded_labels = np.zeros((self.max_labels, 5))
        if self.transform:
            transformed = self.transform(image=image,
                                         mask=mask,
                                         bboxes=bounding_boxes,
                                         category_ids=category_ids)
            image = transformed["image"]
            mask = transformed["mask"]
            h, w, c = image.shape
            bounding_boxes = transformed['bboxes']
            category_ids = transformed['category_ids']
            targets = np.hstack((category_ids, bounding_boxes))
            if isinstance(mask, torch.Tensor):
                mask = mask.type(torch.LongTensor)
        else:
            raise Exception("Transform is required for mtl dataset")
        if len(targets) != 0:
            padded_labels[range(len(targets))[: self.max_labels]] = targets[: self.max_labels]
            padded_labels[:, 1] = padded_labels[:, 1] * w
            padded_labels[:, 2] = padded_labels[:, 2] * h
            padded_labels[:, 3] = padded_labels[:, 3] * w
            padded_labels[:, 4] = padded_labels[:, 4] * h
        return image, padded_labels, mask

    # TODO: @gsoykan - create relations to such as character to balloon, face to balloon...
    def prepare_dataset(self) -> List[DCMMultiTaskLearningData]:
        dataset = self.load_dataset()
        if dataset is not None:
            return dataset
        dataset = []

        def extract_data_from_polygon_element(element):
            points = element.getAttribute("points")
            id = element.getAttribute("id")
            class_name = element.getAttribute("class")
            if class_name == '':
                if "face" in id:
                    class_name = DCMSVGBoundingBoxClass.face
                elif "character" in id:
                    class_name = DCMSVGBoundingBoxClass.character
            return id, class_name, points

        for svg_path in tqdm(self.svg_paths):
            doc = minidom.parse(svg_path)  # parseString also exists
            first_doc_element = doc.getElementsByTagName("svg")[0]
            img_width = first_doc_element.getAttribute("width")
            img_height = first_doc_element.getAttribute("height")
            try:
                title = doc.getElementsByTagName("title")[0].firstChild.data
                title = urllib.parse.unquote(title)
            except:
                print(svg_path + ": could not be loaded")
                continue
            polygons = [extract_data_from_polygon_element(element) for element
                        in doc.getElementsByTagName('polygon')]
            doc.unlink()
            valid_classes = [DCMSVGBoundingBoxClass.gtballoon,
                             DCMSVGBoundingBoxClass.gtNarative,
                             DCMSVGBoundingBoxClass.gtpanel,
                             DCMSVGBoundingBoxClass.face,
                             DCMSVGBoundingBoxClass.character]
            polygons = [item for item in polygons if item[1] in valid_classes and len(item[2]) > 3]
            bounding_box_list = []
            polygon_data_list = []
            for polygon in polygons:
                raw_points = polygon[2].split()
                points = list(map(lambda x: Point(x.split(",")[0], x.split(",")[1]), raw_points))
                bounding_box = BoundingBox(polygon[1], points)
                if bounding_box.check_if_valid():
                    bounding_box_list.append(bounding_box)
                polygon_data_list.append(Polygon(polygon[1], points))
            svg_path_head, svg_path_tail = os.path.split(svg_path)
            comic_name = svg_path_head.split(os.sep)[-1]
            img_path = self.img_paths[comic_name][title]
            if img_path is None:
                raise Exception("img path not found for: " + svg_path)
            dataset.append(DCMMultiTaskLearningData(img_path,
                                                    bounding_boxes=bounding_box_list,
                                                    polygons=polygon_data_list,
                                                    img_width=int(img_width),
                                                    img_height=int(img_height)))
        self._save_dataset(dataset)
        return dataset


if __name__ == '__main__':
    data_dir = "/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/data"
    transformations = A.Compose([
        A.Normalize()
    ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
    dataset = DCMMultiTaskLearningDataset(data_dir, transform=transformations)
    image, bb_labels, segmentation_mask = dataset[1]
    # segmentation mask doğru çalışıyor
    # tried with imgplot = plt.imshow(segmentation_mask, cmap='gray')
    # bounding box lar da ok
    # visualize_all_bboxes(image, bounding_boxes, category_ids, BoundingBox.category_id_to_name())
    print("sample is deconstructed!")
