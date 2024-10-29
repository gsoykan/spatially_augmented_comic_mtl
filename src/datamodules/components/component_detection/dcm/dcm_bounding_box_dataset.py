import itertools
import random
from typing import List, Dict, Tuple, Optional
import os
import torch
from xml.dom import minidom
import numpy as np
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import urllib

from src.datamodules.components.component_detection.dcm.dcm_dataset import DCMDataset, \
    Point, \
    DCMSVGBoundingBoxClass, \
    BoundingBox, \
    DCMBoundingBoxData, DCMDatasetMode
from src.datamodules.components.vision_transform_setting import albu_clip_0_1
from src.utils.ssl.create_ssl_dataset_file_body import box_intersection_rate


class DCMBoundingBoxDataset(DCMDataset):
    def __init__(self,
                 # data_dir: ${work_dir}/data/
                 data_dir: str,
                 element_slice: Tuple[int, int] = None,
                 transform: A.Compose = None,
                 use_instance_masks: bool = False,
                 shuffle: bool = True,
                 min_panel_count: Optional[int] = 3):
        dataset_file_path = os.path.join(data_dir, 'DCM772/bounding_box_dataset.obj')
        super().__init__(dataset_file_path)
        self.min_panel_count = min_panel_count
        self.use_instance_masks = use_instance_masks
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'DCM772/images')
        self.svg_dir = os.path.join(data_dir, 'DCM772/output')
        self.svg_paths = self.get_annotation_svg_paths(self.svg_dir)
        self.img_paths = self.get_image_file_paths(self.image_dir)
        self.banned_items = ['All_humor_01_00_fc/all_humor_01_45']
        self.dataset = self.prepare_dataset(data_dir)
        if shuffle:
            random.seed(123)
            random.shuffle(self.dataset)
        if element_slice is not None:
            self.dataset = self.dataset[element_slice[0]: element_slice[1]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    """
     img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
    source for get_item: https://debuggercafe.com/a-simple-pipeline-to-train-pytorch-faster-rcnn-object-detection-model/
    """

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.dataset[idx].img_path
        # print(img_name)
        image = cv2.imread(img_name)

        if image is None:
            img_path_from_data_dir = os.path.join(self.data_dir, self.dataset[idx].img_path.split('/data/')[1])
            self.dataset[idx].img_path = img_path_from_data_dir
            image = cv2.imread(img_path_from_data_dir)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        boxes, labels, bb_ids = self.dataset[idx].generate_annotations(format='pascal_voc')
        labels = np.array(labels)
        boxes = np.array(boxes)
        balloon_links = self.dataset[idx].balloon_links
        negative_balloon_links = self.dataset[idx].negative_balloon_links
        face_to_char_links = self.dataset[idx].face_to_char_links

        instance_masks = None
        if self.use_instance_masks:
            instance_masks = self.dataset[idx].generate_instance_masks()
            instance_masks = np.array(instance_masks, dtype=np.uint8)
            instance_masks = list(instance_masks)  # np.transpose(instance_masks, (1, 2, 0))

        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        if len(boxes) == 0:
            # print('empty box at index: ', idx)
            return None
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        assert self.transform is not None, "Transform is required for bounding box dataset"

        sample = self.transform(image=image,
                                bboxes=target['boxes'],
                                masks=instance_masks,
                                labels=labels,
                                area=area,
                                iscrowd=iscrowd,
                                bb_ids=bb_ids,
                                indices=np.array(list(range(len(labels)))))
        img_transformed = sample['image']
        target['boxes'] = torch.Tensor(sample['bboxes'])
        target['labels'] = torch.tensor(sample['labels'], dtype=torch.int64)
        target['area'] = torch.tensor(sample['area'])
        target['iscrowd'] = torch.tensor(sample['iscrowd'], dtype=torch.int64)

        if self.use_instance_masks:
            target['masks'] = np.array(sample['masks'])[sample['indices']].astype(np.uint8)
            target['masks'] = torch.tensor(target['masks'])

        bb_ids_to_idx_and_label = {v: (i, sample['labels'][i].item()) for i, v in enumerate(sample['bb_ids'])}
        # handling links
        valid_balloon_links = []
        for balloon, char_id in balloon_links:
            b_info, c_info = bb_ids_to_idx_and_label.get(balloon), bb_ids_to_idx_and_label.get(char_id)
            if None not in [b_info, c_info]:
                (b_idx, b_label), (c_idx, c_label) = b_info, c_info
                valid_balloon_links.append((b_idx, c_idx, b_label, c_label))
        target['links'] = torch.tensor(np.array(valid_balloon_links), dtype=torch.int64)
        # handling negative links
        negative_valid_balloon_links = []
        for balloon, char_id in negative_balloon_links:
            b_info, c_info = bb_ids_to_idx_and_label.get(balloon), bb_ids_to_idx_and_label.get(char_id)
            if None not in [b_info, c_info]:
                (b_idx, b_label), (c_idx, c_label) = b_info, c_info
                negative_valid_balloon_links.append((b_idx, c_idx, b_label, c_label))
        target['negative_links'] = torch.tensor(np.array(negative_valid_balloon_links), dtype=torch.int64)
        # handling face_to_char ids
        valid_face_to_char_links = []
        for face_id, char_id in face_to_char_links:
            f_info, c_info = bb_ids_to_idx_and_label.get(face_id), bb_ids_to_idx_and_label.get(char_id)
            if None not in [f_info, c_info]:
                (f_idx, _), (c_idx, _) = f_info, c_info
                valid_face_to_char_links.append((f_idx, c_idx))
        target['face_to_char_links'] = torch.tensor(np.array(valid_face_to_char_links), dtype=torch.int64)

        if target['boxes'].size()[0] == 0:
            return None

        return img_transformed, target

    def prepare_dataset(self, data_dir: Optional[str] = None) -> List[DCMBoundingBoxData]:
        # our test comic page and svg
        # self.svg_paths = list(filter(lambda x: 'All_humor_01_00_fc/all_humor_01_19' in x, self.svg_paths))
        dataset = self.load_dataset(data_dir)
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

        def extract_balloon_link_from_line_element(element):
            balloon_id = element.getAttribute('idBalloon')
            char_id = element.getAttribute('idCharacter')
            return balloon_id, char_id

        def extract_face_link_from_line_element(element):
            face_id = element.getAttribute('idFace')
            char_id = element.getAttribute('idCharacter')
            return face_id, char_id

        skipped_count_no_face = 0
        skipped_count_no_char = 0
        skipped_count_no_balloon = 0

        for svg_path in tqdm(self.svg_paths):

            skip_svg = False
            for banned_item in self.banned_items:
                if banned_item in svg_path:
                    skip_svg = True
                    break
            if skip_svg:
                continue

            doc = minidom.parse(svg_path)  # parseString also exists
            first_doc_element = doc.getElementsByTagName("svg")[0]
            img_width = first_doc_element.getAttribute("width")
            img_height = first_doc_element.getAttribute("height")
            try:
                title = doc.getElementsByTagName("title")[0].firstChild.data
                title = urllib.parse.unquote(title)
            except:
                print(svg_path + ": could not be loaded")
                doc.unlink()
                continue
            polygons = [extract_data_from_polygon_element(element) for element
                        in doc.getElementsByTagName('polygon')]
            balloon_links = [extract_balloon_link_from_line_element(element) for element
                             in doc.getElementsByTagName('line') if
                             element.hasAttribute('idBalloon') and element.getAttribute('class') != 'unverified']
            face_links = [extract_face_link_from_line_element(element) for element
                          in doc.getElementsByTagName('line') if
                          element.hasAttribute('idCharacter') and element.hasAttribute('idFace')]

            # improvement to face_links - there are some cases where face is not linked to characters...
            # find charless faces
            face_ids_with_char = list(map(lambda x: x[0], face_links))
            face_polygons = list(filter(lambda x: x[1] == DCMSVGBoundingBoxClass.face, polygons))
            charless_face_polygons = list(filter(lambda x: x[0] not in face_ids_with_char, face_polygons))
            if len(charless_face_polygons) != 0:
                # bunları da faceless_char_polygons olarak ayıklamamız lazım...
                char_ids_with_faces = list(map(lambda x: x[1], face_links))
                char_polygons = list(filter(lambda x: x[1] == DCMSVGBoundingBoxClass.character, polygons))
                faceless_char_polygons = list(filter(lambda x: x[0] not in char_ids_with_faces, char_polygons))
                for charless_face_polygon in charless_face_polygons:
                    raw_points = charless_face_polygon[2].split()
                    points = list(map(lambda x: Point(x.split(",")[0], x.split(",")[1]), raw_points))
                    bounding_box = BoundingBox(charless_face_polygon[1], points, charless_face_polygon[0])
                    try:
                        bb_box_face = bounding_box.to_xyxy()
                    except:
                        continue
                    candidates = []
                    for char_polygon in faceless_char_polygons:
                        raw_points = char_polygon[2].split()
                        points = list(map(lambda x: Point(x.split(",")[0], x.split(",")[1]), raw_points))
                        bounding_box = BoundingBox(char_polygon[1], points, char_polygon[0])
                        try:
                            bb_box_char = bounding_box.to_xyxy()
                        except:
                            continue
                        intersection_rate = box_intersection_rate(bb_box_face, bb_box_char)
                        # if intersection_rate != 0:
                        # print('intersection rate:', intersection_rate)
                        if intersection_rate >= 0.20:
                            candidates.append((char_polygon, intersection_rate))
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    if len(candidates) != 0:
                        char_id = candidates[0][0][0]
                        face_id = charless_face_polygon[0]
                        # print('new face link added', face_id, char_id)
                        face_links.append((face_id, char_id))

            additional_links = []
            for balloon_id, char_id in balloon_links:
                if 'face' in char_id:
                    for f_link, c_link in face_links:
                        if f_link == char_id:
                            additional_links.append((balloon_id, c_link))
                            break
                elif 'character' in char_id:
                    for f_link, c_link in face_links:
                        if c_link == char_id:
                            additional_links.append((balloon_id, f_link))
                            break
            balloon_links.extend(additional_links)
            balloon_links = list(set(balloon_links))
            balloon_links.sort()

            doc.unlink()
            valid_classes = [DCMSVGBoundingBoxClass.gtballoon,
                             DCMSVGBoundingBoxClass.gtNarative,
                             DCMSVGBoundingBoxClass.gtpanel,
                             DCMSVGBoundingBoxClass.face,
                             DCMSVGBoundingBoxClass.character]
            polygons = [item for item in polygons if item[1] in valid_classes and len(item[2]) > 3]
            bounding_box_list = []
            panel_count = 0
            for polygon in polygons:
                raw_points = polygon[2].split()
                points = list(map(lambda x: Point(x.split(",")[0], x.split(",")[1]), raw_points))
                bounding_box = BoundingBox(polygon[1], points, polygon[0])
                if bounding_box.check_if_valid():
                    if bounding_box.label == DCMSVGBoundingBoxClass.gtpanel:
                        panel_count += 1
                    bounding_box_list.append(bounding_box)
            svg_path_head, svg_path_tail = os.path.split(svg_path)
            comic_name = svg_path_head.split(os.sep)[-1]
            img_path = self.img_paths[comic_name][title]

            if panel_count < self.min_panel_count:
                print(f'page does not have enough panels with {str(panel_count)} panels at path: ', svg_path)
                continue

            if len(bounding_box_list) == 0:
                print('no bounding box in path: ', svg_path)
                continue

            if img_path is None:
                raise Exception("img path not found for: " + svg_path)

            all_ids = list(map(lambda x: x.id, bounding_box_list))
            balloon_links = list(
                filter(lambda link_tuple: link_tuple[0] in all_ids and link_tuple[1] in all_ids, balloon_links))

            # create negative balloon links => meaning find characters (faces) in the same panel but does not have
            # a relation
            # then match those with speech bubbles in the panel...
            negative_balloon_links = []
            bb_group = itertools.groupby(sorted(bounding_box_list, key=lambda x: x.label), lambda x: x.label)
            bb_group_dict = {}
            for key, group in bb_group:
                bb_group_dict[key] = list(group)
            bb_panels = bb_group_dict[DCMSVGBoundingBoxClass.gtpanel]

            bb_faces = bb_group_dict.get(DCMSVGBoundingBoxClass.face, [])
            if len(bb_faces) == 0:
                skipped_count_no_face += 1
                print('no faces count: ', skipped_count_no_face)
                continue
            bb_chars = bb_group_dict.get(DCMSVGBoundingBoxClass.character, [])
            if len(bb_chars) == 0:
                skipped_count_no_char += 1
                print('no chars count: ', skipped_count_no_char)
                continue
            bb_balloons = bb_group_dict.get(DCMSVGBoundingBoxClass.gtballoon, [])
            if len(bb_balloons) == 0:
                skipped_count_no_balloon += 1
                print('no balloonss count: ', skipped_count_no_balloon)
                continue
            panel_to_items_dict = {}
            for bb_panel in bb_panels:
                bb_panel_box = bb_panel.to_xyxy()
                chars_faces_in_panel = []
                balloons_in_panel = []
                for bb_char_or_face in [*bb_chars, *bb_faces]:
                    bb_box = bb_char_or_face.to_xyxy()
                    intersection_rate = box_intersection_rate(bb_box, bb_panel_box)
                    # print(intersection_rate)
                    if intersection_rate >= 0.95:
                        chars_faces_in_panel.append(bb_char_or_face)
                for bb_balloon in bb_balloons:
                    bb_box = bb_balloon.to_xyxy()
                    intersection_rate = box_intersection_rate(bb_box, bb_panel_box)
                    if intersection_rate >= 0.95:
                        balloons_in_panel.append(bb_balloon)
                panel_to_items_dict[bb_panel.id] = {'chars_or_faces': chars_faces_in_panel,
                                                    'balloons': balloons_in_panel}

            linked_chars_faces = list(map(lambda x: x[1], balloon_links))
            for k, v in panel_to_items_dict.items():
                chars_faces_in_panel = v['chars_or_faces']
                balloons = v['balloons']
                unlinked_chars_faces = [x for x in chars_faces_in_panel if x.id not in linked_chars_faces]
                # now balloons ile unlinked_chars_faces'i çarpıştır...
                for b in balloons:
                    for c in unlinked_chars_faces:
                        negative_balloon_links.append((b.id, c.id))

            dataset.append(DCMBoundingBoxData(img_path,
                                              bounding_boxes=bounding_box_list,
                                              img_width=int(img_width),
                                              img_height=int(img_height),
                                              balloon_links=balloon_links,
                                              negative_balloon_links=negative_balloon_links,
                                              face_to_char_links=face_links))
        print('skipped_count_no_balloon', skipped_count_no_balloon)
        print('skipped_count_no_face', skipped_count_no_face)
        print('skipped_count_no_char', skipped_count_no_char)
        self._save_dataset(dataset)
        return dataset


"""
bboxes = [[5.66, 138.95, 147.09, 164.88], [366.7, 80.84, 132.8, 181.84]]
category_ids = [17, 18]
The `pascal_voc` format
                `[x_min, y_min, x_max, y_max]`, e.g. [97, 12, 247, 212].
"""


def edge_map_check():
    from PIL import Image, ImageFilter
    img_path = '/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/data/DCM772/images/America Greatest 001/AG001-005paper.jpg'
    img = Image.open(img_path)
    # Converting the image to grayscale, as Sobel Operator requires
    # input image to be of mode Grayscale (L)
    img = img.convert("L")

    # method 1: Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
    final = img.filter(ImageFilter.FIND_EDGES)

    # method 2: Calculating Edges using the passed laplacian Kernel
    # final = img.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
    #                                                -1, -1, -1, -1), 1, 0))
    final.save("EDGE_sample.png")


def edge_map_check_laplacian():
    img_path = '/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/data/DCM772/images/America Greatest 001/AG001-005paper.jpg'
    # load the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # remove noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # apply the Laplacian filter
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    # threshold the result to obtain the edges
    edges = np.uint8(np.absolute(laplacian) > 30) * 255

    edges = cv2.resize(edges, (600, 800), interpolation=cv2.INTER_AREA)

    # display the edges
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def edge_map_check_sobel():
    # inanılmaz iyi oldu...
    img_path = '/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/data/DCM772/images/48 Famous Americans/17.jpg'
    # load the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # remove noise
    img = cv2.GaussianBlur(img, (7, 7), 0)  # this works best...

    # apply median filter with kernel size 5x5
    # img = cv2.medianBlur(img, 5)

    # img = cv2.bilateralFilter(img, 5, 75, 75)

    # apply the Sobel filter in the x and y directions
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    # combine the results to obtain the edges
    edges = np.uint8(np.sqrt(np.square(sobel_x) + np.square(sobel_y)))

    edges = cv2.resize(edges, (600, 800), interpolation=cv2.INTER_AREA)

    # display the edges
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # edge_map_check_sobel()
    # edge_map_check()
    data_dir = "/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/data"
    validation_transformations = A.Compose([
        # A.Resize(640, 480, always_apply=True),
        A.LongestMaxSize(640),
        A.ToFloat(max_value=255, always_apply=True),
        ToTensorV2(),
    ],
        bbox_params=A.BboxParams(format='pascal_voc',
                                 label_fields=['labels', 'area', 'iscrowd', 'bb_ids', 'indices']))

    transformations = A.Compose([
        A.ToFloat(max_value=255, always_apply=True),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.OneOf([
            A.LongestMaxSize(640),
            A.RandomResizedCrop(640, 640,
                                interpolation=cv2.INTER_CUBIC, scale=(0.2, 1.0)),
        ], p=1),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ToGray(p=0.25),
        A.OneOf([
            A.ToSepia(p=1),
            A.RandomSnow(brightness_coeff=2, p=1),
            A.RandomFog(fog_coef_lower=0.2,
                        fog_coef_upper=0.4, p=1.0),
            A.ColorJitter(p=1.0),
        ], p=0.5),
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.25),
        A.Lambda(image=albu_clip_0_1),
        ToTensorV2(),
    ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels',
                                                                    'area',
                                                                    'iscrowd',
                                                                    'bb_ids',
                                                                    'indices'], min_area=16,
                                 min_visibility=0.1))
    dataset = DCMBoundingBoxDataset(data_dir,
                                    transform=transformations,
                                    use_instance_masks=True,
                                    shuffle=False)
    sample = dataset[0]

    print(f"Number of training images: {len(dataset)}")


    # function to visualize a single sample
    def visualize_sample(image, target):
        # Convert the tensor to a numpy array
        numpy_image = image.numpy()
        # Convert the numpy array to a cv2 image
        cv2_image = np.transpose(numpy_image, (1, 2, 0))
        image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        COLORS = [(1, 0, 0),  # Red
                  (0, 1, 0),  # Green
                  (0, 0, 1),  # Blue
                  # (0, 0, 0),  # Black
                  (1, 1, 1),  # White
                  (0.5, 0.5, 0.5),  # Gray
                  (1, 1, 0),  # Yellow
                  (0, 1, 1),  # Cyan
                  (1, 0, 1),  # Magenta
                  (1, 0.5, 0)]  # Orange

        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = str(int(target['labels'][box_num]))
            if label not in ["1", "2", "4", "5"]:
                continue
            mask = target['masks'][box_num]
            color = COLORS[random.randrange(0, len(COLORS))]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                color, 2
            )
            cv2.putText(
                image, label, (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

            # drawing masks
            if target['labels'][box_num] in [4]:
                red_map = np.zeros_like(mask).astype(np.uint8)
                green_map = np.zeros_like(mask).astype(np.uint8)
                blue_map = np.zeros_like(mask).astype(np.uint8)
                red_map[mask == 1], green_map[mask == 1], blue_map[mask == 1] = color
                segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
                alpha = 1
                beta = 0.4  # transparency for the segmentation map
                gamma = 0.0  # scalar added to each sum
                cv2.addWeighted(image, alpha, segmentation_map.astype(image.dtype), beta, gamma, image)

        # drawing links (relations)
        color = COLORS[random.randrange(0, len(COLORS))]
        for relation in target['links']:
            # print('relation: ', relation)
            from_idx = relation[0]
            to_idx = relation[1]
            from_box = target['boxes'][from_idx]
            to_box = target['boxes'][to_idx]
            to_label = target['labels'][to_idx]

            def box_center(box):
                # box = (box[0][0], box[0][1], box[1][0], box[1][1])
                return int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)

            cx1, cy1 = box_center(from_box)
            cx2, cy2 = box_center(to_box)

            # print('to_label', to_label)
            if to_label in [1, 2]:
                cv2.line(image, (cx1, cy1), (cx2, cy2), color, 2)

        # drawing negative links (negative relations - wrong relations in the same panel by
        # those who does not have a speech bubble)
        show_negative_links = False
        if show_negative_links:
            color = COLORS[random.randrange(0, len(COLORS))]
            for relation in target['negative_links']:
                # print('relation: ', relation)
                from_idx = relation[0]
                to_idx = relation[1]
                from_box = target['boxes'][from_idx]
                to_box = target['boxes'][to_idx]
                to_label = target['labels'][to_idx]

                def box_center(box):
                    # box = (box[0][0], box[0][1], box[1][0], box[1][1])
                    return int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)

                cx1, cy1 = box_center(from_box)
                cx2, cy2 = box_center(to_box)

                # print('to_label', to_label)
                if to_label in [1, 2]:
                    cv2.line(image, (cx1, cy1), (cx2, cy2), color, 2)

        cv2.imshow('Image', image)
        cv2.waitKey(0)


    NUM_SAMPLES_TO_VISUALIZE = 772
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        print(f'inspecting sample: {str(i)}')
        res = dataset[i]
        if res is None:
            continue
        image, target = res
        visualize_sample(image, target)

    # print(sample)

"""
# define the training tranforms
def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })
# define the validation transforms
def get_valid_transform():
    return A.Compose([
        A.Resize(512, 512, always_apply=True),
        A.ToFloat(max_value=255, always_apply=True),
        ToTensorV2(),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })

# visualizing speech bubble masks...
import matplotlib.pyplot as plt
plt.imshow(target['masks'][target['labels']==4][0].numpy())
plt.show()
"""
