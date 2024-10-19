import itertools
import json
import os
from collections import defaultdict
from typing import Optional, Dict, Tuple, Set, List

import albumentations as A
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datamodules.components.vision_transform_setting import VisionTransformSetting
from src.utils.basic_utils import read_or_get_image


class ComicsSeqFirebaseFaceBodySequenceIteratorDataset(Dataset):
    def __init__(self,
                 transform_face: A.Compose,
                 transform_body: A.Compose,
                 data_dir: str,
                 # path from datadir...
                 sequences_json_path: str = 'comics_seq/train_sequences.json',
                 img_folder_root_dir_face: str = '/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops_large_faces/',
                 img_folder_root_dir_body: str = '/datasets/COMICS/comics_crops/',
                 ):
        self.sequences_json_path = sequences_json_path
        self.img_folder_root_dir_face = img_folder_root_dir_face
        self.img_folder_root_dir_body = img_folder_root_dir_body
        self.data_dir = data_dir
        self.transform_face = transform_face
        self.transform_body = transform_body
        self.dataset_dict = self.load_dataset()
        self.sorted_dataset_keys = list(sorted(self.dataset_dict.keys()))

    def load_dataset(self):
        with open(os.path.join(self.data_dir, self.sequences_json_path), 'r') as file:
            sequences_json = json.load(file)

        def construct_img_path(item_id: str,
                               is_comics_crops_with_body_face: bool,
                               img_folder_root_dir: str,
                               ):
            series_id, page_id, panel_id, img_type, image_idx = item_id.split('_')
            if is_comics_crops_with_body_face:
                return os.path.join(img_folder_root_dir, str(series_id), f'{str(page_id)}_{str(panel_id)}',
                                    'bodies' if img_type == 'body' or img_type == 'bodies' else 'faces',
                                    f'{str(image_idx)}.jpg')
            else:
                return os.path.join(img_folder_root_dir, str(series_id), f'{str(page_id)}_{str(panel_id)}',
                                    f'{str(image_idx)}.jpg')

        dataset_dict = defaultdict(list)
        for element in sequences_json:
            annotations = element['annotations']
            seq_id = element['seq_id']
            for annotation in annotations:
                char_id = annotation['charId']
                for char_instance in annotation['charInstances']:
                    face_instance = char_instance.get('face')
                    body_instance = char_instance.get('body')

                    char_instance_paths = {'char_id': char_id}

                    if face_instance is not None:
                        face_path = construct_img_path(face_instance,
                                                       False,
                                                       self.img_folder_root_dir_face)
                        char_instance_paths['face'] = face_path
                        char_instance_paths['face_id'] = face_instance

                    if body_instance is not None:
                        body_path = construct_img_path(body_instance,
                                                       True,
                                                       self.img_folder_root_dir_body)
                        char_instance_paths['body'] = body_path
                        char_instance_paths['body_id'] = body_instance

                    dataset_dict[seq_id] = [*dataset_dict[seq_id], char_instance_paths]

        return dataset_dict

    def __getitem__(self, index):

        def get_image_data(face_path: Optional[str],
                           body_path: Optional[str]):
            # # TODO: @gsoykan - remove
            # return torch.randn(3, 96, 96), torch.randn(3, 128, 128)
            if face_path is not None:
                anchor_img_face = read_or_get_image(face_path, read_rgb=True)
                anchor_img_face = self.transform_face(image=anchor_img_face)['image']
            else:
                anchor_img_face = torch.randn(3, 96, 96)

            if body_path is not None:
                anchor_img_body = read_or_get_image(body_path, read_rgb=True)
                anchor_img_body = self.transform_body(image=anchor_img_body)['image']
            else:
                anchor_img_body = torch.randn(3, 128, 128)

            return anchor_img_face, anchor_img_body

        selected_seq_id = self.sorted_dataset_keys[index]
        seq_chars = self.dataset_dict[selected_seq_id]

        faces = []
        face_ids = []
        bodies = []
        body_ids = []
        char_ids = []

        for i, char_instance in enumerate(seq_chars):
            char_id, face_path, body_path = char_instance.get('char_id'), \
                                            char_instance.get('face'), \
                                            char_instance.get('body')
            face_id, body_id = char_instance.get('face_id'), \
                               char_instance.get('body_id')
            img_face, img_body = get_image_data(face_path, body_path)
            faces.append(img_face)
            bodies.append(img_body)
            char_ids.append(char_id)
            face_ids.append(face_id)
            body_ids.append(body_id)

        return {
            'faces': faces,
            'face_ids': face_ids,
            'bodies': bodies,
            'body_ids': body_ids,
            'char_ids': char_ids,
            'seq_id': selected_seq_id,
        }

    def __len__(self):
        return len(self.sorted_dataset_keys)


def comics_seq_firebase_face_body_seq_iter_collate_fn(batch):
    raise NotImplementedError


if __name__ == '__main__':
    data_dir = '/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/data'
    transform_args_face = {'N': 96, 'use_padding': False}
    transform_args_body = {'N': 128, 'use_padding': True}
    transform_face = VisionTransformSetting.SIMCLR_TEST.get_transformation(
        **transform_args_face)
    transform_body = VisionTransformSetting.SIMCLR_TEST.get_transformation(
        **transform_args_body)
    dataset = ComicsSeqFirebaseFaceBodySequenceIteratorDataset(data_dir=data_dir,
                                                               transform_face=transform_face,
                                                               transform_body=transform_body,
                                                               sequences_json_path='comics_seq/test_sequences.json',
                                                               img_folder_root_dir_face='/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops_large_faces/',
                                                               img_folder_root_dir_body='/datasets/COMICS/comics_crops/')
    for element in dataset:
        print(element)
    print(len(dataset))
