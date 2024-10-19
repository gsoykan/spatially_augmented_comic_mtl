from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.datamodules.components.face_recognition.comics_seq_firebase_face_body_dataset import \
    ComicsSeqFirebaseFaceBodyDataset
from src.datamodules.components.vision_transform_setting import VisionTransformSetting
from src.utils.basic_utils import read_or_get_image, flatten_list
from src.utils.pml_accuracy_wrapper import PMLAccuracyWrapper


class ComicsSeqFirebaseFaceBodyQueryRefWrapper(Dataset):
    def __init__(self,
                 main_dataset: ComicsSeqFirebaseFaceBodyDataset,
                 is_query: bool = True
                 ):
        self.main_dataset = main_dataset
        self.is_query = is_query
        self.queries, self.references = main_dataset.queries, main_dataset.references

    def __getitem__(self, index):
        def get_image_data(face_path: Optional[str],
                           body_path: Optional[str]):
            if face_path is None:
                anchor_img_face = torch.randn(3, 96, 96)
            if body_path is None:
                anchor_img_body = torch.randn(3, 128, 128)

            if self.main_dataset.is_torch_transform:
                if face_path is not None:
                    anchor_img_face = Image.open(face_path)
                    anchor_img_face = self.main_dataset.transform_face(anchor_img_face)
                if body_path is not None:
                    anchor_img_body = Image.open(body_path)
                    anchor_img_body = self.main_dataset.transform_body(anchor_img_body)
            else:
                # # TODO: @gsoykan - remove
                # return torch.randn(3, 96, 96), torch.randn(3, 96, 96)
                if face_path is not None:
                    anchor_img_face = read_or_get_image(face_path, read_rgb=True)
                    anchor_img_face = self.main_dataset.transform_face(image=anchor_img_face)['image']
                if body_path is not None:
                    anchor_img_body = read_or_get_image(body_path, read_rgb=True)
                    anchor_img_body = self.main_dataset.transform_body(image=anchor_img_body)['image']

            return anchor_img_face, anchor_img_body

        char_id, face_path, body_path, seq_id = self.queries[index] if self.is_query else self.references[index]
        img_face, img_body = get_image_data(face_path, body_path)

        return {'faces': img_face,
                'face_mask': face_path is not None,
                'bodies': img_body,
                'body_mask': body_path is not None,
                'char_ids': char_id,
                'seq_ids': seq_id}

    def __len__(self):
        return len(self.queries) if self.is_query else len(self.references)

    def get_dataloader(self, num_workers: int = 5):
        return DataLoader(
            dataset=self,
            batch_size=32,
            num_workers=num_workers,
            pin_memory=False,
            shuffle=False, )

    @torch.no_grad()
    def evaluate(self, embedder, only_mode: Optional[str] = None):
        # get query embeddings
        self.is_query = True
        query_dataloader = self.get_dataloader()
        query_embeddings = []
        query_labels = []
        for batch in tqdm(query_dataloader, desc=f'iterating over query dataloader...'):
            if only_mode == 'face':
                query_labels.append([char_id for char_id, mask in zip(batch['char_ids'], batch['face_mask']) if mask])
            elif only_mode == 'body':
                query_labels.append([char_id for char_id, mask in zip(batch['char_ids'], batch['body_mask']) if mask])
            else:
                query_labels.append(batch['char_ids'])
            embeddings = embedder(batch['faces'].to('cuda'),
                                  batch['bodies'].to('cuda'),
                                  batch['face_mask'].to('cuda'),
                                  batch['body_mask'].to('cuda'))
            query_embeddings.append(embeddings)
        query_embeddings = torch.cat(query_embeddings, dim=0)
        query_labels = flatten_list(query_labels)
        # get reference embeddings
        self.is_query = False
        ref_dataloader = self.get_dataloader()
        ref_embeddings = []
        ref_labels = []
        for batch in tqdm(ref_dataloader, desc=f'iterating over ref dataloader...'):
            if only_mode == 'face':
                ref_labels.append([char_id for char_id, mask in zip(batch['char_ids'], batch['face_mask']) if mask])
            elif only_mode == 'body':
                ref_labels.append([char_id for char_id, mask in zip(batch['char_ids'], batch['body_mask']) if mask])
            else:
                ref_labels.append(batch['char_ids'])
            embeddings = embedder(batch['faces'].to('cuda'),
                                  batch['bodies'].to('cuda'),
                                  batch['face_mask'].to('cuda'),
                                  batch['body_mask'].to('cuda'))
            ref_embeddings.append(embeddings)
        ref_embeddings = torch.cat(ref_embeddings, dim=0)
        ref_labels = flatten_list(ref_labels)

        # convert labels to tensors
        def convert_ids_to_indices(ids):
            id_to_idx = {}
            idx = 0
            for id in ids:
                if id not in id_to_idx:
                    id_to_idx[id] = idx
                    idx += 1
            # Convert char_ids to indices
            return id_to_idx

        id_to_idx = convert_ids_to_indices([*query_labels, *ref_labels])
        query_labels = torch.tensor(list(map(lambda x: id_to_idx[x], query_labels)), device=query_embeddings.device)
        ref_labels = torch.tensor(list(map(lambda x: id_to_idx[x], ref_labels)), device=ref_embeddings.device)
        # evaluate performance
        calculator = PMLAccuracyWrapper.get_calculator()
        result = calculator.get_accuracy(query_embeddings, query_labels, ref_embeddings, ref_labels)
        return result


if __name__ == '__main__':
    data_dir = '/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/data'
    main_dataset = ComicsSeqFirebaseFaceBodyDataset(data_dir=data_dir,
                                                    transform_face=VisionTransformSetting.VANILLA_VAE_FACE.get_transformation(),
                                                    transform_body=VisionTransformSetting.VANILLA_VAE_FACE.get_transformation(),
                                                    img_folder_root_dir_face='/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops_large_faces/',
                                                    img_folder_root_dir_body='/datasets/COMICS/comics_crops/',
                                                    is_torch_transform=False,
                                                    is_train=False,
                                                    sequences_json_path='comics_seq/validation_sequences.json',
                                                    char_faces_json_path='comics_seq/val_char_faces.json',
                                                    char_bodies_json_path='comics_seq/val_char_bodies.json')
    dataset = ComicsSeqFirebaseFaceBodyQueryRefWrapper(main_dataset, is_query=True)
    dataset.evaluate(lambda x, y: torch.randn(len(x), 64))
    print(len(dataset))
