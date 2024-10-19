import os
from typing import Optional, List, Dict

import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
import cv2

from src.datamodules.components.vision_transform_setting import VisionTransformSetting

from src.models.pml_id_net_fine_tuned_ssl_backbone_face_body_module import PMLIdNetFineTunedSSLBackboneFaceBodyLitModule

from src.utils.character_identity.identity_assignment.comic_seq_firebase_face_body_seq_iter_dataset import \
    ComicsSeqFirebaseFaceBodySequenceIteratorDataset

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances

from src.utils.detection.bounding_box_utils import BOX_COLOR


class UnassignedCharacter:
    def __init__(self,
                 embedding: np.ndarray,
                 instance_id: str,
                 panel_id: Optional[str]):
        self.embedding = embedding
        self.instance_id = instance_id
        self.panel_id = panel_id


class AssignedCharacter:
    def __init__(self,
                 instance_id: str,
                 panel_id: Optional[str],
                 identity_id: str):
        self.instance_id = instance_id
        self.panel_id = panel_id
        self.identity_id = identity_id


"""
Under what conditions can we benefit from the Assigner...
1) (less likely) Image paths (body - face) are provided for each character, and then assignments are requested for these.
2) (likely for me) The `<mode>-sequences.json` file is read. 
   - Since the IDs for each character are known here, the image paths can be derived.
3) (likely) Panel images (in-sequence) are provided.
   - There are bounding boxes for each panel (face-body) from which assignments are made.
4) (MTL likely) A page image is provided -> panels-face-bodies -> character-bundles 
   -> from here, assignment to each character ID.
"""


class CharacterIdentityAssigner:
    # TODO: @gsoykan - handle paths
    @staticmethod
    def assign_from_sequences_json(visualize: bool = True):
        if visualize:
            data_dir = "/scratch/users/gsoykan20/projects/amazing-mysteries-of-gutter-demystified/data/"
            face_body_csv_path = os.path.join(data_dir, 'ssl/merged_face_body.csv')
            face_body_index_df = pd.read_csv(face_body_csv_path)
            by_series_and_page = face_body_index_df.groupby(['series_id',
                                                             'page_id',
                                                             'type',
                                                             'index'])
        model_checkpoint = '/scratch/users/gsoykan20/projects/amazing-mysteries-of-gutter-demystified/logs/experiments/runs/pml_id_net_fine_tuned_sim_clr_backbone_face_body_model_search/2023-06-30_22-46-55/checkpoints/epoch_018.ckpt'  # v9 sum mix series BEST
        trained_model = PMLIdNetFineTunedSSLBackboneFaceBodyLitModule.checkpoint_to_eval(model_checkpoint)
        device = trained_model.device
        # device = 'cuda'

        data_dir = "/scratch/users/gsoykan20/projects/amazing-mysteries-of-gutter-demystified/data/"
        face_root = '/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops_large_faces/'
        body_root = '/datasets/COMICS/comics_crops/'
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
                                                                   img_folder_root_dir_face=face_root,
                                                                   img_folder_root_dir_body=body_root)
        count = 0
        for i, element in enumerate(dataset):
            face_mask = torch.tensor([0 if elem is None else 1 for elem in element['face_ids']]) \
                .to(device)
            body_mask = torch.tensor([0 if elem is None else 1 for elem in element['body_ids']]) \
                .to(device)
            face_batch = torch.stack(element['faces']).to(device)
            body_batch = torch.stack(element['bodies']).to(device)

            with torch.no_grad():
                embeddings = trained_model(face_batch, body_batch, face_mask, body_mask).cpu().numpy()
                # embeddings = torch.randn(len(face_batch), 256)

            if len(embeddings) == 1:
                continue

            if len(set(element['char_ids'])) < 2:
                continue

            print('euclidean')
            dis_mat = pairwise_distances(embeddings, embeddings, metric='euclidean', n_jobs=-1)
            print(dis_mat)
            print('cosine')
            dis_mat = pairwise_distances(embeddings, embeddings, metric='cosine', n_jobs=-1)
            print(dis_mat)
            # AgglomerativeClustering(n_clusters=None, distance_threshold=3, linkage='single',compute_distances=True, affinity='l2').fit(embeddings).labels_
            assigned_labels = CharacterIdentityAssigner.assign(embeddings)
            char_ids_and_labels = list(zip(element['char_ids'], assigned_labels))
            face_id_to_label = {}
            body_id_to_label = {}
            for c, f, b, label in zip(element['char_ids'],
                                      element['face_ids'],
                                      element['body_ids'],
                                      assigned_labels):
                if f is not None:
                    face_id_to_label[f] = label
                if b is not None:
                    body_id_to_label[b] = label

            print(char_ids_and_labels)
            count += 1
            if count % 25 == 0:
                breakpoint()
            if visualize:
                seq_id = element['seq_id']
                visualize_sequence_with_char_ids(seq_id,
                                                 face_id_to_label,
                                                 body_id_to_label,
                                                 by_series_and_page,
                                                 f'{str(count)}')

    @staticmethod
    def assign(embeddings: np.ndarray,
               algo: str = 'agglo',
               distance_threshold=0.82,
               linkage: str = 'average') -> List[int]:
        # DBSCAN(eps=4, min_samples=1).fit(embeddings.cpu().numpy()).labels_
        # AgglomerativeClustering(n_clusters=None, distance_threshold=4.0, linkage='ward', compute_distances=True).fit(embeddings.cpu().numpy()).labels_
        if algo == 'dbscan':
            # Create the DBSCAN object
            dbscan = DBSCAN(eps=distance_threshold, min_samples=1)

            # Fit the data to the DBSCAN model
            dbscan.fit(embeddings)

            # Access the labels assigned to each sample
            labels = dbscan.labels_

            # Access the number of clusters found
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            # Print the labels and number of clusters
            print("Sample Labels:", labels)
            print("Number of Clusters:", n_clusters)
        elif algo == 'agglo':
            # Perform hierarchical clustering
            clustering = AgglomerativeClustering(n_clusters=None,
                                                 distance_threshold=distance_threshold,
                                                 linkage=linkage,
                                                 compute_distances=True).fit(embeddings)
            # Retrieve the assigned labels
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            # Print the labels and number of clusters
            print("Sample Labels:", labels)
            print("Number of Clusters:", n_clusters)

        return labels


def visualize_sequence_with_char_ids(seq_id: str,
                                     face_id_to_char_id: Dict[str, int],
                                     body_id_to_char_id: Dict[str, int],
                                     by_series_and_page,
                                     img_name: Optional[str] = None
                                     ):
    # 1014_14_0 + 1014_14_1 + 1014_14_2 + 1014_14_3
    panel_ids = seq_id.split('+')
    panel_root = '/datasets/COMICS/raw_panel_images'

    num_rows = len(panel_ids)
    fig, axs = plt.subplots(1, num_rows, figsize=(8 * num_rows, 8))

    for panel_idx, (panel_id, ax) in enumerate(zip(panel_ids, axs)):
        # /datasets/COMICS/raw_panel_images/1371/45_6.jpg
        comic_no, page_no, panel_no = panel_id.split('_')
        subpanel = '_'.join([str(page_no), str(panel_no)])
        img_path = os.path.join(panel_root, str(comic_no), f'{subpanel}.jpg')

        panel_image = cv2.imread(img_path)
        ax.imshow(cv2.cvtColor(panel_image, cv2.COLOR_BGR2RGB))

        # find this panels faces and bodies
        panels_faces = {}
        for k, v in face_id_to_char_id.items():
            item_comic_no, item_page_no, item_panel_no = k.split('_')[:3]
            if (comic_no, page_no, panel_no) == (item_comic_no, item_page_no, item_panel_no):
                panels_faces[k] = v

        panels_bodies = {}
        for k, v in body_id_to_char_id.items():
            item_comic_no, item_page_no, item_panel_no = k.split('_')[:3]
            if (comic_no, page_no, panel_no) == (item_comic_no, item_page_no, item_panel_no):
                panels_bodies[k] = v

        bbs = []
        category_ids = []
        # get items bounding boxes...
        for k, v in panels_faces.items():
            item_id = k.split('_')[-1]
            rows = by_series_and_page.get_group((int(comic_no),
                                                 f'{page_no}_{panel_no}',
                                                 'face',
                                                 int(item_id)))
            rows = rows.to_dict('records')
            if len(rows) == 1:
                row = rows[0]
                bb = [row['x_0'], row['y_0'], row['x1'], row['y_1']]
                bbs.append(bb)
                category_ids.append(v)

        for k, v in panels_bodies.items():
            item_id = k.split('_')[-1]
            rows = by_series_and_page.get_group((int(comic_no),
                                                 f'{page_no}_{panel_no}',
                                                 'body',
                                                 int(item_id)))
            rows = rows.to_dict('records')
            if len(rows) == 1:
                row = rows[0]
                bb = [row['x_0'], row['y_0'], row['x1'], row['y_1']]
                bbs.append(bb)
                category_ids.append(v)

        for bbox, category in zip(bbs, category_ids):
            # Extract the coordinates of the bounding box
            x_min, y_min, x_max, y_max = bbox
            # Draw the bounding box on the panel image
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=6,
                                 edgecolor=list(map(lambda x: x / 255, BOX_COLOR[category])), facecolor='none')
            ax.add_patch(rect)

            # Add the category label near the bounding box
            ax.text(x_min, y_min - 10, category, fontsize=18, color=list(map(lambda x: x / 255, BOX_COLOR[category])),
                    bbox=dict(facecolor='white', edgecolor='white', alpha=0.7))
        # Remove axis ticks and labels
        ax.axis('off')

    # Adjust the layout and spacing of the subplots
    fig.tight_layout()
    # Save the combined image
    if img_name == None:
        plt.savefig('combined_image.png')
    else:
        plt.savefig(f'{img_name}.png')
    # Show the combined image
    plt.show()


if __name__ == '__main__':
    CharacterIdentityAssigner.assign_from_sequences_json()
