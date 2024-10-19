import itertools
import json
import os
import random
from typing import Optional, Dict, Tuple, Set, List
import networkx as nx

import albumentations as A
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datamodules.components.vision_transform_setting import VisionTransformSetting
from src.utils.basic_utils import read_or_get_image
from tqdm import tqdm


class ComicsSeqFirebaseFaceBodyDataset(Dataset):
    def __init__(self,
                 transform_face: A.Compose,
                 transform_body: A.Compose,
                 data_dir: str,
                 # path from datadir...
                 sequences_json_path: str = 'comics_seq/train_sequences.json',
                 char_faces_json_path: str = 'comics_seq/train_char_faces.json',
                 char_bodies_json_path: str = 'comics_seq/train_char_bodies.json',
                 img_folder_root_dir_face: str = '/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops_large_faces/',
                 img_folder_root_dir_body: str = '/datasets/COMICS/comics_crops/',
                 lower_idx_bound: Optional[int] = None,
                 higher_idx_bound: Optional[int] = None,
                 is_torch_transform: bool = False,
                 is_train: bool = True,
                 randomly_mask_face_or_body: Optional[float] = 0,
                 # this means using char's that do not have 'non_other' chars
                 # yani sequence içinde negative örnek olabilecek char ı olmayanlar...
                 # only enabled in training -> çünkü gerçek bir evaluation yaratmıcak
                 # val ve test için...
                 use_singular_chars: bool = False
                 ):
        self.use_singular_chars = use_singular_chars
        self.randomly_mask_face_or_body = randomly_mask_face_or_body
        self.lower_idx_bound = lower_idx_bound
        self.higher_idx_bound = higher_idx_bound
        self.sequences_json_path = sequences_json_path
        self.char_faces_json_path = char_faces_json_path
        self.char_bodies_json_path = char_bodies_json_path
        self.img_folder_root_dir_face = img_folder_root_dir_face
        self.img_folder_root_dir_body = img_folder_root_dir_body
        self.data_dir = data_dir
        self.is_torch_transform = is_torch_transform
        self.transform_face = transform_face
        self.transform_body = transform_body
        self.dataset, \
        self.dataset_dict, \
        self.dataset_char_dict, \
        self.char_id_to_certain_other_char_ids = self.load_dataset()
        # set connectivity graph
        self.conn_graph = nx.Graph()
        for char_id, suitable_chars in self.char_id_to_certain_other_char_ids.items():
            self.conn_graph.add_node(char_id)
            for suitable_char in suitable_chars:
                self.conn_graph.add_edge(char_id, suitable_char)
        self.sorted_dataset_keys = list(sorted(self.dataset_dict.keys()))

        self.seq_id_to_char_combos = {}
        for seq_id in tqdm(self.sorted_dataset_keys, desc='finding suitable char combos...'):
            seq_chars = self.dataset_dict[seq_id]
            char_combos = self._find_suitable_char_combinations(seq_chars)
            self.seq_id_to_char_combos[seq_id] = char_combos

        self.is_train = is_train
        if not is_train:
            self.queries, self.references = self.create_query_and_reference_datasets()

    def create_query_and_reference_datasets(self):
        """
        # 347 / 941 min >1
        # 346 / 688 min >1 + 1 ref
        # 96 / 327 min >3 (gayet makul sayılar...) (trainden çıkmış)
        # 69 / 257 min >1 + 1 ref (test)
        # 72 / 240 min >1 + 1 ref (val)
        # new val => 128, 392
        # test için =>
        134 tane seri var
        303 farklı character
        queries: 126
        references: 361
        126 query için 184 tane aynı character id ye ait referans var, 361 - 184 te
            tek instance i olan characterler ve ref datasetini daha challenging yapmak için varlar...
        Returns:

        """
        queries = []
        references = []
        added_q_r_len = []
        char_id_counter = 0

        for series_id, char_instances in self.dataset_dict.items():
            # this may not be deterministic...
            suitable_char_combinations = self.seq_id_to_char_combos[series_id]
            # find each suitable char combinations total instance count pick the most
            best_combo_len = 0
            best_combo = None
            for suitable_combo in suitable_char_combinations:
                total_char_len = sum(map(lambda x: len(self.dataset_char_dict[x]), suitable_combo))
                if total_char_len > best_combo_len:
                    best_combo_len = total_char_len
                    best_combo = suitable_combo

            if best_combo is not None:
                for char_id in best_combo:
                    char_id_counter += 1
                    char_ids_all_instances_list = self.dataset_char_dict[char_id]
                    if len(char_ids_all_instances_list) > 1:
                        queries.append(char_ids_all_instances_list[0])
                        references.extend(char_ids_all_instances_list[1:])
                        added_q_r_len.append([1, len(char_ids_all_instances_list[1:])])
                    else:
                        references.extend(char_ids_all_instances_list)
                        added_q_r_len.append([0, len(char_ids_all_instances_list)])

        return queries, references

    def _create_item_id_to_char_id_dict(self, sequences_json: Dict) -> Tuple[Dict, Dict, Dict]:
        # given face returns body id
        d_f = {}
        # given body returns face id
        d_b = {}
        d = {}
        for element in sequences_json:
            annotations = element['annotations']
            for annotation in annotations:
                char_id = annotation['charId']
                for char_instance in annotation['charInstances']:
                    face_instance = char_instance.get('face')
                    body_instance = char_instance.get('body')
                    if face_instance is not None:
                        d[face_instance] = char_id
                        d_f[face_instance] = body_instance
                    if body_instance is not None:
                        d[body_instance] = char_id
                        d_b[body_instance] = face_instance
        return d, d_f, d_b

    def _create_char_id_to_certain_other_char_ids(self,
                                                  sequences_json: Dict,
                                                  body_id_to_actual_char_id: Dict,
                                                  face_id_to_actual_char_id: Dict
                                                  ) -> Dict[str, Set[str]]:
        char_id_to_certain_other_char_ids: Dict[str, Set[str]] = {}
        for element in sequences_json:
            annotations = element['annotations']
            actual_char_ids = set()
            for annotation in annotations:
                for char_instance in annotation['charInstances']:
                    body_instance = char_instance.get('body')
                    face_instance = char_instance.get('face')
                    if body_instance is not None:
                        actual_char_id = body_id_to_actual_char_id.get(body_instance, None)
                        if actual_char_id is None:
                            continue
                        actual_char_ids.add(actual_char_id)
                    elif face_instance is not None:
                        actual_char_id = face_id_to_actual_char_id.get(face_instance, None)
                        if actual_char_id is None:
                            continue
                        actual_char_ids.add(actual_char_id)
            for char_id in actual_char_ids:
                curr_set = char_id_to_certain_other_char_ids.get(char_id, set())
                curr_set = curr_set.union(actual_char_ids - {char_id})
                char_id_to_certain_other_char_ids[char_id] = curr_set

        return char_id_to_certain_other_char_ids

    def load_dataset(self):
        with open(os.path.join(self.data_dir, self.sequences_json_path), 'r') as file:
            sequences_json = json.load(file)

        with open(os.path.join(self.data_dir, self.char_faces_json_path), 'r') as file:
            char_faces_json = json.load(file)

        with open(os.path.join(self.data_dir, self.char_bodies_json_path), 'r') as file:
            char_bodies_json = json.load(file)

        instance_to_char_id, face_to_body, body_to_face = self._create_item_id_to_char_id_dict(sequences_json)

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

        pre_dataset = []
        body_id_to_actual_char_id = {}
        face_id_to_actual_char_id = {}

        # append bodies and their faces to pre_dataset
        for char_id, body_ids in char_bodies_json.items():
            for body_id in body_ids:
                face_id = body_to_face.get(body_id)
                body_id_to_actual_char_id[body_id] = char_id
                pre_dataset.append((char_id, face_id, body_id))
                if face_id is not None:
                    face_id_to_actual_char_id[face_id] = char_id
        # append faces and their bodies to pre_dataset
        for char_id, face_ids in char_faces_json.items():
            for face_id in face_ids:
                body_id = face_to_body.get(face_id)
                face_id_to_actual_char_id[body_id] = char_id
                pre_dataset.append((char_id, face_id, body_id))
                if body_id is not None:
                    body_id_to_actual_char_id[body_id] = char_id
        pre_dataset = list({*pre_dataset})

        char_id_to_certain_other_char_ids: Dict[str, Set[str]] = self._create_char_id_to_certain_other_char_ids(
            sequences_json, body_id_to_actual_char_id, face_id_to_actual_char_id)

        non_other_count = 0
        non_other_char_dataset = []
        dataset = []
        for char_id, face_id, body_id in pre_dataset:

            seq_id = None
            if face_id is not None:
                face_path = construct_img_path(face_id,
                                               False,
                                               self.img_folder_root_dir_face)
                seq_id = face_id.split('_')[0]
            else:
                face_path = None

            if body_id is not None:
                body_path = construct_img_path(body_id,
                                               True,
                                               self.img_folder_root_dir_body)
                seq_id = body_id.split('_')[0]
            else:
                body_path = None

            if seq_id is None:
                raise Exception('seq_id should be available...')

            other_char_ids = char_id_to_certain_other_char_ids.get(char_id, set())
            if (len(other_char_ids)) == 0:
                non_other_count += 1
                if char_id in char_id_to_certain_other_char_ids:
                    del char_id_to_certain_other_char_ids[char_id]
                if self.use_singular_chars:
                    # Those will be added down to 'dataset'
                    non_other_char_dataset.append((char_id, face_path, body_path, seq_id))
                continue

            dataset.append((char_id, face_path, body_path, seq_id))

        # add non_other_char_dataset to dataset
        # and match them with next sequence's character id's and vice versa
        # then move to next pair...
        added_non_other_count = 0
        if self.use_singular_chars:
            non_other_char_dataset_group = itertools.groupby(sorted(non_other_char_dataset, key=lambda x: x[3]),
                                                             lambda x: x[3])
            non_other_char_dataset_dict = {}
            for key, group in non_other_char_dataset_group:
                non_other_char_dataset_dict[key] = list(group)
            # sorted descending...
            sorted_non_other_keys = sorted(non_other_char_dataset_dict.keys(),
                                           key=lambda k: len(non_other_char_dataset_dict[k]), reverse=True)
            for i in range(0, len(sorted_non_other_keys), 2):
                j = i + 1
                # break at the odd number of seq's
                if j == len(sorted_non_other_keys):
                    break
                seq_id = sorted_non_other_keys[i]
                next_seq_id = sorted_non_other_keys[j]
                char_instances = non_other_char_dataset_dict[seq_id]
                curr_char_ids = list(map(lambda x: x[0], char_instances))
                other_char_instances = non_other_char_dataset_dict[next_seq_id]
                other_curr_char_ids = list(map(lambda x: x[0], other_char_instances))
                min_len = min(len(curr_char_ids), len(other_curr_char_ids))
                for i in range(min_len):
                    curr_char_id = curr_char_ids[i]
                    other_curr_char_id = other_curr_char_ids[i]
                    char_id_to_certain_other_char_ids[curr_char_id] = {other_curr_char_id}
                    char_id_to_certain_other_char_ids[other_curr_char_id] = {curr_char_id}
                    curr_char_instances = list(filter(lambda x: x[0] == curr_char_id, char_instances))
                    dataset.extend(curr_char_instances)
                    other_curr_char_instances = list(filter(lambda x: x[0] == other_curr_char_id, other_char_instances))
                    dataset.extend(other_curr_char_instances)
                    added_non_other_count += 2

        print('element count that has no other', non_other_count)
        print('added has no other count', added_non_other_count)

        dataset = list({*dataset})
        if self.lower_idx_bound is not None and self.higher_idx_bound is None:
            dataset = dataset[self.lower_idx_bound:]
        elif self.higher_idx_bound is not None and self.lower_idx_bound is None:
            dataset = dataset[:self.higher_idx_bound]
        elif self.higher_idx_bound is not None and self.lower_idx_bound is not None:
            dataset = dataset[self.lower_idx_bound:self.higher_idx_bound]

        dataset_group = itertools.groupby(sorted(dataset, key=lambda x: x[3]), lambda x: x[3])
        dataset_dict = {}
        for key, group in dataset_group:
            dataset_dict[key] = list(group)

        dataset_char_group = itertools.groupby(sorted(dataset, key=lambda x: x[0]), lambda x: x[0])
        dataset_char_dict = {}
        for key, group in dataset_char_group:
            dataset_char_dict[key] = list(group)

        return dataset, dataset_dict, dataset_char_dict, char_id_to_certain_other_char_ids

    def _find_suitable_char_combinations(self,
                                         seq_chars: List[Tuple]) -> List[Tuple]:

        seq_chars = list(set(map(lambda x: x[0], seq_chars)))

        combinations = set()

        for start_char in seq_chars:
            stack = [(start_char, [start_char])]
            same_count = 0
            while stack:
                current_char, combination = stack.pop()

                if set(combination) in list(combinations):
                    # TODO: @gsoykan - fix this later...
                    continue
                    same_count += 1
                    if same_count > 100000:
                        continue

                curr_neighbors = list(self.conn_graph.neighbors(current_char))

                if len(combination) >= len(curr_neighbors) + 1:
                    combinations.add(frozenset(sorted(combination)))
                    continue
                elif len(combination) > 1:
                    combinations.add(frozenset(sorted(combination)))

                for next_char in curr_neighbors:
                    # if all(neigh in graph.neighbors(next_char) for neigh in combination) and next_char not in combination:
                    # if all(neighbour in seq_chars for neighbour in graph.neighbors(next_char)):
                    if all(comb_el in self.conn_graph.neighbors(next_char) for comb_el in combination):
                        if next_char not in combination:
                            cand = sorted(combination + [next_char])
                            # if set(cand) not in combinations:
                            # if set(cand) not in list(combinations):
                            # bunu ekledikten sonra stack başlangıcında yapsak durum nasıl olacak...
                            stack.append((next_char, cand))

        combinations = self.eliminate_subset_sets(list(combinations))

        return combinations

    def eliminate_subset_sets(self, sets: List[Set[str]]):
        # Sort the sets by length in ascending order
        sorted_sets = sorted(sets, key=len, reverse=False)

        result = []
        while sorted_sets:
            current_set = sorted_sets.pop(0)
            is_subset = False

            # Check if the current set is a subset of any remaining sets
            for other_set in sorted_sets:
                if current_set.issubset(other_set):
                    is_subset = True
                    break

            # If the current set is not a subset, add it to the result
            if not is_subset:
                result.append(sorted(current_set))

        return sorted(result)

    def __getitem__(self, index):
        def get_image_data(face_path: Optional[str],
                           body_path: Optional[str]):
            if face_path is None:
                anchor_img_face = torch.randn(3, 96, 96)
            if body_path is None:
                anchor_img_body = torch.randn(3, 128, 128)

            if self.is_torch_transform:
                if face_path is not None:
                    anchor_img_face = Image.open(face_path)
                    anchor_img_face = self.transform_face(anchor_img_face)
                if body_path is not None:
                    anchor_img_body = Image.open(body_path)
                    anchor_img_body = self.transform_body(anchor_img_body)
            else:
                # # TODO: @gsoykan - remove
                # return torch.randn(3, 96, 96), torch.randn(3, 128, 128)
                if face_path is not None:
                    anchor_img_face = read_or_get_image(face_path, read_rgb=True)
                    anchor_img_face = self.transform_face(image=anchor_img_face)['image']
                if body_path is not None:
                    anchor_img_body = read_or_get_image(body_path, read_rgb=True)
                    anchor_img_body = self.transform_body(image=anchor_img_body)['image']

            return anchor_img_face, anchor_img_body

        selected_seq_id = self.sorted_dataset_keys[index]
        mutually_inclusive_char_combos = self.seq_id_to_char_combos[selected_seq_id]
        # list(map(lambda a: a[0] == a[1], zip(mutually_inclusive_char_combos, alt)))
        faces = []
        face_mask = []
        bodies = []
        body_mask = []
        char_ids = []
        seq_ids = []
        original_seq_ids = []

        for i, char_combos in enumerate(mutually_inclusive_char_combos):
            selected_chars = []
            for selected_char_id in char_combos:
                selected_chars.extend(self.dataset_char_dict[selected_char_id])

            for char_id, face_path, body_path, seq_id in selected_chars:
                img_face, img_body = get_image_data(face_path, body_path)
                face_mask_value = face_path is not None
                body_mask_value = body_path is not None
                if self.randomly_mask_face_or_body != 0:
                    if face_mask_value and body_mask_value:
                        none_weight = 1 - self.randomly_mask_face_or_body
                        option_weight = self.randomly_mask_face_or_body // 2
                        weights = [option_weight, option_weight, none_weight]
                        selected_option = random.choices(['face', 'body', None],
                                                         weights=weights,
                                                         k=1)[0]
                        if selected_option == 'face':
                            face_mask_value = False
                        elif selected_option == 'body':
                            body_mask_value = False

                faces.append(img_face)
                face_mask.append(face_mask_value)
                bodies.append(img_body)
                body_mask.append(body_mask_value)
                char_ids.append(char_id)
                # we need to consider each uniq combo as a different series
                # so that we can take correct triplets
                seq_ids.append(seq_id + f'+{str(i)}')
                original_seq_ids.append(seq_id)

        return {
            'face': faces,
            'face_mask': face_mask,
            'body': bodies,
            'body_mask': body_mask,
            'char_id': char_ids,
            'seq_id': seq_ids,
            'original_seq_id': original_seq_ids
        }

    def __len__(self):
        return len(self.sorted_dataset_keys)


def comics_seq_firebase_face_body_collate_fn(batch,
                                             char_id_to_certain_other_char_ids: Optional[Dict[str, Set[str]]] = None):
    faces = []
    face_mask = []
    bodies = []
    body_mask = []
    char_ids = []
    seq_ids = []  # seq_ids + char_combo_idx
    original_seq_ids = []

    for sample in batch:
        if isinstance(sample['char_id'], list):
            faces.extend(sample['face'])
            face_mask.extend(sample['face_mask'])
            bodies.extend(sample['body'])
            body_mask.extend(sample['body_mask'])
            char_ids.extend(sample['char_id'])
            seq_ids.extend(sample['seq_id'])
            original_seq_ids.extend(sample['original_seq_id'])
        else:
            faces.append(sample['face'])
            face_mask.append(sample['face_mask'])
            bodies.append(sample['body'])
            body_mask.append(sample['body_mask'])
            char_ids.append(sample['char_id'])
            seq_ids.append(sample['seq_id'])
            original_seq_ids.append(sample['original_seq_id'])

    # Convert faces and bodies to tensors
    faces = torch.stack([face for face in faces])
    bodies = torch.stack([body for body in bodies])

    def convert_ids_to_indices(ids):
        id_to_idx = {}
        idx = 0
        for id in ids:
            if id not in id_to_idx:
                id_to_idx[id] = idx
                idx += 1
        # Convert char_ids to indices
        return [id_to_idx[id] for id in ids]

    # Convert char_ids to indices
    char_indices = convert_ids_to_indices(char_ids)
    seq_indices = convert_ids_to_indices(seq_ids)
    original_seq_ids = convert_ids_to_indices(original_seq_ids)

    char_ids = torch.tensor(char_indices)
    seq_ids = torch.tensor(seq_indices)
    original_seq_ids = torch.tensor(original_seq_ids)

    return {'faces': faces,
            'face_mask': torch.tensor(face_mask),
            'bodies': bodies,
            'body_mask': torch.tensor(body_mask),
            'char_ids': char_ids,
            'seq_ids': seq_ids,
            'original_seq_ids': original_seq_ids}


if __name__ == '__main__':
    data_dir = '/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/data'
    dataset = ComicsSeqFirebaseFaceBodyDataset(data_dir=data_dir,
                                               transform_face=VisionTransformSetting.VANILLA_VAE_FACE.get_transformation(),
                                               transform_body=VisionTransformSetting.VANILLA_VAE_FACE.get_transformation(),
                                               sequences_json_path='comics_seq/test_sequences.json',
                                               char_faces_json_path='comics_seq/test_char_faces.json',
                                               char_bodies_json_path='comics_seq/test_char_bodies.json',
                                               img_folder_root_dir_face='/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops_large_faces/',
                                               img_folder_root_dir_body='/datasets/COMICS/comics_crops/',
                                               is_torch_transform=False,
                                               is_train=False)
    print(len(dataset))
