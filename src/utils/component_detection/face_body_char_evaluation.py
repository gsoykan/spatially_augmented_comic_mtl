import torch
from itertools import groupby
from typing import Optional, Any

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torchvision.ops import boxes as box_ops

from src.utils.basic_utils import bbox_area
from src.utils.ssl.create_ssl_dataset_file_body import box_intersection_rate


class ComicPageFaceBodyCharEvaluator:
    def __init__(self,
                 box_score_threshold: Optional[float] = None):
        self.body_face_preds = []
        self.body_face_gt_labels = []
        self.box_score_threshold = box_score_threshold

    def reset(self):
        self.body_face_preds = []
        self.body_face_gt_labels = []

    def compute(self):
        def get_res(labels, preds):
            results = {
                'precision': precision_score(labels, preds),
                'recall': recall_score(labels, preds),
                'accuracy': accuracy_score(labels, preds),
                'f1': f1_score(labels, preds)
            }
            return results

        body_face_res = get_res(self.body_face_gt_labels, self.body_face_preds)
        body_face_res = {'body_face/' + k: v for k, v in body_face_res.items()}
        return {**body_face_res}

    def step(self, targets, detections):
        for i in range(len(targets)):
            target_boxes = targets[i]['boxes']
            target_face_to_char_links = targets[i]['face_to_char_links']

            chars = ComicPageFaceBodyCharEvaluator.pair_face_body_to_form_char(
                detections[i]['boxes'],
                detections[i]['labels'],
                detections[i]['scores'],
                box_score_threshold=self.box_score_threshold)

            acc_dict = ComicPageFaceBodyCharEvaluator.measure(chars, target_face_to_char_links, target_boxes)
            body_face_gt_labels = list(map(lambda x: x[2], acc_dict['body_face']))
            body_face_preds = list(map(lambda x: x[3], acc_dict['body_face']))

            self.body_face_gt_labels.extend(body_face_gt_labels)
            self.body_face_preds.extend(body_face_preds)

    @staticmethod
    def measure(chars, target_face_to_char_links, target_boxes, box_iou_threshold=0.5):
        # TODO: @gsoykan - currently works only for body_face
        acc_dict = {
            'just_body': [],
            'just_face': [],
            'body_face': []
        }

        face_body_chars = list(filter(lambda x: None not in [x['face'], x['body']], chars))

        matched_char_idxs = []
        for face_to_char in target_face_to_char_links:
            gt_face_box = target_boxes[face_to_char[0]]
            gt_body_box = target_boxes[face_to_char[1]]
            found_pair = False
            for char_idx, char in enumerate(face_body_chars):
                pred_face_box = char['face']
                pred_body_box = char['body']
                face_box_iou = box_ops.box_iou(gt_face_box.view(1, -1), pred_face_box.view(1, -1)).item()
                body_box_iou = box_ops.box_iou(gt_body_box.view(1, -1), pred_body_box.view(1, -1)).item()
                if face_box_iou >= box_iou_threshold and body_box_iou >= box_iou_threshold:
                    acc_dict['body_face'].append([(gt_face_box, gt_body_box), (pred_face_box, pred_body_box), 1, 1])
                    found_pair = True
                    matched_char_idxs.append(char_idx)
                    break
            if not found_pair:
                acc_dict['body_face'].append([(gt_face_box, gt_body_box), None, 1, 0])

        unmatched_face_body_char_idxs = set(range(len(face_body_chars))) - set(matched_char_idxs)
        for idx in unmatched_face_body_char_idxs:
            acc_dict['body_face'].append([None, face_body_chars[idx], 0, 1])

        return acc_dict

    @staticmethod
    def pair_face_body_to_form_char(pred_boxes,
                                    pred_labels,
                                    pred_box_scores,
                                    face_label=2,
                                    body_label=1,
                                    box_score_threshold=0.5,
                                    alternative_box_idxs: Optional[np.ndarray] = None):
        if alternative_box_idxs is not None:
            pred_box_idxs = torch.tensor(alternative_box_idxs, device=pred_boxes.device)
        else:
            pred_box_idxs = torch.tensor(np.array(list(range(len(pred_boxes)))), device=pred_boxes.device)

        if box_score_threshold is not None:
            pred_boxes = pred_boxes[pred_box_scores >= box_score_threshold]
            pred_labels = pred_labels[pred_box_scores >= box_score_threshold]
            pred_box_idxs = pred_box_idxs[pred_box_scores >= box_score_threshold]
            pred_box_scores = pred_box_scores[pred_box_scores >= box_score_threshold]

        body_box_scores = pred_box_scores[pred_labels == body_label]
        body_boxes = pred_boxes[pred_labels == body_label]
        body_idxs = pred_box_idxs[pred_labels == body_label]

        face_box_scores = pred_box_scores[pred_labels == face_label]
        face_boxes = pred_boxes[pred_labels == face_label]
        face_idxs = pred_box_idxs[pred_labels == face_label]

        chars = []
        used_body_indices = []

        def append_to_chars(face_box, body_box, face_idx, body_idx):
            chars.append({
                'face': face_box,
                'face_idx': face_idx,
                'body': body_box,
                'body_idx': body_idx
            })

        for _, (face_box, face_idx) in enumerate(
                sorted(zip(face_boxes, face_idxs), key=lambda x: x[0][1].item(), reverse=False)):
            candidates = []

            for i_b, (body_box, body_idx) in enumerate(zip(body_boxes, body_idxs)):
                if i_b in used_body_indices:
                    continue
                intersection_rate = box_intersection_rate(face_box.tolist(), body_box.tolist())
                if intersection_rate >= 0.20:
                    # center_distance = box_to_box_center_distance(face_box, body_box)
                    # bbox_area(face_box)
                    candidates.append([i_b, intersection_rate,  -1 * face_box[1].item()])  # -1 * center_distance.item()])

            # print('number of candidates => ', len(candidates))
            # if len(candidates) > 1:
            #     print(candidates)
            if len(candidates) != 0:
                # if len(candidates) > 1:
                #     print(candidates)
                # saving chars with face & body
                candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
                body_box_index = candidates[0][0]
                body_box = body_boxes[body_box_index]
                body_box_id = body_idxs[body_box_index]
                used_body_indices.append(body_box_index)
                append_to_chars(face_box, body_box, face_idx.item(), body_box_id.item())
            else:
                # saving bodiless faces
                append_to_chars(face_box, None, face_idx.item(), None)
        # saving faceless bodies
        for i_b, (body_box, body_idx) in enumerate(zip(body_boxes, body_idxs)):
            if i_b not in used_body_indices:
                append_to_chars(None, body_box, None, body_idx.item())

        return chars


def box_to_box_center_distance(box1, box2):
    # Calculate the center coordinates of each box
    center1_x = (box1[0] + box1[2]) / 2  # Scalar
    center1_y = (box1[1] + box1[3]) / 2  # Scalar
    center2_x = (box2[0] + box2[2]) / 2  # Scalar
    center2_y = (box2[1] + box2[3]) / 2  # Scalar

    # Calculate the absolute center-to-center distance between the boxes
    center_distance = torch.sqrt(torch.pow(center1_x - center2_x, 2) + torch.pow(center1_y - center2_y, 2))  # Scalar
    return center_distance
