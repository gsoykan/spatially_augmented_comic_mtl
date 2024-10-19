from itertools import groupby
from typing import Optional, Any, List, Dict

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from torchvision.ops import boxes as box_ops

from src.utils.component_detection.face_body_char_evaluation import ComicPageFaceBodyCharEvaluator


def get_selected_relations(pred_relations,
                           pred_labels,
                           pred_box_scores: Optional[Any] = None,
                           score_threshold: Optional[float] = None,
                           box_score_threshold: Optional[float] = None,
                           chars: Optional[List[Dict[str, Any]]] = None,
                           pred_box_indices: Optional[np.ndarray] = None,
                           # a function to filter possible rels, possibly within panel options...
                           process_possible_rels: Optional[Any] = None):
    if score_threshold is not None:
        pred_relations = list(filter(lambda x: x[2] > score_threshold, pred_relations))

    if box_score_threshold is not None:
        filtered_pred_relations = []
        for pred_relation in pred_relations:
            from_idx = pred_relation[0]
            to_idx = pred_relation[1]

            if pred_box_indices is not None:
                if pred_box_scores[pred_box_indices == from_idx] >= box_score_threshold and pred_box_scores[
                    pred_box_indices == to_idx] >= box_score_threshold:
                    filtered_pred_relations.append(pred_relation)
            else:
                if pred_box_scores[from_idx] >= box_score_threshold and pred_box_scores[to_idx] >= box_score_threshold:
                    filtered_pred_relations.append(pred_relation)
        pred_relations = filtered_pred_relations

    # augmenting relations with face-body pairs...
    # face-body alignment diyebiliriz bu kısma...
    # character consistency post-processing step diyebiliriz..
    if chars is not None:
        def assign_relations_to_chars(pred_relations, assigned_char_element_ids: Optional[set] = None):
            selected_face_relations = []
            selected_body_relations = []
            for char in chars:
                face_idx = char['face_idx']
                possible_face_rels = []
                if face_idx is not None:
                    possible_face_rels = list(filter(lambda x: x[1] == face_idx, pred_relations))
                body_idx = char['body_idx']
                possible_body_rels = []
                if body_idx is not None:
                    possible_body_rels = list(filter(lambda x: x[1] == body_idx, pred_relations))
                possible_rels = [*possible_body_rels, *possible_face_rels]

                if process_possible_rels is not None:
                    # print('------------')
                    # print(possible_rels, face_idx, body_idx)
                    possible_rels = process_possible_rels(face_idx, body_idx, possible_rels)

                relations_by_from_pred_box_idx = groupby(sorted(possible_rels, key=lambda x: x[0]),
                                                         key=lambda x: x[0])

                best_relations_score = 0
                best_relations = None
                for from_box_idx, relations in relations_by_from_pred_box_idx:
                    relations = list(relations)

                    # penalize already assigned char element ids. 0.35 puan düşür...
                    penalty = 0
                    if assigned_char_element_ids is not None:
                        to_idxs = set(map(lambda x: x[1], relations))
                        intersection_count = len(assigned_char_element_ids.intersection(to_idxs))
                        if intersection_count:
                            # print('penalized ---', relations)
                            penalty = 0.35

                    mean_rel_score = max(list(map(lambda x: x[2], relations))) - penalty
                    if mean_rel_score > best_relations_score:
                        best_relations = relations
                        best_relations_score = mean_rel_score

                if best_relations is not None:
                    # add constraint for a speech bubble to belong a single char
                    from_idx = best_relations[0][0]
                    existing_body_relations = [x for x in selected_body_relations if x[0] == from_idx]
                    existing_face_relations = [x for x in selected_face_relations if x[0] == from_idx]
                    existing_relations = [*existing_body_relations, *existing_face_relations]
                    if len(existing_relations) == 0:
                        existing_rel_score = 0
                    else:
                        # we can try mean - max - min etc...
                        existing_rel_score = max(list(map(lambda x: x[2], existing_relations)))
                    # if the new pair of relations has higher score, then replace them.
                    # if not do nothing and keep them
                    if best_relations_score > existing_rel_score:
                        for e in existing_face_relations:
                            selected_face_relations.remove(e)
                        for e in existing_body_relations:
                            selected_body_relations.remove(e)
                        for rel in best_relations:
                            from_idx = rel[0]
                            to_idx = rel[1]

                            if pred_box_indices is not None:
                                to_label = pred_labels[pred_box_indices == to_idx]
                            else:
                                to_label = pred_labels[to_idx]

                            if to_label == 1:
                                selected_body_relations.append(rel)
                            elif to_label == 2:
                                selected_face_relations.append(rel)

            return selected_face_relations, selected_body_relations

        selected_face_relations, selected_body_relations = assign_relations_to_chars(pred_relations)

        # handling cases where a character has 2 speech bubbles...
        # eğer açıkta bir speech bubble kaldıysa onu da var olan charlardan birine iteleyebiliriz...
        selected_relations = [*selected_face_relations, *selected_body_relations]
        selected_speech_bubble_idxs = set(map(lambda x: x[0], selected_relations))
        remaining_pred_relations = list(filter(lambda x: x[0] not in selected_speech_bubble_idxs, pred_relations))
        assigned_char_element_ids = list(map(lambda x: x[1], selected_relations))
        add_selected_face_relations, add_selected_body_relations = assign_relations_to_chars(
            remaining_pred_relations, set(assigned_char_element_ids))

        return [*add_selected_face_relations, *selected_face_relations], [*add_selected_body_relations,
                                                                          *selected_body_relations]

    # make this options if chars is None
    # group relations by speech bubble and pick best for body and face
    relations_by_from_pred_box_idx = groupby(sorted(pred_relations, key=lambda x: x[0]), key=lambda x: x[0])
    selected_face_relations = []
    selected_body_relations = []
    for from_box_idx, relations in relations_by_from_pred_box_idx:
        relations = list(relations)
        relations_by_to_box_label = groupby(sorted(relations, key=lambda x: pred_labels[x[1]]),
                                            key=lambda x: pred_labels[x[1]])
        for to_label, relations in relations_by_to_box_label:
            relations = list(relations)
            max_score_idx = np.array(relations)[:, 2].argmax()
            selected_relation_for_label = relations[max_score_idx]
            if to_label == 1:
                selected_body_relations.append(selected_relation_for_label)
            elif to_label == 2:
                selected_face_relations.append(selected_relation_for_label)
            else:
                raise Exception(f'to label can only be 1 or 2, here it is {str(to_label)}')
    return selected_face_relations, selected_body_relations


def relations_to_labels(gt_links, gt_boxes,
                        pred_boxes, pred_relations, pred_labels, pred_box_scores, box_iou_threshold=0.6,
                        include_not_found_gt_relations: bool = False,
                        include_unmatched_body_relation_idxs: bool = False,
                        score_threshold: Optional[float] = None,
                        box_score_threshold: Optional[float] = None,
                        use_char_alignment: bool = False):
    chars = None
    if use_char_alignment:
        chars = ComicPageFaceBodyCharEvaluator.pair_face_body_to_form_char(
            pred_boxes,
            pred_labels,
            pred_box_scores,
            box_score_threshold=box_score_threshold)
    # group relations by speech bubble and pick best for body and face
    selected_face_relations, selected_body_relations = get_selected_relations(pred_relations,
                                                                              pred_labels,
                                                                              pred_box_scores,
                                                                              score_threshold,
                                                                              box_score_threshold,
                                                                              chars)

    # group links by target label
    # Opt[gt_link], Opt[relation], gt_label(0-1), pred_label(0-1)
    acc_dict = {
        'face': [],
        'body': []
    }
    gt_links = gt_links.detach().cpu().numpy()
    gt_links_by_to_label = groupby(sorted(gt_links, key=lambda x: x[3]), key=lambda x: x[3])
    matched_face_relation_idxs = []
    matched_body_relation_idxs = []
    for to_label, curr_gt_links in gt_links_by_to_label:
        curr_gt_links = list(curr_gt_links)
        for gt_link in curr_gt_links:
            from_gt_box_idx = gt_link[0]
            to_gt_box_idx = gt_link[1]
            from_gt_box = gt_boxes[from_gt_box_idx]
            to_gt_box = gt_boxes[to_gt_box_idx]
            found_relation = False
            if to_label == 1:
                # selected body relations arasında bu gt box lara en yakın olanları bul
                for body_rel_idx, body_relation in enumerate(selected_body_relations):
                    # from ve to su yüzde x iou'un üstündeyse artı kabul et ve selected body relation larından çıkar...
                    from_pred_box = pred_boxes[body_relation[0]]
                    to_pred_box = pred_boxes[body_relation[1]]
                    from_box_iou = box_ops.box_iou(from_gt_box.view(1, -1), from_pred_box.view(1, -1)).item()
                    to_box_iou = box_ops.box_iou(to_gt_box.view(1, -1), to_pred_box.view(1, -1)).item()
                    # print('from - to iou values: ', from_box_iou, to_box_iou)
                    if from_box_iou >= box_iou_threshold and to_box_iou >= box_iou_threshold:
                        acc_dict['body'].append([gt_link, body_relation, 1, 1])
                        found_relation = True
                        matched_body_relation_idxs.append(body_rel_idx)
                        break
                if not found_relation and include_not_found_gt_relations:
                    acc_dict['body'].append([gt_link, None, 1, 0])
            elif to_label == 2:
                # selected face relations arasında bu gt box lara en yakın olanları bul
                for face_rel_idx, face_relation in enumerate(selected_face_relations):
                    # from ve to su yüzde x iou'un üstündeyse artı kabul et ve selected face relation larından çıkar...
                    from_pred_box = pred_boxes[face_relation[0]]
                    to_pred_box = pred_boxes[face_relation[1]]
                    from_box_iou = box_ops.box_iou(from_gt_box.view(1, -1), from_pred_box.view(1, -1)).item()
                    to_box_iou = box_ops.box_iou(to_gt_box.view(1, -1), to_pred_box.view(1, -1)).item()
                    # print('from - to iou values: ', from_box_iou, to_box_iou)
                    if from_box_iou >= box_iou_threshold and to_box_iou >= box_iou_threshold:
                        acc_dict['face'].append([gt_link, face_relation, 1, 1])
                        found_relation = True
                        matched_face_relation_idxs.append(face_rel_idx)
                        break
                if not found_relation and include_not_found_gt_relations:
                    acc_dict['face'].append([gt_link, None, 1, 0])
            else:
                raise Exception(f'to label can only be 1 or 2, here it is {str(to_label)}')

    if include_unmatched_body_relation_idxs:
        unmatched_body_relation_idxs = set(range(len(selected_body_relations))) - set(matched_body_relation_idxs)
        unmatched_face_relation_idxs = set(range(len(selected_face_relations))) - set(matched_face_relation_idxs)
        for idx in unmatched_body_relation_idxs:
            acc_dict['body'].append([None, selected_body_relations[idx], 0, 1])
        for idx in unmatched_face_relation_idxs:
            acc_dict['face'].append([None, selected_face_relations[idx], 0, 1])

    face_gt_labels = list(map(lambda x: x[2], acc_dict['face']))
    face_preds = list(map(lambda x: x[3], acc_dict['face']))
    body_gt_labels = list(map(lambda x: x[2], acc_dict['body']))
    body_preds = list(map(lambda x: x[3], acc_dict['body']))

    return (face_gt_labels, face_preds), (body_gt_labels, body_preds), acc_dict


class ComicPageRelationEvaluator:
    def __init__(self,
                 include_not_found_gt_relations: bool = True,
                 include_unmatched_body_relation_idxs: bool = True,
                 relation_score_threshold: Optional[float] = None,
                 box_score_threshold: Optional[float] = None):
        self.include_not_found_gt_relations = include_not_found_gt_relations
        self.include_unmatched_body_relation_idxs = include_unmatched_body_relation_idxs
        self.relation_score_threshold = relation_score_threshold
        self.box_score_threshold = box_score_threshold
        self.body_preds = []
        self.body_gt_labels = []
        self.face_gt_labels = []
        self.face_preds = []

    def step(self, targets, detections, relation_data, use_char_alignment: bool = False):
        for i in range(len(targets)):
            if len(targets[i]['links']) == 0 or len(relation_data['relations']) == 0:
                continue
            (face_gt_labels, face_preds), (body_gt_labels, body_preds), acc_dict = relations_to_labels(
                targets[i]['links'],
                targets[i]['boxes'],
                detections[i]['boxes'],
                relation_data[
                    'relations'][i],
                detections[i][
                    'labels'],
                detections[i]['scores'],
                include_not_found_gt_relations=self.include_not_found_gt_relations,
                include_unmatched_body_relation_idxs=self.include_unmatched_body_relation_idxs,
                score_threshold=self.relation_score_threshold,
                box_score_threshold=self.box_score_threshold,
                use_char_alignment=use_char_alignment
            )
            self.body_preds.extend(body_preds)
            self.face_preds.extend(face_preds)
            self.face_gt_labels.extend(face_gt_labels)
            self.body_gt_labels.extend(body_gt_labels)

    def reset(self):
        self.body_preds = []
        self.body_gt_labels = []
        self.face_gt_labels = []
        self.face_preds = []

    def show_results(self):
        def print_res(labels, preds):
            print('Precision: %.3f' % precision_score(labels, preds))
            print('Recall: %.3f' % recall_score(labels, preds))
            print('Accuracy: %.3f' % accuracy_score(labels, preds))
            print('F1 Score: %.3f' % f1_score(labels, preds))

        print('******** FACE *********')
        print_res(self.face_gt_labels, self.face_preds)
        print('******** BODY *********')
        print_res(self.body_gt_labels, self.body_preds)

    def compute(self, suffix: str = ''):
        def get_res(labels, preds):
            results = {
                'precision': precision_score(labels, preds),
                'recall': recall_score(labels, preds),
                'accuracy': accuracy_score(labels, preds),
                'f1': f1_score(labels, preds)
            }
            return results

        face_res = get_res(self.face_gt_labels, self.face_preds)
        face_res = {f'face/rel_{suffix}' + k: v for k, v in face_res.items()}
        body_res = get_res(self.body_gt_labels, self.body_preds)
        body_res = {f'body/rel_{suffix}' + k: v for k, v in body_res.items()}
        return {**face_res, **body_res}
