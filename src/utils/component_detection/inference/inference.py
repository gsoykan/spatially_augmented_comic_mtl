import itertools
import os.path
from collections import defaultdict
from typing import Tuple, List, Optional, Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor

from src.models.components.component_detection.pt_detector import PtDetector
from src.models.pml_id_net_fine_tuned_ssl_backbone_face_body_module import PMLIdNetFineTunedSSLBackboneFaceBodyLitModule
from src.models.pt_detector_module import PtDetectorLitModule

from src.utils.basic_utils import merge_pt_boxes, box_intersection_rate, box_to_box_center_distance, \
    sort_elements_by_z_order, extract_bounding_box_from_mask, smooth_edges, bbox_area

from src.utils.component_detection.face_body_char_evaluation import ComicPageFaceBodyCharEvaluator
from src.utils.component_detection.inference.helper_classes import MTLPanel, MTLNarrative, MTLSpeech, MTLCharacter, \
    MTLFace, MTLBody, MTLDanglingComponents, MTLPage, ReadImageError, NoDetectionsError
from src.utils.component_detection.relation_evaluation import get_selected_relations


class MTLInference:
    def __init__(self,
                 ckpt: str,
                 id_net_ckpt: Optional[str] = None,
                 id_net_ssl_ckpt: Optional[str] = None):
        self.ckpt = ckpt
        self.model: PtDetector = PtDetectorLitModule.checkpoint_to_eval(ckpt).model

        self.transform = A.Compose([
            A.LongestMaxSize(800),
            A.ToFloat(max_value=255, always_apply=True),
            ToTensorV2(),
        ])

        self.id_net = None
        if id_net_ckpt is not None and id_net_ssl_ckpt is not None:
            self.id_net = PMLIdNetFineTunedSSLBackboneFaceBodyLitModule.checkpoint_to_eval(id_net_ckpt,
                                                                                           ssl_ckpt_face=id_net_ssl_ckpt,
                                                                                           ssl_ckpt_body=id_net_ssl_ckpt)

    def read_img(self, path):
        try:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)  # H, W, C
        except Exception as e:
            raise ReadImageError(f"An error occurred while reading the image: {e}", path)
        orig_h, orig_w, orig_c = image.shape
        sample = self.transform(image=image)
        img_transformed = sample['image']  # C, H, w
        c, h, w = img_transformed.size()
        return img_transformed, (orig_h, orig_w, orig_c), (h, w, c)

    @torch.no_grad()
    def process_page(self,
                     page_path: str,
                     save_crop_root_folder: str,
                     save_csv_root_folder: str,
                     device: str = 'cuda',
                     visualize_results: bool = False,
                     crop_components: bool = True,
                     save_components_to_csv: bool = True,
                     assign_identities: bool = False) -> Tuple[List[MTLPanel], MTLDanglingComponents]:

        if assign_identities and self.id_net is None:
            raise Exception('Identities can not be assigned without ID-NET')

        box_score_threshold = 0.5
        relation_score_threshold = 0.2
        speech_mask_threshold = 0.6
        # make img and model ready
        img, orig_shape, trans_shape = self.read_img(page_path)
        img = img.to(device).unsqueeze(0)
        self.model.to(device)

        # forward pass
        _, detections, relation_data = self.model(img)
        relations = relation_data.get('relations')
        relations = relations[0] if relations is not None and len(relations) != 0 else []
        detections = detections[0]
        detections['box_idxs'] = np.array(list(range(len(detections['boxes']))))

        detections, relations = MTLInference.filter_contained_boxes(detections, relations)

        if all(detections['scores'] < box_score_threshold):
            raise NoDetectionsError(
                f"All detections are below threshold, num detections => {str(len(detections['scores']))}", page_path)

        # class labels => body, face, panel, speech-bubble, narrative
        panel_idxs = ((detections['labels'] == 3).int() * (
                detections['scores'] >= box_score_threshold).int()).bool().cpu()

        panels = list(zip(detections['boxes'][panel_idxs], detections['box_idxs'][panel_idxs]))

        speech_bubble_idxs = (
                (detections['labels'] == 4).int() * (detections['scores'] >= box_score_threshold).int()).bool().cpu()
        detection_masks = (detections['masks'] > speech_mask_threshold).squeeze().detach().cpu().numpy().astype(
            np.uint8)
        speech_bubbles = list(zip(detections['boxes'][speech_bubble_idxs],
                                  detection_masks[speech_bubble_idxs],
                                  detections['box_idxs'][speech_bubble_idxs]))
        speech_bubbles = MTLInference.handle_speech_bubbles(speech_bubbles, panels)

        narrative_idxs = (
                (detections['labels'] == 5).int() * (detections['scores'] >= box_score_threshold).int()).bool().cpu()
        narrative_boxes = list(zip(detections['boxes'][narrative_idxs], detections['box_idxs'][narrative_idxs]))
        narrative_boxes = MTLInference.handle_narrative_boxes(narrative_boxes, panels)

        chars = MTLInference.handle_chars(detections,
                                          relations,
                                          panels,
                                          box_score_threshold,
                                          relation_score_threshold,
                                          speech_bubbles)

        mtl_panels, dangling_components = MTLInference.panelize_components(panels, speech_bubbles, chars,
                                                                           narrative_boxes)

        if save_components_to_csv:
            # saving csv's out of page
            panel_dfs, narrative_dfs, speech_dfs, char_dfs = [], [], [], []
            for mtl_panel in mtl_panels:
                panel_df, narrative_df, speech_df, char_df = mtl_panel.all_components_to_csv(page_path,
                                                                                             save_csv_root_folder,
                                                                                             orig_shape, trans_shape)
                panel_dfs.append(panel_df)
                narrative_dfs.append(narrative_df)
                speech_dfs.append(speech_df)
                char_dfs.append(char_df)
            # saving all components -> root/{series}/{page}/{type(plural)}.csv
            MTLPanel.combine_all_df_components(page_path,
                                               save_csv_root_folder,
                                               panel_dfs,
                                               char_dfs,
                                               narrative_dfs,
                                               speech_dfs)

        if crop_components:
            # saving all components -> root/{type}/{series}/{page}/{box_id}.jpg
            [mtl_panel.crop_all_components(page_path, save_crop_root_folder, orig_shape, trans_shape) for mtl_panel in
             mtl_panels]

        if assign_identities:
            MTLPage.assign_identities(self.id_net,
                                      page_path,
                                      save_crop_root_folder,
                                      mtl_panels,
                                      dangling_components)

        if visualize_results:
            MTLPage.visualize_page(img.squeeze(0), mtl_panels, dangling_components)

        return mtl_panels, dangling_components

    @staticmethod
    def panelize_components(panels: List[Tuple[Tensor, int]],
                            speech_bubbles,
                            chars,
                            narratives) -> Tuple[List[MTLPanel], MTLDanglingComponents]:
        sorted_panel_order = sort_elements_by_z_order(list(map(lambda x: x[0], panels)))

        mtl_panels = []

        def get_components(panel_id: Optional[int], panel_box, panel_order):
            panel_speeches = list(filter(lambda x: x['panel_id'] == panel_id, speech_bubbles))

            raw_speech_boxes = list(
                map(lambda x:
                    extract_bounding_box_from_mask(smooth_edges(x['speech_mask'])),
                    panel_speeches))
            # panel speeches with None speech boxes are filtered...
            speech_boxes_and_panel_speeches = list(filter(lambda x: x[0] is not None,
                                                          zip(raw_speech_boxes, panel_speeches)))
            speech_boxes = list(
                map(lambda x: torch.tensor(x[0], device=x[1]['speech_box'].device), speech_boxes_and_panel_speeches))
            sorted_speech_bubble_order = sort_elements_by_z_order(speech_boxes, 4, panel_box)
            panel_speeches_with_order = []
            for j, (_, panel_speech) in enumerate(speech_boxes_and_panel_speeches):
                speech_order = np.where(j == sorted_speech_bubble_order)[0][0]
                panel_speeches_with_order.append({
                    **panel_speech,
                    'order': speech_order,
                    'panel_order': panel_order,
                })

            panel_speeches_with_order = list(map(lambda x: MTLSpeech(
                box=x['speech_box'],
                box_id=x['box_idx'],
                panel_id=x['panel_id'],
                order=x['order'],
                segm_mask=x['speech_mask'],
                panel_order=x['panel_order']), panel_speeches_with_order))

            panel_chars = list(filter(lambda x: x['panel_id'] == panel_id, chars))
            panel_chars = list(map(lambda x: MTLCharacter(
                box=x['char_box'],
                panel_id=x['panel_id'],
                panel_order=panel_order,
                face=MTLFace(x['face'], x['face_idx'], panel_id) if x['face_idx'] is not None else None,
                body=MTLBody(x['body'], x['body_idx'], panel_id) if x['body_idx'] is not None else None,
                speech_ids=x['speech_indices']), panel_chars))

            panel_narratives = list(filter(lambda x: x['panel_id'] == panel_id, narratives))
            panel_narratives = list(map(lambda x: MTLNarrative(
                box_id=x['box_idx'],
                panel_id=x['panel_id'],
                panel_order=panel_order,
                box=x['narrative']), panel_narratives))

            return panel_chars, panel_speeches_with_order, panel_narratives

        for i, panel in enumerate(panels):
            order = np.where(i == sorted_panel_order)[0][0]
            panel_box = panel[0]
            panel_id = panel[1]  # panel_box_index

            panel_chars, panel_speeches_with_order, panel_narratives = get_components(panel_id, panel_box, order)
            mtl_panel = MTLPanel(box=panel_box.cpu().numpy(),
                                 box_id=panel_id,
                                 order=order,
                                 narratives=panel_narratives,
                                 speeches=panel_speeches_with_order,
                                 characters=panel_chars)
            mtl_panels.append(mtl_panel)

        dangling_chars, dangling_speeches, dangling_narratives = get_components(None, None, None)
        dangling_components = MTLDanglingComponents(dangling_narratives, dangling_speeches, dangling_chars)

        mtl_panels.sort(key=lambda x: x.order, reverse=True)

        return mtl_panels, dangling_components

    @staticmethod
    def handle_speech_bubbles(speech_bubbles: List[Tuple[Tensor, np.ndarray, int]],
                              panels: List[Tuple[Tensor, int]], ):
        speeches = []
        for speech in speech_bubbles:
            box = speech[0]
            mask = speech[1]
            box_idx = speech[2]
            associated_panel_id = MTLInference.assign_box_to_panel(box, panels)
            speeches.append({
                'speech_box': box,
                'speech_mask': mask,
                'box_idx': box_idx,
                'panel_id': associated_panel_id
            })
        return speeches

    @staticmethod
    def handle_narrative_boxes(narrative_boxes: List[Tuple[Tensor, int]],
                               panels: List[Tuple[Tensor, int]], ):
        narratives = []
        for narrative in narrative_boxes:
            box = narrative[0]
            box_idx = narrative[1]
            associated_panel_id = MTLInference.assign_box_to_panel(box, panels)
            narratives.append({
                'narrative': box,
                'box_idx': box_idx,
                'panel_id': associated_panel_id
            })
        return narratives

    @staticmethod
    def handle_chars(detections,
                     relations,
                     panels: List[Tuple[Tensor, int]],
                     box_score_threshold,
                     relation_score_threshold,
                     speech_bubbles):
        chars = ComicPageFaceBodyCharEvaluator.pair_face_body_to_form_char(
            detections['boxes'],
            detections['labels'],
            detections['scores'],
            box_score_threshold=box_score_threshold,
            alternative_box_idxs=detections['box_idxs'])

        chars_with_rel = []
        for _, char in enumerate(chars):
            face_idx, body_idx = char['face_idx'], char['body_idx']

            if face_idx is None:
                char_box = detections['boxes'][body_idx == detections['box_idxs']][0]
            elif body_idx is None:
                char_box = detections['boxes'][face_idx == detections['box_idxs']][0]
            else:
                # means both of them are available
                char_box = merge_pt_boxes(detections['boxes'][face_idx == detections['box_idxs']][0],
                                          detections['boxes'][body_idx == detections['box_idxs']][0])

            associated_panel_id = MTLInference.assign_box_to_panel(char_box, panels)

            updated_char = {
                **char,
                # 'speech_indices': speech_bubble_idx,
                'char_box': char_box,
                'panel_id': associated_panel_id
            }
            chars_with_rel.append(updated_char)

        def filter_panels_relations(face_idx, body_idx, possible_rels):
            current_chars = list(
                filter(lambda x: x['face_idx'] == face_idx or x['body_idx'] == body_idx, chars_with_rel))
            from_idxs = []
            for current_char in current_chars:
                curr_panel_id = current_char['panel_id']
                panel_speech_bubbles = list(filter(lambda x: x['panel_id'] == curr_panel_id, speech_bubbles))
                from_idxs.extend(list(map(lambda x: x['box_idx'], panel_speech_bubbles)))

            filtered_possible_rels = []
            for rel in possible_rels:
                from_idx, to_idx = rel[0], rel[1]
                if from_idx in from_idxs and to_idx in [face_idx, body_idx]:
                    filtered_possible_rels.append(rel)

            return filtered_possible_rels

        selected_face_relations, selected_body_relations = get_selected_relations(relations,
                                                                                  detections['labels'],
                                                                                  detections['scores'],
                                                                                  score_threshold=relation_score_threshold,
                                                                                  box_score_threshold=box_score_threshold,
                                                                                  chars=chars,
                                                                                  pred_box_indices=detections[
                                                                                      'box_idxs'],
                                                                                  process_possible_rels=filter_panels_relations)

        # assign relations (speeches, panel) to chars
        all_relations = [*selected_face_relations, *selected_body_relations]

        chars_with_rel_updated = []
        for _, char in enumerate(chars_with_rel):
            face_idx, body_idx = char['face_idx'], char['body_idx']
            char_rel = [r for r in all_relations if r[1] in [face_idx, body_idx]]
            speech_bubble_idx = list(set(map(lambda x: x[0], char_rel)))

            updated_char = {
                **char,
                'speech_indices': speech_bubble_idx,
            }
            chars_with_rel_updated.append(updated_char)

        return chars_with_rel_updated

    @staticmethod
    def assign_box_to_panel(box: Tensor,
                            panels: List[Tuple[Tensor, int]]) -> Optional[int]:
        candidates = []

        for panel in panels:
            panel_box = panel[0]
            panel_id = panel[1]
            intersection_rate = box_intersection_rate(box.tolist(), panel_box.tolist())
            if intersection_rate >= 0.20:
                center_distance = box_to_box_center_distance(box, panel_box)
                candidates.append([panel_id, intersection_rate, -1 * center_distance.item()])

        if len(candidates) != 0:
            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            panel_index = candidates[0][0]
            return panel_index

        return None

    @staticmethod
    def filter_contained_boxes(detections,
                               relations,
                               intersection_threshold: float = 0.5,  # lower bound
                               # add to contained list, if the score is lower than this...
                               score_threshold: float = 0.95):
        grouped_detections = itertools.groupby(
            sorted(zip(detections['labels'], detections['boxes'], detections['box_idxs'], detections['scores']),
                   key=lambda x: x[0]),
            lambda x: x[0])
        contained_box_idxs = []
        container_idx_counter = defaultdict(list)
        for key, group in grouped_detections:
            list_group = list(group)
            for i in range(0, len(list_group)):
                label, box, box_idx, score = list_group[i]
                box_area = bbox_area(box)
                for j in range(i + 1, len(list_group)):
                    other_label, other_box, other_box_idx, other_score = list_group[j]

                    # if one of them is already added then don't do anything
                    if other_box_idx in contained_box_idxs or box_idx in contained_box_idxs:
                        continue

                    other_box_area = bbox_area(other_box)
                    intersection_rate = box_intersection_rate(box.tolist(), other_box.tolist())
                    other_intersection_rate = box_intersection_rate(other_box.tolist(), box.tolist())
                    if box_area > other_box_area:
                        if other_intersection_rate > intersection_threshold:
                            if score_threshold > other_score:
                                contained_box_idxs.append(other_box_idx)
                            else:
                                container_idx_counter[box_idx].append(other_box_idx)
                        elif intersection_rate > intersection_threshold:
                            if score_threshold > score:
                                contained_box_idxs.append(box_idx)
                            else:
                                container_idx_counter[other_box_idx].append(box_idx)
                    else:
                        if intersection_rate > intersection_threshold:
                            if score_threshold > score:
                                contained_box_idxs.append(box_idx)
                            else:
                                container_idx_counter[other_box_idx].append(box_idx)
                        elif other_intersection_rate > intersection_threshold:
                            if score_threshold > other_score:
                                contained_box_idxs.append(other_box_idx)
                            else:
                                container_idx_counter[box_idx].append(other_box_idx)

        # if there is container that has more than 2 legit same-type boxes then eliminate it..
        for k, v in container_idx_counter.items():
            if len(v) > 2:
                contained_box_idxs.append(k)

        remaining_indices = np.where(~np.isin(detections['box_idxs'], np.array(contained_box_idxs)))[0]
        filtered_detections = {}
        for k, v in detections.items():
            filtered_detections[k] = v[remaining_indices]

        filtered_relations = list(
            filter(lambda x: x[0] not in contained_box_idxs and x[1] not in contained_box_idxs, relations))
        return filtered_detections, filtered_relations


if __name__ == '__main__':
    project_root = "/home/gsoykan/Desktop/dev/comics_ku_masters_rework/amazing-mysteries-of-gutter-demystified"
    comics_dataset_path = "/home/gsoykan/Desktop/dev/comics_ku_masters_rework/amazing-mysteries-gutter-comics"
    pml_id_net_ckpt = os.path.join(project_root,
                                   'logs/experiments/runs/pml_id_net_fine_tuned_ssl_backbone_face_body_module/epoch_018.ckpt')
    id_net_ssl_ckpt = os.path.join(project_root,
                                   'logs/experiments/runs/pml_id_net_fine_tuned_ssl_backbone_face_body_module/face_body_aligned_ssl.ckpt')
    ckpt = os.path.join(project_root, 'logs/experiments/runs/pt_detector/the_best.ckpt')
    save_crop_root_folder = os.path.join(project_root, 'data/mtl_crop')
    save_csv_root_folder = os.path.join(project_root, 'data/mtl_csv')
    inferencer = MTLInference(ckpt, id_net_ckpt=pml_id_net_ckpt, id_net_ssl_ckpt=id_net_ssl_ckpt)
    for i in range(6, 50):
        sample_page_path = os.path.join(comics_dataset_path, f'raw_page_images/0/{str(i)}.jpg')
        res = inferencer.process_page(sample_page_path,
                                      save_crop_root_folder,
                                      save_csv_root_folder,
                                      visualize_results=True,
                                      assign_identities=False)
        print(res)
