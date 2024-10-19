"""
Implements the Generalized R-CNN framework
"""
import random
from collections import OrderedDict

import numpy as np
import torch
import torchvision.transforms
from einops import rearrange
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union, Any
from torchvision.ops import boxes as box_ops
from torchvision.models.detection import _utils as det_utils
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy

from src.utils.basic_utils import flatten_list
from src.utils.ssl.create_ssl_dataset_file_face import box_intersection_rate

from sklearn.utils import class_weight


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self,
                 backbone,
                 rpn,
                 roi_heads,
                 transform,
                 relation_network,
                 use_filtered_boxes_for_pair_pool,
                 enable_encapsulation_box_masking,
                 to_matcher_iou_threshold_addition,
                 generate_sliding_window_negative_samples,
                 generate_mirrored_by_bubble_center_negative_samples,
                 use_negative_links,
                 base_sample_count,
                 additional_neg_sample_count,
                 select_samples_by_box_intersection_scores: bool = True,
                 filter_body_intersected_generated_negative_face_samples: bool = True,
                 balance_face_char_sample_counts: bool = True,
                 use_edge_maps: bool = False
                 ):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.relation_network = relation_network
        self.is_relation_network_enabled = True
        self.enable_encapsulation_box_masking = enable_encapsulation_box_masking
        self.use_filtered_boxes_for_pair_pool = use_filtered_boxes_for_pair_pool
        self.to_matcher_iou_threshold_addition = to_matcher_iou_threshold_addition
        self.generate_sliding_window_negative_samples = generate_sliding_window_negative_samples
        self.force_get_inference_results = False
        self.generate_mirrored_by_bubble_center_negative_samples = generate_mirrored_by_bubble_center_negative_samples
        self.use_negative_links = use_negative_links
        self.base_sample_count = base_sample_count
        self.additional_neg_sample_count = additional_neg_sample_count
        self.select_samples_by_box_intersection_scores = select_samples_by_box_intersection_scores
        self.filter_body_intersected_generated_negative_face_samples = filter_body_intersected_generated_negative_face_samples
        self.balance_face_char_sample_counts = balance_face_char_sample_counts
        # TODO: @gsoykan - make this an argument
        self.use_loss_weights_face_body = False
        self.use_edge_maps = use_edge_maps
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                            boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        edge_maps = self.extract_edge_maps(images) if self.use_edge_maps else None

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        relation_losses = {}
        relation_additional_data = {}

        def extract_custom_bb_features(img_idx, bb, batch_mode: bool = False):
            is_multiple_img = not isinstance(img_idx, int)

            features_for_image = {}
            if is_multiple_img:
                img_size = np.array(images.image_sizes)[img_idx]
                for k, v in features.items():
                    features_for_image[k] = v[img_idx]
            else:
                img_size = images.image_sizes[img_idx]
                for k, v in features.items():
                    features_for_image[k] = v[img_idx].unsqueeze(dim=0)
            if not batch_mode:
                custom_bb_features = self.roi_heads.box_roi_pool(features_for_image, [bb.view(1, -1)], [img_size])
                return custom_bb_features.squeeze(dim=0)
            else:
                if is_multiple_img:
                    custom_bb_features = self.roi_heads.box_roi_pool(features_for_image, bb, img_size)
                else:
                    custom_bb_features = self.roi_heads.box_roi_pool(features_for_image, [bb], [img_size])
                return custom_bb_features

        if targets is not None:
            relation_additional_data = {
            }
            paired_relation_features, \
            paired_spatial_features, \
            paired_relation_labels, \
            paired_relation_bbs, \
            paired_to_labels, \
            paired_encapsulation_box_masks, \
            paired_encapsulation_edge_maps = self.pair_pool_for_gt_relations(
                detections,
                targets,
                image_sizes=images.image_sizes,
                use_filtered_boxes=self.use_filtered_boxes_for_pair_pool,
                custom_bb_feature_extractor=extract_custom_bb_features,
                enable_encapsulation_box_masking=self.enable_encapsulation_box_masking,
                to_matcher_iou_threshold_addition=self.to_matcher_iou_threshold_addition,
                generate_sliding_window_negative_samples=self.generate_sliding_window_negative_samples,
                generate_mirrored_by_bubble_center_negative_samples=self.generate_mirrored_by_bubble_center_negative_samples,
                use_negative_links=self.use_negative_links,
                base_sample_count=self.base_sample_count,
                additional_neg_sample_count=self.additional_neg_sample_count,
                select_samples_by_box_intersection_scores=self.select_samples_by_box_intersection_scores,
                filter_body_intersected_generated_negative_face_samples=self.filter_body_intersected_generated_negative_face_samples,
                balance_face_char_sample_counts=self.balance_face_char_sample_counts,
                edge_maps=edge_maps) if self.is_relation_network_enabled else (
                None, None, None, None, None, None, None)

            if None not in [paired_relation_features, paired_spatial_features, paired_relation_labels]:
                relation_logits = self.relation_network(paired_relation_features,
                                                        paired_spatial_features,
                                                        paired_to_labels,
                                                        paired_encapsulation_box_masks,
                                                        box_head=self.roi_heads.box_head,
                                                        encapsulation_edge_maps=paired_encapsulation_edge_maps)
                if paired_relation_labels is not None:
                    relation_additional_data = {
                        'relation_bbs': paired_relation_bbs,
                        'relation_logits': relation_logits,
                        'relation_labels': paired_relation_labels
                    }

                    weights = None
                    if self.use_loss_weights_face_body:
                        np_paired_to_labels = paired_to_labels.detach().cpu().numpy()
                        class_weights = class_weight.compute_class_weight('balanced',
                                                                          np.unique(np_paired_to_labels),
                                                                          np_paired_to_labels)
                        weights = torch.ones_like(relation_logits, device=relation_logits.device)
                        for i, l in enumerate(np.unique(np_paired_to_labels)):
                            weights[paired_to_labels == l] = class_weights[i]

                    relation_loss = F.binary_cross_entropy_with_logits(relation_logits,
                                                                       paired_relation_labels.view(-1, 1),
                                                                       weight=weights)
                    relation_losses = {
                        'loss_relation': relation_loss
                    }

            for detection in detections:
                del detection['relational_data']

        if targets is None or self.force_get_inference_results is True:
            score_threshold = 0.0
            with torch.no_grad():
                if self.force_get_inference_results is True:
                    detections, _ = self.roi_heads(features, proposals, images.image_sizes)
                all_box_to_box_idxs, all_box_to_feature_stacks, all_box_to_additional_features = self.pair_pool_for_inference_relations(
                    detections,
                    custom_bb_feature_extractor=extract_custom_bb_features,
                    edge_maps=edge_maps)
                relation_additional_data['relations'] = []
                for box_to_box_idxs, box_to_feature_stacks, box_to_additional_features in zip(
                        all_box_to_box_idxs,
                        all_box_to_feature_stacks,
                        all_box_to_additional_features):
                    related_box_idxs = []
                    if None in [box_to_box_idxs, box_to_feature_stacks, box_to_additional_features]:
                        relation_additional_data['relations'].append(related_box_idxs)
                        continue

                    for k, v in box_to_feature_stacks.items():
                        paired_relation_features = torch.stack(v)
                        spatial_features, to_labels, encapsulation_box_masks, encapsulation_edge_maps = list(
                            zip(*box_to_additional_features[k]))
                        spatial_features = torch.stack(spatial_features)
                        if None not in encapsulation_box_masks:
                            encapsulation_box_masks = torch.stack(encapsulation_box_masks)
                        else:
                            encapsulation_box_masks = None

                        if None not in encapsulation_edge_maps:
                            encapsulation_edge_maps = torch.stack(encapsulation_edge_maps)
                        else:
                            encapsulation_edge_maps = None

                        relation_logits = self.relation_network(paired_relation_features,
                                                                spatial_features,
                                                                torch.stack(to_labels),
                                                                encapsulation_box_masks,
                                                                box_head=self.roi_heads.box_head,
                                                                encapsulation_edge_maps=encapsulation_edge_maps)
                        relation_scores = F.sigmoid(relation_logits).view(-1)
                        for relation_score_idx in (relation_scores > score_threshold).nonzero():
                            relation_to_box_idx = box_to_box_idxs[k][relation_score_idx]
                            related_box_idxs.append(
                                [k.item(), relation_to_box_idx.item(), relation_scores[relation_score_idx].item()])
                    relation_additional_data['relations'].append(related_box_idxs)

            for detection in detections:
                if 'relational_data' in detection:
                    del detection['relational_data']

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(relation_losses)

        return losses, detections, relation_additional_data

    def pair_pool_for_inference_relations(self,
                                          detections,
                                          from_labels=[4],
                                          to_labels=[1, 2],
                                          custom_bb_feature_extractor: Any = None,
                                          edge_maps=None):
        all_box_to_box_idxs = []
        all_box_to_feature_stacks = []
        all_box_to_additional_features = []
        # encapsulation_iou_values = []
        for i in range(len(detections)):
            # detection_box_features = detections[i]['box_features']
            # device = detection_box_features.device
            detection_boxes = detections[i]['boxes']
            detection_labels = detections[i]['labels']

            from_idxs = []
            for from_label in from_labels:
                from_idxs.append(torch.where(detection_labels == from_label)[0])
            from_idxs = torch.cat(from_idxs)
            if len(from_idxs) == 0:
                all_box_to_box_idxs.append(None)
                all_box_to_feature_stacks.append(None)
                all_box_to_additional_features.append(None)
                continue

            to_idxs = []
            for to_label in to_labels:
                to_idxs.append(torch.where(detection_labels == to_label)[0])
            to_idxs = torch.cat(to_idxs)
            if len(to_idxs) == 0:
                all_box_to_box_idxs.append(None)
                all_box_to_feature_stacks.append(None)
                all_box_to_additional_features.append(None)
                continue

            box_to_box_idxs = {}
            box_to_feature_stacks = {}
            box_to_additional_features = {}
            bbs = []
            curr_to_labels = []
            idxs = []
            for from_idx in from_idxs:
                from_box = detection_boxes[from_idx]
                # from_feature = custom_bb_feature_extractor(i, from_box)
                box_to_box_idxs[from_idx] = []
                box_to_feature_stacks[from_idx] = []
                box_to_additional_features[from_idx] = []
                for to_idx in to_idxs:
                    curr_to_label = detection_labels[to_idx]
                    curr_to_labels.append(curr_to_label)
                    to_box = detection_boxes[to_idx]
                    bbs.append(torch.cat((from_box, to_box), dim=0))
                    idxs.append([from_idx, to_idx])
                    box_to_box_idxs[from_idx].append(to_idx)

            bbs = torch.stack(bbs)
            encapsulation_boxes = self.get_encapsulating_box_batched(bbs[:, :4], bbs[:, 4:8])
            bbs = torch.cat((bbs, encapsulation_boxes), dim=1)

            encapsulation_edge_maps = None
            if edge_maps is not None:
                encapsulation_edge_maps = self.crop_from_edge_map(edge_maps[i], bbs[:, 8:])

            if self.enable_encapsulation_box_masking:
                encapsulation_box_masks = self.create_encapsulation_box_mask_batched(bbs)

            spatial_features = self.create_spatial_features_batched(bbs[:, :4], bbs[:, 4:8])

            bbs = torch.cat((bbs[:, 8:], bbs[:, :4], bbs[:, 4:8]), dim=0)
            samples = custom_bb_feature_extractor(i, bbs, True)
            samples = rearrange(samples, "(n b) d h w -> b n d h w", n=3)

            for sample, (from_idx, to_idx) in zip(samples, idxs):
                box_to_feature_stacks[from_idx].append(sample)

            for i, (spatial_feature, curr_to_label, (from_idx, to_idx)) in enumerate(
                    zip(spatial_features, curr_to_labels, idxs)):
                box_mask = None
                if self.enable_encapsulation_box_masking:
                    box_mask = encapsulation_box_masks[i]

                encapsulation_edge_map = None
                if edge_maps is not None:
                    encapsulation_edge_map = encapsulation_edge_maps[i]

                box_to_additional_features[from_idx].append(
                    (spatial_feature, curr_to_label, box_mask, encapsulation_edge_map))

            all_box_to_box_idxs.append(box_to_box_idxs)
            all_box_to_feature_stacks.append(box_to_feature_stacks)
            all_box_to_additional_features.append(box_to_additional_features)

        # print('mean encapsulation iou: ', np.array(encapsulation_iou_values).mean())
        return all_box_to_box_idxs, all_box_to_feature_stacks, all_box_to_additional_features

    def old_pair_pool_for_inference_relations(self,
                                              detections,
                                              from_labels=[4],
                                              to_labels=[1, 2],
                                              custom_bb_feature_extractor: Any = None):
        all_box_to_box_idxs = []
        all_box_to_feature_stacks = []
        all_box_to_additional_features = []
        # encapsulation_iou_values = []
        for i in range(len(detections)):
            # detection_box_features = detections[i]['box_features']
            # device = detection_box_features.device
            detection_boxes = detections[i]['boxes']
            detection_labels = detections[i]['labels']

            from_idxs = []
            for from_label in from_labels:
                from_idxs.append(torch.where(detection_labels == from_label)[0])
            from_idxs = torch.cat(from_idxs)
            if len(from_idxs) == 0:
                all_box_to_box_idxs.append(None)
                all_box_to_feature_stacks.append(None)
                all_box_to_additional_features.append(None)
                continue

            to_idxs = []
            for to_label in to_labels:
                to_idxs.append(torch.where(detection_labels == to_label)[0])
            to_idxs = torch.cat(to_idxs)
            if len(to_idxs) == 0:
                all_box_to_box_idxs.append(None)
                all_box_to_feature_stacks.append(None)
                all_box_to_additional_features.append(None)
                continue

            box_to_box_idxs = {}
            box_to_feature_stacks = {}
            box_to_additional_features = {}
            for from_idx in from_idxs:
                from_box = detection_boxes[from_idx]
                from_feature = custom_bb_feature_extractor(i, from_box)
                box_to_box_idxs[from_idx] = []
                box_to_feature_stacks[from_idx] = []
                box_to_additional_features[from_idx] = []
                for to_idx in to_idxs:
                    curr_to_label = detection_labels[to_idx]
                    to_box = detection_boxes[to_idx]
                    encapsulation_box = self.get_encapsulating_box(from_box, to_box)
                    encapsulation_feature = custom_bb_feature_extractor(i, encapsulation_box)

                    encapsulation_box_mask = None
                    if self.enable_encapsulation_box_masking:
                        encapsulation_box_mask = self.create_encapsulation_box_mask(encapsulation_box, from_box, to_box)

                    to_feature = custom_bb_feature_extractor(i, to_box)
                    feature_stack = torch.stack(
                        [encapsulation_feature, from_feature, to_feature])
                    box_to_feature_stacks[from_idx].append(feature_stack)
                    spatial_features = self.create_spatial_features(from_box, to_box)
                    box_to_additional_features[from_idx].append(
                        (spatial_features, curr_to_label, encapsulation_box_mask, None))
                    box_to_box_idxs[from_idx].append(to_idx)

            all_box_to_box_idxs.append(box_to_box_idxs)
            all_box_to_feature_stacks.append(box_to_feature_stacks)
            all_box_to_additional_features.append(box_to_additional_features)

        # print('mean encapsulation iou: ', np.array(encapsulation_iou_values).mean())
        return all_box_to_box_idxs, all_box_to_feature_stacks, all_box_to_additional_features

    def old_pair_pool_for_gt_relations(self,
                                       detections,
                                       targets,
                                       image_sizes,
                                       use_filtered_boxes: bool = False,
                                       custom_bb_feature_extractor: Any = None,
                                       matcher_iou_threshold: float = 0.6,
                                       enable_encapsulation_box_masking: bool = True,
                                       enable_bb_debugging: bool = False,
                                       to_matcher_iou_threshold_addition: float = 0.1,
                                       generate_sliding_window_negative_samples: bool = True,
                                       generate_mirrored_by_bubble_center_negative_samples: bool = True,
                                       use_negative_links: bool = True,
                                       base_sample_count: int = 75,
                                       additional_neg_sample_count: int = 0,
                                       # if it is False it will be random shuffle...
                                       select_samples_by_box_intersection_scores: bool = True,
                                       filter_body_intersected_generated_negative_face_samples: bool = True,
                                       balance_face_char_sample_counts: bool = True
                                       ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        all_labels = []
        all_to_labels = []
        all_samples = []
        all_encapsulation_box_masks = []
        all_spatial_features = []
        all_relation_bbs = []
        pair_pool_data = []

        for i in range(len(detections)):
            device = detections[i]['boxes'].device

            if use_filtered_boxes:
                relational_data = {
                    'boxes': detections[i]['boxes'],
                    'labels': detections[i]['labels'],
                }
            else:
                relational_data = detections[i]['relational_data']

            from_boxes_key = 'boxes'  # 'from_boxes'
            to_boxes_key = 'boxes'  # 'to_boxes'

            gt_boxes = targets[i]['boxes']
            links = targets[i]['links']
            link_labels = torch.ones(targets[i]['links'].shape[0], device=device, dtype=torch.int64, )
            face_to_char_links = targets[i]['face_to_char_links']
            face_to_char_lookup = {i.item(): j.item() for i, j in face_to_char_links}

            if use_negative_links:
                negative_links = targets[i]['negative_links']
                links = torch.cat((links, negative_links))
                link_label_neg = torch.zeros(targets[i]['negative_links'].shape[0], device=device, dtype=torch.int64, )
                link_labels = torch.cat((link_labels, link_label_neg))
            # merge links with link labels...
            links = torch.cat((links, link_labels.view(-1, 1)), dim=1)

            if len(links) == 0:
                continue

            from_links = links[:, 0]
            to_links = links[:, 1]
            from_idx_lookup = {i: j.item() for i, j in enumerate(from_links)}
            to_idx_lookup = {i: j.item() for i, j in enumerate(to_links)}
            from_to_lookup = {(i.item(), j.item()): k.item() for i, j, k in
                              torch.cat((links[:, :2], links[:, -1].view(-1, 1)), dim=1)}
            to_label_lookup = {i.item(): j.item() for i, j in links[:, [1, 3]]}
            from_gt_boxes = gt_boxes[from_links]
            to_gt_boxes = gt_boxes[to_links]

            from_match_quality_matrix = box_ops.box_iou(from_gt_boxes, relational_data[from_boxes_key])
            to_match_quality_matrix = box_ops.box_iou(to_gt_boxes, relational_data[to_boxes_key])

            if 0 in from_match_quality_matrix.shape or 0 in to_match_quality_matrix.shape:
                continue

            from_matcher = det_utils.Matcher(
                matcher_iou_threshold,
                matcher_iou_threshold,
                allow_low_quality_matches=False)

            from_matched_idxs_in_image = from_matcher(from_match_quality_matrix)
            from_pos_idxs = from_matched_idxs_in_image >= 0
            from_pos_link_0_idxs = from_matched_idxs_in_image[from_pos_idxs]
            from_pos_boxes = relational_data[from_boxes_key][from_pos_idxs]
            from_pos_box_intersection_scores = from_match_quality_matrix[:, from_pos_idxs].amax(dim=0)
            sort_idxs = torch.argsort(from_pos_box_intersection_scores, dim=0, descending=True)
            from_pos_box_intersection_scores = from_pos_box_intersection_scores[sort_idxs]
            from_pos_link_0_idxs = from_pos_link_0_idxs[sort_idxs]
            from_pos_boxes = from_pos_boxes[sort_idxs]

            to_matcher = det_utils.Matcher(
                matcher_iou_threshold + to_matcher_iou_threshold_addition,
                matcher_iou_threshold + to_matcher_iou_threshold_addition,
                allow_low_quality_matches=False)

            to_matched_idxs_in_image = to_matcher(to_match_quality_matrix)
            to_pos_idxs = to_matched_idxs_in_image >= 0
            to_pos_link_1_idxs = to_matched_idxs_in_image[to_pos_idxs]
            to_pos_boxes = relational_data[to_boxes_key][to_pos_idxs]
            to_pos_box_intersection_scores = to_match_quality_matrix[:, to_pos_idxs].amax(dim=0)
            sort_idxs = torch.argsort(to_pos_box_intersection_scores, dim=0, descending=True)
            to_pos_box_intersection_scores = to_pos_box_intersection_scores[sort_idxs]
            to_pos_link_1_idxs = to_pos_link_1_idxs[sort_idxs]
            to_pos_boxes = to_pos_boxes[sort_idxs]

            pair_pool_data.append({'from_pos_boxes': from_pos_boxes,
                                   'from_pos_link_0_idxs': from_pos_link_0_idxs,
                                   'from_idx_lookup': from_idx_lookup,
                                   'from_pos_box_intersection_scores': from_pos_box_intersection_scores,
                                   'to_pos_boxes': to_pos_boxes,
                                   'to_pos_link_1_idxs': to_pos_link_1_idxs,
                                   'to_idx_lookup': to_idx_lookup,
                                   'to_pos_box_intersection_scores': to_pos_box_intersection_scores,
                                   'to_label_lookup': to_label_lookup,
                                   'from_to_lookup': from_to_lookup,
                                   'face_to_char_lookup': face_to_char_lookup,
                                   'detection_idx': i})

        # extracting from & to features in a single batch
        all_pp_boxes = []
        all_pp_box_lengths = []
        for pair_pool_dict in pair_pool_data:
            all_pp_boxes.append(torch.cat([pair_pool_dict['from_pos_boxes'], pair_pool_dict['to_pos_boxes']]))
            all_pp_box_lengths.append(len(pair_pool_dict['from_pos_boxes']))
            all_pp_box_lengths.append(len(pair_pool_dict['to_pos_boxes']))

        if len(all_pp_boxes) == 0:
            return None, None, None, None, None, None

        all_pos_features = custom_bb_feature_extractor(list(map(lambda x: x['detection_idx'], pair_pool_data)),
                                                       all_pp_boxes, True)
        all_pos_features = all_pos_features.split(all_pp_box_lengths, 0)

        for i in range(len(pair_pool_data)):
            pair_pool_data[i]['from_pos_features'] = all_pos_features[i * 2]
            pair_pool_data[i]['to_pos_features'] = all_pos_features[i * 2 + 1]

        for i in range(len(pair_pool_data)):
            detection_idx = pair_pool_data[i]['detection_idx']
            positive_samples = []
            positive_spatial_features = []
            positive_to_labels = []
            positive_bbs = []
            positive_encapsulation_boxes = []
            positive_encapsulation_box_masks = []

            negative_samples = []
            negative_spatial_features = []
            negative_to_labels = []
            negative_bbs = []
            negative_encapsulation_boxes = []
            negative_encapsulation_box_masks = []

            # TODO: @gsoykan - play with this...
            max_sample_len = base_sample_count
            max_sliding_negative_sample_count = 0
            max_mirrored_negative_sample_count = 0
            max_neg_neg_sample_count = 0
            if generate_sliding_window_negative_samples:
                # max_sliding_negative_sample_count = min(20, 5 * self.current_epoch) if not enable_bb_debugging else 35
                max_sliding_negative_sample_count = 15 if not enable_bb_debugging else 20
            if generate_mirrored_by_bubble_center_negative_samples:
                max_mirrored_negative_sample_count = 50 if not enable_bb_debugging else 20
            if use_negative_links:
                max_neg_neg_sample_count = 50 if not enable_bb_debugging else 20

            from_pos_boxes = pair_pool_data[i]['from_pos_boxes']
            from_pos_features = pair_pool_data[i]['from_pos_features']
            from_pos_link_0_idxs = pair_pool_data[i]['from_pos_link_0_idxs']
            # iou of from_box and its gt
            from_pos_box_intersection_scores = pair_pool_data[i]['from_pos_box_intersection_scores']
            from_idx_lookup = pair_pool_data[i]['from_idx_lookup']
            to_pos_boxes = pair_pool_data[i]['to_pos_boxes']
            to_pos_features = pair_pool_data[i]['to_pos_features']
            to_pos_link_1_idxs = pair_pool_data[i]['to_pos_link_1_idxs']
            # iou of to_box and its gt
            to_pos_box_intersection_scores = pair_pool_data[i]['to_pos_box_intersection_scores']
            # given to_link_id returns char or face label
            to_label_lookup = pair_pool_data[i]['to_label_lookup']
            to_idx_lookup = pair_pool_data[i]['to_idx_lookup']
            from_to_lookup = pair_pool_data[i]['from_to_lookup']
            face_to_char_lookup = pair_pool_data[i]['face_to_char_lookup']

            positive_tuples = []
            negative_tuples = []
            neg_negative_tuples = []

            for from_iter_idx, from_link_0_idx in enumerate(from_pos_link_0_idxs):
                from_link_id = from_idx_lookup[from_link_0_idx.item()]
                from_box_intersection_score = from_pos_box_intersection_scores[from_iter_idx]
                for to_iter_idx, to_link_1_idx in enumerate(to_pos_link_1_idxs):
                    to_link_id = to_idx_lookup[to_link_1_idx.item()]
                    # to_label -> char or face..
                    to_label = to_label_lookup[to_link_id]
                    to_box_intersection_score = to_pos_box_intersection_scores[to_iter_idx]
                    # link label -> positive link or negative link...
                    label = from_to_lookup.get((from_link_id, to_link_id), None)
                    content = (from_iter_idx,
                               to_iter_idx,
                               from_box_intersection_score.item(),
                               to_box_intersection_score.item(),
                               from_link_id,
                               to_link_id,
                               to_label)
                    if label is None:
                        negative_tuples.append(content)
                    else:
                        positive_tuples.append(content) if label == 1 else neg_negative_tuples.append(content)

            if select_samples_by_box_intersection_scores:
                positive_tuples.sort(key=lambda x: x[2] + x[3], reverse=True)
                negative_tuples.sort(key=lambda x: x[2] + x[3], reverse=True)
                if use_negative_links:
                    neg_negative_tuples.sort(key=lambda x: x[2] + x[3], reverse=True)
            else:
                random.shuffle(positive_tuples)
                random.shuffle(negative_tuples)
                if use_negative_links:
                    random.shuffle(neg_negative_tuples)

            active_neg_count = max_sample_len \
                               + additional_neg_sample_count \
                               - max_sliding_negative_sample_count \
                               - max_mirrored_negative_sample_count \
                               - max_neg_neg_sample_count
            if balance_face_char_sample_counts:
                def balance_tuples(tuples, sample_len):
                    body_tuples = list(filter(lambda x: x[-1] == 1, tuples))
                    face_tuples = list(filter(lambda x: x[-1] == 2, tuples))
                    body_tuples = body_tuples[:min(sample_len, len(body_tuples))]
                    face_tuples = face_tuples[:min(sample_len, len(face_tuples))]
                    tuples = [*body_tuples, *face_tuples]
                    return tuples

                positive_tuples = balance_tuples(positive_tuples, max_sample_len // 2)
                negative_tuples = balance_tuples(negative_tuples, active_neg_count // 2)
                if use_negative_links:
                    neg_negative_tuples = balance_tuples(neg_negative_tuples, max_neg_neg_sample_count // 2)
                    negative_tuples = [*negative_tuples, *neg_negative_tuples]
            else:
                positive_tuples = positive_tuples[:min(max_sample_len, len(positive_tuples))]
                negative_tuples = negative_tuples[:min(active_neg_count, len(negative_tuples))]
                if use_negative_links:
                    neg_negative_tuples = neg_negative_tuples[:min(max_neg_neg_sample_count, len(neg_negative_tuples))]
                    negative_tuples = [*negative_tuples, *neg_negative_tuples]

            def create_sample(from_idx, to_idx, to_link_id, is_positive: bool):
                from_box = from_pos_boxes[from_idx]
                to_box = to_pos_boxes[to_idx]
                from_feature = from_pos_features[from_idx]
                to_feature = to_pos_features[to_idx]
                encapsulation_box = self.get_encapsulating_box(from_box, to_box)
                to_label = to_label_lookup[to_link_id]

                if enable_encapsulation_box_masking:
                    encapsulation_box_mask = self.create_encapsulation_box_mask(encapsulation_box, from_box, to_box)

                sample = torch.stack([from_feature, to_feature])
                spatial_features = self.create_spatial_features(from_box, to_box)
                if is_positive:
                    positive_encapsulation_boxes.append(encapsulation_box)
                    positive_samples.append(sample)
                    positive_spatial_features.append(spatial_features)
                    positive_to_labels.append(to_label)
                    if enable_bb_debugging:
                        positive_bbs.append(torch.stack([*from_box, *to_box, *encapsulation_box]))
                    if enable_encapsulation_box_masking:
                        positive_encapsulation_box_masks.append(encapsulation_box_mask)
                else:
                    negative_encapsulation_boxes.append(encapsulation_box)
                    negative_samples.append(sample)
                    negative_spatial_features.append(spatial_features)
                    negative_to_labels.append(to_label)
                    if enable_bb_debugging:
                        negative_bbs.append(torch.stack([*from_box, *to_box, *encapsulation_box]))
                    if enable_encapsulation_box_masking:
                        negative_encapsulation_box_masks.append(encapsulation_box_mask)

            def create_sliding_window_negative_samples_from_positive_sample(from_idx,
                                                                            to_idx,
                                                                            from_link_id,
                                                                            to_link_id):
                # this can definitely be optimized but let's go for now...
                from_box = from_pos_boxes[from_idx]
                original_to_box = to_pos_boxes[to_idx]
                to_label = to_label_lookup[to_link_id]
                candidates = []

                to_associated_box = None
                if filter_body_intersected_generated_negative_face_samples:
                    # at this point we don't know whether to is face or not
                    char_link_id = face_to_char_lookup.get(to_link_id, None)
                    if char_link_id is not None:
                        to_associated_box = targets[detection_idx]['boxes'][char_link_id]

                new_to_boxes = []
                if generate_sliding_window_negative_samples:
                    new_to_boxes.extend(self.slide_box_in_encapsulating_box(original_to_box, from_box,
                                                                            image_sizes[detection_idx],
                                                                            to_associated_box))

                if generate_mirrored_by_bubble_center_negative_samples:
                    new_to_boxes.extend(self.generate_mirrored_samples_by_from_box_center(original_to_box, from_box,
                                                                                          image_sizes[detection_idx],
                                                                                          to_associated_box))

                if len(new_to_boxes) == 0:
                    return candidates

                from_feature = from_pos_features[from_idx]
                for (new_to_box, new_encapsulating_box) in new_to_boxes:
                    to_feature = custom_bb_feature_extractor(detection_idx, new_to_box)
                    encapsulation_box_mask = None
                    if enable_encapsulation_box_masking:
                        encapsulation_box_mask = self.create_encapsulation_box_mask(new_encapsulating_box, from_box,
                                                                                    new_to_box)
                    sample = torch.stack([from_feature, to_feature])
                    spatial_features = self.create_spatial_features(from_box, new_to_box)
                    canditate = {'encapsulation_box': new_encapsulating_box,
                                 'sample': sample,
                                 'spatial_features': spatial_features,
                                 'to_label': to_label,
                                 'from_box': from_box,
                                 'new_to_box': new_to_box,
                                 'encapsulation_box_mask': encapsulation_box_mask}
                    candidates.append(canditate)
                return candidates

            def handle_negative_sliding_window_candidates(candidates):
                random.shuffle(candidates)
                sliding_neg_count = max_sliding_negative_sample_count + max_mirrored_negative_sample_count
                candidates = candidates[:min(sliding_neg_count, len(candidates))]
                for candidate in candidates:
                    negative_encapsulation_boxes.append(candidate['encapsulation_box'])
                    negative_samples.append(candidate['sample'])
                    negative_spatial_features.append(candidate['spatial_features'])
                    negative_to_labels.append(candidate['to_label'])
                    if enable_bb_debugging:
                        negative_bbs.append(torch.stack(
                            [*candidate['from_box'], *candidate['new_to_box'], *candidate['encapsulation_box']]))
                    if enable_encapsulation_box_masking:
                        negative_encapsulation_box_masks.append(candidate['encapsulation_box_mask'])

            [create_sample(x[0], x[1], x[5], True) for x in positive_tuples]
            [create_sample(x[0], x[1], x[5], False) for x in negative_tuples]
            if generate_sliding_window_negative_samples or generate_mirrored_by_bubble_center_negative_samples:
                candidates = flatten_list(
                    [create_sliding_window_negative_samples_from_positive_sample(x[0], x[1], x[4], x[5]) for x in
                     positive_tuples])
                handle_negative_sliding_window_candidates(candidates)

            if len(positive_samples) == 0:
                continue

            all_encapsulation_features = custom_bb_feature_extractor(detection_idx, torch.stack(
                [*positive_encapsulation_boxes, *negative_encapsulation_boxes]), True)
            positive_encapsulation_features, negative_encapsulation_features = all_encapsulation_features.split(
                [len(positive_encapsulation_boxes), len(negative_encapsulation_boxes)], 0)

            if enable_encapsulation_box_masking:
                positive_encapsulation_box_masks = torch.stack(positive_encapsulation_box_masks)
                if len(negative_samples) != 0:
                    negative_encapsulation_box_masks = torch.stack(negative_encapsulation_box_masks)

            positive_samples = torch.stack(positive_samples)
            positive_samples = torch.cat([positive_encapsulation_features.unsqueeze(dim=1), positive_samples], dim=1)
            positive_spatial_features = torch.stack(positive_spatial_features)
            positive_to_labels = torch.tensor(positive_to_labels, device=device)
            if enable_bb_debugging:
                positive_bbs = torch.stack(positive_bbs)

            if len(negative_samples) != 0:
                negative_samples = torch.stack(negative_samples)
                negative_samples = torch.cat([negative_encapsulation_features.unsqueeze(dim=1), negative_samples],
                                             dim=1)
                negative_spatial_features = torch.stack(negative_spatial_features)
                negative_to_labels = torch.tensor(negative_to_labels, device=device)
                if enable_bb_debugging:
                    negative_bbs = torch.stack(negative_bbs)

            labels = torch.cat([torch.ones(len(positive_samples)), torch.zeros(len(negative_samples))]).to(device)
            to_labels = torch.cat([positive_to_labels, negative_to_labels]) if len(
                negative_samples) != 0 else positive_to_labels
            samples = torch.cat([positive_samples, negative_samples]) if len(
                negative_samples) != 0 else positive_samples

            if enable_encapsulation_box_masking:
                encapsulation_box_masks = torch.cat(
                    [positive_encapsulation_box_masks, negative_encapsulation_box_masks]) if len(
                    negative_samples) != 0 else positive_encapsulation_box_masks
                all_encapsulation_box_masks.append(encapsulation_box_masks)

            spatial_features = torch.cat([positive_spatial_features, negative_spatial_features]) if len(
                negative_samples) != 0 else positive_spatial_features

            # gather all spatial features...
            all_labels.append(labels)
            all_to_labels.append(to_labels)
            all_samples.append(samples)
            all_spatial_features.append(spatial_features)
            if enable_bb_debugging:
                relation_bbs = {
                    'positive': positive_bbs,
                    'negative': negative_bbs
                }
                all_relation_bbs.append(relation_bbs)

        if len(all_labels) == 0:
            return None, None, None, None, None, None

        all_labels = torch.cat(all_labels)
        all_to_labels = torch.cat(all_to_labels)
        all_samples = torch.cat(all_samples)
        all_spatial_features = torch.cat(all_spatial_features)
        if len(all_encapsulation_box_masks) != 0:
            all_encapsulation_box_masks = torch.cat(all_encapsulation_box_masks)
        else:
            all_encapsulation_box_masks = None
        # print('mean encapsulation iou: ', np.array(encapsulation_iou_values).mean())
        return all_samples, all_spatial_features, all_labels, all_relation_bbs, all_to_labels, all_encapsulation_box_masks

    def pair_pool_for_gt_relations(self,
                                   detections,
                                   targets,
                                   image_sizes,
                                   use_filtered_boxes: bool = False,
                                   custom_bb_feature_extractor: Any = None,
                                   matcher_iou_threshold: float = 0.6,
                                   enable_encapsulation_box_masking: bool = True,
                                   enable_bb_debugging: bool = False,
                                   to_matcher_iou_threshold_addition: float = 0.1,
                                   generate_sliding_window_negative_samples: bool = True,
                                   generate_mirrored_by_bubble_center_negative_samples: bool = True,
                                   use_negative_links: bool = True,
                                   base_sample_count: int = 75,
                                   additional_neg_sample_count: int = 0,
                                   # if it is False it will be random shuffle...
                                   select_samples_by_box_intersection_scores: bool = True,
                                   filter_body_intersected_generated_negative_face_samples: bool = True,
                                   balance_face_char_sample_counts: bool = True,
                                   edge_maps: Optional[List[torch.Tensor]] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        all_labels = []
        all_to_labels = []
        all_samples = []
        all_encapsulation_box_masks = []
        all_encapsulation_edge_maps = []
        all_spatial_features = []
        all_relation_bbs = []
        pair_pool_data = []

        for i in range(len(detections)):
            device = detections[i]['boxes'].device

            if use_filtered_boxes:
                relational_data = {
                    'boxes': detections[i]['boxes'],
                    'labels': detections[i]['labels'],
                }
            else:
                relational_data = detections[i]['relational_data']

            from_boxes_key = 'boxes'  # 'from_boxes'
            to_boxes_key = 'boxes'  # 'to_boxes'

            gt_boxes = targets[i]['boxes']
            links = targets[i]['links']
            link_labels = torch.ones(targets[i]['links'].shape[0], device=device, dtype=torch.int64, )
            face_to_char_links = targets[i]['face_to_char_links']
            face_to_char_lookup = {i.item(): j.item() for i, j in face_to_char_links}

            if use_negative_links:
                negative_links = targets[i]['negative_links']
                links = torch.cat((links, negative_links))
                link_label_neg = torch.zeros(targets[i]['negative_links'].shape[0], device=device, dtype=torch.int64, )
                link_labels = torch.cat((link_labels, link_label_neg))
            # merge links with link labels...
            links = torch.cat((links, link_labels.view(-1, 1)), dim=1)

            if len(links) == 0:
                continue

            from_links = links[:, 0]
            to_links = links[:, 1]
            from_idx_lookup = {i: j.item() for i, j in enumerate(from_links)}
            to_idx_lookup = {i: j.item() for i, j in enumerate(to_links)}
            from_to_lookup = {(i.item(), j.item()): k.item() for i, j, k in
                              torch.cat((links[:, :2], links[:, -1].view(-1, 1)), dim=1)}
            to_label_lookup = {i.item(): j.item() for i, j in links[:, [1, 3]]}
            from_gt_boxes = gt_boxes[from_links]
            to_gt_boxes = gt_boxes[to_links]

            from_match_quality_matrix = box_ops.box_iou(from_gt_boxes, relational_data[from_boxes_key])
            to_match_quality_matrix = box_ops.box_iou(to_gt_boxes, relational_data[to_boxes_key])

            if 0 in from_match_quality_matrix.shape or 0 in to_match_quality_matrix.shape:
                continue

            from_matcher = det_utils.Matcher(
                matcher_iou_threshold,
                matcher_iou_threshold,
                allow_low_quality_matches=False)

            from_matched_idxs_in_image = from_matcher(from_match_quality_matrix)
            from_pos_idxs = from_matched_idxs_in_image >= 0
            from_pos_link_0_idxs = from_matched_idxs_in_image[from_pos_idxs]
            from_pos_boxes = relational_data[from_boxes_key][from_pos_idxs]
            from_pos_box_intersection_scores = from_match_quality_matrix[:, from_pos_idxs].amax(dim=0)
            sort_idxs = torch.argsort(from_pos_box_intersection_scores, dim=0, descending=True)
            from_pos_box_intersection_scores = from_pos_box_intersection_scores[sort_idxs]
            from_pos_link_0_idxs = from_pos_link_0_idxs[sort_idxs]
            from_pos_boxes = from_pos_boxes[sort_idxs]

            to_matcher = det_utils.Matcher(
                matcher_iou_threshold + to_matcher_iou_threshold_addition,
                matcher_iou_threshold + to_matcher_iou_threshold_addition,
                allow_low_quality_matches=False)

            to_matched_idxs_in_image = to_matcher(to_match_quality_matrix)
            to_pos_idxs = to_matched_idxs_in_image >= 0
            to_pos_link_1_idxs = to_matched_idxs_in_image[to_pos_idxs]
            to_pos_boxes = relational_data[to_boxes_key][to_pos_idxs]
            to_pos_box_intersection_scores = to_match_quality_matrix[:, to_pos_idxs].amax(dim=0)
            sort_idxs = torch.argsort(to_pos_box_intersection_scores, dim=0, descending=True)
            to_pos_box_intersection_scores = to_pos_box_intersection_scores[sort_idxs]
            to_pos_link_1_idxs = to_pos_link_1_idxs[sort_idxs]
            to_pos_boxes = to_pos_boxes[sort_idxs]

            pair_pool_data.append({'from_pos_boxes': from_pos_boxes,
                                   'from_pos_link_0_idxs': from_pos_link_0_idxs,
                                   'from_idx_lookup': from_idx_lookup,
                                   'from_pos_box_intersection_scores': from_pos_box_intersection_scores,
                                   'to_pos_boxes': to_pos_boxes,
                                   'to_pos_link_1_idxs': to_pos_link_1_idxs,
                                   'to_idx_lookup': to_idx_lookup,
                                   'to_pos_box_intersection_scores': to_pos_box_intersection_scores,
                                   'to_label_lookup': to_label_lookup,
                                   'from_to_lookup': from_to_lookup,
                                   'face_to_char_lookup': face_to_char_lookup,
                                   'detection_idx': i})

        # extracting from & to features in a single batch
        all_pp_boxes = []
        all_pp_box_lengths = []
        for pair_pool_dict in pair_pool_data:
            all_pp_boxes.append(torch.cat([pair_pool_dict['from_pos_boxes'], pair_pool_dict['to_pos_boxes']]))
            all_pp_box_lengths.append(len(pair_pool_dict['from_pos_boxes']))
            all_pp_box_lengths.append(len(pair_pool_dict['to_pos_boxes']))

        if len(all_pp_boxes) == 0:
            return None, None, None, None, None, None

        for i in range(len(pair_pool_data)):
            detection_idx = pair_pool_data[i]['detection_idx']
            positive_to_labels = []
            positive_bbs = []

            negative_to_labels = []
            negative_bbs = []

            # TODO: @gsoykan - play with this...
            max_sample_len = base_sample_count
            max_sliding_negative_sample_count = 0
            max_mirrored_negative_sample_count = 0
            max_neg_neg_sample_count = 0
            if generate_sliding_window_negative_samples:
                # max_sliding_negative_sample_count = min(20, 5 * self.current_epoch) if not enable_bb_debugging else 35
                max_sliding_negative_sample_count = 5 if not enable_bb_debugging else 20
            if generate_mirrored_by_bubble_center_negative_samples:
                max_mirrored_negative_sample_count = 5 if not enable_bb_debugging else 20
            if use_negative_links:
                max_neg_neg_sample_count = 10 if not enable_bb_debugging else 20

            from_pos_boxes = pair_pool_data[i]['from_pos_boxes']
            from_pos_link_0_idxs = pair_pool_data[i]['from_pos_link_0_idxs']
            # iou of from_box and its gt
            from_pos_box_intersection_scores = pair_pool_data[i]['from_pos_box_intersection_scores']
            from_idx_lookup = pair_pool_data[i]['from_idx_lookup']
            to_pos_boxes = pair_pool_data[i]['to_pos_boxes']
            to_pos_link_1_idxs = pair_pool_data[i]['to_pos_link_1_idxs']
            # iou of to_box and its gt
            to_pos_box_intersection_scores = pair_pool_data[i]['to_pos_box_intersection_scores']
            # given to_link_id returns char or face label
            to_label_lookup = pair_pool_data[i]['to_label_lookup']
            to_idx_lookup = pair_pool_data[i]['to_idx_lookup']
            from_to_lookup = pair_pool_data[i]['from_to_lookup']
            face_to_char_lookup = pair_pool_data[i]['face_to_char_lookup']

            positive_tuples = []
            negative_tuples = []
            neg_negative_tuples = []

            for from_iter_idx, from_link_0_idx in enumerate(from_pos_link_0_idxs):
                from_link_id = from_idx_lookup[from_link_0_idx.item()]
                from_box_intersection_score = from_pos_box_intersection_scores[from_iter_idx]
                for to_iter_idx, to_link_1_idx in enumerate(to_pos_link_1_idxs):
                    to_link_id = to_idx_lookup[to_link_1_idx.item()]
                    # to_label -> char or face..
                    to_label = to_label_lookup[to_link_id]
                    to_box_intersection_score = to_pos_box_intersection_scores[to_iter_idx]
                    # link label -> positive link or negative link...
                    label = from_to_lookup.get((from_link_id, to_link_id), None)
                    content = (from_iter_idx,
                               to_iter_idx,
                               from_box_intersection_score.item(),
                               to_box_intersection_score.item(),
                               from_link_id,
                               to_link_id,
                               to_label)
                    if label is None:
                        negative_tuples.append(content)
                    else:
                        positive_tuples.append(content) if label == 1 else neg_negative_tuples.append(content)

            # TODO: @gsoykan - arrange this...
            # WIP - mix negative links with regular neg links

            if select_samples_by_box_intersection_scores:
                positive_tuples.sort(key=lambda x: x[2] + x[3], reverse=True)
                negative_tuples.sort(key=lambda x: x[2] + x[3], reverse=True)
                if use_negative_links:
                    negative_tuples = [*negative_tuples, *neg_negative_tuples]
                    negative_tuples.sort(key=lambda x: x[2] + x[3], reverse=True)
            else:
                random.shuffle(positive_tuples)
                random.shuffle(negative_tuples)
                # if use_negative_links:
                #     random.shuffle(neg_negative_tuples)

            active_neg_count = max_sample_len \
                               + additional_neg_sample_count \
                               - max_sliding_negative_sample_count \
                               - max_mirrored_negative_sample_count \
                            #   - max_neg_neg_sample_count
            if balance_face_char_sample_counts:
                def balance_tuples(tuples, sample_len):
                    body_tuples = list(filter(lambda x: x[-1] == 1, tuples))
                    face_tuples = list(filter(lambda x: x[-1] == 2, tuples))
                    body_tuples = body_tuples[:min(sample_len, len(body_tuples))]
                    face_tuples = face_tuples[:min(sample_len, len(face_tuples))]
                    tuples = [*body_tuples, *face_tuples]
                    return tuples

                positive_tuples = balance_tuples(positive_tuples, max_sample_len // 2)
                negative_tuples = balance_tuples(negative_tuples, active_neg_count // 2)
                # if use_negative_links:
                #     neg_negative_tuples = balance_tuples(neg_negative_tuples, max_neg_neg_sample_count // 2)
                #     negative_tuples = [*negative_tuples, *neg_negative_tuples]
            else:
                positive_tuples = positive_tuples[:min(max_sample_len, len(positive_tuples))]
                negative_tuples = negative_tuples[:min(active_neg_count, len(negative_tuples))]
                # if use_negative_links:
                #     neg_negative_tuples = neg_negative_tuples[:min(max_neg_neg_sample_count, len(neg_negative_tuples))]
                #     negative_tuples = [*negative_tuples, *neg_negative_tuples]

            def create_sample(from_idx, to_idx, to_link_id, is_positive: bool):
                from_box = from_pos_boxes[from_idx]
                to_box = to_pos_boxes[to_idx]
                to_label = to_label_lookup[to_link_id]
                if is_positive:
                    positive_to_labels.append(to_label)
                    positive_bbs.append(torch.stack([*from_box, *to_box]))
                else:
                    negative_to_labels.append(to_label)
                    negative_bbs.append(torch.stack([*from_box, *to_box]))

            def create_sliding_window_negative_samples_from_positive_sample(from_idx,
                                                                            to_idx,
                                                                            from_link_id,
                                                                            to_link_id):
                # this can definitely be optimized but let's go for now...
                from_box = from_pos_boxes[from_idx]
                original_to_box = to_pos_boxes[to_idx]
                to_label = to_label_lookup[to_link_id]
                candidates = []

                to_associated_box = None
                if filter_body_intersected_generated_negative_face_samples:
                    # at this point we don't know whether to is face or not
                    char_link_id = face_to_char_lookup.get(to_link_id, None)
                    if char_link_id is not None:
                        to_associated_box = targets[detection_idx]['boxes'][char_link_id]

                new_to_boxes = []
                if generate_sliding_window_negative_samples:
                    new_to_boxes.extend(self.slide_box_in_encapsulating_box(original_to_box, from_box,
                                                                            image_sizes[detection_idx],
                                                                            to_associated_box))

                if generate_mirrored_by_bubble_center_negative_samples:
                    new_to_boxes.extend(self.generate_mirrored_samples_by_from_box_center(original_to_box, from_box,
                                                                                          image_sizes[detection_idx],
                                                                                          to_associated_box))

                if len(new_to_boxes) == 0:
                    return candidates

                for new_to_box in new_to_boxes:
                    canditate = {'to_label': to_label,
                                 'from_box': from_box,
                                 'new_to_box': new_to_box}
                    candidates.append(canditate)
                return candidates

            def handle_negative_sliding_window_candidates(candidates):
                random.shuffle(candidates)
                sliding_neg_count = max_sliding_negative_sample_count + max_mirrored_negative_sample_count
                candidates = candidates[:min(sliding_neg_count, len(candidates))]
                for candidate in candidates:
                    negative_to_labels.append(candidate['to_label'])
                    negative_bbs.append(torch.stack(
                        [*candidate['from_box'], *candidate['new_to_box']]))

            [create_sample(x[0], x[1], x[5], True) for x in positive_tuples]
            [create_sample(x[0], x[1], x[5], False) for x in negative_tuples]
            if generate_sliding_window_negative_samples or generate_mirrored_by_bubble_center_negative_samples:
                candidates = flatten_list(
                    [create_sliding_window_negative_samples_from_positive_sample(x[0], x[1], x[4], x[5]) for x in
                     positive_tuples])
                handle_negative_sliding_window_candidates(candidates)

            if len(positive_to_labels) == 0:
                continue

            positive_bbs = torch.stack(positive_bbs)
            if len(negative_to_labels) != 0:
                negative_bbs = torch.stack(negative_bbs)

            all_bbs = torch.cat([positive_bbs, negative_bbs]) if len(negative_bbs) != 0 else positive_bbs
            encapsulation_boxes = self.get_encapsulating_box_batched(all_bbs[:, :4], all_bbs[:, 4:8])
            all_bbs = torch.cat((all_bbs, encapsulation_boxes), dim=1)

            encapsulation_edge_maps = None
            if edge_maps is not None:
                encapsulation_edge_maps = self.crop_from_edge_map(edge_maps[detection_idx], all_bbs[:, 8:])
                all_encapsulation_edge_maps.append(encapsulation_edge_maps)

            stacked_sample_bbs = torch.cat((all_bbs[:, 8:], all_bbs[:, :4], all_bbs[:, 4:8]), dim=0)
            samples = custom_bb_feature_extractor(detection_idx, stacked_sample_bbs, True)
            samples = rearrange(samples, "(n b) d h w -> b n d h w", n=3)

            positive_to_labels = torch.tensor(positive_to_labels, device=device)
            if len(negative_to_labels) != 0:
                negative_to_labels = torch.tensor(negative_to_labels, device=device)

            labels = torch.cat([torch.ones(len(positive_to_labels)), torch.zeros(len(negative_to_labels))]).to(device)
            to_labels = torch.cat([positive_to_labels, negative_to_labels]) if len(
                negative_to_labels) != 0 else positive_to_labels

            # gather all spatial features...
            all_labels.append(labels)
            all_to_labels.append(to_labels)
            all_samples.append(samples)

            spatial_features = self.create_spatial_features_batched(all_bbs[:, :4], all_bbs[:, 4:8])
            all_spatial_features.append(spatial_features)

            if enable_encapsulation_box_masking:
                encapsulation_box_masks = self.create_encapsulation_box_mask_batched(all_bbs)
                all_encapsulation_box_masks.append(encapsulation_box_masks)

            del all_bbs
            if enable_bb_debugging:
                relation_bbs = {
                    'positive': positive_bbs,
                    'negative': negative_bbs
                }
                all_relation_bbs.append(relation_bbs)
            else:
                del positive_bbs
                del negative_bbs

        if len(all_labels) == 0:
            return None, None, None, None, None, None, None

        all_labels = torch.cat(all_labels)
        all_to_labels = torch.cat(all_to_labels)
        all_samples = torch.cat(all_samples)
        all_spatial_features = torch.cat(all_spatial_features)
        if len(all_encapsulation_box_masks) != 0:
            all_encapsulation_box_masks = torch.cat(all_encapsulation_box_masks)
        else:
            all_encapsulation_box_masks = None
        if len(all_encapsulation_edge_maps) != 0:
            all_encapsulation_edge_maps = torch.cat(all_encapsulation_edge_maps)
        else:
            all_encapsulation_edge_maps = None
        # print('mean encapsulation iou: ', np.array(encapsulation_iou_values).mean())
        return all_samples, all_spatial_features, all_labels, all_relation_bbs, all_to_labels, all_encapsulation_box_masks, all_encapsulation_edge_maps

    def create_spatial_features(self, from_box: torch.Tensor, to_box: torch.Tensor) -> torch.Tensor:
        def to_xc_yc_w_h(box):
            x_c = (box[2] + box[0]) / 2
            y_c = (box[3] + box[1]) / 2
            w = box[2] - box[0]
            h = box[3] - box[1]
            return x_c, y_c, w, h

        x_b, y_b, w_b, h_b = to_xc_yc_w_h(from_box)
        x_c, y_c, w_c, h_c = to_xc_yc_w_h(to_box)
        f_0 = (x_b - x_c) / w_b
        f_1 = (y_b - y_c) / y_b
        f_2 = (x_b - x_c) / w_c
        f_3 = (y_b - y_c) / y_c
        f_4 = box_ops.box_iou(from_box.view(1, -1),
                              to_box.view(1, -1))[0]
        spatial_features = torch.cat([f_0.view(1), f_1.view(1), f_2.view(1), f_3.view(1), f_4])
        return spatial_features

    def create_spatial_features_batched(self,
                                        from_box: torch.Tensor,
                                        to_box: torch.Tensor) -> torch.Tensor:
        def to_xc_yc_w_h(box):
            x_c = (box[:, 2] + box[:, 0]) / 2
            y_c = (box[:, 3] + box[:, 1]) / 2
            w = box[:, 2] - box[:, 0]
            h = box[:, 3] - box[:, 1]
            return x_c, y_c, w, h

        x_b, y_b, w_b, h_b = to_xc_yc_w_h(from_box)
        x_c, y_c, w_c, h_c = to_xc_yc_w_h(to_box)
        f_0 = (x_b - x_c) / w_b
        f_1 = (y_b - y_c) / y_b
        f_2 = (x_b - x_c) / w_c
        f_3 = (y_b - y_c) / y_c
        f_4 = self.box_iou_diag(from_box, to_box)
        spatial_features = torch.cat(
            [f_0.unsqueeze(1),
             f_1.unsqueeze(1),
             f_2.unsqueeze(1),
             f_3.unsqueeze(1),
             f_4.unsqueeze(1)],
            dim=1)
        return spatial_features

    def create_encapsulation_box_mask(self,
                                      encapsulation_box: torch.Tensor,
                                      from_box: torch.Tensor,
                                      to_box: torch.Tensor,
                                      version: int = 3) -> torch.Tensor:
        """
        we can have couple different versions
        - 1) lighting from and to boxes withing encapsulation box
        - 2) lighting from and to boxes withing encapsulation box and their line connections
        - 3) lighting from and to boxes withing encapsulation box and their line connections + surrounding = 1
        - 4) lighting from and to boxes withing encapsulation box - 0 değil 0.25 ile masklemek
        - 5) lighting from and to boxes withing encapsulation box and their line connections + surrounding = 1
        + line üstünde olmayanların effecti azalsın
        """

        def translate_box(box, min_x, min_y):
            box[0] -= min_x
            box[2] -= min_x
            box[1] -= min_y
            box[3] -= min_y
            return box

        def rescale_box(box, scale_factor_width, scale_factor_height):
            rescaled_box = box * torch.tensor(
                [scale_factor_width, scale_factor_height, scale_factor_width, scale_factor_height], device=box.device)
            rescaled_box = torch.round(torch.tensor(rescaled_box)).int()
            return rescaled_box

        encapsulation_box = encapsulation_box.clone()
        from_box = from_box.clone()
        to_box = to_box.clone()
        if version in [1, 2, 3, 4]:
            mask = torch.zeros((7, 7), device=encapsulation_box.device)
            if version == 4:
                mask += 0.25
            # the first thing is to translate and then scale
            min_x = encapsulation_box[0].item()
            min_y = encapsulation_box[1].item()
            encapsulation_box = translate_box(encapsulation_box, min_x, min_y)
            encapsulation_w = encapsulation_box[2].item() - encapsulation_box[0].item()
            encapsulation_h = encapsulation_box[3].item() - encapsulation_box[1].item()
            scale_factor_width = 7 / encapsulation_w
            scale_factor_height = 7 / encapsulation_h

            from_box = translate_box(from_box, min_x, min_y)
            to_box = translate_box(to_box, min_x, min_y)

            from_box = rescale_box(from_box, scale_factor_width, scale_factor_height)
            to_box = rescale_box(to_box, scale_factor_width, scale_factor_height)
            # this is because feature dimensions are C x H x W
            mask[from_box[1]:from_box[3], from_box[0]:from_box[2]] = 1
            mask[to_box[1]:to_box[3], to_box[0]:to_box[2]] = 1
            if version in [2, 3]:
                def get_box_center(box):
                    x_c = (box[2] + box[0]) // 2
                    y_c = (box[3] + box[1]) // 2
                    return x_c, y_c

                start_point = get_box_center(from_box)
                end_point = get_box_center(to_box)
                mask = self.draw_line_in_mask(mask, start_point, end_point, light_neighboring=version == 3)
            return mask
        else:
            raise Exception('unhandled version for create_encapsulation_box_mask')

    def create_encapsulation_box_mask_batched(self,
                                              batch_bbs: torch.Tensor,
                                              version: int = 3) -> torch.Tensor:

        encapsulation_box = batch_bbs[:, 8:].clone()
        from_box = batch_bbs[:, :4].clone()
        to_box = batch_bbs[:, 4:8].clone()

        def translate_box(box, min_x, min_y):
            box[:, 0] -= min_x
            box[:, 2] -= min_x
            box[:, 1] -= min_y
            box[:, 3] -= min_y
            return box

        def rescale_box(batch_box, batch_scale_factor_width, batch_scale_factor_height):
            rescaled_box = batch_box * torch.stack([
                batch_scale_factor_width,
                batch_scale_factor_height,
                batch_scale_factor_width,
                batch_scale_factor_height
            ], dim=1)
            rescaled_box = torch.round(rescaled_box).int()
            return rescaled_box

        batch_size = encapsulation_box.shape[0]

        if version in [1, 2, 3, 4]:
            mask = torch.zeros((batch_size, 7, 7), device=encapsulation_box.device)
            if version == 4:
                mask += 0.25

            # the first thing is to translate and then scale
            min_x = encapsulation_box[:, 0]
            min_y = encapsulation_box[:, 1]
            encapsulation_w = encapsulation_box[:, 2] - encapsulation_box[:, 0]
            encapsulation_h = encapsulation_box[:, 3] - encapsulation_box[:, 1]
            scale_factor_width = 7 / encapsulation_w
            scale_factor_height = 7 / encapsulation_h

            fb = translate_box(from_box, min_x, min_y)
            tb = translate_box(to_box, min_x, min_y)

            fb = rescale_box(fb, scale_factor_width, scale_factor_height)
            tb = rescale_box(tb, scale_factor_width, scale_factor_height)

            def mask_with_box(mask, box):
                y_indices = torch.arange(7, device=box.device).unsqueeze(0).unsqueeze(-1)
                x_indices = torch.arange(7, device=box.device).unsqueeze(0).unsqueeze(1)
                mask[(y_indices >= box[:, 1].unsqueeze(1).unsqueeze(1)) & (
                        y_indices < box[:, 3].unsqueeze(1).unsqueeze(1)) & (
                             x_indices >= box[:, 0].unsqueeze(1).unsqueeze(1)) & (
                             x_indices < box[:, 2].unsqueeze(1).unsqueeze(1))] = 1

            mask_with_box(mask, fb)
            mask_with_box(mask, tb)

            if version in [2, 3]:
                def get_box_center(box):
                    x_c = (box[:, 2] + box[:, 0]) // 2
                    y_c = (box[:, 3] + box[:, 1]) // 2
                    return x_c, y_c

                start_point, end_point = get_box_center(fb), get_box_center(tb)

                # for testing purposes...
                # tms = []
                # for i, m in enumerate(mask):
                #     r = self.draw_line_in_mask(m.clone(), start_point[i], end_point[i], light_neighboring=version==3)
                #     tms.append(r)

                mask = self.draw_line_in_mask_batched(mask, start_point, end_point, light_neighboring=version == 3)

            return mask

    def get_encapsulating_box(self, box1: torch.Tensor, box2) -> torch.Tensor:
        """
        Returns a bounding box that encapsulates both input bounding boxes.

        Args:
            box1 (torch.Tensor): Tensor of shape (4,) representing the coordinates of
                the first bounding box in the format [xmin, ymin, xmax, ymax].
            box2 (torch.Tensor): Tensor of shape (4,) representing the coordinates of
                the second bounding box in the format [xmin, ymin, xmax, ymax].

        Returns:
            torch.Tensor: Tensor of shape (4,) representing the coordinates of the
                encapsulating bounding box in the format [xmin, ymin, xmax, ymax].
        """
        xmin = torch.min(box1[0], box2[0])
        ymin = torch.min(box1[1], box2[1])
        xmax = torch.max(box1[2], box2[2])
        ymax = torch.max(box1[3], box2[3])

        return torch.tensor([xmin, ymin, xmax, ymax], dtype=box1.dtype, device=box1.device)

    def get_encapsulating_box_batched(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Returns a bounding box that encapsulates both input bounding boxes.

        Args:
            boxes1 (torch.Tensor): Tensor of shape (N, 4) representing the coordinates of
                the first set of bounding boxes in the format [xmin, ymin, xmax, ymax].
            boxes2 (torch.Tensor): Tensor of shape (N, 4) representing the coordinates of
                the second set of bounding boxes in the format [xmin, ymin, xmax, ymax].

        Returns:
            torch.Tensor: Tensor of shape (N, 4) representing the coordinates of the
                encapsulating bounding boxes in the format [xmin, ymin, xmax, ymax].
        """
        xmin = torch.min(boxes1[:, 0], boxes2[:, 0])
        ymin = torch.min(boxes1[:, 1], boxes2[:, 1])
        xmax = torch.max(boxes1[:, 2], boxes2[:, 2])
        ymax = torch.max(boxes1[:, 3], boxes2[:, 3])

        return torch.stack([xmin, ymin, xmax, ymax], dim=1)

    def draw_line_in_mask(self, mask, start_point, end_point, light_neighboring: bool = False):
        # Extract x and y coordinates of start and end points
        H, W = mask.shape
        x0, y0 = min(W - 1, start_point[0]), min(H - 1, start_point[1])
        x1, y1 = min(W - 1, end_point[0]), min(H - 1, end_point[1])

        # Compute differences between start and end points
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        # Determine direction of the line
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        # Compute the initial error
        error = dx - dy

        # Initialize the current position to the start point
        x, y = x0, y0

        # Loop until we reach the end point
        while x != x1 or y != y1:
            # Append the current position to the list of points
            mask[y, x] = 1

            if light_neighboring:
                for n_x, n_y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    n_x = x + n_x
                    n_y = y + n_y
                    if n_x >= 0 and n_x < W and n_y >= 0 and n_y < H:
                        mask[n_y, n_x] = 1

            # Compute the error for the next position
            e2 = 2 * error

            # Determine which direction to move
            if e2 > -dy:
                error = error - dy
                x = x + sx
            if e2 < dx:
                error = error + dx
                y = y + sy

        # Append the final position to the list of points
        mask[y1, x1] = 1

        return mask

    # slightly different than the single one but more or less the same...
    def draw_line_in_mask_batched(self, mask, start_point, end_point, light_neighboring=True):
        """
        Draws a line in the given batch of masks.

        Args:
            mask (torch.Tensor): Batch of masks with shape (B, H, W)
            start_point (List of Tuples): Batch of start points with shape (B, 2)
            end_point (List of Tuples): Batch of end points with shape (B, 2)

        Returns:
            torch.Tensor: Updated batch of masks with the line drawn, with shape (B, H, W)
        """
        B, H, W = mask.shape
        device = mask.device

        start_point = torch.stack(start_point, dim=1)
        end_point = torch.stack(end_point, dim=1)

        # Extract x and y coordinates of start and end points
        start_point = torch.min(torch.tensor([W - 1, H - 1], device=device),
                                torch.max(torch.tensor([0, 0], device=device), start_point))
        end_point = torch.min(torch.tensor([W - 1, H - 1], device=device),
                              torch.max(torch.tensor([0, 0], device=device), end_point))
        x0, y0 = start_point[:, 0], start_point[:, 1]
        x1, y1 = end_point[:, 0], end_point[:, 1]

        # Compute differences between start and end points
        dx = torch.abs(x1 - x0)
        dy = torch.abs(y1 - y0)

        # Determine direction of the line
        sx = torch.where(x0 < x1, torch.ones_like(x0), -torch.ones_like(x0))
        sy = torch.where(y0 < y1, torch.ones_like(y0), -torch.ones_like(y0))

        # Compute the initial error
        error = dx - dy

        # Initialize the current position to the start point
        x, y = x0.clone(), y0.clone()

        # Loop until we reach the end point
        while ((x != x1) & (y != y1)).any():
            # Append the current position to the list of points
            mask[torch.arange(B), y, x] = 1

            if light_neighboring:
                for n_x, n_y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    n_x = x + n_x
                    n_y = y + n_y
                    valid_x = (n_x >= 0) & (n_x < W)
                    valid_y = (n_y >= 0) & (n_y < H)
                    valid = valid_x & valid_y
                    mask[torch.arange(B, device=mask.device)[valid], n_y[valid], n_x[valid]] = 1

            # Compute the error for the next position
            e2 = 2 * error

            # Determine which direction to move
            i_e_dy = e2 > -dy
            error[i_e_dy] -= dy[i_e_dy]
            x[i_e_dy] += sx[i_e_dy]
            x[i_e_dy] = torch.clamp(x[i_e_dy], torch.min(x0[i_e_dy], x1[i_e_dy]), torch.max(x0[i_e_dy], x1[i_e_dy]))

            i_e_dx = e2 < dx
            error[i_e_dx] += dx[i_e_dx]
            y[i_e_dx] += sy[i_e_dx]
            y[i_e_dx] = torch.clamp(y[i_e_dx], torch.min(y0[i_e_dx], y1[i_e_dx]), torch.max(y0[i_e_dx], y1[i_e_dx]))

        # Append the final position to the list of points
        mask[torch.arange(B), y1, x1] = 1

        return mask

    def slide_box_in_encapsulating_box(self,
                                       box: torch.Tensor,
                                       from_box: torch.Tensor,
                                       image_size: Tuple[int, int],
                                       to_associated_box: Optional[torch.Tensor],
                                       min_area_ratio: float = 0.5,
                                       slide_stride_ratio: int = 1.25) -> List[torch.Tensor]:
        width = box[2].item() - box[0].item()
        height = box[3].item() - box[1].item()
        box_area = self.bbox_area(box)
        img_h, img_w = image_size
        to_associated_box: List = to_associated_box.tolist() if to_associated_box is not None else None

        new_boxes = []

        def append_to_new_boxes(new_box):
            if to_associated_box is not None:
                to_associated_box_intersection_rate = box_intersection_rate(new_box.tolist(), to_associated_box)
                if to_associated_box_intersection_rate >= 0.50:
                    return
            if (self.bbox_area(new_box) / box_area) >= min_area_ratio:
                new_boxes.append(new_box)

        left_box = box.clone()
        left_box[0] -= (width * slide_stride_ratio)
        left_box[2] -= (width * slide_stride_ratio)
        left_box[0] = max(left_box[0], 0)
        left_box[2] = max(left_box[2], 0)
        if left_box[0].item() != left_box[2].item():
            append_to_new_boxes(left_box)

        right_box = box.clone()
        right_box[0] += (width * slide_stride_ratio)
        right_box[2] += (width * slide_stride_ratio)
        right_box[0] = min(right_box[0], img_w)
        right_box[2] = min(right_box[2], img_w)
        if right_box[0].item() != right_box[2].item():
            append_to_new_boxes(right_box)

        up_box = box.clone()
        up_box[1] -= (height * slide_stride_ratio)
        up_box[3] -= (height * slide_stride_ratio)
        up_box[1] = max(up_box[1], 0)
        up_box[3] = max(up_box[3], 0)
        if up_box[1].item() != up_box[3].item():
            append_to_new_boxes(up_box)

        down_box = box.clone()
        down_box[1] += (height * slide_stride_ratio)
        down_box[3] += (height * slide_stride_ratio)
        down_box[1] = min(down_box[1], img_h)
        down_box[3] = min(down_box[3], img_w)
        if down_box[1].item() != down_box[3].item():
            append_to_new_boxes(down_box)

        return new_boxes

    def generate_mirrored_samples_by_from_box_center(self,
                                                     box: torch.Tensor,
                                                     from_box: torch.Tensor,
                                                     image_size: Tuple[int, int],
                                                     to_associated_box: Optional[torch.Tensor]) -> List[torch.Tensor]:
        img_h, img_w = image_size
        box = box.clone()
        to_associated_box: List = to_associated_box.tolist() if to_associated_box is not None else None
        new_boxes = []

        # Calculate the center coordinates of from_box and to_box
        from_center_x = (from_box[2].item() + from_box[0].item()) / 2
        from_center_y = (from_box[3].item() + from_box[1].item()) / 2
        to_center_x = (box[2].item() + box[0].item()) / 2
        to_center_y = (box[3].item() + box[1].item()) / 2

        # Calculate the x and y distances between the centers of from_box and to_box
        dist_x = to_center_x - from_center_x
        dist_y = to_center_y - from_center_y

        # Calculate the symmetrical bounding box
        sym_x_0 = from_center_x - dist_x - (box[2] - to_center_x)
        sym_x_1 = from_center_x - dist_x + (to_center_x - box[0])
        sym_y_0 = from_center_y - dist_y - (box[3] - to_center_y)
        sym_y_1 = from_center_y - dist_y + (to_center_y - box[1])
        symmetric_box = torch.tensor([sym_x_0, sym_y_0, sym_x_1, sym_y_1], device=box.device)
        vertical_sym_box = torch.tensor([box[0], sym_y_0, box[2], sym_y_1], device=box.device)
        horizontal_sym_box = torch.tensor([sym_x_0, box[1], sym_x_1, box[3]], device=box.device)
        new_boxes.extend([symmetric_box, vertical_sym_box, horizontal_sym_box])
        # 1) these should not intersect with GT
        new_boxes = list(filter(lambda x: box_ops.box_iou(x.view(1, -1),
                                                          box.view(1, -1))[0] == 0, new_boxes))

        def check_bb_in_img_boundary(x: torch.Tensor) -> bool:
            zero_check = (x >= 0).all()
            return zero_check and x[0] <= img_w and x[2] <= img_w and x[1] <= img_h and x[3] <= img_h

        def check_intersection_with_associated_box(x: torch.Tensor) -> bool:
            to_associated_box_intersection_rate = box_intersection_rate(x.tolist(), to_associated_box)
            if to_associated_box_intersection_rate >= 0.50:
                return False
            else:
                return True

        # 2) these should stay in image boundaries
        new_boxes = list(filter(lambda x: check_bb_in_img_boundary(x), new_boxes))

        if to_associated_box is not None:
            new_boxes = list(filter(lambda x: check_intersection_with_associated_box(x), new_boxes))

        return new_boxes

    def bbox_area(self, bbox):
        """
        Calculates the area of a bounding box in PyTorch.

        Args:
            bbox: a 1D tensor of shape (4,) representing the coordinates of a bounding box
                in the form (x1, y1, x2, y2).

        Returns:
            A scalar tensor representing the area of the bounding box.
        """
        width = bbox[2].item() - bbox[0].item()
        height = bbox[3].item() - bbox[1].item()
        area = width * height
        return area

    def box_iou_diag(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Return intersection-over-union (Jaccard index) between two sets of boxes.

        Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
        ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

        Args:
            boxes1 (Tensor[N, 4]): first set of boxes
            boxes2 (Tensor[N, 4]): second set of boxes

        Returns:
            Tensor[N, 1]: the Nx1 matrix containing the pairwise IoU values for every pair of element in boxes1 and boxes2
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = torch.max(boxes1[:, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
        iou = inter / (area1 + area2 - inter)

        return iou

    def extract_edge_maps(self, images):
        edge_maps = []
        for image in images.tensors:
            # Add a batch dimension to the input tensor
            grayscale_img = torchvision.transforms.Compose([
                torchvision.transforms.GaussianBlur(kernel_size=7),
                torchvision.transforms.Grayscale()
            ])(image)

            input_tensor = grayscale_img.unsqueeze(0)

            # Define the Sobel filter kernels for x and y direction
            sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
                                          device=input_tensor.device)
            sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,
                                          device=input_tensor.device)

            # Apply the Sobel filter to the input tensor for both x and y direction
            edge_tensor_x = F.conv2d(input_tensor, sobel_kernel_x.unsqueeze(0).unsqueeze(0), padding=1)
            edge_tensor_y = F.conv2d(input_tensor, sobel_kernel_y.unsqueeze(0).unsqueeze(0), padding=1)

            # Compute the magnitude of the edge tensor
            edge_tensor = torch.sqrt(edge_tensor_x ** 2 + edge_tensor_y ** 2)

            # Remove the batch and channel dimensions from the edge tensor
            edge_tensor = edge_tensor.squeeze(0).squeeze(0)
            # edge_tensor = edge_tensor / edge_tensor.max()
            edge_maps.append(edge_tensor)
        return edge_maps

    def crop_from_edge_map(self,
                           edge_map: torch.Tensor,
                           boxes: torch.Tensor,
                           crop_size=(64, 64)) -> torch.Tensor:
        # create an empty tensor to store the cropped boxes
        cropped_boxes = []

        # loop over the boxes and crop each one
        for b in range(boxes.shape[0]):
            # extract the coordinates of the current box
            x_min, y_min, x_max, y_max = [int(i.item()) for i in boxes[b, :]]

            # crop the edge map using the current box
            cropped_box = torchvision.transforms.functional.crop(edge_map, top=y_min, left=x_min, height=y_max - y_min,
                                                                 width=x_max - x_min)

            cropped_box = torchvision.transforms.Compose([
                SquarePad(),
                torchvision.transforms.Resize(crop_size)
            ])(cropped_box.unsqueeze(0)).squeeze(0)

            # resize the cropped box to the desired size
            # cropped_box = F.interpolate(cropped_box.unsqueeze(0).unsqueeze(0), size=crop_size, mode='bilinear',
            #                             align_corners=False).squeeze(0).squeeze(0)

            # append the cropped box to the list of cropped boxes
            cropped_boxes.append(cropped_box)

        # stack the cropped boxes along the batch dimension to create a tensor of shape [B, H, W]
        cropped_boxes = torch.stack(cropped_boxes, dim=0)
        return cropped_boxes


import torchvision.transforms.functional as T_F


class SquarePad:
    def __call__(self, image):
        h, w = image.size()[-2:]
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return T_F.pad(image, padding, 0, 'constant')


"""
import matplotlib.pyplot as plt
plt.imshow(  cropped_box.cpu().unsqueeze(0).permute(1, 2, 0)  )
plt.show()
"""
