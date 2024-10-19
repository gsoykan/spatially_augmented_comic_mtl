from collections import OrderedDict
from typing import Optional, Dict, List, Tuple

import torch
import torchvision
from torch import Tensor
from torch import nn
from torchinfo import summary

from src.models.components.component_detection.customized_faster_rcnn import faster_rcnn
from src.models.components.component_detection.customized_faster_rcnn import mask_rcnn
from src.models.components.component_detection.customized_faster_rcnn.faster_rcnn import FastRCNNPredictor, FasterRCNN
from src.models.components.component_detection.customized_faster_rcnn.mask_rcnn import MaskRCNN, MaskRCNNPredictor

from src.models.components.component_detection.customized_faster_rcnn_112 import faster_rcnn as faster_rcnn_112, \
    mask_rcnn as mask_rcnn_112

"""
WIP:
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

https://pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/
    Faster R-CNN with a ResNet50 backbone (more accurate, but slower)
    Faster R-CNN with a MobileNet v3 backbone (faster, but less accurate)
    RetinaNet with a ResNet50 backbone (good balance between speed and accuracy)
"""


class PtDetector(nn.Module):
    def __init__(self,
                 model_name: str = "fasterrcnn_resnet50_fpn",
                 num_classes=6,
                 use_filtered_boxes_for_pair_pool: bool = True,
                 relation_network_first_layer_type='linear',
                 enable_encapsulation_box_masking: bool = True,
                 to_matcher_iou_threshold_addition: float = 0.0,
                 generate_sliding_window_negative_samples: bool = False,
                 generate_mirrored_by_bubble_center_negative_samples: bool = True,
                 use_negative_links: bool = True,
                 base_sample_count: int = 75,
                 additional_neg_sample_count: int = 0,
                 relation_network_feat_embedding_type: Optional[str] = None,
                 relation_network_encapsulation_box_masks_strategy: Optional[str] = None,
                 use_object_relation_modules: bool = False,
                 box_head_output_strategy: Optional[str] = None,
                 select_samples_by_box_intersection_scores: bool = True,
                 filter_body_intersected_generated_negative_face_samples: bool = True,
                 balance_face_char_sample_counts: bool = True,
                 relation_network_representation_size: int = 256,
                 use_edge_maps: bool = False
                 ):
        """
        During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses.

        During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detections:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each detection
        - scores (``Tensor[N]``): the scores of each detection
        Args:
            model_name (): "frcnn-resnet",  "frcnn-mobilenet",  "retinanet"
            num_classes (): including background
        """
        super().__init__()
        MODELS = {
            'fasterrcnn_resnet50_fpn': faster_rcnn.fasterrcnn_resnet50_fpn,
            'maskrcnn_resnet50_fpn': mask_rcnn.maskrcnn_resnet50_fpn,
            'fasterrcnn_resnet50_fpn_v2': faster_rcnn_112.fasterrcnn_resnet50_fpn_v2,
            'maskrcnn_resnet50_fpn_v2': mask_rcnn_112.maskrcnn_resnet50_fpn_v2,
        }
        self.use_filtered_boxes_for_pair_pool = use_filtered_boxes_for_pair_pool
        self.uses_mask = True if 'mask' in model_name else False
        if 'v2' in model_name:
            self.model = MODELS[model_name](weights="DEFAULT",
                                            use_filtered_boxes_for_pair_pool=use_filtered_boxes_for_pair_pool,
                                            relation_network_first_layer_type=relation_network_first_layer_type,
                                            enable_encapsulation_box_masking=enable_encapsulation_box_masking,
                                            to_matcher_iou_threshold_addition=to_matcher_iou_threshold_addition,
                                            generate_sliding_window_negative_samples=generate_sliding_window_negative_samples,
                                            generate_mirrored_by_bubble_center_negative_samples=generate_mirrored_by_bubble_center_negative_samples,
                                            use_negative_links=use_negative_links,
                                            base_sample_count=base_sample_count,
                                            additional_neg_sample_count=additional_neg_sample_count,
                                            relation_network_feat_embedding_type=relation_network_feat_embedding_type,
                                            relation_network_encapsulation_box_masks_strategy=relation_network_encapsulation_box_masks_strategy,
                                            use_object_relation_modules=use_object_relation_modules,
                                            box_head_output_strategy=box_head_output_strategy,
                                            select_samples_by_box_intersection_scores=select_samples_by_box_intersection_scores,
                                            filter_body_intersected_generated_negative_face_samples=filter_body_intersected_generated_negative_face_samples,
                                            balance_face_char_sample_counts=balance_face_char_sample_counts,
                                            relation_network_representation_size=relation_network_representation_size,
                                            use_edge_maps=use_edge_maps
                                            )
        else:
            self.model: FasterRCNN = MODELS[model_name](pretrained=True,
                                                        use_filtered_boxes_for_pair_pool=use_filtered_boxes_for_pair_pool,
                                                        relation_network_first_layer_type=relation_network_first_layer_type,
                                                        enable_encapsulation_box_masking=enable_encapsulation_box_masking,
                                                        to_matcher_iou_threshold_addition=to_matcher_iou_threshold_addition,
                                                        generate_sliding_window_negative_samples=generate_sliding_window_negative_samples,
                                                        generate_mirrored_by_bubble_center_negative_samples=generate_mirrored_by_bubble_center_negative_samples,
                                                        use_negative_links=use_negative_links,
                                                        base_sample_count=base_sample_count,
                                                        additional_neg_sample_count=additional_neg_sample_count,
                                                        relation_network_feat_embedding_type=relation_network_feat_embedding_type,
                                                        relation_network_encapsulation_box_masks_strategy=relation_network_encapsulation_box_masks_strategy,
                                                        use_object_relation_modules=use_object_relation_modules,
                                                        box_head_output_strategy=box_head_output_strategy,
                                                        select_samples_by_box_intersection_scores=select_samples_by_box_intersection_scores,
                                                        filter_body_intersected_generated_negative_face_samples=filter_body_intersected_generated_negative_face_samples,
                                                        balance_face_char_sample_counts=balance_face_char_sample_counts,
                                                        relation_network_representation_size=relation_network_representation_size,
                                                        use_edge_maps=use_edge_maps
                                                        )

        # get the number of input features
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        if self.uses_mask:
            # now get the number of input features for the mask classifier
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            # and replace the mask predictor with a new one
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                                    hidden_layer,
                                                                    num_classes=num_classes)

    def forward(self,
                images: Tensor,
                targets: Optional[List[Dict[str, Tensor]]] = None,
                current_epoch: Optional[int] = None) -> Tuple[
        Optional[Dict[str, Tensor]], List[Dict[str, Tensor]]]:
        """
        idea source: https://stackoverflow.com/a/72321898/8265079
        """
        self.model.current_epoch = current_epoch
        return self.model(images, targets)

    def enable_relation_network(self, enable: bool):
        self.model.is_relation_network_enabled = enable

    def set_force_get_inference_results(self, value: bool):
        self.model.force_get_inference_results = value

    def get_force_inference_results(self):
        return self.model.force_get_inference_results


if __name__ == '__main__':
    detector = PtDetector(model_name='fasterrcnn_resnet50_fpn_v2')
    detector.to('cuda')
    detector.eval()
    summary(detector, input_size=(1, 3, 256, 256))
