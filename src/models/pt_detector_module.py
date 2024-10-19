import random
from itertools import groupby
from pprint import pprint
from typing import Any, List, Optional, Dict

import cv2
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, Accuracy
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.models.components.component_detection.pt_detector import PtDetector
from src.utils.basic_utils import flatten_list
from src.utils.component_detection.face_body_char_evaluation import ComicPageFaceBodyCharEvaluator
from src.utils.component_detection.relation_evaluation import ComicPageRelationEvaluator
from src.utils.pickle_helper import PickleHelper
from torchvision.ops import boxes as box_ops
import torch.nn.functional as F

from src.utils.scheduler.cosine_warmup_scheduler import CosineWarmupScheduler


class PtDetectorLitModule(LightningModule):
    def __init__(
            self,
            model_name: str = "fasterrcnn_resnet50_fpn",
            num_classes=6,
            use_filtered_boxes_for_pair_pool: bool = True,
            relation_network_first_layer_type='linear',
            enable_encapsulation_box_masking: bool = True,
            to_matcher_iou_threshold_addition: float = 0.0,
            generate_sliding_window_negative_samples: bool = False,
            generate_mirrored_by_bubble_center_negative_samples: bool = False,
            use_negative_links: bool = False,
            base_sample_count: int = 75,
            additional_neg_sample_count: int = 0,
            relation_network_feat_embedding_type: Optional[str] = None,
            relation_network_encapsulation_box_masks_strategy: Optional[str] = None,
            use_object_relation_modules: bool = False,
            box_head_output_strategy: Optional[str] = None,
            select_samples_by_box_intersection_scores: bool = True,
            filter_body_intersected_generated_negative_face_samples: bool = True,
            balance_face_char_sample_counts: bool = True,
            relation_network_representation_size: int = 128,
            use_edge_maps: bool = False,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            freeze_relation_head_10_ep: bool = False,
            log_relation_score_every_n_epoch: int = 5
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = PtDetector(model_name,
                                num_classes,
                                use_filtered_boxes_for_pair_pool,
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
        self.train_mAP = MeanAveragePrecision(class_metrics=True)
        self.val_mAP = MeanAveragePrecision(class_metrics=True)
        self.test_mAP = MeanAveragePrecision(class_metrics=True)
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.val_acc_best = MaxMetric()
        self.val_mAP_best = MaxMetric()

        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()
        self.val_f1_best = MaxMetric()
        self.comic_page_relation_evaluator = ComicPageRelationEvaluator(relation_score_threshold=0.75,
                                                                        box_score_threshold=0.5)
        self.comic_page_relation_evaluator_char_aligned = ComicPageRelationEvaluator(relation_score_threshold=0.75,
                                                                                     box_score_threshold=0.5)

        self.optimizer = None
        self.lr_scheduler = None
        self.set_initial_optimizer()

    def set_initial_optimizer(self):
        # freezes the backbone and relation network
        self.model.enable_relation_network(not self.hparams.freeze_relation_head_10_ep)
        for name, params in self.named_parameters():
            condition = 'model.backbone' in name or 'model.relation_network' in name if self.hparams.freeze_relation_head_10_ep else 'model.backbone' in name
            if condition:
                params.requires_grad = False
        non_frozen_parameters = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params=non_frozen_parameters, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.lr_scheduler = CosineWarmupScheduler(self.optimizer, warmup=20, max_iters=10 * 75)

    def forward(self, x: torch.Tensor, y):
        return self.model(x, y, self.current_epoch)

    def step(self, x, y):
        loss_dict, detections, relation_data = self.forward(x, y)
        return loss_dict, detections, relation_data

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        loss_dict, detections, relation_data = self.step(x, y)
        losses = sum(loss for loss in loss_dict.values())
        self.log_detailed_loss(loss_dict, mode='train')
        self.log_map(detections, y, self.train_mAP, 'train')
        self.log("train/loss", losses.item(), on_step=False, on_epoch=True, prog_bar=False)
        if 'relation_logits' in relation_data and 'relation_labels' in relation_data:
            # print('****** training step *******')
            # print('relation_logits: ', F.sigmoid(relation_data['relation_logits'].detach()).view(-1))
            # print('relation_labels: ', relation_data['relation_labels'].view(-1))
            relation_acc = self.train_acc(relation_data['relation_logits'].detach().view(-1),
                                          relation_data['relation_labels'].view(-1))
            self.log("train/rel_acc", relation_acc, on_step=True, on_epoch=True, prog_bar=True)
            relation_f1 = self.train_f1(relation_data['relation_logits'].detach().view(-1),
                                        relation_data['relation_labels'].view(-1))
            self.log("train/rel_f1", relation_f1, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": losses, }

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        x, y = batch
        loss_dict, detections, relation_data = self.step(x, y)
        losses = sum(loss for loss in loss_dict.values())
        self.log_detailed_loss(loss_dict, mode='val')
        self.log_map(detections, y, self.val_mAP, 'val')
        self.log("val/loss", losses.item(), on_step=False, on_epoch=True, prog_bar=False)
        if 'relation_logits' in relation_data and 'relation_labels' in relation_data:
            # print('****** val step *******')
            # print('relation_scores: ', F.sigmoid(relation_data['relation_logits'].detach()).view(-1))
            # print('relation_labels: ', relation_data['relation_labels'].view(-1))
            relation_acc = self.val_acc(relation_data['relation_logits'].detach().view(-1),
                                        relation_data['relation_labels'].view(-1))
            self.log("val/rel_acc", relation_acc, on_step=False, on_epoch=True, prog_bar=True)
            relation_f1 = self.val_f1(relation_data['relation_logits'].detach().view(-1),
                                      relation_data['relation_labels'].view(-1))
            self.log("val/rel_f1", relation_f1, on_step=False, on_epoch=True, prog_bar=True)
        if self.model.get_force_inference_results():
            self.comic_page_relation_evaluator.step(y, detections, relation_data)
            self.comic_page_relation_evaluator_char_aligned.step(y, detections, relation_data, use_char_alignment=True)
        return {"loss": losses, }

    def validation_epoch_end(self, outputs: List[Any]):
        map_value = self.val_mAP.compute()['map']
        self.val_mAP_best.update(map_value)
        self.log("val/map_best", self.val_mAP_best.compute(), on_epoch=True, prog_bar=True)
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/rel_acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        f1 = self.val_f1.compute()  # get val accuracy from current epoch
        self.val_f1_best.update(f1)
        self.log("val/rel_f1_best", self.val_f1_best.compute(), on_epoch=True, prog_bar=True)
        if self.model.get_force_inference_results():
            self.log_relation_scores('val')

    def test_step(self, batch: Any, batch_idx: int):
        x, y = batch
        loss_dict, detections, relation_data = self.step(x, y)
        losses = sum(loss for loss in loss_dict.values())
        self.log_detailed_loss(loss_dict, mode='test')
        self.log_map(detections, y, self.test_mAP, 'test')
        self.log("test/loss", losses.item(), on_step=False, on_epoch=True)
        if 'relation_logits' in relation_data and 'relation_labels' in relation_data:
            relation_acc = self.test_acc(relation_data['relation_logits'].detach().view(-1),
                                         relation_data['relation_labels'].view(-1))
            self.log("test/rel_acc", relation_acc, on_step=False, on_epoch=True)
            relation_f1 = self.test_f1(relation_data['relation_logits'].detach().view(-1),
                                       relation_data['relation_labels'].view(-1))
            self.log("test/rel_f1", relation_f1, on_step=False, on_epoch=True)
        self.comic_page_relation_evaluator.step(y, detections, relation_data)
        self.comic_page_relation_evaluator_char_aligned.step(y, detections, relation_data, use_char_alignment=True)
        return {"loss": losses, }

    def test_epoch_end(self, outputs: List[Any]):
        self.log_relation_scores('test')

    def on_epoch_end(self):
        self.train_mAP.reset()
        self.val_mAP.reset()
        self.test_mAP.reset()
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()
        self.train_f1.reset()
        self.val_f1.reset()
        self.test_f1.reset()
        self.comic_page_relation_evaluator.reset()
        self.comic_page_relation_evaluator_char_aligned.reset()

    def log_relation_scores(self, mode: str):
        computed = self.comic_page_relation_evaluator.compute()
        relation_scores = {f'{mode}/' + k: v for k, v in computed.items()}
        self.log_dict(relation_scores, sync_dist=True)
        computed = self.comic_page_relation_evaluator_char_aligned.compute(suffix='aligned_')
        relation_scores = {f'{mode}/' + k: v for k, v in computed.items()}
        self.log_dict(relation_scores, sync_dist=True)

    def log_map(self, detections, targets, map_fn, mode: str):
        computed = map_fn(detections, targets)
        mAPs = {f'{mode}/' + k: v for k, v in computed.items()}
        mAPs_per_class = mAPs.pop(f"{mode}/map_per_class")
        mARs_per_class = mAPs.pop(f"{mode}/mar_100_per_class")
        # TODO: @gsoykan - log per_class values too
        self.log_dict(mAPs, sync_dist=True)

    def log_detailed_loss(self, loss_dict, mode: str):
        losses_with_mode_keys = {f'{mode}/' + k: v for k, v in loss_dict.items()}
        self.log_dict(losses_with_mode_keys, sync_dist=True)

    def configure_optimizers(self):
        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}

    def on_validation_start(self) -> None:
        if self.current_epoch % self.hparams.log_relation_score_every_n_epoch == 0:
            self.comic_page_relation_evaluator.reset()
            self.comic_page_relation_evaluator_char_aligned.reset()
            self.model.set_force_get_inference_results(True)

    def on_validation_end(self) -> None:
        if self.current_epoch % self.hparams.log_relation_score_every_n_epoch == 0:
            self.comic_page_relation_evaluator.reset()
            self.comic_page_relation_evaluator_char_aligned.reset()
            self.model.set_force_get_inference_results(False)

    def on_test_start(self) -> None:
        self.comic_page_relation_evaluator.reset()
        self.comic_page_relation_evaluator_char_aligned.reset()
        self.model.set_force_get_inference_results(True)

    def on_test_end(self) -> None:
        self.comic_page_relation_evaluator.reset()
        self.comic_page_relation_evaluator_char_aligned.reset()
        self.model.set_force_get_inference_results(False)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.lr_scheduler, CosineWarmupScheduler):
            self.lr_scheduler.step()  # Step per iteration

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == 10:
            print('full fine-tuning starts...')
            self.model.enable_relation_network(True)
            for name, params in self.named_parameters():
                params.requires_grad = True
            non_frozen_parameters = [p for p in self.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                params=non_frozen_parameters, lr=self.hparams.lr / 10, weight_decay=self.hparams.weight_decay
            )
            # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
            one_epoch_iter = len(self.trainer.train_dataloader)
            self.lr_scheduler = CosineWarmupScheduler(self.optimizer, warmup=one_epoch_iter,
                                                      max_iters=(self.trainer.max_epochs - 10) * one_epoch_iter)
            self.trainer.optimizers[0] = self.optimizer
            self.trainer.lr_schedulers[0]['scheduler'] = self.lr_scheduler

    @classmethod
    def checkpoint_to_eval(cls, checkpoint_path: str, strict: bool = False):
        trained_model = cls.load_from_checkpoint(checkpoint_path, strict=strict).to('cuda')
        trained_model.eval()
        trained_model.freeze()
        trained_model.model.enable_relation_network(True)
        return trained_model


def forward_pass_test():
    lit_module = PtDetectorLitModule().to('cuda')
    batch = PickleHelper.load_object(PickleHelper.faster_rcnn_batch)
    images, targets = batch
    images = list(image.to('cuda') for image in images)
    targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
    with torch.no_grad():
        lit_module.eval()
        losses_eval, detections_eval = lit_module(images, targets)
        lit_module.train()
        losses_train, detections_train = lit_module(images, targets)
        print(losses_train, losses_eval)


def forward_pass_for_detection_and_relation():
    # the best reported model ckpt = /scratch/users/gsoykan20/projects/amazing-mysteries-of-gutter-demystified/logs/experiments/runs/dcm_detection/2023-05-10_20-45-13/checkpoints/epoch_012.ckpt
    ckpt = '/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/logs/experiments/runs/pt_detector/the_best.ckpt'
    lit_module = PtDetectorLitModule.checkpoint_to_eval(ckpt)
    # lit_module = PtDetectorLitModule(model_name='maskrcnn_resnet50_fpn_v2',
    #                                  box_head_output_strategy=None,
    #                                  relation_network_first_layer_type='mha_box_dot_product',
    #                                  use_filtered_boxes_for_pair_pool=True).to(
    #    'cuda').eval()
    batch = PickleHelper.load_object(PickleHelper.faster_rcnn_batch)
    images, targets = batch
    images = list(image.to('cuda') for image in images)
    targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
    with torch.no_grad():
        # to test training...
        # losses, detections, relation_data = lit_module(images, targets)
        # to test inference...
        losses, detections, relation_data = lit_module(images, None)
        print(losses)
    # mAP source: https://www.v7labs.com/blog/mean-average-precision
    metric = MeanAveragePrecision(class_metrics=True)
    metric.update(detections, targets)
    pprint(metric.compute())

    face_body_char_eval = ComicPageFaceBodyCharEvaluator(box_score_threshold=0.5)
    face_body_char_eval.step(targets, detections)
    face_body_eval_res = face_body_char_eval.compute()
    print(face_body_eval_res)

    # visualize training relations (works only with training setting, meaning having targets)
    # visualize_relation_bbs(images, relation_data['relation_bbs'], relation_data['relation_labels'],
    #                        F.sigmoid(relation_data['relation_logits']))
    # return

    # function to visualize a single sample
    def visualize_detections(image, detection, relations, score_threshold: float = 0.9):
        filtered_det = {'boxes': [], 'labels': [], 'scores': [], 'box_idxs': [], 'masks': []}
        detection_masks = (detection['masks'] > 0.5).squeeze().detach().cpu().numpy().astype(np.uint8)
        detection_labels = detection['labels'].detach().cpu().numpy()
        detection_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in
                           detection['boxes'].detach().cpu()]
        for i, (b, l, s, m) in enumerate(
                zip(detection_boxes, detection_labels, detection['scores'], detection_masks)):
            if s >= score_threshold:
                filtered_det['boxes'].append(b)
                filtered_det['labels'].append(l)
                filtered_det['scores'].append(s)
                filtered_det['masks'].append(m)
                filtered_det['box_idxs'].append(i)

        filtered_rel = []
        for relation in relations:
            if relation[0] in filtered_det['box_idxs'] and relation[1] in filtered_det['box_idxs']:
                filtered_rel.append(relation)

        detection = filtered_det
        # Convert the tensor to a numpy array
        numpy_image = image.detach().cpu().numpy()
        # Convert the numpy array to a cv2 image
        cv2_image = np.transpose(numpy_image, (1, 2, 0))
        image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        COLORS = [(1, 0, 0),  # Red
                  (0, 1, 0),  # Green
                  (0, 0, 1),  # Blue
                  (0, 0, 0),  # Black
                  (1, 1, 1),  # White
                  (0.5, 0.5, 0.5),  # Gray
                  (1, 1, 0),  # Yellow
                  (0, 1, 1),  # Cyan
                  (1, 0, 1),  # Magenta
                  (1, 0.5, 0)]  # Orange
        # visualizes boxes - masks - and best relations of them...
        for box_num in range(len(detection['boxes'])):
            if detection['labels'][box_num] in [1, 2, 4]:
                box = detection['boxes'][box_num]
                mask = detection['masks'][box_num]
                label = str(int(detection['labels'][box_num]))
                color = COLORS[random.randrange(0, len(COLORS))]
                cv2.rectangle(image, box[0], box[1], color, 2)
                cv2.putText(
                    image, label, (int(box[0][0]), int(box[0][1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )
            # draw mask
            if detection['labels'][box_num] in [4]:
                red_map = np.zeros_like(mask).astype(np.uint8)
                green_map = np.zeros_like(mask).astype(np.uint8)
                blue_map = np.zeros_like(mask).astype(np.uint8)
                red_map[mask == 1], green_map[mask == 1], blue_map[mask == 1] = color
                segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
                alpha = 1
                beta = 0.4  # transparency for the segmentation map
                gamma = 0.0  # scalar added to each sum
                cv2.addWeighted(image, alpha, segmentation_map.astype(image.dtype), beta, gamma, image)

        color = COLORS[random.randrange(0, len(COLORS))]
        # visualizes filtered relations
        for relation in filtered_rel:
            from_idx = filtered_det['box_idxs'].index(relation[0])
            to_idx = filtered_det['box_idxs'].index(relation[1])
            from_box = filtered_det['boxes'][from_idx]
            to_box = filtered_det['boxes'][to_idx]
            to_label = filtered_det['labels'][to_idx]

            def box_center(box):
                box = (box[0][0], box[0][1], box[1][0], box[1][1])
                return int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)

            cx1, cy1 = box_center(from_box)
            cx2, cy2 = box_center(to_box)

            if to_label in [1, 2]:
                cv2.line(image, (cx1, cy1), (cx2, cy2), color, 2)

        # visualizes all selected relations
        # for relation in relations:
        #     print('score: ',  relation[2])
        #     numpy_image = np.copy(numpy_image)
        #     # Convert the numpy array to a cv2 image
        #     cv2_image = np.transpose(numpy_image, (1, 2, 0))
        #     image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        #
        #     from_idx = relation[0]
        #     to_idx = relation[1]
        #     from_box = detection_boxes[from_idx]
        #     to_box = detection_boxes[to_idx]
        #     to_label = detection_labels[to_idx]
        #
        #     cv2.rectangle(image, from_box[0], from_box[1], color, 2)
        #     cv2.rectangle(image, to_box[0], to_box[1], color, 2)
        #
        #     def box_center(box):
        #         box = (box[0][0], box[0][1], box[1][0], box[1][1])
        #         return int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)
        #
        #     cx1, cy1 = box_center(from_box)
        #     cx2, cy2 = box_center(to_box)
        #
        #     if to_label in [1, 2]:
        #         cv2.line(image, (cx1, cy1), (cx2, cy2), color, 2)
        #
        #     cv2.imshow('Image', image)
        #     cv2.waitKey(0)

        cv2.imshow('Image', image)
        cv2.waitKey(0)

    relations = relation_data.get('relations')
    for i, image in enumerate(images):
        curr_relations = relations[i] if relations is not None else []
        pred_labels = detections[i]['labels']

        chars = ComicPageFaceBodyCharEvaluator.pair_face_body_to_form_char(
            detections[i]['boxes'],
            detections[i]['labels'],
            detections[i]['scores'],
            box_score_threshold=0.5)

        selected_face_relations, selected_body_relations = get_selected_relations(curr_relations,
                                                                                  pred_labels,
                                                                                  detections[i]['scores'],
                                                                                  score_threshold=0.75,
                                                                                  box_score_threshold=0.5,
                                                                                  chars=chars)
        visualize_detections(image,
                             detections[i],
                             [*selected_face_relations, *selected_body_relations],
                             0.5)

    for i in range(len(images)):
        from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
        (face_gt_labels, face_preds), (body_gt_labels, body_preds), acc_dict = evaluate_relation(targets[i]['links'],
                                                                                                 targets[i]['boxes'],
                                                                                                 detections[i]['boxes'],
                                                                                                 relation_data[
                                                                                                     'relations'][i],
                                                                                                 detections[i][
                                                                                                     'labels'],
                                                                                                 detections[i][
                                                                                                     'scores'])

        def show_res(labels, preds):
            print('Precision: %.3f' % precision_score(labels, preds))
            print('Recall: %.3f' % recall_score(labels, preds))
            print('Accuracy: %.3f' % accuracy_score(labels, preds))
            print('F1 Score: %.3f' % f1_score(labels, preds))

        print('******** FACE *********')
        show_res(face_gt_labels, face_preds)
        print('******** BODY *********')
        show_res(body_gt_labels, body_preds)


def visualize_relation_bbs(images, relation_bbs, relation_labels, relation_scores):
    split_counts = flatten_list(list(map(lambda x: [len(x['positive']), len(x['negative'])], relation_bbs)))
    relation_labels = relation_labels.split(split_counts, 0)
    relation_scores = relation_scores.split(split_counts, 0)

    # visualizes in-training relations one-by-one
    def visualize_relations(image, bbs, labels, logits):
        bbs = bbs.detach().cpu().numpy()
        numpy_image = image.detach().cpu().numpy()
        for i in range(len(bbs)):
            # Convert the numpy array to a cv2 image
            cv2_image = np.transpose(np.copy(numpy_image), (1, 2, 0))
            image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            curr_relation_bbs = bbs[i]

            print('label: ', labels[i])
            print('score: ', logits[i])

            box_from = curr_relation_bbs[:4]
            box_from = (int((480 * box_from[0]) // 800), int((640 * box_from[1]) // 1088)), (int(
                (480 * box_from[2]) // 800), int((640 * box_from[3]) // 1088))
            box_to = curr_relation_bbs[4:8]
            box_to = (int((480 * box_to[0]) // 800), int((640 * box_to[1]) // 1088)), (int(
                (480 * box_to[2]) // 800), int((640 * box_to[3]) // 1088))
            box_encapsulation = curr_relation_bbs[8:]
            box_encapsulation = (int((480 * box_encapsulation[0]) // 800), int((640 * box_encapsulation[1]) // 1088)), (
                int(
                    (480 * box_encapsulation[2]) // 800), int((640 * box_encapsulation[3]) // 1088))
            color = (1, 0.5, 0)
            cv2.rectangle(image, box_from[0], box_from[1], color, 2)
            cv2.rectangle(image, box_to[0], box_to[1], color, 2)
            cv2.rectangle(image, box_encapsulation[0], box_encapsulation[1], color, 2)

            def box_center(box):
                box = (box[0][0], box[0][1], box[1][0], box[1][1])
                return int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)

            cx1, cy1 = box_center(box_from)
            cx2, cy2 = box_center(box_to)

            cv2.line(image, (cx1, cy1), (cx2, cy2), color, 2)

            cv2.imshow('Image', image)
            cv2.waitKey(0)

    for i, image in enumerate(images):
        curr_relation_bbs = relation_bbs[i]
        print('******** POSITIVE *********')
        curr_labels = relation_labels[i * 2]
        curr_logits = relation_scores[i * 2]
        positive_bbs = curr_relation_bbs['positive']
        print('positive bbs len: ', len(positive_bbs))
        visualize_relations(image, positive_bbs, curr_labels, curr_logits)
        print('******** NEGATIVE *********')
        curr_labels = relation_labels[i * 2 + 1]
        curr_logits = relation_scores[i * 2 + 1]
        negative_bbs = curr_relation_bbs['negative']
        print('negative bbs len: ', len(negative_bbs))
        visualize_relations(image, negative_bbs, curr_labels, curr_logits)


def forward_pass_with_mask():
    lit_module = PtDetectorLitModule(model_name='maskrcnn_resnet50_fpn', num_classes=6).to(
        'cuda').eval()
    batch = PickleHelper.load_object(PickleHelper.faster_rcnn_batch)
    images, targets = batch
    images = list(image.to('cuda') for image in images)
    targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]
    with torch.no_grad():
        # to test training...
        losses, detections, relation_data = lit_module(images, targets)
        # to test inference...
        # losses, detections, relation_data = lit_module(images, None)
        print(losses)


def get_selected_relations(pred_relations,
                           pred_labels,
                           pred_box_scores: Optional[Any] = None,
                           score_threshold: Optional[float] = None,
                           box_score_threshold: Optional[float] = None,
                           chars: Optional[List[Dict[str, Any]]] = None):
    if score_threshold is not None:
        pred_relations = list(filter(lambda x: x[2] > score_threshold, pred_relations))

    if box_score_threshold is not None:
        filtered_pred_relations = []
        for pred_relation in pred_relations:
            from_idx = pred_relation[0]
            to_idx = pred_relation[1]
            if pred_box_scores[from_idx] >= box_score_threshold and pred_box_scores[to_idx] >= box_score_threshold:
                filtered_pred_relations.append(pred_relation)
        pred_relations = filtered_pred_relations

    # augmenting relations with face-body pairs...
    # face-body alignment diyebiliriz bu kısma...
    # character consistency post-processing step diyebiliriz..
    if chars is not None:
        def assign_relations_to_chars(pred_relations):
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
                relations_by_from_pred_box_idx = groupby(sorted(possible_rels, key=lambda x: x[0]), key=lambda x: x[0])
                best_relations_score = 0
                best_relations = None
                for from_box_idx, relations in relations_by_from_pred_box_idx:
                    relations = list(relations)
                    mean_rel_score = np.array(list(map(lambda x: x[2], relations))).mean()
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
                        existing_rel_score = np.array(list(map(lambda x: x[2], existing_relations))).mean()
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
        add_selected_face_relations, add_selected_body_relations = assign_relations_to_chars(remaining_pred_relations)

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


def evaluate_relation(gt_links, gt_boxes,
                      pred_boxes, pred_relations, pred_labels, pred_scores, box_iou_threshold=0.6):
    chars = ComicPageFaceBodyCharEvaluator.pair_face_body_to_form_char(
        pred_boxes,
        pred_labels,
        pred_scores,
        box_score_threshold=0.5)
    # group relations by speech bubble and pick best for body and face
    selected_face_relations, selected_body_relations = get_selected_relations(pred_relations, pred_labels, chars=chars)

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
                if not found_relation:
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
                if not found_relation:
                    acc_dict['face'].append([gt_link, None, 1, 0])
            else:
                raise Exception(f'to label can only be 1 or 2, here it is {str(to_label)}')
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


if __name__ == '__main__':
    forward_pass_for_detection_and_relation()
