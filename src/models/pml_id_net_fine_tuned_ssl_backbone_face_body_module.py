import warnings
from typing import Any, List, Tuple, Optional, Dict, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning import LightningModule
from pytorch_metric_learning import distances as pml_distances, losses as pml_losses, miners as pml_miners, \
    reducers as pml_reducers, regularizers as pml_regularizers
from torch import Tensor
from torch import nn
from torchmetrics import Accuracy, MaxMetric
from torchmetrics.classification import BinaryAUROC
from torchvision.models.resnet import resnet18, resnet50, wide_resnet50_2
import torch.nn.init as init

from src.datamodules.components.face_recognition.comics_seq_firebase_face_body_query_ref_wrapper import \
    ComicsSeqFirebaseFaceBodyQueryRefWrapper
from src.datamodules.components.vision_transform_setting import VisionTransformSetting
from src.models.components.autoencoder.triplet_loss import TripletLoss
from src.models.components.component_detection.customized_faster_rcnn.relation_network import LambdaLayer
from src.models.components.face_body_fusion.coeff_sum_fusion import CoefficientSumModule
from src.models.components.face_body_fusion.face_to_body_attention_fusion import FaceToBodyAttentionFusion
from src.models.components.face_body_fusion.self_attention_fusion import SelfAttentionFusion
from src.models.components.face_body_fusion.weighted_sum_fusion import WeightedSumModule
from src.models.components.ssl_module.sim_clr import SimCLR
from src.models.components.ssl_module.ssl_backbone import SSLBackbone
from src.models.sim_clr_module import SimCLRLitModule
from src.utils.basic_utils import read_or_get_image
from src.utils.color_histogram.color_histogram_utils import get_histogram_vector_batched
from src.utils.pml_accuracy_wrapper import PMLAccuracyWrapper
from src.utils.pml_seq_miner_wrapper import PMLSeqMinerWrapper


class PMLIdNetFineTunedSSLBackboneFaceBodyLitModule(LightningModule):
    def __init__(
            self,
            ssl_ckpt_face: str,
            ssl_ckpt_body: str,
            is_face_body_aligned_module: bool = False,
            # face params
            backbone_latent_dim_face: int = 512,
            ssl_backbone_face: SSLBackbone = SSLBackbone.CORINFOMAX,
            # body params
            backbone_latent_dim_body: int = 512,
            ssl_backbone_body: SSLBackbone = SSLBackbone.CORINFOMAX,
            override_backbones: Optional[Dict] = None,
            id_latent_dim: int = 128,
            lr: float = 0.001,
            weight_decay: float = 0.01,
            scheduler_gamma: float = 0.95,
            # id-net configs
            by_pass_id_net: bool = False,
            use_linear: bool = True,
            start_id_net_with_relu: bool = False,
            use_dropout: Optional[bool] = False,
            only_mode: Optional[str] = None,
            l2_normalize_embeddings: Optional[bool] = False,
            use_trainable_padding_vectors: bool = False,
            use_triplets_dataset_for_val_test: Optional[bool] = True,
            fusion_strategy: str = 'cat',
            # cat - sum - mean - favor_body - favor_face - weighted_sum - self_attn - face_to_body_attn
            pml_setup_v: Optional[int] = 1,
            mix_series_for_mining: bool = False
    ):
        super().__init__()
        if override_backbones is None:
            if isinstance(ssl_backbone_face, str):
                self.ssl_backbone_face = SSLBackbone(ssl_backbone_face)
            else:
                self.ssl_backbone_face = ssl_backbone_face
            if isinstance(ssl_backbone_body, str):
                self.ssl_backbone_body = SSLBackbone(ssl_backbone_body)
            else:
                self.ssl_backbone_body = ssl_backbone_body if ssl_backbone_body is not None else ssl_backbone_face
        else:
            self.backbone_face, self.backbone_body = self._override_backbones(override_backbones)

        self.save_hyperparameters(logger=False)

        if override_backbones is None:
            if only_mode == 'face':
                self.backbone_face = self.load_backbone(self.ssl_backbone_face, ssl_ckpt_face)
            elif only_mode == 'body':
                self.backbone_body = self.load_backbone(self.ssl_backbone_body, ssl_ckpt_body)
            else:
                if is_face_body_aligned_module:
                    self.backbone_face = self.load_backbone(self.ssl_backbone_face, ssl_ckpt_face)
                    self.backbone_body = self.backbone_face
                else:
                    self.backbone_face = self.load_backbone(self.ssl_backbone_face, ssl_ckpt_face)
                    self.backbone_body = self.load_backbone(self.ssl_backbone_body, ssl_ckpt_body)

        if only_mode == 'face':
            total_backbone_latent_dim = backbone_latent_dim_face
        elif only_mode == 'body':
            total_backbone_latent_dim = backbone_latent_dim_body
        else:
            if self.hparams.fusion_strategy in ['coeff_sum',
                                                'weighted_sum',
                                                'sum',
                                                'mean',
                                                'favor_body',
                                                'favor_face']:
                total_backbone_latent_dim = backbone_latent_dim_face
            elif self.hparams.fusion_strategy == 'self_attn':
                total_backbone_latent_dim = backbone_latent_dim_face
            elif self.hparams.fusion_strategy == 'face_to_body_attn':
                total_backbone_latent_dim = backbone_latent_dim_face
            else:
                total_backbone_latent_dim = backbone_latent_dim_face + backbone_latent_dim_body

            if self.hparams.fusion_strategy == 'weighted_sum':
                self.fusion_model = WeightedSumModule(backbone_latent_dim_face)
            if self.hparams.fusion_strategy == 'coeff_sum':
                self.fusion_model = CoefficientSumModule(backbone_latent_dim_face)
            elif self.hparams.fusion_strategy == 'self_attn':
                self.fusion_model = SelfAttentionFusion(backbone_latent_dim_face,
                                                        total_backbone_latent_dim // 2,
                                                        use_positional_encoding=True)
            elif self.hparams.fusion_strategy == 'face_to_body_attn':
                self.fusion_model = FaceToBodyAttentionFusion(backbone_latent_dim_face,
                                                              total_backbone_latent_dim,
                                                              use_positional_encoding=True)

        if by_pass_id_net:
            self.id_net = nn.Identity()
        else:
            if use_linear:
                self.id_net = nn.Sequential(
                    nn.ReLU(inplace=True) if start_id_net_with_relu else nn.Identity(),
                    nn.Linear(total_backbone_latent_dim, id_latent_dim),
                    LambdaLayer(lambda x: F.normalize(x, p=2, dim=1)) if l2_normalize_embeddings else nn.Identity()
                )
            else:
                self.id_net = nn.Sequential(
                    nn.ReLU(inplace=True) if start_id_net_with_relu else nn.Identity(),
                    nn.Linear(total_backbone_latent_dim, total_backbone_latent_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2) if use_dropout else nn.Identity(),
                    nn.Linear(total_backbone_latent_dim // 2, id_latent_dim),
                    LambdaLayer(lambda x: F.normalize(x, p=2, dim=1)) if l2_normalize_embeddings else nn.Identity()
                )

        if use_trainable_padding_vectors:
            # Initialize the padding vectors with 2 dimensions
            self.padding_vector_face = nn.Parameter(
                torch.zeros((1, backbone_latent_dim_face), requires_grad=True)).cuda()
            self.padding_vector_body = nn.Parameter(
                torch.zeros((1, backbone_latent_dim_body), requires_grad=True)).cuda()
            # Initialize the padding vectors using Xavier/Glorot initialization
            init.xavier_uniform_(self.padding_vector_face)
            init.xavier_uniform_(self.padding_vector_body)
        else:
            self.padding_vector_face = torch.zeros(backbone_latent_dim_face, requires_grad=False).cuda()
            self.padding_vector_body = torch.zeros(backbone_latent_dim_body, requires_grad=False).cuda()

        # pml stuff
        if pml_setup_v == 1:
            distance = pml_distances.CosineSimilarity()
            reducer = pml_reducers.ThresholdReducer(low=0)
            self.pml_loss_func = pml_losses.TripletMarginLoss(margin=0.2,
                                                              distance=distance,
                                                              reducer=reducer)
            self.pml_mining_func = pml_miners.TripletMarginMiner(
                margin=0.2, distance=distance, type_of_triplets="semihard"
            )
        elif pml_setup_v == 2:
            distance = pml_distances.LpDistance()
            reducer = pml_reducers.ThresholdReducer(low=0)
            self.pml_regularizer = pml_regularizers.LpRegularizer()
            self.pml_loss_func = pml_losses.TripletMarginLoss(margin=1,
                                                              distance=distance,
                                                              reducer=reducer,
                                                              embedding_regularizer=self.pml_regularizer)
            self.pml_mining_func = pml_miners.TripletMarginMiner(
                margin=0.2, distance=distance, type_of_triplets="all"
            )
        elif pml_setup_v == 3:
            self.pml_loss_func = pml_losses.TripletMarginLoss(margin=0.1)
            self.pml_mining_func = pml_miners.MultiSimilarityMiner(epsilon=0.1)
        elif pml_setup_v == 4:
            distance = pml_distances.CosineSimilarity()
            reducer = pml_reducers.MeanReducer()
            self.pml_loss_func = pml_losses.MultiSimilarityLoss(alpha=2,
                                                                beta=50,
                                                                base=0.5,
                                                                distance=distance,
                                                                reducer=reducer)
            self.pml_mining_func = pml_miners.MultiSimilarityMiner(
                epsilon=0.1, distance=distance,
            )
        elif pml_setup_v == 5:
            distance = pml_distances.DotProductSimilarity()
            reducer = pml_reducers.MeanReducer()
            self.pml_loss_func = pml_losses.NPairsLoss(distance=distance,
                                                       reducer=reducer)
            self.pml_mining_func = None
        elif pml_setup_v == 6:
            self.pml_loss_func = pml_losses.CircleLoss(m=0.25, gamma=256)
            self.pml_mining_func = pml_miners.HDCMiner(filter_percentage=0.25)
        elif pml_setup_v == 7:
            # default'u hep buymuş => LpDistance(normalize_embeddings=True, p=2, power=1)
            distance = pml_distances.LpDistance()
            self.pml_loss_func = pml_losses.ContrastiveLoss(distance=distance)
            self.pml_mining_func = None
        elif pml_setup_v == 8:
            distance = pml_distances.CosineSimilarity()
            self.pml_loss_func = pml_losses.ContrastiveLoss(distance=distance,
                                                            pos_margin=1,
                                                            neg_margin=0)
            self.pml_mining_func = None
        elif pml_setup_v == 9:
            distance = pml_distances.LpDistance()
            self.pml_loss_func = pml_losses.TripletMarginLoss(margin=0.2)
            self.pml_mining_func = pml_miners.MultiSimilarityMiner(epsilon=0.2, distance=distance)
        elif pml_setup_v == 10:
            distance = pml_distances.LpDistance()
            self.pml_loss_func = pml_losses.TripletMarginLoss(margin=0.5)
            self.pml_mining_func = pml_miners.MultiSimilarityMiner(epsilon=0.5, distance=distance)
        elif pml_setup_v == 11:
            main_loss = pml_losses.TupletMarginLoss()
            var_loss = pml_losses.IntraPairVarianceLoss()
            self.pml_loss_func = pml_losses.MultipleLosses([main_loss, var_loss], weights=[1, 0.5])
            self.pml_mining_func = None
        elif pml_setup_v == 12:
            distance = pml_distances.LpDistance()
            reducer = pml_reducers.MeanReducer()
            self.pml_loss_func = pml_losses.MultiSimilarityLoss(alpha=2,
                                                                beta=50,
                                                                base=0.5,
                                                                distance=distance,
                                                                reducer=reducer)
            self.pml_mining_func = pml_miners.MultiSimilarityMiner(
                epsilon=0.1, distance=distance,
            )
        elif pml_setup_v == 13:
            distance = pml_distances.LpDistance()
            triplet_loss = pml_losses.TripletMarginLoss(margin=0.2, distance=distance)
            contrastive_loss = pml_losses.ContrastiveLoss(distance=distance)
            self.pml_loss_func = pml_losses.MultipleLosses([triplet_loss, contrastive_loss], weights=[1, 1])
            self.pml_mining_func = None

        if self.pml_mining_func is not None:
            self.pml_mining_func = PMLSeqMinerWrapper(self.pml_mining_func)

        self.train_pml_accuracy_wrapper = PMLAccuracyWrapper()
        self.val_pml_accuracy_wrapper = PMLAccuracyWrapper()
        self.test_pml_accuracy_wrapper = PMLAccuracyWrapper()

        self.criterion = TripletLoss()
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')
        self.val_acc_best = MaxMetric()
        self.val_auroc = BinaryAUROC(thresholds=None)
        self.test_auroc = BinaryAUROC(thresholds=None)
        self.val_auroc_best = MaxMetric()
        self.euclidian_threshold = 0

    def _override_backbones(self, override_backbone_configs: Optional[Dict]) -> Tuple[nn.Module, nn.Module]:
        model = override_backbone_configs['model']
        if model == 'pretrained_resnet50':
            convnet = resnet50(pretrained=True)
            convnet.fc = nn.Identity()
            convnet.eval()
            for param in convnet.parameters():
                param.requires_grad = False
            return convnet, convnet
        elif model == 'color_histogram':
            # This can not be trained as expected :)
            def get_batched_color_histogram_vectors(batch: torch.Tensor):
                device = batch.device
                batch = rearrange(batch, "b c h w-> b h w c")
                batch = batch.detach().cpu().numpy()
                np_vectors = get_histogram_vector_batched(batch)
                return torch.from_numpy(np_vectors).to(device)

            model = LambdaLayer(get_batched_color_histogram_vectors)
            return model, model
        elif model == 'random':
            def generate_random_tensors(batch: torch.Tensor):
                return torch.randn((batch.size()[0], 256), device=batch.device)

            model = LambdaLayer(generate_random_tensors)
            return model, model
        else:
            raise Exception(f'Unhandled override backbone model {model}')

    def load_backbone(self, ssl_backbone: SSLBackbone, ssl_ckpt) -> nn.Module:
        if ssl_backbone in [SSLBackbone.SIM_CLR,
                            SSLBackbone.SIM_CLR_FINETUNE,
                            SSLBackbone.SIM_CLR_DEEPER_LAST,
                            SSLBackbone.SIM_CLR_KEEP_ALL,
                            SSLBackbone.SIM_CLR_KEEP_ALL_ACTIVATE_LAST_LINEAR]:
            backbone: SimCLRLitModule = SimCLRLitModule.load_from_checkpoint(ssl_ckpt,
                                                                             strict=False)
            if 'FINETUNE' not in self.ssl_backbone_face.value:
                backbone.eval()
                backbone.freeze()
            backbone: SimCLR = backbone.model
            if ssl_backbone == SSLBackbone.SIM_CLR_DEEPER_LAST:
                # getting rid of final Linear->BatchNorm1d combo...
                backbone.convnet.fc = torch.nn.Sequential(*list(backbone.convnet.fc.children())[:-2])
            elif ssl_backbone == SSLBackbone.SIM_CLR_KEEP_ALL:
                print('keeps sim_clr projection head')
            elif ssl_backbone == SSLBackbone.SIM_CLR_KEEP_ALL_ACTIVATE_LAST_LINEAR:
                frozen_projector_parts = torch.nn.Sequential(*list(backbone.convnet.fc.children())[:-2])
                unfrozen_projector_parts = torch.nn.Sequential(*list(backbone.convnet.fc.children())[-2:])
                # only activate final linear layer and batch-norm...
                for param in unfrozen_projector_parts.parameters():
                    param.requires_grad = True
                unfrozen_projector_parts.train()
                backbone.convnet.fc = nn.Sequential(frozen_projector_parts, unfrozen_projector_parts)
            else:
                backbone.convnet.fc = nn.Identity()
            return backbone
        else:
            raise Exception(f'unhandled backbone for triplet id net fine-tuning: {self.ssl_backbone_face}')

    def forward(self,
                x_all_face: Tensor,
                x_all_body: Tensor,
                all_face_mask: Optional[Tensor],
                all_body_mask: Optional[Tensor],
                return_features: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.hparams.only_mode == 'face':
            feature_all = self.backbone_face(x_all_face)
            if all_face_mask is not None:
                feature_all[~all_face_mask.bool()] = self.padding_vector_face.view(-1).to(feature_all.dtype)
        elif self.hparams.only_mode == 'body':
            feature_all = self.backbone_body(x_all_body)
            if all_body_mask is not None:
                feature_all[~all_body_mask.bool()] = self.padding_vector_body.view(-1).to(feature_all.dtype)
        else:
            feature_all_face = self.backbone_face(x_all_face)
            feature_all_body = self.backbone_body(x_all_body)
            # TODO: @gsoykan - you can try to imagine different padding vectors...
            if all_face_mask is not None:
                feature_all_face[~all_face_mask.bool()] = self.padding_vector_face.view(-1).to(feature_all_face.dtype)
            if all_body_mask is not None:
                feature_all_body[~all_body_mask.bool()] = self.padding_vector_body.view(-1).to(feature_all_body.dtype)

            # TODO: @all - this can all be moved to a generalized fusion model...
            if self.hparams.fusion_strategy == 'cat':
                feature_all = torch.cat([feature_all_face, feature_all_body], dim=-1)
            elif self.hparams.fusion_strategy == 'sum':
                feature_all = feature_all_face + feature_all_body
            elif self.hparams.fusion_strategy in ['coeff_sum',
                                                  'weighted_sum',
                                                  'self_attn',
                                                  'face_to_body_attn']:
                feature_all = self.fusion_model(feature_all_face, feature_all_body)
            elif self.hparams.fusion_strategy == 'mean':
                feature_all = torch.mean(torch.stack([feature_all_face, feature_all_body]), dim=0)
            elif self.hparams.fusion_strategy == 'favor_body':
                feature_all = feature_all_body
                # if there is no body fill it with face...
                feature_all[~all_body_mask.bool()] = feature_all_face[~all_body_mask.bool()]
            elif self.hparams.fusion_strategy == 'favor_face':
                feature_all = feature_all_face
                # if there is no face fill it with body...
                feature_all[~all_face_mask.bool()] = feature_all_body[~all_face_mask.bool()]
            else:
                raise Exception('set a valid fusion_strategy', self.hparams.fusion_strategy)

        if return_features:
            return feature_all, self.id_net(feature_all)

        return self.id_net(feature_all)

    # make this a utility function...
    def _generate_preds_and_y(self, anchor, positive, negative, return_distances: bool = False) -> Tuple[
        Tensor, Tensor, Optional[Dict]]:
        neg_gt = torch.ones(anchor.shape[0], device=anchor.device)
        pos_gt = torch.zeros(anchor.shape[0], device=anchor.device)
        gt = torch.cat([neg_gt, pos_gt]).type(torch.LongTensor)

        neg_cos_sim = F.cosine_similarity(anchor, negative)
        pos_cos_sim = F.cosine_similarity(anchor, positive)

        neg_euc_dist = F.pairwise_distance(anchor, negative)
        pos_euc_dist = F.pairwise_distance(anchor, positive)

        preds = (torch.cat([neg_euc_dist, pos_euc_dist]) > self.euclidian_threshold).type(
            torch.LongTensor)
        if return_distances:
            distance_and_similarities = {
                'neg_euc_dist': neg_euc_dist,
                'pos_euc_dist': pos_euc_dist,
                'neg_cos_sim': neg_cos_sim,
                'pos_cos_sim': pos_cos_sim
            }
            return preds.view(-1, 1).to(anchor.device), gt.view(-1, 1).to(anchor.device), distance_and_similarities
        else:
            return preds.view(-1, 1).to(anchor.device), gt.view(-1, 1).to(anchor.device), None

    def step(self, batch: Any, return_distances: bool = False):
        x_all_face = torch.cat([batch['anchor_face'][0], batch['positive_face'][0], batch['negative_face'][0]])
        all_face_valid_idx = torch.cat([batch['anchor_face'][1], batch['positive_face'][1], batch['negative_face'][1]])
        x_all_body = torch.cat([batch['anchor_body'][0], batch['positive_body'][0], batch['negative_body'][0]])
        all_body_valid_idx = torch.cat([batch['anchor_body'][1], batch['positive_body'][1], batch['negative_body'][1]])
        id_embedding_all = self.forward(x_all_face, x_all_body, all_face_valid_idx, all_body_valid_idx)
        id_embedding_anchor, id_embedding_positive, id_embedding_negative = torch.tensor_split(id_embedding_all, 3)
        if self.hparams.only_mode == 'face':
            face_existing_idx = batch['anchor_face'][1] * batch['positive_face'][1] * batch['negative_face'][1]
            id_embedding_anchor, id_embedding_positive, id_embedding_negative = id_embedding_anchor[face_existing_idx], \
                id_embedding_positive[
                    face_existing_idx], \
                id_embedding_negative[face_existing_idx]
        elif self.hparams.only_mode == 'body':
            body_existing_idx = batch['anchor_body'][1] * batch['positive_body'][1] * batch['negative_body'][1]
            id_embedding_anchor, id_embedding_positive, id_embedding_negative = id_embedding_anchor[body_existing_idx], \
                id_embedding_positive[
                    body_existing_idx], \
                id_embedding_negative[body_existing_idx]

        triplet_output = self.criterion(id_embedding_anchor, id_embedding_positive, id_embedding_negative)
        preds, y, distances = self._generate_preds_and_y(id_embedding_anchor.detach(), id_embedding_positive.detach(),
                                                         id_embedding_negative.detach(), return_distances)
        return triplet_output['loss'], preds, y, distances

    def _pml_step(self,
                  batch,
                  pml_accuracy_wrapper: Optional[PMLAccuracyWrapper],
                  is_train: bool = False):
        labels = batch['char_ids']
        all_faces = batch['faces']
        all_bodies = batch['bodies']
        seq_ids = batch['seq_ids']
        original_seq_ids = batch['original_seq_ids']
        face_mask = batch['face_mask']
        body_mask = batch['body_mask']
        embeddings = self.forward(all_faces, all_bodies, face_mask, body_mask)

        if self.hparams.only_mode == 'face':
            embeddings = embeddings[face_mask]
            labels = labels[face_mask]
            seq_ids = seq_ids[face_mask]
            original_seq_ids = original_seq_ids[face_mask]
        elif self.hparams.only_mode == 'body':
            embeddings = embeddings[body_mask]
            labels = labels[body_mask]
            seq_ids = seq_ids[body_mask]
            original_seq_ids = original_seq_ids[body_mask]

        mix_series_for_mining = self.hparams.mix_series_for_mining if is_train else False
        indices_tuple = self.pml_mining_func(embeddings,
                                             labels,
                                             seq_ids,
                                             original_seq_ids,
                                             mix_series_for_mining) if self.pml_mining_func is not None else None

        # handling edge case where indices_tuple is empty
        if not indices_tuple and self.pml_mining_func is not None:
            warnings.warn(
                f'indices_tuple is empty, using 0 loss tensor...',
                UserWarning,
            )
            return torch.tensor([0.0], requires_grad=True, device=labels.device), {}

        loss = self.pml_loss_func(embeddings, labels, indices_tuple)
        pml_acc_results = pml_accuracy_wrapper(embeddings=embeddings.detach(),
                                               char_ids=labels,
                                               seq_ids=seq_ids) if pml_accuracy_wrapper is not None else {}
        return loss, pml_acc_results

    def training_step(self, batch: Any, batch_idx: int):
        loss, pml_acc_results = self._pml_step(batch, None, is_train=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_pml_acc(pml_acc_results, mode='train')
        return {"loss": loss}

    def log_pml_acc(self,
                    pml_acc_results,
                    mode: str,
                    suffix: str = '',
                    prog_bar: bool = False,
                    on_epoch: bool = False,
                    prefix: str = ''):
        scores = {f'{mode}/{prefix}' + k + f'{suffix}': v for k, v in pml_acc_results.items()}
        self.log_dict(scores, sync_dist=True, prog_bar=prog_bar, on_epoch=on_epoch)

    def training_epoch_end(self, outputs: List[Any]):
        self.train_pml_accuracy_wrapper.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        if self.hparams.use_triplets_dataset_for_val_test:
            loss, preds, targets, distances = self.step(batch, return_distances=True)
            acc = self.val_acc(preds, targets)
            auroc = self.compute_auroc(targets, distances, mode='val')
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
            return {"loss": loss, "preds": preds, "targets": targets, "distances": distances}
        else:
            loss, pml_acc_results = self._pml_step(batch, self.val_pml_accuracy_wrapper)
            self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            # self.log_pml_acc(pml_acc_results, mode='val')
            return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        if self.hparams.use_triplets_dataset_for_val_test:
            acc = self.val_acc.compute()  # get val accuracy from current epoch
            auroc = self.val_auroc.compute()
            self.val_acc_best.update(acc)
            self.val_auroc_best.update(auroc)
            self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
            self.log("val/auroc_best", self.val_auroc_best.compute(), on_epoch=True, prog_bar=True)
            self.compute_mean_std_of_distances_for_epoch(outputs, mode='val')
        else:
            pml_acc_results = self.val_pml_accuracy_wrapper.compute()
            self.log_pml_acc(pml_acc_results, mode='val', prog_bar=True, on_epoch=True)
            self.val_pml_accuracy_wrapper.reset()
            val_dataset = self.trainer.datamodule.val_dataloader().dataset
            query_ref_wrapper = ComicsSeqFirebaseFaceBodyQueryRefWrapper(val_dataset)
            query_ref_result = query_ref_wrapper.evaluate(
                embedder=lambda x_f, x_b, f_m, b_m: self.forward(x_f, x_b, f_m, b_m) if self.hparams.only_mode not in [
                    'face', 'body'] else self.forward(x_f[f_m], x_b[b_m], None, None), only_mode=self.hparams.only_mode)
            self.log_pml_acc(query_ref_result, 'val', prefix='query/', prog_bar=True, on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int):
        if self.hparams.use_triplets_dataset_for_val_test:
            loss, preds, targets, distances = self.step(batch, return_distances=True)
            acc = self.test_acc(preds, targets)
            auroc = self.compute_auroc(targets, distances, mode='test')
            self.log("test/loss", loss, on_step=False, on_epoch=True)
            self.log("test/acc", acc, on_step=False, on_epoch=True)
            self.log("test/auroc", auroc, on_step=False, on_epoch=True)
            return {"loss": loss, "preds": preds, "targets": targets, "distances": distances}
        else:
            loss, pml_acc_results = self._pml_step(batch, self.test_pml_accuracy_wrapper)
            self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log_pml_acc(pml_acc_results, mode='test', on_epoch=True, prog_bar=True)
            return {"loss": loss}

    def test_epoch_end(self, outputs: List[Any]):
        if self.hparams.use_triplets_dataset_for_val_test:
            self.compute_mean_std_of_distances_for_epoch(outputs, mode='test')
        else:
            pml_acc_results = self.test_pml_accuracy_wrapper.compute()
            self.log_pml_acc(pml_acc_results, mode='test', prog_bar=True, on_epoch=True)
            self.test_pml_accuracy_wrapper.reset()
            test_dataset = self.trainer.datamodule.test_dataloader().dataset
            query_ref_wrapper = ComicsSeqFirebaseFaceBodyQueryRefWrapper(test_dataset)
            query_ref_result = query_ref_wrapper.evaluate(
                embedder=lambda x_f, x_b, f_m, b_m: self.forward(x_f, x_b, f_m, b_m) if self.hparams.only_mode not in [
                    'face', 'body'] else self.forward(x_f[f_m], x_b[b_m], None, None), only_mode=self.hparams.only_mode)
            self.log_pml_acc(query_ref_result, 'test', prefix='query/', prog_bar=True, on_epoch=True)

    def on_epoch_end(self):
        for metric in [self.val_auroc,
                       self.val_acc,
                       self.test_auroc,
                       self.test_acc]:
            metric.reset()

    def configure_optimizers(self):
        optims = []
        scheds = []
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay)
        optims.append(optimizer)
        if getattr(self.hparams, 'scheduler_gamma', None) is not None:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optims[0],
                                                               gamma=self.hparams.scheduler_gamma)
            scheds.append(scheduler)
            return optims, scheds
        return optims

    def compute_auroc(self, gt: Tensor, distances: Dict, mode: str):
        gt = gt.view(-1)
        distances = torch.cat((distances['neg_euc_dist'], distances['pos_euc_dist'])).view(-1)
        # distances = torch.cat((1 - distances['neg_cos_sim'], 1 - distances['pos_cos_sim'])).view(-1)
        if mode == 'val':
            return self.val_auroc(distances, gt)
        elif mode == 'test':
            return self.test_auroc(distances, gt)
        else:
            raise Exception(f'Unknown mode for auroc {mode}!')

    def compute_mean_std_of_distances_for_epoch(self, outputs: List[Any], mode: str):
        neg_eucs = []
        pos_eucs = []
        neg_coses = []
        pos_coses = []
        for output in outputs:
            distances = output['distances']
            neg_eucs.append(distances['neg_euc_dist'])
            pos_eucs.append(distances['pos_euc_dist'])
            neg_coses.append(distances['neg_cos_sim'])
            pos_coses.append(distances['pos_cos_sim'])
        neg_eucs = torch.cat(neg_eucs)
        pos_eucs = torch.cat(pos_eucs)
        neg_coses = torch.cat(neg_coses)
        pos_coses = torch.cat(pos_coses)
        neg_eucs_std, neg_eucs_mean = torch.std_mean(neg_eucs, unbiased=False)
        pos_eucs_std, pos_eucs_mean = torch.std_mean(pos_eucs, unbiased=False)
        neg_coses_std, neg_coses_mean = torch.std_mean(neg_coses, unbiased=False)
        pos_coses_std, pos_coses_mean = torch.std_mean(pos_coses, unbiased=False)
        self.euclidian_threshold = (pos_eucs_mean + pos_eucs_std).item()
        val_dict = {
            'neg_eucs_std': neg_eucs_std,
            'neg_eucs_mean': neg_eucs_mean,
            'pos_eucs_std': pos_eucs_std,
            'pos_eucs_mean': pos_eucs_mean,
            'neg_coses_std': neg_coses_std,
            'neg_coses_mean': neg_coses_mean,
            'pos_coses_std': pos_coses_std,
            'pos_coses_mean': pos_coses_mean
        }
        for k, v in val_dict.items():
            self.log(f"{mode}/{k}", v, on_epoch=True, prog_bar=True)

    @classmethod
    def checkpoint_to_eval(cls,
                           checkpoint_path: str,
                           **kwargs, ):
        trained_model = cls.load_from_checkpoint(checkpoint_path, strict=False, **kwargs).to('cuda')
        trained_model.eval()
        trained_model.freeze()
        return trained_model

    @staticmethod
    def get_embeddings_from_imgs(trained_model: LightningModule,
                                 img_paths_face: List[Optional[str]],
                                 img_paths_body: List[Optional[str]],
                                 transform_face: Optional[Any] = None,
                                 transform_body: Optional[Any] = None) -> torch.Tensor:
        transform_args_face = {'N': 96, 'use_padding': False}
        transform_args_body = {'N': 128, 'use_padding': True}
        transform_face = transform_face if transform_face is not None else VisionTransformSetting.SIMCLR_TEST.get_transformation(
            **transform_args_face)
        transform_body = transform_body if transform_body is not None else VisionTransformSetting.SIMCLR_TEST.get_transformation(
            **transform_args_body)

        def get_batch(img_paths: List[str], is_face: bool):
            batch = []
            for img_path in img_paths:
                if img_path is None:
                    dim = 96 if is_face else 128
                    source_img = torch.zeros(3, dim, dim).to(trained_model.device)
                else:
                    source_img = read_or_get_image(img_path, read_rgb=True)
                    if is_face:
                        source_img = transform_face(image=source_img)['image'].to(trained_model.device)
                    else:
                        source_img = transform_body(image=source_img)['image'].to(trained_model.device)
                batch.append(source_img)
            return torch.stack(batch)

        face_batch = get_batch(img_paths_face, True)
        face_mask = torch.tensor([0 if elem is None else 1 for elem in img_paths_face]).to(trained_model.device)
        body_batch = get_batch(img_paths_body, False)
        body_mask = torch.tensor([0 if elem is None else 1 for elem in img_paths_body]).to(trained_model.device)

        with torch.no_grad():
            embeddings = trained_model(face_batch, body_batch, face_mask, body_mask)

        return embeddings


######################################

def check_embedding_generation():
    # SSL f-b aligned last checkpoint
    # /scratch/users/gsoykan20/projects/amazing-mysteries-of-gutter-demystified/logs/experiments/runs/sim_clr_comics_crops_face_body_aligned/2023-06-08_17-30-49/checkpoints/last.ckpt
    # face+body mean fusion (düşününce bu da süper mantıklı değil -> face+body mean result ile sadece face+0 mean ı farklı...)
    model_checkpoint = '/scratch/users/gsoykan20/projects/amazing-mysteries-of-gutter-demystified/logs/experiments/runs/pml_id_net_fine_tuned_sim_clr_backbone_face_body/2023-06-10_01-02-25/checkpoints/epoch_004.ckpt'
    trained_model = PMLIdNetFineTunedSSLBackboneFaceBodyLitModule.checkpoint_to_eval(model_checkpoint)
    # TODO: @gsoykan - bunları large faces ile değiştir...
    face_img_paths = [
        '/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops_large_faces/0/0_0/0.jpg',
        '/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops_large_faces/0/0_0/1.jpg']
    body_img_paths = [
        '/datasets/COMICS/comics_crops/0/0_0/bodies/0.jpg',
        '/datasets/COMICS/comics_crops/0/0_0/bodies/1.jpg']
    embeddings = PMLIdNetFineTunedSSLBackboneFaceBodyLitModule \
        .get_embeddings_from_imgs(trained_model, face_img_paths, body_img_paths)
    print(embeddings)


if __name__ == '__main__':
    check_embedding_generation()
