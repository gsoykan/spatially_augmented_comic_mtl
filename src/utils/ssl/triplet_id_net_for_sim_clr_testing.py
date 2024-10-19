import os
from copy import deepcopy
from functools import partial
from typing import Any, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import Tensor
from torch import nn
from torch import optim
from torchmetrics import Accuracy, MaxMetric
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.datamodules.components.face_recognition.triplet_comics_seq_firebase_dataset import \
    TripletComicsSeqFirebaseDataset
from src.datamodules.components.vision_transform_setting import VisionTransformSetting
from src.models.components.autoencoder.triplet_loss import TripletLoss
from tqdm import tqdm

from src.models.sim_clr_module import SimCLRLitModule


class TripletIdNetForSimCLRTesting(LightningModule):
    def __init__(self, backbone_feature_dim: int = 128,
                 feature_dim: int = 128,
                 lr: float = 0.001,
                 weight_decay: float = 0.01, ):
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(backbone_feature_dim, feature_dim)
        self.criterion = TripletLoss()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.val_acc_best = MaxMetric()

    @staticmethod
    def prepare_data_features(model: SimCLRLitModule, dataset):
        # TODO: @gsoykan - this part can be generalizable
        network: nn.Module = deepcopy(model.model)
        network.fc = nn.Identity()  # Removing projection head g(.)
        network.eval()
        network.to(model.device)
        # Encode all images
        data_loader = DataLoader(dataset,
                                 batch_size=64,
                                 num_workers=5,
                                 shuffle=False,
                                 drop_last=False)
        anchor_feats, positive_feats, negative_feats = [], [], []
        for batch in tqdm(data_loader, desc='preparing data features for id net testing...',
                          leave=False):
            x_all = torch.cat([batch['anchor'], batch['positive'], batch['negative']])
            batch_feats = network(x_all.to(model.device))
            batch_feats.detach().cpu()
            id_embedding_anchor, id_embedding_positive, id_embedding_negative = torch.tensor_split(batch_feats, 3)
            anchor_feats.append(id_embedding_anchor)
            positive_feats.append(id_embedding_positive)
            negative_feats.append(id_embedding_negative)
        anchor_feats = torch.cat(anchor_feats, dim=0)
        positive_feats = torch.cat(positive_feats, dim=0)
        negative_feats = torch.cat(negative_feats, dim=0)
        return TensorDataset(anchor_feats, positive_feats, negative_feats)

    @staticmethod
    def train_triplet_id_net_in_ssl_training(simclr_model: SimCLRLitModule):
        dataset = partial(TripletComicsSeqFirebaseDataset,
                          triplet_index_csv_path='comics_seq/face_triplets.csv',
                          img_folder_root_dir='/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops/',
                          data_dir='/scratch/users/gsoykan20/amazing-mysteries-of-gutter-demystified/data',
                          transform=VisionTransformSetting.SIMCLR_TEST_TORCH.get_transformation(),
                          positive_transform=VisionTransformSetting.SIMCLR_TEST_TORCH.get_transformation(),
                          negative_transform=VisionTransformSetting.SIMCLR_TEST_TORCH.get_transformation(),
                          is_comics_crops_with_body_face=True,
                          is_torch_transform=False
                          )
        train_img_data = dataset(
            lower_idx_bound=0,
            higher_idx_bound=-4000,
            is_train=False
        )
        test_img_data = dataset(
            lower_idx_bound=-4000,
            higher_idx_bound=None,
            is_train=False
        )
        train_feats_data = TripletIdNetForSimCLRTesting.prepare_data_features(simclr_model, train_img_data)
        test_feats_data = TripletIdNetForSimCLRTesting.prepare_data_features(simclr_model, test_img_data)
        CHECKPOINT_PATH = '/scratch/users/gsoykan20/projects/amazing-mysteries-of-gutter-demystified/logs/experiments/runs/'
        batch_size = 128
        trainer = pl.Trainer(
            default_root_dir=os.path.join(CHECKPOINT_PATH, "TripletIdNetForSimCLRTesting"),
            gpus=1,
            max_epochs=20,
            callbacks=[
                ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                LearningRateMonitor("epoch"),
            ],
            progress_bar_refresh_rate=0,
            check_val_every_n_epoch=10,
        )
        trainer.logger._default_hp_metric = None

        # Data loaders
        train_loader = DataLoader(
            train_feats_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=0
        )
        test_loader = DataLoader(
            test_feats_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=0
        )

        model = TripletIdNetForSimCLRTesting(backbone_feature_dim=128,
                                             feature_dim=128,
                                             lr=0.0005,
                                             weight_decay=0.01, )
        trainer.fit(model, train_loader, test_loader)
        model = TripletIdNetForSimCLRTesting.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        # Test best model on train and validation set
        train_result = trainer.test(model, dataloaders=train_loader, verbose=False)
        test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
        result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}
        return model, result

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(self.hparams.max_epochs * 0.6), int(self.hparams.max_epochs * 0.8)], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    # make this a utility function...
    def _generate_preds_and_y(self, anchor, positive, negative) -> Tuple[Tensor, Tensor]:
        neg_gt = torch.zeros(anchor.shape[0], device=anchor.device)
        pos_gt = torch.ones(anchor.shape[0], device=anchor.device)
        gt = torch.cat([neg_gt, pos_gt]).type(torch.LongTensor)
        preds = (torch.cat([F.cosine_similarity(anchor, negative), F.cosine_similarity(anchor, positive)]) > 0.5).type(
            torch.LongTensor)
        return preds.view(-1, 1), gt.view(-1, 1)

    def step(self, batch: Any):
        # TODO: @gsoykan - measure mean & std of cos sim and euclidian dist
        x_all = torch.cat([batch[0], batch[1], batch[1]])
        id_embedding_all = self.forward(x_all)
        id_embedding_anchor, id_embedding_positive, id_embedding_negative = torch.tensor_split(id_embedding_all, 3)
        triplet_output = self.criterion(id_embedding_anchor, id_embedding_positive, id_embedding_negative)
        preds, y = self._generate_preds_and_y(id_embedding_anchor.detach(), id_embedding_positive.detach(),
                                              id_embedding_negative.detach())
        return triplet_output['loss'], preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.step(batch)
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "targets": targets}
