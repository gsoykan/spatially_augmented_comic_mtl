from typing import Any, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor

from src.datamodules.components.vision_transform_setting import VisionTransformSetting
from src.models.components.ssl_module.info_nce_loss import InfoNCELoss
from src.models.components.ssl_module.nt_xent_loss import NTXentLoss
from src.models.components.ssl_module.sim_clr import SimCLR
from src.utils.basic_utils import read_or_get_image


class SimCLRLitModule(LightningModule):
    def __init__(
            self,
            # model
            model_name: str = 'resnet18',
            encoder_dim: Optional[int] = None,
            use_deeper_proj_head: Optional[bool] = False,
            normalize_projections: bool = False,
            hidden_dim: int = 128,
            # loss
            temperature: float = 0.07,  # 0.5 for nt-xent loss
            loss_fn='info_nce',  # 'info_nce' or 'nt_xent'
            # training
            max_epochs: int = 500,
            lr: float = 5e-4,
            weight_decay: float = 0.0001,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        self.model = SimCLR(model_name=model_name,
                            hidden_dim=hidden_dim,
                            encoder_dim=encoder_dim,
                            use_deeper_proj_head=use_deeper_proj_head,
                            normalize=normalize_projections)
        if loss_fn == 'info_nce':
            self.criterion = InfoNCELoss(temperature)
        elif loss_fn == 'nt_xent':
            self.criterion = NTXentLoss(temperature)

    def forward(self, x_all: Tensor) -> Tensor:
        return self.model(x_all)

    def step(self, batch: Any):
        x_all = torch.cat([batch[0], batch[1]])
        representation_all = self.forward(x_all)
        embedding_anchor, embedding_prime = torch.tensor_split(representation_all, 2)
        loss, logging_metrics = self.criterion(embedding_anchor, embedding_prime)
        return loss, logging_metrics

    def training_step(self, batch: Any, batch_idx: int):
        loss, logging_metrics = self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in logging_metrics.items():
            self.log(f"train/{k}", v, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, **logging_metrics}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logging_metrics = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        for k, v in logging_metrics.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, **logging_metrics}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, logging_metrics = self.step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        for k, v in logging_metrics.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True)
        return {"loss": loss, **logging_metrics}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # make a system that trains a logreg model
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    @classmethod
    def checkpoint_to_eval(cls, checkpoint_path: str):
        trained_model = cls.load_from_checkpoint(checkpoint_path).to('cuda')
        trained_model.eval()
        trained_model.freeze()
        return trained_model

    @staticmethod
    def get_embeddings_from_imgs(trained_model: LightningModule, img_paths: List[str]) -> torch.Tensor:
        # TODO: @gsoykan - update this...
        transform = VisionTransformSetting.CORINFOMAX_EVAL_TEST.get_transformation()
        batch = []
        for img_path in img_paths:
            source_img = read_or_get_image(img_path, read_rgb=True)
            source_img = transform(image=source_img)['image'].to(trained_model.device)
            batch.append(source_img)
        batch = torch.stack(batch)
        with torch.no_grad():
            embeddings = trained_model.model(batch)
        return embeddings


######################################

def check_embedding_generation():
    # TODO: @gsoykan - make sure this works...
    ckpt_path = '/scratch/users/gsoykan20/projects/amazing-mysteries-of-gutter-demystified/logs/experiments/runs/corinfomax_comics_crops_body/2022-12-17_21-43-50/checkpoints/last.ckpt'
    trained_model: SimCLRLitModule = SimCLRLitModule.checkpoint_to_eval(ckpt_path)
    img_paths = [
        '/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops/1008/10_5/bodies/1.jpg',
        '/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops/100/12_3/bodies/1.jpg',
        '/scratch/users/gsoykan20/projects/DASS_Det_Inference/comics_crops/1008/10_4/bodies/0.jpg']
    embeddings = SimCLRLitModule.get_embeddings_from_imgs(trained_model, img_paths).detach().cpu()
    return trained_model, embeddings


if __name__ == '__main__':
    loaded_module, embeddings = check_embedding_generation()
    eigs = loaded_module.cov_criterion.save_eigs()
    dists = [[torch.linalg.norm(e1 - e2).item() for e2 in embeddings] for e1 in embeddings]
    print(pd.DataFrame(dists, columns=['same 1', 'diff', 'same 2', ], index=['same 1', 'diff', 'same 2', ]))
    cos_sims = [[F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))[0].item() for e2 in embeddings] for e1 in
                embeddings]
    print(pd.DataFrame(cos_sims, columns=['same 1', 'diff', 'same 2', ], index=['same 1', 'diff', 'same 2', ]))
