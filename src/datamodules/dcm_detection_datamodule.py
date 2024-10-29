from functools import partial
from typing import Optional, Tuple

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.components.component_detection.dcm.dcm_bounding_box_dataset import DCMBoundingBoxDataset
from src.datamodules.components.component_detection.faster_rcnn_collate_fn import faster_rcnn_collate_fn
from src.datamodules.components.vision_transform_setting import albu_clip_0_1
from src.utils.pickle_helper import PickleHelper


class DCMDetectionDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            train_val_test_split: Tuple[float, float, float] = (0, -50, -25),
            img_dims_hw: Tuple[int, Optional[int]] = (512, 512),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            use_instance_masks: bool = False,
            use_aug_for_training: bool = True,
            shuffle: bool = True
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 6

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = partial(DCMBoundingBoxDataset,
                              data_dir=self.hparams.data_dir,
                              use_instance_masks=self.hparams.use_instance_masks,
                              shuffle=self.hparams.shuffle
                              )
            boundary_low = self.hparams.train_val_test_split[0]
            boundary_mid = self.hparams.train_val_test_split[1]
            boundary_high = self.hparams.train_val_test_split[2]

            validation_transformations = A.Compose([
                A.LongestMaxSize(self.hparams.img_dims_hw[0]) if self.hparams.img_dims_hw[1] is None else A.Resize(
                    self.hparams.img_dims_hw[0], self.hparams.img_dims_hw[1], always_apply=True),
                A.ToFloat(max_value=255, always_apply=True),
                ToTensorV2(),
            ],
                bbox_params=A.BboxParams(format='pascal_voc',
                                         label_fields=['labels', 'area', 'iscrowd', 'bb_ids', 'indices']))

            train_transformations = A.Compose([
                A.ToFloat(max_value=255, always_apply=True),
                A.LongestMaxSize(self.hparams.img_dims_hw[0]) if self.hparams.img_dims_hw[1] is None else A.Resize(
                        self.hparams.img_dims_hw[0], self.hparams.img_dims_hw[1]),
#                 A.OneOf([
#                     A.LongestMaxSize(self.hparams.img_dims_hw[0]) if self.hparams.img_dims_hw[1] is None else A.Resize(
#                         self.hparams.img_dims_hw[0], self.hparams.img_dims_hw[1]),
#                     A.RandomResizedCrop(self.hparams.img_dims_hw[0],
#                                         self.hparams.img_dims_hw[1] if self.hparams.img_dims_hw[1] is not None else
#                                         self.hparams.img_dims_hw[0],
#                                         interpolation=cv2.INTER_CUBIC, scale=(0.2, 1.0)),
#                 ], p=1),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(5, 7), p=1),
                    A.MotionBlur(p=1),
                    A.MedianBlur(blur_limit=3, p=1),
                    A.Blur(blur_limit=3, p=1),
                ], p=0.1),
                A.ToGray(p=0.2),
                A.OneOf([
                    A.ToSepia(p=1),
#                     A.RandomSnow(brightness_coeff=2, p=1),
#                     A.RandomFog(fog_coef_lower=0.2,
#                                 fog_coef_upper=0.4, p=1.0),
                    A.ColorJitter(p=1.0),
                ], p=0.1),
                A.HorizontalFlip(),
                A.Lambda(image=albu_clip_0_1),
                ToTensorV2(),
            ],
                bbox_params=A.BboxParams(format='pascal_voc',
                                         label_fields=['labels', 'area', 'iscrowd', 'bb_ids', 'indices'],
                                         min_area=16,
                                         min_visibility=0.1))

            self.data_train = dataset(
                element_slice=(boundary_low, boundary_mid),
                transform=train_transformations if self.hparams.use_aug_for_training else validation_transformations
            )
            print('train dataset size => ', len(self.data_train))
            self.data_val = dataset(
                element_slice=(boundary_mid, boundary_high),
                transform=validation_transformations,
            )
            self.data_test = dataset(
                element_slice=(boundary_high, None),
                transform=validation_transformations,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=faster_rcnn_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=faster_rcnn_collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=faster_rcnn_collate_fn
        )


if __name__ == '__main__':
    data_dir = '/home/gsoykan20/Desktop/self_development/amazing-mysteries-of-gutter-demystified/data'
    datamodule = DCMDetectionDataModule(data_dir=data_dir,
                                        batch_size=4,
                                        use_instance_masks=True,
                                        img_dims_hw=(640, 480),
                                        use_aug_for_training=False)
    datamodule.prepare_data()
    datamodule.setup()
    dataloader = iter(datamodule.test_dataloader())
    batch = next(dataloader)
    PickleHelper.save_object(PickleHelper.faster_rcnn_batch, batch)
    loaded_batch = PickleHelper.load_object(PickleHelper.faster_rcnn_batch)
    # print(batch)
