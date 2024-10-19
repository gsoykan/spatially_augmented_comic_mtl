from enum import Enum, unique
from typing import Optional, Dict, Any, Tuple
import random

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as tv_transforms
from albumentations.augmentations import functional as A_F
import matplotlib.pyplot as plt
from albumentations.core.transforms_interface import ImageOnlyTransform


class BlackOutWithPolygonMasks(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1.0):
        super(BlackOutWithPolygonMasks, self).__init__(always_apply, p)

    @property
    def targets_as_params(self):
        return ["image", "polygon_masks", "start_coordinates"]

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params

    def apply(self, img, **params):
        masks = params.get('polygon_masks')
        start_coordinates = params.get('start_coordinates')
        if masks is None or len(masks) == 0:
            return img

        if start_coordinates is not None:
            masks = list(map(lambda x: x[:, 0] - start_coordinates, masks))

        # Create a black mask with the same shape as the image
        black_mask = np.ones_like(img) * 255

        for mask in masks:
            # Draw the polygon on the black mask
            cv2.fillPoly(black_mask, [mask], (0, 0, 0))

        # Use the black mask to black out the image where the polygon is
        img_blackout = cv2.bitwise_and(img, black_mask)

        return img_blackout


def albu_clip_0_1(image, **kwargs):
    return A_F.clip(image, image.dtype, 1)


def concat_edge_map_channel(image, **kwargs):
    """
    plt.figure()
    plt.imshow(sample_image)
    plt.show()  # display it
    Args:
        image ():
        **kwargs ():

    Returns:
    """
    gray = cv2.cvtColor(image * 255, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # apply the Sobel filter in the x and y directions
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # combine the results to obtain the edges
    edges = np.uint8(np.sqrt(np.square(sobel_x) + np.square(sobel_y)))
    edges = edges / 255
    edges = np.expand_dims(edges, axis=2).astype(image.dtype)
    merged = np.concatenate((image, edges), axis=2)
    return merged


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.solarize(img)

# TODO: @gsoykan - you can delete unused settings
@unique
class VisionTransformSetting(str, Enum):
    VGG_TRAIN_SETTING_ONE = 'VGG_TRAIN_SETTING_ONE'
    VGG_STANDARD = 'VGG_STANDARD'
    # source: https://github.dev/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/models/common.py'
    # https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet#model-overview
    EFFICIENT_NET_STANDARD = 'EFFICIENT_NET_STANDARD'
    RAW_FACE_TRAINING = 'RAW_FACE_TRAINING'
    RAW_FACE_TESTING = 'RAW_FACE_TESTING'
    VANILLA_AUTOENCODER_FACE = 'VANILLA_AUTOENCODER_FACE'
    VAE_FACE_POSITIVE = 'VAE_FACE_POSITIVE'
    VAE_FACE_NEGATIVE = 'VAE_FACE_NEGATIVE'
    VAE_FACE_LIGHT = 'VAE_FACE_LIGHT'
    VANILLA_VAE_FACE = 'VANILLA_VAE_FACE'
    CORINFOMAX_PRETRAIN = 'CORINFOMAX_PRETRAIN'
    CORINFOMAX_PRETRAIN_TORCH = 'CORINFOMAX_PRETRAIN_TORCH'
    CORINFOMAX_PRETRAIN_PRIME = 'CORINFOMAX_PRETRAIN_PRIME'
    CORINFOMAX_PRETRAIN_PRIME_TORCH = 'CORINFOMAX_PRETRAIN_PRIME_TORCH'
    CORINFOMAX_EVAL_VAL = 'CORINFOMAX_EVAL_VAL',
    CORINFOMAX_EVAL_VAL_TORCH = 'CORINFOMAX_EVAL_VAL_TORCH',
    CORINFOMAX_FINE_TUNING = 'CORINFOMAX_FINE_TUNING'
    CORINFOMAX_EVAL_TEST = 'CORINFOMAX_EVAL_TEST',
    SIMCLR_PRETRAIN_TORCH = 'SIMCLR_PRETRAIN_TORCH',
    SIMCLR_TEST = 'SIMCLR_TEST'
    SIMCLR_FINE_TUNING = 'SIMCLR_FINE_TUNING'
    SIMCLR_FINE_TUNING_INTENSE = 'SIMCLR_FINE_TUNING_INTENSE'

    # nice transformation source: https://tugot17.github.io/data-science-blog/albumentations/data-augmentation/tutorial/2020/09/20/Pixel-level-transforms-using-albumentations-package.html#CLAHE
    def get_transformation(self,
                           should_get_standard: Optional[bool] = None,
                           **kwargs) -> A.Compose:
        def get_pad_and_crop():
            N = 224
            PADDED_N = 32
            transforms = A.Compose([
                A.LongestMaxSize(max_size=PADDED_N, interpolation=1),
                A.PadIfNeeded(min_height=PADDED_N, min_width=PADDED_N, border_mode=0, value=(0, 0, 0)),
                A.RandomCrop(height=N, width=N),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2()
            ])
            return transforms

        def get_universal_standard():
            N = 224
            transforms = A.Compose([
                A.LongestMaxSize(max_size=N, interpolation=1),
                A.PadIfNeeded(min_height=N, min_width=N, border_mode=0, value=(0, 0, 0)),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2()
            ])
            return transforms

        if should_get_standard:
            return get_universal_standard()
        if self == VisionTransformSetting.VGG_STANDARD:
            return get_universal_standard()
        elif self == VisionTransformSetting.VGG_TRAIN_SETTING_ONE:
            # TODO: @gsoykan apply some transformations here
            return get_universal_standard()
        elif self == VisionTransformSetting.EFFICIENT_NET_STANDARD:
            # TODO: @gsoykan - you might want to add some
            #  more augmentations for training
            N = 224
            transforms = A.Compose([
                A.LongestMaxSize(max_size=N + 32, interpolation=1),
                A.PadIfNeeded(min_height=N + 32, min_width=N + 32, border_mode=0, value=(0, 0, 0)),
                A.CenterCrop(N, N),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2()
            ])
            return transforms
        elif self == VisionTransformSetting.RAW_FACE_TRAINING:
            N = 80
            transforms = A.Compose([
                A.LongestMaxSize(max_size=N, interpolation=1),
                A.PadIfNeeded(min_height=N, min_width=N, border_mode=0, value=(0, 0, 0)),
                A.HorizontalFlip(),
                ToTensorV2()
            ])
            return transforms
        elif self == VisionTransformSetting.RAW_FACE_TESTING:
            N = 80
            transforms = A.Compose([
                A.LongestMaxSize(max_size=N, interpolation=1),
                A.PadIfNeeded(min_height=N, min_width=N, border_mode=0, value=(0, 0, 0)),
                ToTensorV2()
            ])
            return transforms
        elif self == VisionTransformSetting.VANILLA_AUTOENCODER_FACE:
            W = 96
            H = 96
            transforms = A.Compose([
                A.Resize(height=H, width=W),
                A.Normalize(0.5, 0.5),
                ToTensorV2(),
            ])
            return transforms
        elif self == VisionTransformSetting.VAE_FACE_POSITIVE:
            # source: https://albumentations.ai/docs/examples/showcase/
            # https://albumentations.ai/docs/examples/pytorch_classification/
            W = 96
            H = 96
            transforms = A.Compose([
                A.Resize(height=H, width=W),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.25),
                A.OneOf([
                    # light
                    A.GaussNoise(var_limit=(100, 150), p=1),
                    # medium
                    A.MotionBlur(blur_limit=17, p=1),
                    # strong
                    A.Compose([
                        A.Blur(blur_limit=11, p=1),
                        A.RandomBrightness(p=1),
                        A.CLAHE(p=1),
                    ])
                ], p=0.5),
                A.ToFloat(255),
                ToTensorV2()
            ])
            return transforms
        elif self == VisionTransformSetting.VAE_FACE_NEGATIVE:
            # source: https://albumentations.ai/docs/examples/showcase/
            # https://albumentations.ai/docs/examples/pytorch_classification/
            W = 96
            H = 96
            transforms = A.Compose([
                A.Resize(height=H, width=W),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.25),
                A.OneOf([
                    # light
                    A.GaussNoise(var_limit=(100, 150), p=1),
                    # medium
                    A.MotionBlur(blur_limit=17, p=1),
                ], p=0.5),
                A.ToFloat(255),
                ToTensorV2()
            ])
            return transforms
        elif self == VisionTransformSetting.VAE_FACE_LIGHT:
            W = 96
            H = 96
            transforms = A.Compose([
                A.Resize(height=H, width=W),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.ToFloat(255),
                ToTensorV2()
            ])
            return transforms
        elif self == VisionTransformSetting.VANILLA_VAE_FACE:
            W = 96
            H = 96
            transforms = A.Compose([
                A.Resize(height=H, width=W),
                A.ToFloat(255),
                ToTensorV2()
            ])
            return transforms
        elif self == VisionTransformSetting.CORINFOMAX_PRETRAIN:
            N = 224
            transforms = A.Compose([
                A.RandomResizedCrop(N, N, interpolation=cv2.INTER_CUBIC, scale=(0.2, 1.0), ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
                A.ToGray(p=0.2),
                A.GaussianBlur(sigma_limit=[0.1, 2], p=1),
                A.Solarize(p=0),
                A.Normalize(
                    mean=[0.531, 0.480, 0.387],
                    std=[0.249, 0.225, 0.192],
                ),
                ToTensorV2(),
            ])
            return transforms
        elif self == VisionTransformSetting.CORINFOMAX_PRETRAIN_TORCH:
            data_normalize_mean = [0.531, 0.480, 0.387]
            data_normalize_std = [0.249, 0.225, 0.192]
            min_scale = kwargs['min_scale'] if kwargs.get(
                'min_scale') is not None else 0.2  # 0.2 #0.08 in some versions
            N = kwargs['N'] if kwargs.get('N') is not None else 224
            transforms = tv_transforms.Compose(
                [
                    tv_transforms.RandomResizedCrop(
                        N,
                        scale=(min_scale, 1.0),
                        interpolation=tv_transforms.InterpolationMode.BICUBIC,  # Only in VicReg
                    ),
                    tv_transforms.RandomHorizontalFlip(p=0.5),
                    tv_transforms.RandomApply(
                        [tv_transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                    ),
                    tv_transforms.RandomGrayscale(p=0.2),
                    tv_transforms.RandomApply([GaussianBlur()], p=1.0),  # only for TinyImageNet
                    tv_transforms.RandomApply([Solarization()], p=0.0),  # Only in VicReg
                    tv_transforms.ToTensor(),
                    tv_transforms.Normalize(data_normalize_mean, data_normalize_std),
                ]
            )
            return transforms
        elif self == VisionTransformSetting.CORINFOMAX_PRETRAIN_PRIME:
            min_scale = kwargs['min_scale'] if kwargs.get(
                'min_scale') is not None else 0.08  # 0.2 #0.08 in some versions
            N = kwargs['N'] if kwargs.get('N') is not None else 224

            transforms = A.Compose([
                A.RandomResizedCrop(N, N, interpolation=cv2.INTER_CUBIC,
                                    scale=(min_scale, 1.0), ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
                A.ToGray(p=0.2),
                A.GaussianBlur(sigma_limit=[0.1, 2], p=0.1),
                A.Solarize(p=0.2),
                A.Normalize(
                    mean=[0.531, 0.480, 0.387],
                    std=[0.249, 0.225, 0.192],
                ),
                ToTensorV2(),
            ])
            return transforms

        elif self == VisionTransformSetting.CORINFOMAX_PRETRAIN_PRIME_TORCH:
            data_normalize_mean = [0.531, 0.480, 0.387]
            data_normalize_std = [0.249, 0.225, 0.192]
            min_scale = kwargs['min_scale'] if kwargs.get(
                'min_scale') is not None else 0.2  # 0.2 #0.08 in some versions
            N = kwargs['N'] if kwargs.get('N') is not None else 224
            transforms = tv_transforms.Compose(
                [
                    tv_transforms.RandomResizedCrop(
                        N,
                        scale=(min_scale, 1.0),
                        interpolation=tv_transforms.InterpolationMode.BICUBIC,  # Only in VicReg
                    ),
                    tv_transforms.RandomHorizontalFlip(p=0.5),
                    tv_transforms.RandomApply(
                        [tv_transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                    ),
                    tv_transforms.RandomGrayscale(p=0.2),
                    tv_transforms.RandomApply([GaussianBlur()], p=0.1),  # only for TinyImageNet
                    tv_transforms.RandomApply([Solarization()], p=0.2),  # Only in VicReg
                    tv_transforms.ToTensor(),
                    tv_transforms.Normalize(data_normalize_mean, data_normalize_std),
                ]
            )
            return transforms
        elif self == VisionTransformSetting.CORINFOMAX_EVAL_VAL:
            min_scale = kwargs['min_scale'] if kwargs.get(
                'min_scale') is not None else 0.08  # 0.2 #0.08 in some versions
            N = kwargs['N'] if kwargs.get('N') is not None else 224
            transforms = A.Compose([
                A.RandomResizedCrop(N, N, interpolation=cv2.INTER_CUBIC, scale=(min_scale, 1.0), ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=[0.531, 0.480, 0.387],
                    std=[0.249, 0.225, 0.192],
                ),
                ToTensorV2(),
            ])
            return transforms
        elif self == VisionTransformSetting.CORINFOMAX_EVAL_VAL_TORCH:
            data_normalize_mean = [0.531, 0.480, 0.387]
            data_normalize_std = [0.249, 0.225, 0.192]
            min_scale = kwargs['min_scale'] if kwargs.get(
                'min_scale') is not None else 0.2  # 0.2 #0.08 in some versions
            N = kwargs['N'] if kwargs.get('N') is not None else 224
            transforms = tv_transforms.Compose(
                [
                    tv_transforms.RandomResizedCrop(
                        N,
                        scale=(min_scale, 1.0),
                        interpolation=tv_transforms.InterpolationMode.BICUBIC,  # Only in VicReg
                    ),
                    tv_transforms.RandomHorizontalFlip(p=0.5),
                    tv_transforms.ToTensor(),
                    tv_transforms.Normalize(data_normalize_mean, data_normalize_std),
                ]
            )
            return transforms
        elif self == VisionTransformSetting.CORINFOMAX_EVAL_TEST:
            N = kwargs['N'] if kwargs.get('N') is not None else 224
            use_padding = kwargs['use_padding'] if kwargs.get('use_padding') is not None else True
            if use_padding:
                transforms = A.Compose([
                    A.LongestMaxSize(max_size=N + 32, interpolation=1),
                    A.PadIfNeeded(min_height=N + 32, min_width=N + 32, border_mode=0, value=(0, 0, 0)),
                    A.CenterCrop(N, N),
                    A.Normalize(
                        mean=[0.531, 0.480, 0.387],
                        std=[0.249, 0.225, 0.192],
                    ),
                    ToTensorV2(),
                ])
            else:
                transforms = A.Compose([
                    A.Resize(N, N),
                    A.Normalize(
                        mean=[0.531, 0.480, 0.387],
                        std=[0.249, 0.225, 0.192],
                    ),
                    ToTensorV2(),
                ])
            return transforms
        elif self == VisionTransformSetting.CORINFOMAX_FINE_TUNING:
            N = kwargs['N'] if kwargs.get('N') is not None else 224
            use_padding = kwargs['use_padding'] if kwargs.get('use_padding') is not None else True
            min_scale = kwargs['min_scale'] if kwargs.get(
                'min_scale') is not None else 0.2  # 0.2 #0.08 in some versions
            common_augs = [
                A.RandomResizedCrop(N, N, interpolation=cv2.INTER_CUBIC, scale=(min_scale, 1.0), p=0.25),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.GaussianBlur(sigma_limit=[0.1, 2], p=0.1),
                A.Solarize(p=0.2),
            ]
            if use_padding:
                transforms = A.Compose([
                    A.LongestMaxSize(max_size=N + 32, interpolation=1),
                    A.PadIfNeeded(min_height=N + 32, min_width=N + 32, border_mode=0, value=(0, 0, 0)),
                    A.CenterCrop(N, N),
                    *common_augs,
                    A.Normalize(
                        mean=[0.531, 0.480, 0.387],
                        std=[0.249, 0.225, 0.192],
                    ),
                    ToTensorV2(),
                ])
            else:
                transforms = A.Compose([
                    A.Resize(N, N),
                    *common_augs,
                    A.Normalize(
                        mean=[0.531, 0.480, 0.387],
                        std=[0.249, 0.225, 0.192],
                    ),
                    ToTensorV2(),
                ])
            return transforms
        elif self == VisionTransformSetting.SIMCLR_PRETRAIN_TORCH:
            N = kwargs['N'] if kwargs.get('N') is not None else 224
            min_scale = kwargs['min_scale'] if kwargs.get(
                'min_scale') is not None else 0.2  # 0.2 #0.08 in some versions
            contrast_transforms = tv_transforms.Compose(
                [
                    tv_transforms.RandomHorizontalFlip(),
                    tv_transforms.RandomResizedCrop(size=N, scale=(min_scale, 1)),
                    tv_transforms.RandomApply(
                        [tv_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
                    tv_transforms.RandomGrayscale(p=0.2),
                    tv_transforms.GaussianBlur(kernel_size=9),
                    tv_transforms.ToTensor(),
                    tv_transforms.Normalize((0.5,), (0.5,)),
                ]
            )
            return contrast_transforms
        elif self == VisionTransformSetting.SIMCLR_TEST:
            N = kwargs['N'] if kwargs.get('N') is not None else 224
            use_padding = kwargs['use_padding'] if kwargs.get('use_padding') is not None else True
            if use_padding:
                transforms = A.Compose([
                    A.LongestMaxSize(max_size=N, interpolation=1),
                    A.PadIfNeeded(min_height=N, min_width=N, border_mode=0, value=(0, 0, 0)),
                    A.Normalize(
                        mean=0.5,
                        std=0.5,
                    ),
                    ToTensorV2(),
                ])
            else:
                transforms = A.Compose([
                    A.Resize(N, N),
                    A.Normalize(
                        mean=0.5,
                        std=0.5,
                    ),
                    ToTensorV2(),
                ])
            return transforms
        elif self == VisionTransformSetting.SIMCLR_FINE_TUNING:
            N = kwargs['N'] if kwargs.get('N') is not None else 224
            use_padding = kwargs['use_padding'] if kwargs.get('use_padding') is not None else True
            min_scale = kwargs['min_scale'] if kwargs.get(
                'min_scale') is not None else 0.2  # 0.2 #0.08 in some versions
            common_augs = [
                A.RandomResizedCrop(N, N, interpolation=cv2.INTER_CUBIC, scale=(min_scale, 1.0), p=0.25),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(5, 7), p=1),
                    A.MotionBlur(p=1),
                    A.MedianBlur(blur_limit=3, p=1),
                    A.Blur(blur_limit=3, p=1),
                ], p=0.1),
                A.ToGray(p=0.2),
            ]
            if use_padding:
                transforms = A.Compose([
                    A.LongestMaxSize(max_size=N, interpolation=1),
                    A.PadIfNeeded(min_height=N, min_width=N, border_mode=0, value=(0, 0, 0)),
                    *common_augs,
                    A.Normalize(
                        mean=0.5,
                        std=0.5,
                    ),
                    ToTensorV2(),
                ])
            else:
                transforms = A.Compose([
                    A.Resize(N, N),
                    *common_augs,
                    A.Normalize(
                        mean=0.5,
                        std=0.5,
                    ),
                    ToTensorV2(),
                ])
            return transforms
        elif self == VisionTransformSetting.SIMCLR_FINE_TUNING_INTENSE:
            N = kwargs['N'] if kwargs.get('N') is not None else 224
            use_padding = kwargs['use_padding'] if kwargs.get('use_padding') is not None else True
            min_scale = kwargs['min_scale'] if kwargs.get(
                'min_scale') is not None else 0.2  # 0.2 #0.08 in some versions
            common_augs = [
                # p => 0.25 in lighter version
                A.RandomResizedCrop(N, N, interpolation=cv2.INTER_CUBIC, scale=(min_scale, 1.0), p=0.5),
                # A.HorizontalFlip(p=0.5),
                A.Flip(),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(5, 7), p=1),
                    A.MotionBlur(p=1),
                    A.MedianBlur(blur_limit=3, p=1),
                    A.Blur(blur_limit=3, p=1),
                ], p=0.5),  # in light: p=0.1
                A.ToGray(p=0.2),
            ]
            if use_padding:
                transforms = A.Compose([
                    A.LongestMaxSize(max_size=N, interpolation=1),
                    A.PadIfNeeded(min_height=N, min_width=N, border_mode=0, value=(0, 0, 0)),
                    *common_augs,
                    A.Normalize(
                        mean=0.5,
                        std=0.5,
                    ),
                    ToTensorV2(),
                ])
            else:
                transforms = A.Compose([
                    A.Resize(N, N),
                    *common_augs,
                    A.Normalize(
                        mean=0.5,
                        std=0.5,
                    ),
                    ToTensorV2(),
                ])
            return transforms
        # We will not have this - since longest max size works better for us...
        # elif self == VisionTransformSetting.CORINFOMAX_EVAL_TEST_TORCH:
        #     data_normalize_mean = [0.531, 0.480, 0.387]
        #     data_normalize_std = [0.249, 0.225, 0.192]
        #     min_scale = 0.2  # 0.2 #0.08 in some versions
        #     N = 224
        #     transforms = tv_transforms.Compose(
        #         [
        #             tv_transforms.Resize(int(self.random_crop_size * (8 / 7)),
        #                                  interpolation=tv_transforms.InterpolationMode.BICUBIC),
        #             # In Imagenet: 224 -> 256
        #             tv_transforms.CenterCrop(self.random_crop_size),
        #             tv_transforms.ToTensor(),
        #             tv_transforms.Normalize(data_normalize_mean, data_normalize_std),
        #         ]
        #     )
        #     return transforms
