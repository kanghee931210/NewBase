'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2022, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2022.04.20.
'''
import torch.utils.data
import numpy as np
from torchvision import transforms
from skimage import io, transform, color

from albumentations.core.transforms_interface import ImageOnlyTransform

import albumentations as A
import albumentations.pytorch

from PIL import Image

from rembg import remove
from pathlib import Path
import os

import onnxruntime as ort

class AugMix(ImageOnlyTransform):
    """Augmentations mix to Improve Robustness and Uncertainty.
    Args:
        image (np.ndarray): Raw input image of shape (h, w, c)
        severity (int): Severity of underlying augmentation operators.
        width (int): Width of augmentation chain
        depth (int): Depth of augmentation chain. -1 enables stochastic depth uniformly
          from [1, 3]
        alpha (float): Probability coefficient for Beta and Dirichlet distributions.
        augmentations (list of augmentations): Augmentations that need to mix and perform.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, width=2, depth=2, alpha=0.5, augmentations=[], always_apply=False, p=0.5):
        super(AugMix, self).__init__(always_apply, p)
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.augmentations = augmentations
        self.ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        self.m = np.float32(np.random.beta(self.alpha, self.alpha))

    def apply_op(self, image, op):
        image = op(image=image)["image"]
        return image

    def apply(self, img, **params):
        mix = np.zeros_like(img)
        for i in range(self.width):
            image_aug = img.copy()

            for _ in range(self.depth):
                op = np.random.choice(self.augmentations)
                image_aug = self.apply_op(image_aug, op)

            mix = np.add(mix, self.ws[i] * image_aug, out=mix, casting="unsafe")

        mixed = (1 - self.m) * img + self.m * mix
        if img.dtype in ["uint8", "uint16", "uint32", "uint64"]:
            mixed = np.clip((mixed), 0, 255).astype(np.uint8)
        return mixed

    def get_transform_init_args_names(self):
        return ("width", "depth", "alpha")

class BackGround(object):
    """Operator that resizes to the desired size while maintaining the ratio
        fills the remaining part with a black background

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size.
    """

    def __init__(self, output_size,input_size):
        self.output_size = output_size
        self.input_size = input_size // 2
    def __call__(self, image, landmarks, sub_landmarks=None):
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size * h / w, self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), mode='constant')

        if landmarks is not None:
            landmarks = landmarks * [new_w / w, new_h / h]

            new_image = np.zeros((self.output_size, self.output_size, 3))

            if h > w:
                new_image[:,(self.input_size - new_w//2):(self.input_size - new_w//2 + new_w),:] = img
                landmarks = landmarks + [self.input_size - new_w//2, 0]
            else:
                new_image[(self.input_size - new_h//2):(self.input_size - new_h//2 + new_h), :, :] = img
                landmarks = landmarks + [0, self.input_size - new_h//2]

            if sub_landmarks is not None:
                sub_landmarks = sub_landmarks * [new_w / w, new_h / h]
                if h > w:
                    sub_landmarks = sub_landmarks + [self.input_size - new_w // 2, 0]
                else:
                    sub_landmarks = sub_landmarks + [0, self.input_size - new_h // 2]
                return new_image, landmarks, sub_landmarks
            else:
                return new_image, landmarks
        else:
            new_image = np.zeros((self.output_size, self.output_size, 3))
            if h > w:
                new_image[:,(self.input_size - new_w//2):(self.input_size - new_w//2 + new_w),:] = img
            else:
                new_image[(self.input_size - new_h//2):(self.input_size - new_h//2 + new_h), :, :] = img

            return new_image


class BBoxCrop(object):
    """ Operator that crops according to the given bounding box coordinates. """

    def __call__(self, image, x_1, y_1, x_2, y_2):
        h, w = image.shape[:2]

        top = y_1
        left = x_1
        new_h = y_2 - y_1
        new_w = x_2 - x_1

        image = image[top: top + new_h,
                      left: left + new_w]

        return image


class ETRIDataset_emo(torch.utils.data.Dataset):
    """ Dataset containing emotion categories (Daily, Gender, Embellishment). """

    def __init__(self, df, base_path,img_size):
        self.df = df
        self.base_path = base_path
        self.bbox_crop = BBoxCrop()
        self.background = BackGround(img_size,img_size)

        self.transforms = A.Compose([
                    A.Resize(img_size, img_size),
                    A.OneOf([
                        A.HorizontalFlip(p=1),
                        A.ShiftScaleRotate(
                            shift_limit=0.0625,
                            scale_limit=0.1,
                            rotate_limit=30,
                            p=1.0),
                        A.VerticalFlip(p=1)
                    ], p=1),
                    A.OneOf([
                        A.MotionBlur(p=1),
                        A.OpticalDistortion(p=1),
                        A.GaussNoise(p=1)
                    ], p=1),
                    A.CoarseDropout(p=1, max_holes=20, max_height=24, max_width=24
                                    , min_holes=1, min_height=8, min_width=8),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    A.pytorch.transforms.ToTensorV2()
                            ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # # for vis
        # self.unnormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        #                                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        # self.to_pil = transforms.ToPILImage()

    def __getitem__(self, i):
        sample = self.df.iloc[i]
        image = io.imread(self.base_path + sample['image_name'].split('.')[0]+'.png')
        if image.shape[2] != 3:
            image = color.rgba2rgb(image)
        daily_label = sample['Daily']
        gender_label = sample['Gender']
        embel_label = sample['Embellishment']
        bbox_xmin = sample['BBox_xmin']
        bbox_ymin = sample['BBox_ymin']
        bbox_xmax = sample['BBox_xmax']
        bbox_ymax = sample['BBox_ymax']

        image = self.bbox_crop(image, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
        image = self.background(image, None)

        image_ = image.copy()

        image_ = self.transforms(image = image_)

        ret = {}
        ret['ori_image'] = image
        ret['image'] = image_['image']
        ret['daily_label'] = daily_label
        ret['gender_label'] = gender_label
        ret['embel_label'] = embel_label

        return ret

    def __len__(self):
        return len(self.df)

class ETRIDataset_emo_val(torch.utils.data.Dataset):
    """ Dataset containing emotion categories (Daily, Gender, Embellishment). """

    def __init__(self, df, base_path,img_size):
        self.df = df
        self.base_path = base_path
        self.bbox_crop = BBoxCrop()
        self.background = BackGround(img_size,img_size)

        self.transforms = A.Compose([
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    A.pytorch.transforms.ToTensorV2()
                            ])

    def __getitem__(self, i):
        sample = self.df.iloc[i]
        image = io.imread(self.base_path + sample['image_name'].split('.')[0]+'.png')
        if image.shape[2] != 3:
            image = color.rgba2rgb(image)
        daily_label = sample['Daily']
        gender_label = sample['Gender']
        embel_label = sample['Embellishment']
        bbox_xmin = sample['BBox_xmin']
        bbox_ymin = sample['BBox_ymin']
        bbox_xmax = sample['BBox_xmax']
        bbox_ymax = sample['BBox_ymax']

        image = self.bbox_crop(image, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
        image = self.background(image,None)

        image_ = image.copy()

        image_ = self.transforms(image = image_)

        ret = {}
        ret['ori_image'] = image
        ret['image'] = image_['image']
        ret['daily_label'] = daily_label
        ret['gender_label'] = gender_label
        ret['embel_label'] = embel_label

        return ret

    def __len__(self):
        return len(self.df)
class ETRIDataset_submit(torch.utils.data.Dataset):
    """ Dataset containing emotion categories (Daily, Gender, Embellishment). """

    def __init__(self, df, base_path,img_size):
        self.df = df
        self.base_path = base_path
        self.bbox_crop = BBoxCrop()
        self.background = BackGround(img_size,img_size)

        self.transforms = A.Compose([
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                    A.pytorch.transforms.ToTensorV2()
                            ])

    def __getitem__(self, i):
        sample = self.df.iloc[i]
        image = io.imread(self.base_path + sample['image_name'].split('.')[0]+'.png')
        image = Image.fromarray(image)
        image = np.array(remove(image))
        if image.shape[2] != 3:
            image = color.rgba2rgb(image)
        daily_label = sample['Daily']
        gender_label = sample['Gender']
        embel_label = sample['Embellishment']
        bbox_xmin = sample['BBox_xmin']
        bbox_ymin = sample['BBox_ymin']
        bbox_xmax = sample['BBox_xmax']
        bbox_ymax = sample['BBox_ymax']

        image = self.bbox_crop(image, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax)
        image = self.background(image,None)

        image_ = image.copy()

        image_ = self.transforms(image = image_)

        ret = {}
        ret['ori_image'] = image
        ret['image'] = image_['image']
        ret['daily_label'] = daily_label
        ret['gender_label'] = gender_label
        ret['embel_label'] = embel_label

        return ret

    def __len__(self):
        return len(self.df)