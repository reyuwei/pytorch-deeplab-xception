from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class SBODYSegmentation(Dataset):
    """
    Semantic body dataset
    """
    NUM_CLASSES = 11  # include background

    CLASSES = (
        "__background__",
        "Torso",
        "Rarm",
        "LArm",
        "RHand",
        "LHand",
        "LLeg",
        "RLeg",
        "LFoot",
        "RFoot",
        "Head",
    )

    def __init__(self, args, base_dir=Path.db_root_dir('semantic_body'), split='train'):
        """
        :param base_dir: path to dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir + 'real/'
        self._list_dir = self._base_dir + 'list/'
        # self.transforms = transforms

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        image = []
        initial_mask = []
        gt_semantic_mask = []
        for splt in self.split:
            ann_file = os.path.join(os.path.join(self._list_dir, splt + 'lst.txt'))

            with open(ann_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                items = line.split("\t")
                if len(items) == 2:  # for demo
                    items.append(items[1])
                    self.is_demo = True
                else:
                    self.is_demo = False
                image.append(items[0].replace("\n", ""))
                initial_mask.append(items[1].replace("\n", ""))
                gt_semantic_mask.append(items[2].replace("\n", ""))

        self.imagelist = image
        self.initialmsklist = initial_mask
        self.gtsemanticmsklist = gt_semantic_mask
        self.length = len(image)
        self.classes = SBODYSegmentation.NUM_CLASSES
        assert (len(self.imagelist) == len(self.initialmsklist))
        assert (len(self.initialmsklist) == len(self.gtsemanticmsklist))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, self.length))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)

    def _make_img_gt_point_pair(self, idx):
        """
        :param idx: image index in list
        :return: input image, ground truth segmentation

        ground truth segmentation: init mask input is [0, 10, 20, ... , 100], change to [0, 1, 2, ... 10]
        """
        _target = Image.open(self._base_dir + self.gtsemanticmsklist[idx]).convert("L")
        # _img = Image.open(self._base_dir + self.imagelist[idx]).convert('RGB')


        # real input path
        init_mask_path = self._base_dir + self.initialmsklist[idx]
        init_mask = Image.open(init_mask_path).convert("L")  # gray
        downsample_width, downsample_height = init_mask.size

        imgname_split = self.imagelist[idx].split('/')
        real_input_folder = self._base_dir + imgname_split[0] + "/" + imgname_split[1] + "/real_input/"

        if os.path.exists(real_input_folder):
            real_input_file = real_input_folder + imgname_split[2]
            input_img = Image.open(real_input_file).convert("RGB")
            _img = input_img.resize((downsample_width, downsample_height), Image.ANTIALIAS)
        else:
            _img = Image.open(self._base_dir + self.imagelist[idx]).convert("RGB")
            _img = img.resize((downsample_width, downsample_height), Image.ANTIALIAS)

            # for real data
            init_mask_array = np.array(init_mask)
            init_mask_array_empty = init_mask_array.copy()
            init_mask_array_empty[:] = 0
            for cc in range(0, 110, 10):
                init_mask_array_empty[init_mask_array == cc] = cc / 10
            init_mask = Image.fromarray(init_mask_array_empty)
            r, g, b = img.split()
            _img = Image.merge("RGB", (r, init_mask, b))

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'Semantic Body Data(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = SBODYSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='semantic_body')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


