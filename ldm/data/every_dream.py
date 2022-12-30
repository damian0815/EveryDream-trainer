from torch.utils.data import Dataset
from ldm.data.data_loader import DataLoaderMultiAspect as dlma
import math
import ldm.data.dl_singleton as dls
from ldm.data.image_train_item import ImageTrainItem
import random

class EveryDreamBatch(Dataset):
    """
    data_root: root path of all your training images, will be recursively searched for images
    repeats: how many times to repeat each image in the dataset
    flip_p: probability of flipping the image horizontally
    debug_level: 0=none, 1=print drops due to unfilled batches on aspect ratio buckets, 2=debug info per image, 3=save crops to disk for inspection
    batch_size: how many images to return in a batch
    conditional_dropout: probability of dropping the caption for a given image
    resolution: max resolution (relative to square)
    jitter: number of pixels to jitter the crop by, only for non-square images
    """
    def __init__(self,
                 data_root,
                 repeats=1,
                 flip_p=0.0,
                 debug_level=0,
                 batch_size=1,
                 set='train',
                 conditional_dropout=0.02,
                 resolution=512,
                 crop_jitter=20,
                 seed=555,
                 test_pct=0.15,
                 validate_pct=0.15
                 ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.debug_level = debug_level
        self.conditional_dropout = conditional_dropout
        self.crop_jitter = crop_jitter
        self.unloaded_to_idx = 0
        if seed == -1:
            seed = random.randint(0, 9999)

        if not dls.shared_dataloader:
            print(" * Creating new dataloader singleton")
            dls.shared_dataloader = dlma(data_root=data_root,
                                         seed=seed,
                                         debug_level=debug_level,
                                         batch_size=self.batch_size,
                                         flip_p=flip_p,
                                         resolution=resolution,
                                         test_pct=test_pct,
                                         validate_pct=validate_pct)

        self.image_train_items = dls.shared_dataloader.get_train_images()
        
        self.num_images = len(self.image_train_items)

        self._length = math.trunc(self.num_images * repeats)

        print()
        print(f" ** Trainer Set: {set}, steps: {self._length / batch_size:.0f}, num_images: {self.num_images}, batch_size: {self.batch_size}, length w/repeats: {self._length}")
        print()

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        idx = i % self.num_images
        image_train_item = self.image_train_items[idx]
        example = self.__get_image_for_trainer(image_train_item, self.debug_level)

        if self.unloaded_to_idx > idx:
            self.unloaded_to_idx = 0

        if idx % (self.batch_size*3) == 0 and idx > (self.batch_size * 5):
            start_del = self.unloaded_to_idx
            self.unloaded_to_idx = int(idx / self.batch_size)*self.batch_size - self.batch_size*4
            
            for j in range(start_del, self.unloaded_to_idx):
                if hasattr(self.image_train_items[j], 'image'):
                    del self.image_train_items[j].image
            if self.debug_level > 1: print(f" * Unloaded images from idx {start_del} to {self.unloaded_to_idx}")

        return example

    def __get_image_for_trainer(self, image_train_item: ImageTrainItem, debug_level=0):
        example = {}

        save = debug_level > 2

        image_train_tmp = image_train_item.hydrate(crop=False, save=save, crop_jitter=self.crop_jitter)

        example["image"] = image_train_tmp.image
        
        if random.random() > self.conditional_dropout:
            example["caption"] = image_train_tmp.caption
        else:
            example["caption"] = " "

        return example
