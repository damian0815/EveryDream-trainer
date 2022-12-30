import numpy as np
from torch.utils.data import Dataset
from ldm.data.data_loader import DataLoaderMultiAspect as dlma
import math
import ldm.data.dl_singleton as dls
from ldm.data.image_train_item import ImageTrainItem

class EDValidateBatch(Dataset):
    def __init__(self,
                 data_root,
                 batch_size=1,
                 set='val',
                 ):
        self.set = set
        self.data_root = data_root
        self.batch_size = batch_size

        if not dls.shared_dataloader:
            raise RuntimeError(f"{type(self).__name__} must be instantiated after EveryDreamBatch")

        self.image_train_items = dls.shared_dataloader.get_validation_images() if set=='val' else dls.shared_dataloader.get_test_images()
        
        self.num_images = len(self.image_train_items)

        self._length = self.num_images

        print()
        print(f" ** Validation/Test Set: {set}, steps: {self._length / batch_size:.0f}")
        print()

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        assert(i < self.num_images)
        image_train_item: ImageTrainItem = self.image_train_items[i]
        print(f"image {i} summoned for dataset '{self.set}': {image_train_item.pathname}, caption '{image_train_item.caption}'")

        example = self.__get_image_for_trainer(image_train_item)
        return example

    @staticmethod
    def __get_image_for_trainer(image_train_item: ImageTrainItem):
        example = {}

        image_train_tmp = image_train_item.hydrate()

        example["image"] = image_train_tmp.image
        example["caption"] = image_train_tmp.caption

        return example
        