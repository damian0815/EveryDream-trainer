# script to test data loader by itself
# run from training root, edit the data_root manually
# python ldm/data/test_dl.py
import time

from ldm.data.ed_validate import EDValidateBatch
from ldm.data.data_loader import DataLoaderMultiAspect
import ldm.data.dl_singleton as dls

s = time.perf_counter()

data_root = "../stablediffusion/buzzybee/cropped"
num_images = 11 # count by hand

batch_size = 1
repeats = 1

dls.shared_dataloader = DataLoaderMultiAspect(data_root=data_root,
                                              seed=555,
                                              debug_level=0,
                                              batch_size=batch_size,
                                              flip_p=0.0,
                                              test_pct=0,
                                              validate_pct=1)

ed_val_batch = EDValidateBatch()

print(f"batch type: {type(ed_val_batch)}")
i = 0
is_next = True
curr_batch = []
while is_next and i < num_images:
    try:
        example = ed_val_batch[i]
        if example is not None:
            #print(f"example type: {type(example)}") # dict
            #print(f"example keys: {example.keys()}") # dict_keys(['image', 'caption'])
            #print(f"example image type: {type(example['image'])}") # numpy.ndarray
            if i%batch_size == 0:
                curr_batch = example['image'].shape
            img_in_right_batch = curr_batch == example['image'].shape
            print(f"example image shape: {example['image'].shape} {i%batch_size} {img_in_right_batch}") # (256, 256, 3)

            if not img_in_right_batch:
                raise Exception("Current image in wrong batch")
            #print(f"example caption: {example['caption']}") # str
        else:
            is_next = False
        i += 1
    except IndexError:
        is_next = False
        print(f"IndexError: {i}")
        pass
    # for idx, batches in every_dream_batch:
    # print(f"inner example type: {type(batches)}")
    # print(type(batches))
    # print(type(batches[0]))
    # print(dir(batches))
    #h, w = batches.image.size
    #print(f"{idx:05d}-{idx%6:02d}EveryDreamBatch image caption pair: w:{w} h:{h} {batches.caption[1]}")

print(f" *TEST* test cycles: {i}")
print(f" *TEST* EDValidateBatch epoch image length: {len(ed_val_batch)}")
elapsed = time.perf_counter() - s
print(f"{__file__} executed in {elapsed:5.2f} seconds.")