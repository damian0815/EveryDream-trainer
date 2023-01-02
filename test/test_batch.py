# script to test data loader by itself
# run from training root, edit the data_root manually

from  ldm.data.every_dream import EveryDreamBatch
import time

s = time.perf_counter()

#data_root = "r:/everydream-trainer/test/input"
data_root = "../stablediffusion/buzzybee/cropped"
num_images = 11 # count by hand

batch_size = 1
repeats=1
every_dream_batch = EveryDreamBatch(data_root=data_root, flip_p=0.0, debug_level=2, batch_size=batch_size, repeats=repeats, crop_jitter=25, conditional_dropout=0.3, resolution=512)

print(f" *TEST* EveryDreamBatch epoch image length: {len(every_dream_batch)}")
print(f" max test cycles: {int(len(every_dream_batch) / batch_size)}, batch_size: {batch_size}, repeats: {repeats}")
i = 0

while i < 99: # and i < len(every_dream_batch):
    curr_batch = []
    for j in range(i,i+batch_size):
        curr_batch.append(every_dream_batch[j])
    
    # all in batch must have the same image size
    assert all(x == curr_batch[0]['image'].shape for x in [e['image'].shape for e in curr_batch])
    assert all(x[0] > 2 for x in [e['image'].shape for e in curr_batch])

    #print(f"idx: {i}, batch sample: shape: {curr_batch[0]['image'].shape}: {curr_batch[0]['caption']}")

    i += batch_size

print(f" *TEST* test cycles: {i}")
print(f" *TEST* EveryDreamBatch epoch image length: {len(every_dream_batch)}")
elapsed = time.perf_counter() - s
print(f"{__file__} executed in {elapsed:5.2f} seconds.")