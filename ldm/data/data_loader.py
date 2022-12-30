import os
from PIL import Image
import PIL
import random
from ldm.data.image_train_item import ImageTrainItem
import ldm.data.aspects as aspects
from tqdm import tqdm

PIL.Image.MAX_IMAGE_PIXELS = 933120000
        
class DataLoaderMultiAspect():
    """
    Data loader for multi-aspect-ratio training and bucketing

    data_root: root folder of training data
    batch_size: number of images per batch
    flip_p: probability of flipping image horizontally (i.e. 0-0.5)
    """
    def __init__(self,
                 data_root,
                 seed=555,
                 debug_level=0,
                 batch_size=1,
                 flip_p=0.0,
                 resolution=512,
                 test_pct=0.15,
                 validate_pct=0.15):
        self.image_paths = []
        self.debug_level = debug_level
        self.flip_p = flip_p

        self.aspects = aspects.get_aspect_buckets(resolution)
        print(f"* DLMA resolution {resolution}, buckets: {self.aspects}")
        print(" Preloading images...")

        self.__recurse_data_root(self=self, recurse_root=data_root)
        random.Random(seed).shuffle(self.image_paths)
        prepared_train_data = self.__prescan_images(debug_level, self.image_paths, flip_p) # ImageTrainItem[]
        self.image_caption_pairs = self.__bucketize_images(prepared_train_data, batch_size=batch_size, debug_level=debug_level)

        # automatically split all image indices (which are already shuffled) into test/train indices
        image_count = len(self.image_caption_pairs)
        train_split_pos = int((image_count * (1.0 - (test_pct + validate_pct))) // batch_size) * batch_size
        if train_split_pos <= 0:
            raise ValueError(f"test_pct {test_pct} and validation_pct {validate_pct} are too high for image count {image_count}")
        test_split_pos = int((image_count * (1.0 - (validate_pct))) // batch_size) * batch_size

        self.train_indices = list(range(train_split_pos))
        self.test_indices = list(range(train_split_pos,test_split_pos))
        self.validation_indices = list(range(test_split_pos,image_count))
        if len(self.test_indices) == 0:
            raise ValueError(f"test_pct {test_pct} results in a test split index {test_split_pos} that is the same as train split index {train_split_pos}")
        if len(self.validation_indices) == 0:
            raise ValueError(f"test_pct {test_pct} results in a test split index {test_split_pos} that leaves no images left for validation")
        assert((len(self.train_indices) % batch_size) == 0)
        assert((len(self.test_indices) % batch_size) == 0)
        assert((len(self.validation_indices) % batch_size) == 0)

        if debug_level > 0: print(f" * DLMA Example: {self.image_caption_pairs[0]} images")
        

    def get_train_images(self):
        return [self.image_caption_pairs[i] for i in self.train_indices]

    def get_test_images(self):
        return [self.image_caption_pairs[i] for i in self.test_indices]

    def get_validation_images(self) -> [ImageTrainItem]:
        return [self.image_caption_pairs[i] for i in self.validation_indices]

    def get_all_images(self):
        return self.image_caption_pairs

    @staticmethod
    def __read_caption_from_file(file_path, fallback_caption):
        caption = fallback_caption
        try:
            with open(file_path, encoding='utf-8', mode='r') as caption_file:
                caption = caption_file.read()
        except:
            print(f" *** Error reading {file_path} to get caption, falling back to filename")
            caption = fallback_caption
            pass
        return caption

    def __prescan_images(self, debug_level: int, image_paths: list, flip_p=0.0):
        """
        Create ImageTrainItem objects with metadata for hydration later 
        """
        decorated_image_train_items = []

        for pathname in tqdm(image_paths):
            caption_from_filename = os.path.splitext(os.path.basename(pathname))[0].split("_")[0]

            txt_file_path = os.path.splitext(pathname)[0] + ".txt"
            caption_file_path = os.path.splitext(pathname)[0] + ".caption"

            if os.path.exists(txt_file_path):
                caption = self.__read_caption_from_file(txt_file_path, caption_from_filename)                
            elif os.path.exists(caption_file_path):
                caption = self.__read_caption_from_file(caption_file_path, caption_from_filename)                
            else:
                caption = caption_from_filename

            #if debug_level > 1: print(f" * DLMA file: {pathname} with caption: {caption}")
            
            image = Image.open(pathname)
            width, height = image.size
            image_aspect = width / height

            target_wh = min(self.aspects, key=lambda aspects:abs(aspects[0]/aspects[1] - image_aspect))

            image_train_item = ImageTrainItem(image=None, caption=caption, target_wh=target_wh, pathname=pathname, flip_p=flip_p)

            decorated_image_train_items.append(image_train_item)

        return decorated_image_train_items

    @staticmethod
    def __bucketize_images(prepared_train_data: list, batch_size=1, debug_level=0):
        """
        Put images into buckets based on aspect ratio with batch_size*n images per bucket, discards remainder
        """
        # TODO: this is not terribly efficient but at least linear time
        buckets = {}

        for image_caption_pair in prepared_train_data:
            target_wh = image_caption_pair.target_wh

            if (target_wh[0],target_wh[1]) not in buckets:
                buckets[(target_wh[0],target_wh[1])] = []
            buckets[(target_wh[0],target_wh[1])].append(image_caption_pair) 
        
        print(f" ** Number of buckets: {len(buckets)}")

        if len(buckets) > 1: 
            for bucket in buckets:
                truncate_count = len(buckets[bucket]) % batch_size
                current_bucket_size = len(buckets[bucket])
                buckets[bucket] = buckets[bucket][:current_bucket_size - truncate_count]

                if debug_level > 0:
                    print(f"  ** Bucket {bucket} with {current_bucket_size} will drop {truncate_count} images due to batch size {batch_size}")

        # flatten the buckets
        image_caption_pairs = []
        for bucket in buckets:
            image_caption_pairs.extend(buckets[bucket])
        
        return image_caption_pairs

    @staticmethod
    def __recurse_data_root(self, recurse_root):
        multiply = 1
        multiply_path = os.path.join(recurse_root, "multiply.txt")
        if os.path.exists(multiply_path):
            try: 
                with open(multiply_path, encoding='utf-8', mode='r') as f:
                    multiply = int(float(f.read().strip()))
                    print(f" * DLMA multiply.txt in {recurse_root} set to {multiply}")
            except:
                print(f" *** Error reading multiply.txt in {recurse_root}, defaulting to 1")
                pass

        for f in os.listdir(recurse_root):
            current = os.path.join(recurse_root, f)

            if os.path.isfile(current):
                ext = os.path.splitext(f)[1]
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif']:
                    # add image multiplyrepeats number of times
                    for _ in range(multiply):
                        self.image_paths.append(current)

        sub_dirs = []        

        for d in os.listdir(recurse_root):
            current = os.path.join(recurse_root, d)
            if os.path.isdir(current):                
                sub_dirs.append(current)

        for dir in sub_dirs:
            self.__recurse_data_root(self=self, recurse_root=dir)
