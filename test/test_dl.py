# script to test data loader by itself
# run from trainer root directory
# Ex:
# python test/test_dl.py --data_root "x:/mytestdata/input" --batch_size 2
import argparse

import ldm.data.data_loader as dl

def main(data_root, batch_size):
    data_loader = dl.DataLoaderMultiAspect(data_root=data_root, batch_size=batch_size, debug_level=1)

    image_caption_pairs = data_loader.get_all_images()

    print(f"Loaded {len(image_caption_pairs)} image-caption pairs")

    print(f"**** Done loading. Loaded {len(image_caption_pairs)} images from data_root: {data_root} ****")

if __name__ == "__main__":
    """
    test the data loader by itself, outputs buckets and image counts
    data_root: root folder of training data
    batch_size: number of images per batch
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="input", help="root folder of training data")
    parser.add_argument("--batch_size", type=int, default=4, help="number of images per batch")
    args = parser.parse_args()
    main(data_root=args.data_root, batch_size=args.batch_size)