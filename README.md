# Every Dream trainer for Stable Diffusion

This is a bit of a divergence from other fine tuning methods out there for Stable Diffusion.  This is a general purpose fine-tuning codebase meant to bridge the gap from small scales (ex Texual Inversion, Dreambooth) and large scale (i.e. full fine tuning on large clusters of GPUs).  It is designed to run on a local 24GB Nvidia GPU, currently the 3090, 3090 Ti, 4090, or other various Quadrios and datacenter cards (A5500, A100, etc), or on Runpod with any of those GPUs. 

Please join us on Discord! https://discord.gg/uheqxU6sXN

If you find this tool useful, please consider subscribing to the project on [Patreon](https://www.patreon.com/everydream) or buy me a [Ko-fi](https://ko-fi.com/everydream). The tools are open source and free, but it is a lot of work to maintain and develop and donations will allow me to expand capabilties and spend more time on the project.

## Main features

* **Supervised Learning** - Caption support reads the filename (or if present a .txt file) for each image as opposed to just token/class of dream booth implementations.  This also means you can train multiple subjects, multiple art styles, or whatever multiple-anything-you-want in one training session into one model, including the context around your characters, like their clothing, background, cityscapes, or the common artstyle shared across them.  
* **Multiple Aspect Ratios** - Supports everything from 1:1 (square) to 4:1 (super tall) or 1:4 (super wide) all at the same time with no fuss.
* **Auto-Scaling** - Automatically resizes the image to the aspect ratios of the model.  No need to crop or resize images.  Just throw them in and let the code do the work. 
* **Recursive load** - Loads all images in a folder and subfolders so you can organize your data set however you like. 
* **[Runpod notebook](doc/RUNPOD.MD)** - Run on a 24GB+ GPU on Runpod.
* **[Google Colab](https://colab.research.google.com/github/victorchall/EveryDream-trainer/blob/main/Train-Runpod.ipynb)** - Currently requires A100, you can bump up batch size 
* **[Micro mode](doc/MICROMODELS.MD)** - Skip perservation and train a smaller model fast.

## Onward to Every Dream

This trainer is focused on enabling fine tuning with new training data plus weaving in original, ground truth images scraped from the web via Laion dataset or other publically available ML image sets.  Compared to DreamBooth, concepts such as regularization have been removed in favor of support for adding back ground truth data (ex. Laion), and token/class concepts are removed and replaced by per-image captioning for training, more or less equal to how Stable Diffusion was trained itself. This is a shift back to the original training code and methodology for fine tuning for general cases.

To get the most out of this trainer, you will need to curate your data with captions.  Luckily, there are additional tools below to help enable that, and will grow over time.

Check out the tools repo here: [Every Dream Tools](https://github.com/victorchall/everydream#tools) for automated captioning and Laion web scraper tools so you can use real images for model preservation if you wish to step beyond micro models. 

## Installation

You will need Anaconda or Miniconda to run locally on your own GPU.

1. Clone the repo:  `git clone https://www.github.com/victorchall/everydream-trainer.git`
2. Create a new conda environment with the provided environment.yaml file: `conda env create -f environment.yaml`
3. Activate the environment: `conda activate everydream`

*Please note other repos are using older versions of some packages like torch, torchvision, and transformers that are known to be less VRAM efficient and cause problems.  Please make a new conda environment for this repo and use the provided environment.yaml file.  I will be updating packages as work progresses as well.  Watch #change-log in the discord.*

### Docker option

Entmike has created a dockerfile for EveryDream tools and trainer available here: https://github.com/entmike/docker-images/tree/main/everydream

## Techniques

This is a *general purpose fine tuning app*.  You can train large or small scale with it and everything in between.

Check out [MICROMODELS.MD](./doc/MICROMODELS.MD) for a quickstart guide and example for quick model creation with a small data set.  It is suited for training one or two subects with 20-50 images each with no preservation in 10-30 minutes depending on your content.

Or [README-FF7R.MD](./doc/README-FF7R.MD) for an example of large scale training of many characters with model preservation trained on 1000s of images with 7 characters and many citscapes from the video game Final Fantasy 7 Remake.

You can scale up or down from there.  The code is designed to be flexible by adjusting the yamls.  If you need help, join the discord for advice on your project.  Many people are working on exciting large scale fine tuning projects with hundreds or *thousands* of images.  You can do it too!

## Tracking progress

Logs are in the /logs folder along with your test image samples.

You can also watch your training progress through Tensorboard.  You'll need to launch a second terminal and activate the conda environment again, then run the following command.  It will be available at http://localhost:6006/ or http://localhost:6006/ if you are running locally (URL will be in the terminal output).

    (everydream) R:\everydream-trainer>tensorboard --logdir logs


## Image Captioning

This trainer is built to use the filenames of your images as "captions" on a per-image basis, or reads a .txt file that is in the same folder with the same filename, *so the entire Stable Diffusion model can be trained effectively.*  **Image captioning is a big step forward.** I strongly suggest you use the tools repo to caption your images.  This will help it learn more effectively and mix concepts (styles, characters, cityscapes and more) more freely. 

[More detailed info on captioning](doc/CAPTIONING.MD)

## Data prep and cropping

With the multiple-aspect ratio support, it is important to follow cropping guidelines. Please read here for advice: 

[More detailed info on cropping](./doc/CROPPING.MD)

### Formatting

The filenames are using for captioning, with a split on underscore so you can have "duplicate" captioned images.  Examples of valid filenames:

    a photo of John Jacob Jingleheimerschmidt riding a bicycle.webp
    a pencil drawing of john jacob jingleheimerscmidt.jpg
    john jacob jingleheimerschmidt sitting on a bench in a park with trees in the background_(1).png
    john jacob jingleheimerschmidt sitting on a bench in a park with trees in the background_(2).png

In the 3rd and 4th example above, the _(1) and _(2) are ignored and not considered by the trainer.  This is useful if you end up with duplicate filenames but different image contents for whatever reason, but that is generally a rare case.  

The trainer will also look for a .txt file in the same folder with the same filename as the image.  *If it finds one, it will use that instead of the filename.*  You can mix and match if you want to use filenames or .txt files, it will prefer the .txt file and fall back to the image filename if no .txt is present

    1234myphoto.webp
    1234myphoto.txt
    a pencil drawing of john jacob jingleheimerscmidt.jpg
    big_john.png
    big_john.txt
    random.txt

In the above example, "1234myphoto.txt" could contain "John Jacob Jingleheimerschmidt riding a bicycle" and it will apply that caption to 1234myphoto.webp, and "big_john.txt" could contain "big john mcarthy in a black shirt wearing black gloves standing in the octagon".
  
Since no .txt file is present for "a pencil drawing of john jacob jingleheimerscmidt.jpg", it will use the filename as the caption which would be "a pencil drawing of john jacob jingleheimerscmidt".

random.txt does not have a matching image, so it will be ignored.

### Data set organization

You can place all your images in some sort of "root" training folder and the traniner will recurvisely locate and find them all from any number of subfolders and add them to the queue for training.

You may wish to organize with subfolders so you can adjust your training data mix, something like this:

    /training_samples/MyProject
    /training_samples/MyProject/man
    /training_samples/MyProject/man_laion
    /training_samples/MyProject/man_nvflickr
    /training_samples/MyProject/paintings_laion
    /training_samples/MyProject/drawings_laion

In the above example, "training_samples/MyProject" will be the "--data_root" folder for the command line.  

As you build your data set, you may find it is easiest to organize in this way to track your balance between new training data and ground truth used to preserve the model integrity.  For instance, if you have 500 new training images in "training_samples/MyProject/man" you may with to use 300  in the "man_laion" and another 200 in "/"man_nvflickr".  You can then experiment by removing different folders to see the effects on training quality and model preservation. 

You can also organize subfolders for each character if you wish to train many characters so you can add and remove them, and easily track that you are balancing the number of images for each.

If you are training multiple subjects, it is best to balance the amount of training data for each.  Subjects should have an even mix per subject.  Some styles will take at the same time as subjects with fewer training images of them. 

## Ground truth data sources and data engineering for larger scale training

Visit [EveryDream Data Engineering Tools](https://github.com/victorchall/EveryDream) to find a **web scraper** that can pull down images from the Laion dataset along with an **Auto Caption** script to prepare your data.  You should consider that your first step before using this trainer if you wish to train a significant number of characters and if you wish to keep them or the general shared style of your subjects or art styles from bleeding into the rest of the model. 

The more data you add from ground truth data sets such as Laion, the more training you will get away with without "damaging" the original model.  The wider variety of data in the ground truth portion of your dataset, the less likely your training images are to "bleed" into the rest of your model, losing qualities like the ability to generate images of other styles you are not training.  This is about knowledge retention in the model by refeeding it the same data it was originally trained on.  This is a big part of the reason why the original training code on Stable Diffusion was so effective.  It was able to train on a wide variety of data and manages to understand possibly millions of concepts and mix them. 

If you don't care to preserve the model you can skip this and train only on your new data.  For a single subject, aka "fast" or "micro" mode, you can usually get away with putting one character or artstyle in without ruining the model you create. 

## Starting training

An example comand to start training: **make sure you activate the conda environment first** 

    conda activate everydream

    python main.py --base configs/stable-diffusion/v1-finetune_everydream.yaml -t --actual_resume sd_v1-5_vae.ckpt -n MyProjectName --data_root training_samples\MyProject

In the above, the source training data is expected to be laid out in subfolders of training_samples\MyProject as described in above sections. It will resume from the checkpoint named "sd_v1-5_vae.ckpt" but you can change this to most Stable Diffusion checkpoints (ex. 1.4, 1.5, 1.5 + new vae, WD, or others that people have shared online). Inpainting model is not yet supported.  "-n MyProjectName" is merely a name for the folder where logs will be written during training, which appear under /logs. 

## Managing training runs

Each project is different, but consider carefully reading below to adjust your YAML file that configures your training run.  You can make your own copies of the YAML files for differenet projects then use --config to change which one you use.  I will tend to update the YAMLs in future releases so making your own copy also avoids a collision when you "git pull" a new version.  
## Testing

I strongly recommend attempting to undertrain via the repeats and instead tend to set max_epoch higher *compared to typical dream booth recommendations* so you will get a few different ckpts along the course of your training session.  The ckpt files will be dumped to a folder such as "_\logs\MyPrject2022-10-25T20-37-40_MyProject_" date stamped to the start of training. There are also test images in the _\logs\images\train_ folder that spit out periodically based on another finetune yaml setting.

The images will often not all be fully formed, and are randomly selected based on the last few training images, but it's a good idea to watch those images and learn to understand how they look compared to when you go try your new model out in a normal Stable Diffusion inference repo. 

If you are close, consider lowering repeats!
## Finetune yaml adjustments

The finetune yamls are your best friend.

Depending on your project, a few settings may be useful to tweak or adjust.  In [Starting Training](#starting_training) I'm using __v1-finetune_everydream.yaml__ here but you can make your own copies if you like with different adjustments and save them for your projects.  It is a good idea to get familar with this file as tweaking can be useful as you train.

I'll highlight the following settings at the end of the file: 

    trainer:
      benchmark: True
      max_epochs: 4
      max_steps: 99000

"max_epochs" will halt training.  I suggest ending on a clean end of an epoch rather than using a steps limit, so defaults are configured as such.  3-5 epochs will give you a few copies to try.  If you are unsure how many epochs to run, setting a higher value and lower repeats below will give you more ckpt files to test after training concludes.  You can always [continue training](#resuming_training) if needed.

      train:
        target: ldm.data.every_dream.EveryDreamBatch
        params:
            repeats: 20
            debug_level: 1

Above, the "repeats" defines the number of times each training image is trained on per epoch.  For large scale training with 500+ images per subject you may find just 10-15 repeats with 3-4 epochs.  As you add more and more data you can slowly use lower repeat values.  For very small training sets, try the micro YAML that has higher repeats (40-60) with a few epochs.

debug_level: 1 will show in the console when you have multiple aspect ratio images that are dropped because they cannot be fit in.  

You are also free to move data in and out of your training_samples/MyProject folder between training sessions.  If you have multiple subjects and your number of images between them is a bit mismatched in number, say, 100 for one and only 60 for another, you can try running one epoch 25 repeats, then remove the character with 100 images and train just the one with the 60 images for another epoch at 5 repeats.  It's best to try to keep the data evenly spread, but sometimes that is diffcult.  You may also find certain characters are harder to train, and need more on their own.  Again, test!  Go generate images between 

    data:
      target: main.DataModuleFromConfig
      params:
        batch_size: 6

Batch size determine how many images are loaded and trained on in parallel. batch_size 6 will work on a 24GB GPU, 1 will only reduce VRAM use to about 19.5GB.  The batch size will divide the number of steps used as well, but one epoch is still "repeats" number of trainings on each image.  Higher batch sizes are desired to give better generalization as the gradient is calculated across the entire batch.  More images in a batch will also decrease training time by keep your GPU utilization higher.

I recommend not worrying about step count so much. Focus on epochs and repeats instead.  Steps are a result of the number of training images you have.

    callbacks:
      image_logger:
        target: main.ImageLogger
        params:
          batch_frequency: 250

Image logger batch frequency determines how often a test image is placed into the logs folder.  150-300 is recommended.  Lower values produce more images but slow training down a bit. 

    modelcheckpoint:
      params:
        every_n_epochs: 1  # produce a ckpt every epoch, leave 1!
        save_top_k: 4   # save the best N ckpts according to loss, can reduce to save disk space but suggest at LEAST 2, more if you have max_epochs below higher!


"every_n_epochs" will make the trainer create a ckpt file at the end of every epoch.  I do not recommend changing this.  If you want checkpoints less frequently, increase your repeats instead.  "save_top_k" will save the "best" N ckpts based on a loss value the trainer is tracking.  If you are training 10 epochs and use save_top_k 4, it will only save the "best" 4, saving some disk space.  *It's possible the last few epochs may not save because they are getting worse over time according to the loss value the trainer calculates as it goes.*  If you want all the ckpts to always be saved you can set save_top_k to 99 or any value over max_epochs

    validation:
      target: ldm.data.ed_validate.EDValidateBatch
      params:
        repeats: 0.4

Repeats for validation adjusts how much of the training set is used for validation.  I've added support to reduce this to a decimal value.  For large training where you only use 5-15 repeats, setting this lower speeds up training but stills allows the trainer to run validation to make sure nothing has broken along the way wasting future compute time if something goes wrong.  You can generally leave this untouched.

## Resuming training

If you find even your best or last ckpt from a training run seems "undertrained" you can cut and paste a trained ckpt from your logs into the root folder and resume by running the trainer again and chnage the --ckpt to point to your file.

    python main.py --base configs/stable-diffusion/v1-finetune_everydream.yaml -t  --actual_resume epoch=03-step=01437.ckpt -n MyProjectName --data_root training_samples\MyProject

or

    python main.py --base configs/stable-diffusion/v1-finetune_everydream.yaml -t  --actual_resume last.ckpt -n MyProjectName --data_root training_samples\MyProject

Note above the "epoch=03-step=01437.ckpt" or "last.ckpt" instead of "sd-v1-4-pruned.ckpt".  The full 11GB ckpt file contains the ema weights, non-ema weights, and optimizer state so resuming will have the full trainer state.

## Pruning

To prune your file down from 11GB to 2GB file use:

    python prune_ckpt.py --ckpt last.ckpt

(where last.ckpt is whatever your trained filename is).  This will remove training state and nonema weights and save a new file called "last-pruned.ckpt" in the root folder and leave the last.ckpt in place in case you need to resume.  

I do not suggest using a pruned 2GB file to resume later training.  If you want to resume training, use the full 11GB file.  You can move your 2GB file to whatever your favorite Stable Diffusion webui is, test it out, and delete all the 11GB files and your log folder once you are satisfied with the results.

### Autoprune All

To prune all the ckpts at once from your logs folders, use this:

    python scripts/autoprune_all.py

or
  
    python scripts/autoprune_all.py --delete

This will sweep all the ckpts in your logs folder, copy them all down to the root trainer folder, and prune them down to 2GB files.  If you use the --delete flag, it will delete the original 11GB files. 

### Additional notes

Thanks go to the CompVis team for the original training code, Xaiver Xiao for the DreamBooth implementation and tweaking of trainer configs to stuff it into a 24GB card, and Kane Wallmann for the first implementation of image caption from the filenames.

References:

[Compvis Stable Diffusion](https://github.com/CompVis/stable-diffusion)

[Xaiver Xiao's DreamBooth implementation](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion)

[Kane Wallmann](https://github.com/kanewallmann/Dreambooth-Stable-Diffusion)

# Troubleshooting

**Cuda out of memory:**  You should have <600MB used before starting training to use batch size 6.  People have reported issues with 

* Precision X1 running in the background 
* Microsoft's system tray weather widget 
* Using the conda environment of another repo that uses older package versions

You can disable hardware acceleration in apps like Discord and VS Code to reduce VRAM use, and close as many Chrome tabs as you can bear.  While using a batch_size of 1 only uses about 19.5GB it will have a significant impact on training speed and quality.

