import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch

import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data import dl_singleton
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

## Un-comment this for windows
## os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"ckpt: {ckpt} has {pl_sd['global_step']} steps")
        
    ## sd = pl_sd["state_dict"]
    if "state_dict" in pl_sd:
        print("load_state_dict from state_dict")
        sd = pl_sd["state_dict"]        
    else:
        print("load_state_dict from directly")
        sd = pl_sd
        
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if torch.cuda.is_available():
        model.cuda()
    elif torch.backends.mps.is_available():
        model.to('mps')
    return model

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=False,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )

    parser.add_argument(
        "--datadir_in_name",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Prepend the final directory in the data_root to the output directory name")

    parser.add_argument("--actual_resume", 
        type=str,
        required=False,
        help="Path to model to actually resume from")

    parser.add_argument("--data_root", 
        type=str, 
        required=True, 
        help="Path to directory with training images")

    parser.add_argument("--embedding_manager_ckpt", 
        type=str, 
        default="", 
        help="Initialize embedding manager from a checkpoint")

    parser.add_argument("--init_word",
        type=str, 
        help="Word to use as source for initial token embedding")

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            train.params.batch_size = self.batch_size
            train.params.set = 'train'
            self.dataset_configs["train"] = train
        
        self.train_dataloader = self._train_dataloader
        
        if validation is not None:
            #validation.params.batch_size = self.batch_size
            validation.params.set = 'val'
            print(f" ****** validation: {validation}")
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            #test.params.batch_size = self.batch_size
            test.params.set = 'test'
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            #predict.params.batch_size = self.batch_size
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        #for data_cfg in self.dataset_configs.values():
        #    instantiate_from_config(data_cfg)
        pass

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        dataset = self.datasets["train"]
        return DataLoader(dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=False)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['test'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Keyboard interrupt. Summoning checkpoint.")
            print(f"Steps completed: {trainer.global_step} {trainer.current_epoch}")
            # "{epoch:02d}-{step:05d}"
            ckpt_path = os.path.join(self.ckptdir, f"interrupted_epoch={trainer.current_epoch:02d}-step={trainer.global_step:05d}.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 extra_captions=None, log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.extra_captions = None if extra_captions is None else list(extra_captions)
        self.extra_captions_x_T = None

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            filename = "{}_gs-{:05}_ep-{:02}_batch-{:04}.{}".format(
                k,
                global_step,
                current_epoch,
                batch_idx,
                'txt' if k == 'caption' else 'jpg')
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            if k == 'caption':
                captions_joined = "\n".join(images[k])
                with open(path, 'w') as f:
                    f.write(captions_joined)
                print('saved captions to ', path)
            else:
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                Image.fromarray(grid).save(path)
                print('saved images to ', path)
        print('log_local done')


    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            # images are generated here
            # batch['caption'] contains incoming captions
            all_captions = batch['caption']
            with torch.no_grad():
                # this calls through to LatentDiffusion.DDPM
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
                if self.extra_captions is not None and len(self.extra_captions)>0:
                    batch_size = batch['image'].shape[0]
                    captions_to_generate = self.extra_captions
                    if len(captions_to_generate) > batch_size:
                        print(f" - more extra_captions than batch size, truncating to {batch_size} (out of {self.extra_captions}).")
                        captions_to_generate = captions_to_generate[:batch_size]
                    # append '' to make a full batch
                    if len(captions_to_generate) < batch_size:
                        print(f" - fewer extra_captions than batch size, padding with ''. consider writing more extra_captions, up to {batch_size} in total.")
                        captions_to_generate.append([''] * (batch_size-len(captions_to_generate)))
                    c = pl_module.get_learned_conditioning(captions_to_generate)

                    print("making extra captions internal function")

                    def generate_extra_captions_images(x_T, prefix):
                        print("generating extra captions for prefix", prefix)
                        extra_images = pl_module.log_images_direct(z=x_T,
                                                                   c=c,
                                                                   N=batch_size,
                                                                   z_is_premade_x_T=True,
                                                                   sample_includes_unscaled=False,
                                                                   **self.log_images_kwargs)
                        for k,v in extra_images.items():
                            images[prefix + k] = extra_images[k]

                    # make a fixed set of random seed images the first time, then re-use them
                    if self.extra_captions_x_T is None or self.extra_captions_x_T.shape[0] != batch_size:
                        print("making fixed seed x_T")
                        x_shape = [batch_size, 4, 64, 64] # 512x512 samples
                        self.extra_captions_x_T = torch.randn(x_shape, device='cpu')
                    x_T_fixed = self.extra_captions_x_T.detach().clone().to(pl_module.device)
                    generate_extra_captions_images(x_T_fixed, 'extra_captions_fixedseed_')
                    print("making random x_T")
                    x_T_random = torch.randn_like(x_T_fixed)
                    generate_extra_captions_images(x_T_random, 'extra_captions_random_')
                    print("all extra caption images generated")
                    all_captions.extend(captions_to_generate)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            # logger_log_images apparently doesn't like extra keys in images dict
            local_images = images.copy()
            local_images['caption'] = all_captions
            print('about to log_local')
            self.log_local(pl_module.logger.save_dir, split, local_images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)
            print('log_local done')

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            print('about to logger_log_images')
            logger_log_images(pl_module, images, pl_module.global_step, split)
            print('logger_log_images done')

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        pass
        #if not self.disabled and pl_module.global_step > 0:
            #self.log_img(pl_module, batch, batch_idx, split="val")
        #if hasattr(pl_module, 'calibrate_grad_norm'):
            #if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                #self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

class ShuffleCallback(Callback):
    def __int__(self):
        print("instantiated ShuffleCallback")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print("shuffling indices")
        dl_singleton.shared_dataloader.shuffle()

class CUDACallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(trainer.strategy.root_device.index)
            torch.cuda.synchronize(trainer.strategy.root_device.index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.start_time
        max_memory = None
        if torch.cuda.is_available():
            torch.cuda.synchronize(trainer.strategy.root_device.index)
            max_memory = torch.cuda.max_memory_allocated(trainer.strategy.root_device.index) / 2 ** 20

        try:
            epoch_time = trainer.strategy.reduce(epoch_time)
            epoch_time_msg =f"Average Epoch time: {epoch_time:.2f} seconds"
            rank_zero_info(epoch_time_msg)
            if max_memory is not None:
                max_memory = trainer.strategy.reduce(max_memory)
                epoch_peak_mem_msg = f"Average Peak memory {max_memory:.2f}MiB"
                rank_zero_info(epoch_peak_mem_msg)

        except AttributeError:
            pass

class ModeSwapCallback(Callback):

    def __init__(self, swap_step=2000):
        super().__init__()
        self.is_frozen = False
        self.swap_step = swap_step

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.global_step < self.swap_step and not self.is_frozen:
            self.is_frozen = True
            trainer.optimizers = [pl_module.configure_opt_embedding()]

        if trainer.global_step > self.swap_step and self.is_frozen:
            self.is_frozen = False
            trainer.optimizers = [pl_module.configure_opt_model()]

if __name__ == "__main__":
    
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        print(f"** Resume: overwriting actual_resume {opt.actual_resume} with {ckpt}, logdir {logdir}")
        opt.actual_resume = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        print(f"** Resume: Will pull config from {base_configs}")
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""

        if opt.datadir_in_name:
            now = os.path.basename(os.path.normpath(opt.data_root)) + now
            
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())

        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        model = None
        if opt.actual_resume:
            model = load_model_from_config(config, opt.actual_resume)
        else:
            model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        #modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        #specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:03}-{global_step:05}",
                "verbose": True,
            }
        }

        if hasattr(config.model.params, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = config.model.params.monitor
            #default_modelckpt_cfg["params"]["save_top_k"] = 3 #moved to yaml

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 500,
                    "max_images": 8,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
            "shuffle_callback": {
                "target": "main.ShuffleCallback"
            }
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        # data
        config.data.params.train.params.data_root = opt.data_root
        if config.data.params.validation.params is None:
            config.data.params.validation.params = {}
        if config.data.params.test.params is None:
            config.data.params.test.params = {}

        data = instantiate_from_config(config.data)

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                last_ckpt_name = "last.ckpt"
                print(f"Training halted. Summoning checkpoint as {last_ckpt_name}")
                ckpt_path = os.path.join(ckptdir, last_ckpt_name)
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal


        # Changed to work with windows
        signal.signal(signal.SIGTERM, melk)
        #signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGTERM, divein)
        #signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            print('testing..')
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer is not None and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print("Training complete. max_steps or max_epochs reached, or we blew up.")
            print(trainer.profiler.summary())
