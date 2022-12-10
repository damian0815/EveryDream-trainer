import os
import torch
import argparse
import glob

def prune_it(p, full_precision=False, keep_only_ema=False):
    print(f"prunin' in path: {p}")
    size_initial = os.path.getsize(p)
    nsd = dict()
    sd = torch.load(p, map_location="cpu")
    print(sd.keys())
    for k in sd.keys():
        if k != "optimizer_states":
            nsd[k] = sd[k]
    else:
        print(f"removing optimizer states for path {p}")
    if "global_step" in sd:
        print(f"This is global step {sd['global_step']}.")
    if keep_only_ema:
        sd = nsd["state_dict"].copy()
        # infer ema keys
        ema_keys = {k: "model_ema." + k[6:].replace(".", ".") for k in sd.keys() if k.startswith("model.")}
        new_sd = dict()

        for k in sd:
            if k in ema_keys:
                if full_precision:
                    new_sd[k] = sd[ema_keys[k]]
                else:
                    new_sd[k] = sd[ema_keys[k]].half()
        new_sd = dict()
        for k in sd:
            if full_precision:
                new_sd[k] = sd[k]
            else:
                new_sd[k] = sd[k].half()
        nsd['state_dict'] = new_sd

    fn = f"{os.path.splitext(p)[0]}-pruned.ckpt" if not keep_only_ema else f"{os.path.splitext(p)[0]}-ema-pruned.ckpt"
    print(f"saving pruned checkpoint at: {fn}")
            elif not k.startswith("model_ema.") or k in ["model_ema.num_updates", "model_ema.decay"]:
                if full_precision:
                    new_sd[k] = sd[k]
                else:
                    new_sd[k] = sd[k].half()

        assert len(new_sd) == len(sd) - len(ema_keys)
        nsd["state_dict"] = new_sd
    else:
        sd = nsd['state_dict'].copy()
    torch.save(nsd, fn)
    newsize = os.path.getsize(fn)
    MSG = f"New ckpt size: {newsize*1e-9:.2f} GB. " + \
          f"Saved {(size_initial - newsize)*1e-9:.2f} GB by removing optimizer states"
    if keep_only_ema:
        MSG += " and non-EMA weights"
    print(MSG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pruning')
    parser.add_argument('--ckpt', type=str, default=False, help='path to model ckpt')
    parser.add_argument("--full", action="store_true", help="Whether or not to save the model in full precision.")
    args = parser.parse_args()
    ckpt = args.ckpt
    full_precision = args.full
    prune_it(ckpt, full_precision)
