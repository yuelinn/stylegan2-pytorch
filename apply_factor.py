import argparse

import torch
from torchvision import utils

from model import Generator
import numpy as np
import pdb


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=7, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )
    parser.add_argument(
        "factor",
        type=str,
        help="name of the closed form factorization result factor file",
    )

    args = parser.parse_args()

    eigvec = torch.load(args.factor)["eigvec"].to(args.device)
    ckpt = torch.load(args.ckpt)
    g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    trunc = g.mean_latent(4096)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)

    # direction = args.degree * eigvec[:, args.index].unsqueeze(0)
    
    #dir_arr = np.arange(-2.5, 0.25, 0.25)
    #np.set_printoptions(precision=3)

    dir_arr=[0.0, -3., -2., -1., 0, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]
    # dir_arr = [0.0, -1.0 -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    # dir_arr = [0.0, -5.0 -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    print(dir_arr)
    img_arr=[]
    pdb.set_trace()

    for degree in dir_arr:
        direction = degree * eigvec[:, args.index].unsqueeze(0)
        # pdb.set_trace()
        img, _ = g(
            [latent+ direction],
            truncation=args.truncation,
            truncation_latent=trunc,
            input_is_latent=True,
        )
        img_arr.append(img)

    grid = utils.save_image(
        torch.cat(img_arr, 0),
        f"{args.out_prefix}_index-{args.index}_degree-{args.degree}.png",
        normalize=True,
        range=(-1, 1),
        nrow=args.n_sample,
    )
