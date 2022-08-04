import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
import ipdb


def generate(args, g_ema, device, mean_latent, i):

    with torch.no_grad():
        g_ema.eval()

        # Linn: create more z's by interpolation
        perturb_delta=0.05  # TODO make an args
        perturb_n = 5
        delta_np = np.linspace(perturb_n*perturb_delta*-1, (perturb_n)*perturb_delta, num=perturb_n*2+1)

        sample_z_pts = np.random.normal(0., 1., (args.sample, args.latent))
        sample_z_np =np.repeat(sample_z_pts, delta_np.shape, axis=0) + np.expand_dims(delta_np, axis=1)

        # ipdb.set_trace()
        # sample_z_np=sample_z_np.reshape(-1, sample_z_np.shape[-1])

        sample_z = torch.from_numpy(sample_z_np).float().to(device)

        sample, _ = g_ema(
            [sample_z], truncation=args.truncation, truncation_latent=mean_latent
        ) # TODO itemize so that batch size is okay

        utils.save_image(
            sample,
            f"{args.out_path}/grid_{i}.png",
            nrow=perturb_n*2+1,
            normalize=True,
            range=(-1, 1),
        )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="generator",
        help="path of output dir",
    )   
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    for i in range(10): #TODO make an args
        generate(args, g_ema, device, mean_latent, i)
