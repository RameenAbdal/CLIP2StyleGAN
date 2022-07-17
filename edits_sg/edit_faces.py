import argparse
import math
import random
import os

import numpy as np
import torch

from torchvision import transforms, utils
from tqdm import tqdm

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def edit(g_ema):

    with torch.no_grad():
                    
                    g_ema.eval()
                    mean_latent = g_ema.mean_latent(4096)
    
                    if args.edit_type == 'to_male':
                       
                        latent_ = torch.tensor(np.load('latents_/gender.npy')).cuda().float()
                        base_latent = torch.tensor(np.load('data_latents/women.npy').reshape(-1,1, 18, 512)).cuda().float() *0.7
                        scale = 1500
                    elif args.edit_type == 'to_female':
                        
                        latent_ = torch.tensor(np.load('latents_/gender.npy')).cuda().float()
                        base_latent = torch.tensor(np.load('data_latents/men.npy').reshape(-1,1, 18, 512)).cuda().float() *0.7
                        scale = -1500
                       

                    elif args.edit_type == 'glasses':
                        latent_ = torch.tensor(np.load('latents_/glass_new.npy')).cuda().float()
                        base_latent = torch.tensor(np.load('data_latents/women.npy').reshape(-1,1, 18, 512)).cuda().float() *0.7  
                        scale = -2500
                       

                    elif args.edit_type == 'smile':
                        latent_ = torch.tensor(np.load('latents_/smile_only.npy')).cuda().float()
                        base_latent = torch.tensor(np.load('data_latents/women.npy').reshape(-1,1, 18, 512)).cuda().float() *0.7
                        scale = -500
                       

                    elif args.edit_type == 'kids':
                        latent_ = torch.tensor(np.load('latents_/kids_text.npy')).cuda().float()
                        base_latent = torch.tensor(np.load('data_latents/women.npy').reshape(-1,1, 18, 512)).cuda().float() *0.7
                        scale = 900

                    elif args.edit_type == 'beard':
                        latent_ = torch.tensor(np.load('latents_/beard_glass.npy')).cuda().float()
                        base_latent = torch.tensor(np.load('data_latents/men.npy').reshape(-1,1, 18, 512)).cuda().float() *0.7
                        scale = - 500


                    pbar = tqdm(range(len(base_latent))[:10])

                    for ll in pbar:
                        for ii in range(5):

                            cal_latent = base_latent[ll] + scale* (ii/5)*latent_

                            if args.edit_type == 'to_male' or args.edit_type == 'to_female' or args.edit_type == 'kids' or args.edit_type == 'beard':
                               
                                cal_latent[0, 8:, :] =  base_latent[ll ][0,8:, :]

                            elif args.edit_type == 'glasses' :
                                cal_latent[0, 3:, :] =  base_latent[ll ][0,3:, :]

                            elif args.edit_type == 'smile' :
                                cal_latent[0, :3, :] =  base_latent[ll ][0,:3, :]
                                cal_latent[0, 6:, :] =  base_latent[ll ][0,6:, :]

                          
                            if ii == 0:
                                ori , _ = g_ema([base_latent[ll]], truncation=1.0,truncation_latent=mean_latent, input_is_latent=True, return_latents = True)
                                samplex, _ = g_ema([cal_latent], truncation=1.0,truncation_latent=mean_latent, input_is_latent=True, return_latents = True)
                            else:
                                sample, _ = g_ema([cal_latent], truncation=1.0,truncation_latent=mean_latent, input_is_latent=True, return_latents = True)
                                samplex = torch.cat((samplex, sample), dim = 0)
                           
                    
                        utils.save_image(
                                samplex,
                                 f"edited_results/{str(ll).zfill(3)}.png",
                                 nrow=int(10),
                                 normalize=True,
                                 range=(-1, 1),)
                

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 generator")
    parser.add_argument("--edit_type", type=str, help="edit type")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--size", type=int, default=1024, help="image sizes for the model"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default='faces.pt',
        help="path to the checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )

    args = parser.parse_args()


    args.latent = 512
    args.n_mlp = 8

    if args.arch == 'stylegan2':
        from model import Generator, Discriminator

    elif args.arch == 'swagan':
        from swagan import Generator, Discriminator

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)


    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

    
        g_ema.load_state_dict(ckpt["g_ema"])

    

    edit(g_ema)
