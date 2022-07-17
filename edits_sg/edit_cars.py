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


def edit(gma):
   

    with torch.no_grad():
                mean_latent = g_ema.mean_latent(4096)
                with torch.no_grad():
                    
                    g_ema.eval()
                  

                    if args.edit_type == 'scrap_to_car':
                        latent_ = torch.tensor(np.load('latents_/scrap.npy')).cuda().float()
                        base_latent = torch.tensor(np.load('data_latents/scrap.npy').reshape(-1,1, 16, 512)).cuda().float() *1.5
                        scale = 180

                    elif args.edit_type == 'red_car':
                    
                        latent_ = torch.tensor(np.load('latents_/red_car.npy')).cuda().float()
                        base_latent = torch.tensor(np.load('data_latents/cars.npy').reshape(-1,1, 16, 512)).cuda().float() *1.5
                        scale = -50

                    elif args.edit_type == 'capri':
                        latent_ = torch.tensor(np.load('latents_/capri.npy')).cuda().float()
                        base_latent = torch.tensor(np.load('data_latents/cars.npy').reshape(-1,1, 16, 512)).cuda().float() *1.5
                        scale = -150

                    elif args.edit_type == 'race_car':
                        latent_ = torch.tensor(np.load('latents_/race_car.npy')).cuda().float()
                        base_latent = torch.tensor(np.load('data_latents/cars.npy').reshape(-1,1, 16, 512)).cuda().float() *1.5
                        scale = -150

                    elif args.edit_type == 'blazer':
                        latent_ = torch.tensor(np.load('latents_/blazzer_car.npy')).cuda().float()
                        base_latent = torch.tensor(np.load('data_latents/cars.npy').reshape(-1,1, 16, 512)).cuda().float() *1.5
                        scale = -150


                  
                    pbar = tqdm(range(len(base_latent))[0:100])
                    for ll in pbar:
                        for ii in range(10):
                           
                            cal_latent = base_latent[ll] + scale* (ii/10)*latent_

                            if args.edit_type == 'scrap_to_car' or args.edit_type =='race_car' or args.edit_type =='blazer':
            
                                cal_latent[0, 8:, :] =  base_latent[ll ][0,8:, :]

                            elif args.edit_type == 'red_car':
                                
                                cal_latent[0, :8, :] =  base_latent[ll ][0,:8, :]

                            elif args.edit_type == 'capri':
                                cal_latent[0, 5:, :] =  base_latent[ll ][0,5:, :]


                          
                            if ii == 0:
                                ori , _ = g_ema([base_latent[ll]], truncation=1.0,truncation_latent=mean_latent, input_is_latent=True, return_latents = True, randomize_noise=False)
                                samplex, _ = g_ema([cal_latent], truncation=1.0,truncation_latent=mean_latent, input_is_latent=True, return_latents = True, randomize_noise=False)
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
        "--size", type=int, default=512, help="image sizes for the model"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default='car.pt',
        help="path to the checkpoints to resume training",
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
