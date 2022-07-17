import torch
import clipmod

import glob
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import shutil, os
from natsort import natsorted
from sklearn.decomposition import PCA

import argparse




def generate_pca_images( image_features):

    mean_vector = torch.mean(image_features, dim = 0, keepdim = True)

    clusterable_embedding = PCA(n_components=512, svd_solver='full').fit(image_features.detach().cpu().numpy())
   
    new_data = (image_features.unsqueeze(0) - mean_vector.unsqueeze(1)).squeeze(0).cuda()
    
    base_address = args.save_path

    ii = 0

    for direction in tqdm(clusterable_embedding.components_): 

        indices_ = []
       

        highest_vector = torch.from_numpy(direction).unsqueeze(0).cuda()
    
        scores = F.normalize( new_data) @ highest_vector.t()
        
        values_new, indices_new = scores.squeeze(1).topk(new_data.shape[0] , largest = args.similarity)

        indices_ = indices_new[values_new > 0.00]
        

        original_data = new_data[indices_]
        highest_vector = new_data.squeeze(0)[indices_[0].cpu().numpy()].unsqueeze(0)
        redifined_dot_product = original_data @ highest_vector.t()    
        values_red, indices_red = redifined_dot_product.squeeze(1).topk(len(indices_), largest = args.similarity)


        

        directory = str(ii)
        path = os.path.join(base_address, directory)
        try:
            os.mkdir(path)
        except:
            print('Directory already exists..... overwritting...')

        print("Directory '% s' created" % directory, 'with similarity:', values_new[0].item())

        ii += 1

        jj = 0 
        for xx in range(100):

            shutil.copy(f"{args.image_path}{str(indices_[indices_red[xx]].cpu().numpy()).zfill(5)}.png", path)
            os.rename(path + "/" +str(indices_[indices_red[xx]].cpu().numpy()).zfill(5)+ ".png", path + "/" +str(jj) + ".png")
            jj += 1

    



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="PCA")
    parser.add_argument('image_path', type=str, help = 'path to the datset')
    parser.add_argument("--extracted_features_path", type=str, default='extracted_features/image_features.npy', help = 'path to the extarcted features')
    parser.add_argument("--save_path", type=str, default = 'PCA_images')
    parser.add_argument("--similarity", type=bool, default = True, help="False to sample the negative set")
 
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clipmod.load("ViT-B/32", device=device)
    image_features = torch.tensor(np.load(args.extracted_features_path)).float()


    generate_pca_images( image_features)







