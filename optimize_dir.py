import torch
import clipmod
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob
from natsort import natsorted
from sklearn.decomposition import PCA
import argparse
from torch.autograd import Variable

def run_disentanglement(model, preprocess):

    image_features = torch.tensor(np.load(args.extracted_features_path)).float()
    mean_vector = torch.mean(image_features, dim = 0, keepdim = True)
    clusterable_embedding = PCA(n_components=512, svd_solver='full').fit(image_features.detach().cpu().numpy())
    new_data = (image_features.unsqueeze(0) - mean_vector.unsqueeze(1)).squeeze(0).cuda()


    image_data = natsorted(glob.glob(args.path_image_data))

    with torch.no_grad():
        for i in range(len(image_data)):
            if i == 0:
                i_x = preprocess(Image.open(image_data[i])).unsqueeze(0).to(device)
                image_features_pred = model.encode_image(i_x)
            else:
                i_x = preprocess(Image.open(image_data[i])).unsqueeze(0).to(device)
                image_features_pred = torch.cat((image_features_pred, model.encode_image(i_x)),dim = 0) 
        
        image_features_pred =  image_features_pred / image_features_pred.norm(dim=-1, keepdim = True)
        image_features_pred =  image_features_pred.mean(dim = 0,  keepdim = True)



    if args.pca_axis == 0:
        predictions = ['kids', 'smiling']

    elif args.pca_axis == 19:
        predictions = ['A person with a beard', 'A person with glasses']
    else: 
        print('Not an option')
        exit(0)

    text__ = clipmod.tokenize(predictions).cuda()

    with torch.no_grad():
        text_encodings = model.encode_text(text__)

        image_features_pred /= image_features_pred.norm(dim=-1, keepdim=True)
        text_encodings /= text_encodings.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features_pred @ text_encodings.T).softmax(dim=-1)
        values, indices = similarity[0].topk(2)

    print('Similarities before optimization')

    for ii in range(len(values)):
        print(predictions[indices[ii].cpu().numpy()], ' ', values[ii].item())

    V =  torch.tensor(clusterable_embedding.components_[args.pca_axis]).unsqueeze(0).cuda() 

    W = torch.tensor([values[0], values[1]]).unsqueeze(0).float().cuda()

    B = Variable(torch.zeros(2, 512).cuda(), requires_grad = True)

    optZ = torch.optim.Adam([B], lr=args.lr)

    text_encodings = text_encodings.float()

    for ii in range(args.steps):

        X =  1* torch.norm( B@B.T - torch.eye(B.shape[0]).cuda()) - B[0] @ text_encodings[indices[0].cpu().numpy()] - B[1] @ text_encodings[indices[1].cpu().numpy()] + 0.1*torch.norm(V - W@B) #+ B[0] @ text_encodings[indices[1].cpu().numpy()]  + B[1] @ text_encodings[indices[0].cpu().numpy()]

        if (ii % 100 == 0):
            print('Loss:' , X.clone().detach().cpu().numpy())

        optZ.zero_grad()
        X.backward()
        optZ.step()

    with torch.no_grad():
       

        B /= B.norm(dim=-1, keepdim=True)
        similarity = (100.0 * B @ text_encodings.float().T).softmax(dim=-1)
        values, indices = similarity[0].topk(2)

        print('Similarities after optimization')

        for ii in range(len(values)):
            print(predictions[indices[ii].cpu().numpy()], ' ', values[ii].item())


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Disentanglement optimizer")

    parser.add_argument("--path_image_data", type=str, default = "disentanglement_results/0/*.png", help="path to the Entangled PCA data")
    parser.add_argument("--extracted_features_path", type=str, default = "extracted_features/StyleGAN_image_features.npy", help="path to the SG CLIP Image encoded features")
    parser.add_argument("--pca_axis", type=int, default=0, help="PCA axis index : (0, 19) shown in the paper")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--steps", type=int, default=1000, help="number of top predictions used for refining")
 

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clipmod.load("ViT-B/32", device=device)

    run_disentanglement(model, preprocess)