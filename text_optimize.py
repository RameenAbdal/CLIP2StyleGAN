import torch
import clipmod
from PIL import Image
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import glob
from natsort import natsorted
from torch import nn
import torch.nn.functional as F
import argparse


   
def run_optimizer(model, preprocess):

   image_data = natsorted(glob.glob(args.path))
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
      
   

   model.eval()

   # parameters have require grad True
   for params in model.parameters():
      params.requires_grad = True 

   # Normalize the image features
   image_features_pred = image_features_pred / image_features_pred.norm(dim=-1, keepdim = True)

   pbar = tqdm(range(1))

   for ii in pbar :
   # Intialize the soft variable
      init_txt = torch.zeros((model.token_embedding.weight.shape[0])).cuda()

      if ii > 0:
         init_txt[other[0][indices[0]]] = init_txt[other[0][indices[0]]] + 1
         positive_text_features =  text_encodings[indices[0]]

      # keep the require grad true
      optim_var = Variable(init_txt, requires_grad = True)
   

      # initialize the optimizer
      optZ = torch.optim.Adam([optim_var], lr=args.lr)

      # start the loop to optimize the text given the embeddings
   
      for kk in range(args.iter):

         output_text , loss , entropy,  index, values , other = model.optimize_text(optim_var)
         
         output_text =  output_text/ output_text.norm(dim=-1, keepdim=True)
         
      
         loss_=   0.0001* entropy 
         
         loss_final_embedding =  -3* (output_text @ image_features_pred.T) + loss_
         optZ.zero_grad()
         loss_final_embedding.backward()
         
         optZ.step()
         pbar.set_description(
                           f'loss: {loss_final_embedding.item():.5f}, label_pred: {index.item()}, value: {values.item()}, text: {clipmod.reverse_tokenize([index.item()])}, loss2: {loss_.item()} , entropy: {entropy.item()}')
      


   # Refine the predictions and print the final outputs
      list__ = []

      for i in range(args.top_refine):
         list__.append(clipmod.reverse_tokenize([other[0][i].item()]))


      _text = ['A picture of a {}.']

      ff = 0

      for classname in tqdm(list__):

         if ff == 0:
            text_raw = clipmod.tokenize([_text.format(classname) for _text in _text]).to(device)
         else:
            text_raw = torch.cat((text_raw, clipmod.tokenize([_text.format(classname) for _text in _text]).to(device)), axis = 0) 

         ff += 1

      with torch.no_grad():

         text_encodings = model.encode_text(text_raw)

         image_features_pred /= image_features_pred.norm(dim=-1, keepdim=True)
         text_encodings /= text_encodings.norm(dim=-1, keepdim=True)

         similarity = (100.0 * image_features_pred @ text_encodings.T).softmax(dim=-1)
         print(similarity.shape)
         values, indices = similarity[0].topk(100)
      

      print("Predictions:")
      for ii in range(len(values)):
               
               print(" Label: ", list__[indices[ii].cpu().numpy()], ' Confidence: ', values[ii].item())



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Text Optimizer")

    parser.add_argument("--path", type=str, default = 'PCA_FFHQ/0/*.png', help="path to the PCA dataset")
    parser.add_argument("--iter", type=int, default=150, help="total training iterations")
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    parser.add_argument("--top_refine", type=int, default=5000, help="number of top predictions used for refining")
    parser.add_argument("--preds", type=int, default=100, help="number of predictions")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clipmod.load("ViT-B/32", device=device)

    run_optimizer(model, preprocess)