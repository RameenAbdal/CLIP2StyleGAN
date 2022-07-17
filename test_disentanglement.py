import torch
import clipmod
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob
from natsort import natsorted
import argparse

def compute_scores():
  # load folders and relevant prompts
  if args.attribute == 'beard_glass':

    enlangled = natsorted(glob.glob("disentanglement_results/19/*.png"))
    disentangled1 = natsorted(glob.glob("disentanglement_results/beard/*.png"))
    disentangled2 = natsorted(glob.glob("disentanglement_results/glasses/*.png"))
    predictions = ['A picture of a person with beard', 'A picture of a person with glasses']

  elif args.attribute == 'kids_smile':

    enlangled = natsorted(glob.glob("disentanglement_results/1/*.png"))
    disentangled1 = natsorted(glob.glob("disentanglement_results/kids/*.png"))
    disentangled2 = natsorted(glob.glob("disentanglement_results/smile/*.png"))

    predictions = [' kids', ' smile']

  else:

    print('Option not supported')
    exit(0)


  # Load entangled images into CLIP embeddings
  with torch.no_grad():
      for i in range(len(enlangled)):
          if i == 0:
            i_x = preprocess(Image.open(enlangled[i])).unsqueeze(0).to(device)
            image_features_predGS = model.encode_image(i_x)
          else:
            i_x = preprocess(Image.open(enlangled[i])).unsqueeze(0).to(device)
            image_features_predGS = torch.cat((image_features_predGS, model.encode_image(i_x)),dim = 0) 
      
      image_features_predGS =  image_features_predGS / image_features_predGS.norm(dim=-1, keepdim = True)
      image_features_predGS =  image_features_predGS.mean(dim = 0,  keepdim = True)

  # Load disentangled1 images into CLIP embeddings
  with torch.no_grad():
      for i in range(len(disentangled1)):
          if i == 0:
            i_x = preprocess(Image.open(disentangled1[i])).unsqueeze(0).to(device)
            image_features_pred = model.encode_image(i_x)
          else:
            i_x = preprocess(Image.open(disentangled1[i])).unsqueeze(0).to(device)
            image_features_pred = torch.cat((image_features_pred, model.encode_image(i_x)),dim = 0) 
      
      image_features_pred =  image_features_pred / image_features_pred.norm(dim=-1, keepdim = True)
      image_features_pred =  image_features_pred.mean(dim = 0,  keepdim = True)

  # Load disentangled2 images into CLIP embeddings
  with torch.no_grad():
      for i in range(len(disentangled2)):
          if i == 0:
            i_x = preprocess(Image.open(disentangled2[i])).unsqueeze(0).to(device)
            image_features_pred2 = model.encode_image(i_x)
          else:
            i_x = preprocess(Image.open(disentangled2[i])).unsqueeze(0).to(device)
            image_features_pred2 = torch.cat((image_features_pred2, model.encode_image(i_x)),dim = 0) 
      
      image_features_pred2 =  image_features_pred2 / image_features_pred2.norm(dim=-1, keepdim = True)
      image_features_pred2 =  image_features_pred2.mean(dim = 0,  keepdim = True)



  # Tokenize the prompts
  text__ = clipmod.tokenize(predictions).cuda()


  # calculate scores
  with torch.no_grad():
    text_encodings = model.encode_text(text__)

    image_features_predGS /= image_features_predGS.norm(dim=-1, keepdim=True)
    text_encodings /= text_encodings.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features_predGS @ text_encodings.T).softmax(dim=-1)
    values, indices = similarity[0].topk(2)


    for ii in range(len(values)):
      print("Enlangled predictions : ", predictions[indices[ii].cpu().numpy()], ' ', values[ii].item())


  # calculate scores
  with torch.no_grad():
    text_encodings = model.encode_text(text__)

    image_features_pred /= image_features_pred.norm(dim=-1, keepdim=True)
    text_encodings /= text_encodings.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features_pred @ text_encodings.T).softmax(dim=-1)
    values, indices = similarity[0].topk(2)


    for ii in range(len(values)):
      print("Disentangled1 predictions : ", predictions[indices[ii].cpu().numpy()], ' ', values[ii].item())

  # calculate scores
  with torch.no_grad():
    text_encodings = model.encode_text(text__)

    image_features_pred2 /= image_features_pred2.norm(dim=-1, keepdim=True)
    text_encodings /= text_encodings.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features_pred2 @ text_encodings.T).softmax(dim=-1)
    values, indices = similarity[0].topk(2)


    for ii in range(len(values)):
      print("Disentangled2 predictions : ", predictions[indices[ii].cpu().numpy()], ' ', values[ii].item())
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test 2")
    parser.add_argument('--attribute', type=str, default = 'beard_glass', help = 'choose beard_glass, kids_smile')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clipmod.load("ViT-B/32", device=device)

    compute_scores()