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
  if args.attribute == 'gender':
    our = natsorted(glob.glob("identity_results/gender/edited/*.png"))
    GS = natsorted(glob.glob("identity_results/ganspace/gender/*.png"))
    predictions = ['A male person', 'A picture a masculine person', 'A picture of a female' ]

  elif args.attribute == 'beard':
    our = natsorted(glob.glob("identity_results/beard/edited/*.png"))
    GS = natsorted(glob.glob("identity_results/ganspace/Beard/*.png"))
    predictions = ['A picture of a bearded person', 'A picture of a person with facial hair',  'A clean shaved person']

  elif args.attribute == 'age':
    our = natsorted(glob.glob("identity_results/kids2/edited/*.png"))
    GS = natsorted(glob.glob("identity_results/ganspace/Age/*.png"))
    predictions = ['kids','Teens', 'adults']

  elif args.attribute == 'smile':
    our = natsorted(glob.glob("identity_results/smile/edited/*.png"))
    GS = natsorted(glob.glob("identity_results/ganspace/smile/*.png"))
    predictions = ['A picture of a smiling person', 'A picture of a happy person',  'A picture of a sad person']

  else:
    print('Not an attribute')
    exit(0)


  # Load GS images into CLIP embeddings
  with torch.no_grad():
      for i in range(len(GS)):
          if i == 0:
            i_x = preprocess(Image.open(GS[i])).unsqueeze(0).to(device)
            image_features_predGS = model.encode_image(i_x)
          else:
            i_x = preprocess(Image.open(GS[i])).unsqueeze(0).to(device)
            image_features_predGS = torch.cat((image_features_predGS, model.encode_image(i_x)),dim = 0) 
      
      image_features_predGS =  image_features_predGS / image_features_predGS.norm(dim=-1, keepdim = True)
      image_features_predGS =  image_features_predGS.mean(dim = 0,  keepdim = True)

  # Load CLIP2SG images into CLIP embeddings
  with torch.no_grad():
      for i in range(len(our)):
          if i == 0:
            i_x = preprocess(Image.open(our[i])).unsqueeze(0).to(device)
            image_features_pred = model.encode_image(i_x)
          else:
            i_x = preprocess(Image.open(our[i])).unsqueeze(0).to(device)
            image_features_pred = torch.cat((image_features_pred, model.encode_image(i_x)),dim = 0) 
      
      image_features_pred =  image_features_pred / image_features_pred.norm(dim=-1, keepdim = True)
      image_features_pred =  image_features_pred.mean(dim = 0,  keepdim = True)



  # Tokenize the prompts
  text__ = clipmod.tokenize(predictions).cuda()


  # calculate scores
  with torch.no_grad():
    text_encodings = model.encode_text(text__)

    image_features_pred /= image_features_pred.norm(dim=-1, keepdim=True)
    text_encodings /= text_encodings.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features_pred @ text_encodings.T).softmax(dim=-1)
    values, indices = similarity[0].topk(3)


    for ii in range(len(values)):
      print("our predictions : ", predictions[indices[ii].cpu().numpy()], ' ', values[ii].item())
      
  # calculate scores
  with torch.no_grad():
    text_encodings = model.encode_text(text__)

    image_features_predGS /= image_features_predGS.norm(dim=-1, keepdim=True)
    text_encodings /= text_encodings.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features_predGS @ text_encodings.T).softmax(dim=-1)
    values, indices = similarity[0].topk(3)


    for ii in range(len(values)):
      print("GS predictions : ", predictions[indices[ii].cpu().numpy()], ' ', values[ii].item())

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test 1")
    parser.add_argument('--attribute', type=str, default = 'gender', help = 'choose gender, beard , age, smile')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clipmod.load("ViT-B/32", device=device)

    compute_scores()