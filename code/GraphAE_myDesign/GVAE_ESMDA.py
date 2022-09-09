# Load trained GVAE 
import torch
import torch.nn as nn
import numpy as np
import Variational_GAE as graphAE
import graphAE_param as Param
import graphAE_dataloader as Dataloader
from datetime import datetime
from plyfile import PlyData
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from torch.distributions import MultivariateNormal
import ESMDA as esmda

surface_indeces = range(100)
N_ensambels =100
measurment_tensor = torch.ones(surface_indeces.shape[0],3)


def get_faces_from_ply(ply):
    faces_raw = ply['face']['vertex_index']
    faces = np.zeros((faces_raw.shape[0], 3)).astype(np.int32)
    for i in range(faces_raw.shape[0]):
        faces[i][0]=faces_raw[i][0]
        faces[i][1]=faces_raw[i][1]
        faces[i][2]=faces_raw[i][2]
    
    
    return faces

def obtain_surface_point(input_mesh,surface_indeces):
    return input_mesh[surface_indeces,:]

def model_to_predict_output_points(Latent_ensambels):
    return obtain_surface_point(model.decoder(Latent_ensambels),surface_indeces)





param_enc = Param.Parameters()
param_dec = Param.Parameters()
param_enc.read_config("/content/MeshConvolution_v2/train/graphAE_Breast/encoder.config")
param_dec.read_config("/content/MeshConvolution_v2/train/graphAE_Breast/decoder.config")
model=graphAE.VariationalAutoencoder(param_enc,param_dec,test_mode=True)

model.cuda()

if(param_enc.read_weight_path!=""):
    print ("load "+param_enc.read_weight_path)
    checkpoint = torch.load(param_enc.read_weight_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.init_test_mode()
    
    
model.eval()

    
template_plydata = PlyData.read(param_enc.template_ply_fn)
faces = get_faces_from_ply(template_plydata)




#param.augmented_data=True
param_enc.batch =1

param_enc.read_weight_path = "/home/mehrn/projects/def-mtavakol/mehrn/MeshConvolution_v2/train/graphAE_Breast/model_epoch0198.weight"
print (param_enc.read_weight_path)

test_npy_fn = "../../data/Breast/test.npy"

out_test_folder = "../../train/graphAE_Breast/test_20/epoch198/"

out_ply_folder = out_test_folder+"ply/"

batch = param_enc.batch

print ("**********Get test pcs**********", test_npy_fn)
##get ply file lst
pc_lst= np.load(test_npy_fn)
print (pc_lst.shape[0], "meshes in total.")

pcs = pc_lst[:batch]
height = pcs[:,:,1].mean(1)
pcs[:,:,0:3] -= pcs[:,:,0:3].mean(1).reshape((-1,1,3)).repeat(param_enc.point_num, 1) ##centralize each instance

pcs_torch = torch.FloatTensor(pcs).cuda()



# out_put of encoder 

latent_mu, latent_var = model.encoder(pcs_torch)   

# Generate ensamble in the latent space 
covarinace = torch.diag(latent_var)
m = MultivariateNormal(latent_mu, covarinace)
Latent_ensambels = m.sample(torch.Size([N_ensambels]))

# Calculte the out-put of each ensable 
Ensambel_predictions = obtain_surface_point(model.decoder(Latent_ensambels),surface_indeces)

# Define the ESMDA parameters 
Na = 4 # Number of esmda iterations 
N_obs = surface_indeces.shape[0]
alpha = 0.01
Cd = 0.05*torch.eye(N_obs)
esmda_model = esmda( Na, N_ensambels, N_obs, Latent_ensambels, Ensambel_predictions, Cd, alpha, model_to_predict_output_points)
esmda_model.assimilate (measurment_tensor)