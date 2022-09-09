import torch
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
from renderer.renderer import Mesh, Renderer
from matplotlib import cm
from torch.distributions import MultivariateNormal
import ESMDA as esmda


def get_faces_from_ply(ply):
    faces_raw = ply['face']['vertex_index']
    faces = np.zeros((faces_raw.shape[0], 3)).astype(np.int32)
    for i in range(faces_raw.shape[0]):
        faces[i][0]=faces_raw[i][0]
        faces[i][1]=faces_raw[i][1]
        faces[i][2]=faces_raw[i][2]
    
    
    return faces
    




# Load weights 
def Load(param_enc,param_dec,test_npy_fn,test_npy_gt_fn, out_ply_folder, out_img_folder, is_render_mesh=False, skip_frames =0):
    
    # For now the test_npy_fn is similar to test_npy_gt

    
   
    print ("**********Initiate Netowrk**********")
    model = graphAE.VariationalAutoencoder(param_enc,param_dec)
    param = param_enc
    if(param.read_weight_path!=""):
        print ("load "+param.read_weight_path)
        checkpoint = torch.load(param.read_weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        #model.init_test_mode()
    
    model.cuda()
    model.eval()
    
    template_plydata = PlyData.read(param.template_ply_fn)
    faces = get_faces_from_ply(template_plydata)
    
    print ("**********Get test pcs**********", test_npy_fn)
    ##get ply file lst
    pc_lst= np.load(test_npy_fn)
    pc_gt_lst= np.load(test_npy_gt_fn)


    # print (pc_lst.shape[0], "meshes in total.")
    pc_lst[:,:,0:3] -= pc_lst[:,:,0:3].mean(1).reshape((-1,1,3)).repeat(param.point_num, 1)
    pc_gt_lst[:,:,0:3] -= pc_gt_lst[:,:,0:3].mean(1).reshape((-1,1,3)).repeat(param.point_num, 1)

    pc_lst = torch.FloatTensor(pc_lst)
    pc_gt_lst = torch.FloatTensor(pc_gt_lst)
    
    return   model, pc_lst, pc_gt_lst



def measurment_function (model, Latent_ensambels, surface_index):
    surface_index = surface_index.cuda()
    out_put_ensamble_mesh =  model.decoder(Latent_ensambels)
    out_put_surface_nodes = out_put_ensamble_mesh[:,surface_index,:]
    return out_put_surface_nodes
    
def encoder_function (model,N_ensambels, pc_lst_sample):
    # a = torch.FloatTensor(pc_lst_sample)
    pc_lst_sample = torch.unsqueeze(pc_lst_sample,0).cuda()
    latent_mu, latent_var =  model.encoder(pc_lst_sample)
    covarinace = torch.diag(torch.squeeze((0.5*latent_var).exp(),0))
    m = MultivariateNormal(torch.squeeze(latent_mu,0), covarinace)
    Latent_ensambels = m.sample(torch.Size([N_ensambels]))
    return Latent_ensambels



param_enc=Param.Parameters()
param_dec=Param.Parameters()
param_enc.read_config("../../train/graphAE_Breast/encoder.config")
param_dec.read_config("../../train/graphAE_Breast/decoder.config") 

#param.augmented_data=True
param = param_enc
param.batch =1  # this should be one since we are feeding data one-by-one in ESMDA 

param.read_weight_path = "../../train/graphAE_Breast/weight_00/model_epoch0070.weight"
print (param.read_weight_path)

test_npy_fn = "../../data/Breast/test.npy"
test_npy_gt_fn = "../../data/Breast/test.npy"
surface_indeces_fn = "../../data/Breast/index.npy"

out_test_folder = "../../train/graphAE_Breast/ESMDA/"

out_ply_folder = out_test_folder
out_img_folder = out_test_folder+"img/"

out_weight_visualize_folder=out_test_folder+"weight_vis/"

out_interpolation_folder = out_test_folder+"interpolation/"

# LOad weights and load data sets 

model, pc_lst, pc_gt_lst = Load(param_enc,param_dec,test_npy_fn,test_npy_gt_fn, out_ply_folder, out_img_folder)


# Build output functions 


# Apply ESMDA 

# Define the ESMDA parameters 
Na = 100 # Number of esmda iterations 
num=150
N_ensambels = 300
# surface_indeces= np.load(surface_indeces_fn)
surface_indeces = torch.tensor([354,1732,1712,905,1466,332,805,1648,1447,1715,1212,166,1546,1677,1251,1394,1364,748,\
                                1212,377,1296,159,559,178,259,1504,1283,619,1700,164,824,731,1402,1460,404,931,846,1202,1312,1390,\
                                558,1252,1212,365,294,941,1715,670,1090,466],dtype=torch.long)
measurment_tensor = pc_gt_lst[num+1,:][surface_indeces,:]
measurment_tensor = measurment_tensor.view(-1)
N_obs = measurment_tensor.shape[0]
alpha = 0.01
Cd = 0.01*torch.eye(N_obs,dtype=torch.float32)



#a = torch.FloatTensor(pc_lst[num,:])
#pc_lst_sample = torch.unsqueeze(a,0).cuda()
#latent_mu, latent_var =  model.encoder(pc_lst_sample)
#print(latent_var.shape)
#print((0.5*latent_var).exp())
#covarinace = torch.diag(torch.squeeze(latent_var,0))

Latent_ensambels = encoder_function (model,N_ensambels, pc_lst[num,:]) 
out_put_surface_nodes_prediction =  measurment_function (model, Latent_ensambels, surface_indeces)
Ensambel_predictions = out_put_surface_nodes_prediction.view(out_put_surface_nodes_prediction.size(0),-1)



esmda_model = esmda.ESMDA( Na, N_ensambels, N_obs, Latent_ensambels.t(), Ensambel_predictions.t(), Cd, alpha,measurment_function,model,surface_indeces)
a , b = esmda_model.assimilate (measurment_tensor)


gt = measurment_tensor.cuda()

print(torch.norm(torch.mean(b,1)-gt))
print(torch.norm(torch.mean(Ensambel_predictions,0)-gt))

final_latent = torch.unsqueeze(torch.mean(a,1),0)
first_latent = torch.unsqueeze(torch.mean(Latent_ensambels,0),0)


final_refined_mesh =  model.decoder(final_latent)
first_unrefined_mesh =  model.decoder(first_latent)


#print(pc_gt_lst[num,:].cuda()-torch.squeeze(first_unrefined_mesh,0))
#print(pc_gt_lst[num,:].cuda()-torch.squeeze(final_refined_mesh,0))

#print("before refinment",torch.norm(pc_gt_lst[num,:].cuda()-torch.squeeze(first_unrefined_mesh,0),dim=1))
#print("after refinment",torch.norm(pc_gt_lst[num,:].cuda()-torch.squeeze(final_refined_mesh,0),dim=1))

print("before refinment",torch.mean(torch.norm(pc_gt_lst[num+1,:].cuda()-torch.squeeze(first_unrefined_mesh,0),dim=1)))
print("after refinment",torch.mean(torch.norm(pc_gt_lst[num+1,:].cuda()-torch.squeeze(final_refined_mesh,0),dim=1)))

# Save Assimilated out_put 

pc_out = np.array(torch.squeeze(final_refined_mesh,0).data.tolist())
pc_out0 = np.array(torch.squeeze(first_unrefined_mesh,0).data.tolist())
pc_outgt = np.array(pc_gt_lst[num+1,:].data.tolist())
template_plydata = PlyData.read(param.template_ply_fn)

if not os.path.exists(out_ply_folder):
    os.makedirs(out_ply_folder)

Dataloader.save_pc_into_ply(template_plydata, pc_out, out_ply_folder+"f_out3.ply")
Dataloader.save_pc_into_ply(template_plydata, pc_out0, out_ply_folder+"first3.ply")
Dataloader.save_pc_into_ply(template_plydata, pc_outgt, out_ply_folder+"gt3.ply")
