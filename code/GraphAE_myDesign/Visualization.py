import numpy as np
import pandas as pd
import torch
import Variational_GAE as graphAE
import graphAE_param as Param
import graphAE_dataloader as Dataloader
from plyfile import PlyData
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from renderer.renderer import Mesh, Renderer

# For plotting





#PCA
from sklearn.decomposition import PCA
#TSNE
from sklearn.manifold import TSNE
#UMAP
#import umap
#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def plot_2d(component1, component2):
    
   fig = plt.figure(figsize=(12, 12))
   ax = fig.add_subplot()
   ax.scatter(component1,component2)
   fig.show()


def plot_3d(component1,component2,component3):

   fig = plt.figure(figsize=(12, 12))
   ax = fig.add_subplot(projection='3d')
   ax.scatter(component1,component2,component3)
   fig.show()

   


def Load(param_enc,param_dec,test_npy_fn,test_npy_gt_fn, out_ply_folder, out_img_folder, is_render_mesh=False, skip_frames =0):
    
    # For now the test_npy_fn is similar to test_npy_gt

    n=10
    batch=100
   
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
  
    
    print ("**********Get test pcs**********", test_npy_fn)
    ##get ply file lst
    pc_lst= np.load(test_npy_fn)
    # print (pc_lst.shape[0], "meshes in total.")
    pc_lst[:,:,0:3] -= pc_lst[:,:,0:3].mean(1).reshape((-1,1,3)).repeat(param.point_num, 1)
    pcs = pc_lst[n:n+batch]

    pcs = torch.FloatTensor(pcs)
    
    return   model, pcs


# Data preperation 
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

out_test_folder = "../../train/graphAE_Breast/ESMDA/"

out_ply_folder = out_test_folder
out_img_folder = out_test_folder+"img/"

out_weight_visualize_folder=out_test_folder+"weight_vis/"

out_interpolation_folder = out_test_folder+"interpolation/"

# LOad weights and load data sets 

model, pc_lst = Load(param_enc,param_dec,test_npy_fn,test_npy_gt_fn, out_ply_folder, out_img_folder)


original_dimension = pc_lst
latent_dimension = model(pc_lst.cuda())

x = original_dimension.view(original_dimension.shape[0],-1)
print(x.shape)
# Implementing PCA 

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principal = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3'])


plot_2d(principalComponents[:, 0],principalComponents[:, 1])
plot_3d(principalComponents[:, 0],principalComponents[:, 1],principalComponents[:, 2])


# t-SNE Implementation 

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(x)
tsne = TSNE(random_state = 42, n_components=3,verbose=0, perplexity=40, n_iter=400).fit_transform(pca_result_50)

plot_2d(tsne[:, 0],tsne[:, 1])
plot_3d(tsne[:, 0],tsne[:, 1],tsne[:, 2])

"""
# Implement MAP 
reducer = umap.UMAP(random_state=42,n_components=3)
embedding = reducer.fit_transform(x)

plot_2d(reducer.embedding_[:, 0],reducer.embedding_[:, 1])
plot_3d(reducer.embedding_[:, 0],reducer.embedding_[:, 1],reducer.embedding_[:, 2])

# Implement LDA 
#X_LDA = LDA(n_components=3).fit_transform(standardized_data,y)
#plot_2d(X_LDA[:, 0],X_LDA[:, 1])
#plot_3d(X_LDA[:, 0],X_LDA[:, 1],X_LDA[:, 2])

"""