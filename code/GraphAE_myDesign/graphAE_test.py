# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import numpy as np
import new_AE as graphAE
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

def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    color = mpl.colors.to_hex((1-mix)*c1 + mix*c2)
    (r,g,b) = mpl.colors.ColorConverter.to_rgb(color)
    return np.array([r,g,b])

def get_colors_from_diff_pc(diff_pc, min_error, max_error):
    colors = np.zeros((diff_pc.shape[0],3))
    mix = (diff_pc-min_error)/(max_error-min_error)
    mix = np.clip(mix, 0,1) #point_num
    cmap=cm.get_cmap('coolwarm')
    colors = cmap(mix)[:,0:3]
    return colors

def get_faces_colors_from_vertices_colors(vertices_colors, faces):
    faces_colors = vertices_colors[faces]
    faces_colors = faces_colors.mean(1)
    return faces_colors




def get_faces_from_ply(ply):
    faces_raw = ply['face']['vertex_index']
    faces = np.zeros((faces_raw.shape[0], 3)).astype(np.int32)
    for i in range(faces_raw.shape[0]):
        faces[i][0]=faces_raw[i][0]
        faces[i][1]=faces_raw[i][1]
        faces[i][2]=faces_raw[i][2]
    
    
    return faces
    
  

def evaluate(param, model, pc_lst , template_plydata):
    geo_error_sum = 0
    pc_num = len(pc_lst)
    n = 0

    while (n<(pc_num-1)):
        batch = min(pc_num-n, param.batch)
        pcs = pc_lst[n:n+batch]
        height = pcs[:,:,1].mean(1)
        pcs_torch = torch.FloatTensor(pcs).cuda()
        if(param.augmented_data==True):
            pcs_torch = Dataloader.get_augmented_pcs(pcs_torch)
        if(batch<param.batch):
            pcs_torch = torch.cat((pcs_torch, torch.zeros(param.batch-batch, param.point_num, 3).cuda()),0)
        out_pcs_torch = model(pcs_torch)
        geo_error = model.compute_geometric_mean_euclidean_dist_error(pcs_torch, out_pcs_torch)
        geo_error_sum = geo_error_sum + geo_error*batch
        # print(n, geo_error)

        if(n % 128 ==0):
            print (height[0])
            pc_gt = np.array(pcs_torch[0].data.tolist()) 
            pc_gt[:,1] +=height[0]
            pc_out = np.array(out_pcs_torch[0].data.tolist())
            pc_out[:,1] +=height[0]

            diff_pc = np.sqrt(pow(pc_gt-pc_out, 2).sum(1))
            color = get_colors_from_diff_pc(diff_pc, 0, 0.02)*255
            Dataloader.save_pc_into_ply(template_plydata, pc_out, out_ply_folder+"%08d"%(n)+"_out.ply")
            Dataloader.save_pc_into_ply(template_plydata, pc_gt, out_ply_folder+"%08d"%(n)+"_gt.ply")

        n = n+batch
        

    geo_error_avg=geo_error_sum.item()/pc_num
    
     
    return geo_error_avg


def test(param,test_npy_fn, out_ply_folder, out_img_folder, is_render_mesh=False, skip_frames =0):
    
    
    print ("**********Initiate Netowrk**********")
    model=graphAE.Model(param)
    
    model.cuda()
    
    if(param.read_weight_path!=""):
        print ("load "+param.read_weight_path)
        checkpoint = torch.load(param.read_weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        #model.init_test_mode()
    
    
    model.eval()
    
    template_plydata = PlyData.read(param.template_ply_fn)
    faces = get_faces_from_ply(template_plydata)
    
    print ("**********Get test pcs**********", test_npy_fn)
    ##get ply file lst
    pc_lst= np.load(test_npy_fn)

    # print (pc_lst.shape[0], "meshes in total.")
    pc_lst[:,:,0:3] -= pc_lst[:,:,0:3].mean(1).reshape((-1,1,3)).repeat(param.point_num, 1)

    geo_error_avg=evaluate(param, model, pc_lst ,template_plydata)

    print ("geo error:", geo_error_avg)#, "laplace error:", laplace_error_avg)
    
        
    
def visualize_weights(param, out_folder):
    print ("**********Initiate Netowrk**********")
    model=graphAE.Model(param, test_mode=False)
    
    model.cuda()
    
    if(param.read_weight_path!=""):
        print ("load "+param.read_weight_path)
        checkpoint = torch.load(param.read_weight_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    
    model.eval()
    
    model.quantify_and_draw_w_weight_histogram(out_folder)
    
    #draw_w_weight_histogram(model, out_folder )

    
    

param=Param.Parameters()
param.read_config("../../train/graphAE_Breast/00_conv_pool_Cheb_Mehr01.config")
print (param.use_vanilla_pool)
#param.augmented_data=True
param.batch =16

param.read_weight_path = "../../train/graphAE_Breast/weight_00/model_epoch0050.weight"
print (param.read_weight_path)

test_npy_fn = "../../data/Breast/test.npy"

out_test_folder = "../../train/graphAE_Breast/test/epoch200/"

out_ply_folder = out_test_folder+"ply/"
out_img_folder = out_test_folder+"img/"

out_weight_visualize_folder=out_test_folder+"weight_vis/"

out_interpolation_folder = out_test_folder+"interpolation/"

if not os.path.exists(out_ply_folder):
    os.makedirs(out_ply_folder)
    
if not os.path.exists(out_img_folder+"/gt"):
    os.makedirs(out_img_folder+"gt/")
    
if not os.path.exists(out_img_folder+"/out"):
    os.makedirs(out_img_folder+"out/")
    
if not os.path.exists(out_img_folder+"/out_color"):
    os.makedirs(out_img_folder+"out_color/")
    
if not os.path.exists(out_weight_visualize_folder):
    os.makedirs(out_weight_visualize_folder)

pc_lst= np.load(test_npy_fn)
# print (pc_lst[:,:,1].max()-pc_lst[:,:,1].min())

with torch.no_grad():
    torch.manual_seed(2)
    np.random.seed(2)
    test(param, test_npy_fn, out_ply_folder,out_img_folder, is_render_mesh=False, skip_frames=0)
    test(param, param.pcs_train, out_ply_folder,out_img_folder, is_render_mesh=False, skip_frames=0)
    visualize_weights(param, out_weight_visualize_folder)
    #test_interpolation(param, middle_layer_id=7, inter_num=10, test_npy_fn=test_npy_fn, id1=40, id2=2000, out_folder=out_interpolation_folder)


        
        

