import torch
import numpy as np
from plyfile import PlyData
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
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