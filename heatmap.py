import os
import sys
import numpy as np
import pickle as pickle
import sys,glob,io,random
import pandas as pd
import math
import random
fps = 30
import csv
import numpy as np
import matplotlib.pyplot as plt  
import tensorflow as tf


def discretization(lat,lon):
    lat = lat-np.pi/2
    lon = lon+np.pi
    n = lat.shape[0]
    bin_size = 10
    one_hot_code_matrix = np.zeros((n,int(180/bin_size),int(360/bin_size)))
    for i in range(n):
        theta = lon[i]
        phi = lat[i]
        theta = theta/np.pi*180   # 0 to 360
        phi = phi/np.pi*180       # 90 to -90
        col = math.floor(theta/bin_size)
        row = math.floor(phi/bin_size)
        if phi == 180:
            row = 17
        if theta == 180:
            col = 17
        if col == 36:
            col = 0
        if row == 18:
            row = 0

        one_hot_code_matrix[i,int(row),int(col)] = 1

    return one_hot_code_matrix
    
def gauss_2d(one_hot):
   
    IMAGE_WIDTH = 36
    IMAGE_HEIGHT = 18
    center = np.where(one_hot == 1)

    R = 1#np.sqrt(center[0]**2 + center[1]**2)/10
    Gauss_map = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    for i in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            dis = np.sqrt((i-center[1])**2+(j-center[0])**2)
            Gauss_map[i, j] = np.exp(-0.5*dis/R)

    return Gauss_map
    
    
    
    with open('./179._2017-10-13-10-27_ori_0.txt','r') as csvfile:
    reader = csv.reader(csvfile)
    rows= [row for row in reader]
    
    
    data=np.array(rows)
    a=np.zeros(2940).reshape(1470,2)
    for i in range(1470):
    a[i][0]=float(data[i][6])
    a[i][1]=float(data[i][7])
    
    heatmap=[]
    for i in range(49):

    a1=a[i*30:(i+1)*30]
    a1.shape
    oneshot=discretization(a1[:,0],a1[:,1])
    b=np.zeros((18,36))
    for j in range(30):
        k=gauss_2d(oneshot[j])  
        b+=k
    heatmap.append(b)
    
    encoder_inputs1=heatmap[:5] 
    decoder_inputs1=heatmap[4] 
    decoder_target1=heatmap[5:10]
    
    
    encoder_inputs1=np.array(encoder_inputs1) 
    decoder_inputs1=np.array(decoder_inputs1) 
    decoder_target1=np.array(decoder_target1)
    
    encoder_inputs1= tf.convert_to_tensor(encoder_inputs1, tf.float32, name='t') 
    decoder_inputs1= tf.convert_to_tensor(decoder_inputs1, tf.float32, name='t') 
    decoder_target1= tf.convert_to_tensor(decoder_target1, tf.float32, name='t')
