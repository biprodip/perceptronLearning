#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 01:22:14 2018

@author: biprodip
"""

def plotLearning(X,Y,W):
    import matplotlib.pyplot as plt
    import numpy as np
    #points forming current classifier line
    #get  10 points to draw lines
    linePointsX=np.linspace(-3,8,10,dtype="int32");
    linePointsX=np.vstack([linePointsX,np.ones(linePointsX.shape[0])]); #2 x 10
    linePointsY=W.dot(linePointsX) #(1x 2) x (2 x 10)= (1 x 10)  

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(linePointsX[0,], linePointsY, color='blue', linewidth=3)
    ax.scatter(X[0,Y>0],Y[Y>0],color='darkgreen',marker='^')
    ax.scatter(X[0,Y<1],Y[Y<1],color='red',marker='s')
    #plot the data and current classifier line
    plt.show()
