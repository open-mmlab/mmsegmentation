import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot1():
       x = np.array([3.72,13.68,24.72,44.6,61.37,81.97])
       #y =  np.array([90.06,90.33,90.5,90.64,91.06,90.94])
       # y =  np.array([90.13,90.35,90.61,90.68,91.08,90.94]) #best aAcc
       y =  np.array([79.49,80.38,81.19,81.4,82.56,82.15]) #best aAcc
       plt.xlabel("Params (Millions)") 
       plt.ylabel("Vaihingen(mIoU)") 
       plt.plot(x,y,'-ob', label="Segformer B0-B5")
       txt=['B0','B1','B2','B3','B4','B5']
       for i in range(len(txt)):
              plt.annotate(txt[i], xy = (x[i], y[i]), xytext = (x[i]+2, y[i]-0.05))

       x = np.array([7.03,26.82,48.91,88.67,122.21,163.41])
       #y = np.array([90.47,90.71,90.98,90.99,90.9,91.0])
       # y = np.array([90.47,90.73,90.99,91.0,90.96,90.99])
       y = np.array([80.14,81.04,81.74,82.2,82.74,82.35])
       plt.plot(x,y,'-og', label="two-stream-same") 

       #x = np.array([5.74,21.76,32.81,52.68,69.45,90.05])
       x = np.array([4.2,15.6,26.65,46.52,63.29,83.89])
       #y = np.array([90.35,90.67,90.96,90.99,90.96,91.0])
       # y = np.array([90.6,90.74,90.96,90.99,91.24,91.13])
       y = np.array([79.93,81.02,81.78,82.11,82.26,82.31])
       plt.plot(x,y,'-Dc', label="two-stream-differ") 

       x = np.array([4.49,16.76,27.81,47.68,64.46,85.06])
       y = np.array([80.49,81.28,82.17,82.27,83.02,82.48])
       plt.plot(x,y,'-Dr', label="EDFT") 

       plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4,
              ncol=4, mode="expand", borderaxespad=0.)
       plt.show()

def plot3():
       x = np.array([3.72,13.68,24.72,44.6,61.37,81.97])
       #y =  np.array([90.06,90.33,90.5,90.64,91.06,90.94])
       # y =  np.array([90.13,90.35,90.61,90.68,91.08,90.94]) #best aAcc
       y =  np.array([79.49,80.38,81.19,81.4,82.56,82.15]) #best aAcc
       plt.xlabel("Params (Millions)") 
       plt.ylabel("Vaihingen(mIoU)") 
       plt.plot(x,y,'-ob', label="Segformer B0-B5")

       x = np.array([7.03,26.82,48.91,88.67,122.21,163.41])
       #y = np.array([90.47,90.71,90.98,90.99,90.9,91.0])
       # y = np.array([90.47,90.73,90.99,91.0,90.96,90.99])
       y = np.array([80.14,81.04,81.74,82.2,82.74,82.35])
       plt.plot(x,y,'-og', label="add-same") 

       #x = np.array([5.74,21.76,32.81,52.68,69.45,90.05])
       x = np.array([4.2,15.6,26.65,46.52,63.29,83.89])
       #y = np.array([90.35,90.67,90.96,90.99,90.96,91.0])
       # y = np.array([90.6,90.74,90.96,90.99,91.24,91.13])
       y = np.array([79.93,81.02,81.78,82.11,82.26,82.31])
       plt.plot(x,y,'-Dr', label="add-differ") 

       plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=3, mode="expand", borderaxespad=0.)
       plt.show()

def plot2():
       x = np.arange(7)
       ss=[82.39,82.97,82.42,82.43,82.26,82.74,82.56]
       ms=[83.32,83.81,83.52,83.39,83.0,83.6,83.63]


       total_width, n = 0.6, 2
       width = total_width / n
       x = x - (total_width - width) / 2
       
       plt.ylim((82, 84))
       label_list =["λ=1.0","λ=0.5","λ=0.1","DSA-concat","add-differ","add-same","Segformer-B4"]
       plt.xticks([index + 0.2 for index in x], label_list)

       rects1=plt.bar(x, ss,  width=width, label='single-scale')
       for rect in rects1:
              height = rect.get_height()
              plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha="center", va="bottom")

       rects1=plt.bar(x + width, ms, width=width, label='multi-scale')
       for rect in rects1:
              height = rect.get_height()
              plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha="center", va="bottom")
       
       plt.xlabel("Method") 
       plt.ylabel("Vaihingen(mIoU)") 
       plt.legend()
       plt.show()

plot1()