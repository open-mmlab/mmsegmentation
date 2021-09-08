import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

x = np.array([3.72,13.68,24.72,44.6,61.37,81.97])
#y =  np.array([90.06,90.33,90.5,90.64,91.06,90.94])
y =  np.array([90.13,90.35,90.61,90.68,91.08,90.94]) #best aAcc
plt.xlabel("Params (Millions)") 
plt.ylabel("Vaihingen(OA)") 
plt.plot(x,y,'-ob', label="Segformer B0-B5")

x = np.array([7.03,26.82,48.91,88.67,122.21,163.41])
#y = np.array([90.47,90.71,90.98,90.99,90.9,91.0])
y = np.array([90.47,90.73,90.99,91.0,90.96,90.99])
plt.plot(x,y,'-og', label="Traditiional Two Stream") 

x = np.array([5.74,21.76,32.81,52.68,69.45,90.05])
#y = np.array([90.35,90.67,90.96,90.99,90.96,91.0])
y = np.array([90.6,90.74,90.96,90.99,91.24,91.13])
plt.plot(x,y,'-Dr', label="Our Method") 

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=3, mode="expand", borderaxespad=0.)
plt.show()