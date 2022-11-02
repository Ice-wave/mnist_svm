import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt

a = np.zeros(shape=(8391, 64), dtype=float)
b = np.zeros(shape=(8391), dtype=float)
path = "D:\edge下载\deeploc_embedding3.pkl"
f = open(path, 'rb')
data = pickle.load(f)
# y = data['membrane']
y = data['loc']
X = data['z_mean']
for i in range(0, X.shape[0]):
    a[i] = X[i]
print(a)
print(a.shape)

for i in range(0, y.shape[0]):
    if y[i]=='Cytoplasm':
        b[i]=0
    if y[i]=='Cell.membrane':
        b[i]=1
    if y[i]=='Nucleus':
        b[i]=2
    if y[i]=='Golgi.apparatus':
        b[i]=3
    if y[i]=='Plastid':
        b[i]=4
    if y[i]=='Mitochondrion':
        b[i]=5
    if y[i]=='Extracellular':
        b[i]=6
    if y[i]=='Endoplasmic.reticulum':
        b[i]=7
    if y[i]=='Peroxisome':
        b[i]=8
    if y[i]=='Lysosome/Vacuole':
        b[i]=9

print(b)
print(b.shape)

'''
# 二分类
for i in range(0, y.shape[0]):
    if y[i]=='S':
        b[i]=0
    if y[i]=='M':
        b[i]=1
    else:
        b[i]=2
print(b)
print(b.shape)
'''

clf = SVC(kernel='rbf')
clf.fit(a, b)
result=clf.predict(a)
print(result)
score=clf.score(a, b)
print(score)
plt.figure()
plt.subplot(111)
plt.scatter(a[:,0],a[:,3],c =b.reshape((-1)),edgecolor='k',s=50)
plt.show()

