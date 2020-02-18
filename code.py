import numpy as np
from numpy import linalg as LA
from PIL import Image
import os
import cv2

#loding the dataset
images = os.listdir('./YaleFaceDatabase')
test = []
train = []
#dividing the data set into training and test 
for i in images:
	if images.index(i)%11<=7:
		train.append(i)
	else:
		test.append(i)


a = np.zeros(77760)
b = np.zeros((77760,24))
k=0
l=0
a = np.zeros(77760)
b=np.zeros((77760,120))
phi=np.zeros((77760,120))
t=np.zeros((77760,45))
thi=np.zeros((77760,45))

for i in train:
	pic = Image.open('YaleFaceDatabase/'+i)
	pix = np.array(pic)
	img = pix.flatten()
	a = a + img
	b[:,train.index(i)] = img
	k+=1

#Finding the mean
mean = a/k

#images can be seen by uncommenting the foloowing code
#avgprint = np.reshape(mean, (243, 320))
#cv2.imshow("Average", np.array(avgprint, dtype=np.uint8))
#cv2.waitKey()


for i in range(k):
	phi[:,i] = b[:,i] - mean


#images can be seen by uncommenting the foloowing code
#phiprint = np.reshape(phi[:;][0], (243, 320))
#cv.imshow("Phi", np.array(phiprint, dtype=np.uint8))
#cv.waitKey()


C = phi.transpose().dot(phi) #-------------#Getting the covariance matrix A'A

w,v = LA.eigh(C) #finding the eigen vectors
w=w.tolist()
v=v.tolist()

e=[]
E=[]
#getting top k(=30) eigen vectors
for i in range(30):
	ind = w.index(max(w))
	e.append(max(w))
	E.append(v[ind])
	w.remove(w[ind])
	v.remove(v[ind])

E = np.asarray(E)

U = E.dot(phi.transpose()) #Generating eigen faces

for i in range(len(U)):
	norm = LA.norm(U[i])
	U[i]=U[i]/norm

weight = U.dot(phi) #Generating Weights for training data set


#Lets find the accuracy for test dataset
k=0
for i in test:
	pic = Image.open('YaleFaceDatabase/'+i)
	pix = np.array(pic)
	img = pix.flatten()
	t[:,test.index(i)] = img
	k+=1


for i in range(k):
	thi[:,i] = t[:,i] - mean
weights = U.dot(thi)


d=[]
for i in range(45):
	l=[]
	for j in range(120):
		norm = LA.norm(weight[:,j] - weights[:,i])
		l.append(norm)
	d.append(l)
m=[]
for i in range(len(d)):
	m.append(d[i].index(min(d[i]))//8+1)
n=[1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10,11,11,11,12,12,12,13,13,13,14,14,14,15,15,15]
y=0
for i in range(len(m)):
	if m[i]==n[i]:
		y+=1
#printing the accuracy
print('For ' + str(8/11*100) + '% training data and ' + str(300/11) + '% test data, we are getting '+str(y*100/len(m)) +'%  accuracy')
