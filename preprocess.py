import numpy as np
import cv2
import os


import glob


a=[]
with open("train.txt")as ff:
	
	for line in ff:
		a.append(str(line))

#counter=0

os.system("mkdir train")
##uncomment to create dir
for ii in range(0,len(a)):

	xx=str("mkdir train/"+str(ii))
	os.system(xx)
#uncomment to get data
for ii in range(0,len(a)):
	xx=(str("wget https://storage.googleapis.com/uga-dsp/project4/data/"+str(a[ii].strip())+".tar -O train/"+str(ii)+"/1.tar"))
	print(xx)
	os.system(xx)
#uncomment to get masks
for ii in range(0,len(a)):
	xx=(str("wget https://storage.googleapis.com/uga-dsp/project4/masks/"+str(a[ii].strip())+".png -O train/"+str(ii)+"/mask.png"))
	print(xx)
	os.system(xx)

#uncomment to decompress the train data
for ii in range(0,len(a)):
	xx=str("tar -xvf train/"+str(ii)+"/1.tar")
	os.system(xx)

##uncomment to create dir again
for ii in range(0,len(a)):
	xx=str("mkdir data/"+str(ii))
	os.system(xx)
#copy stuff (just to verify if the ordering is correct) folder i will have the line from train.txt make sure to note the ordering of train.txt starts with 1 while these start from 0 
for ii in range(0,len(a)):
	xx=str("cp -rf data/"+str(a[ii].strip())+" data/"+str(ii))
	os.system(xx)

#remove old dir doesnt work (sadly it doesnt)
#for ii in range(0,len(a)):
#	xx=str("rm -rf data/"+str(a[ii].strip()))




#creates seperate X_train and y_train for different folders #duplicates the y_train for as many X_train images in a folder

for i in range(0,len(a)):
	X_train = []
	y_train=[]
	
	num_files=glob.glob("data/"+str(a[i].strip())+"/*.png")
	print(num_files[0])

	gt="train/"+str(i)+"/mask.png"
	print gt
	for j in range(0,len (num_files)):
		#print(num_files[j])

		X_train.append(cv2.imread(num_files[j],0))
		y_train.append(cv2.imread(gt,0))

	np.save("X_train_"+str(i),((np.array(X_train))))
	np.save("y_train_"+str(i),((np.array(y_train))))



















