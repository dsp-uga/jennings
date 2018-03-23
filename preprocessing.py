import os;
import sys;
#https://docs.python.org/2/library/tarfile.html
import tarfile
import shutil
import glob;
import numpy as np
import cv2
import wget
from matplotlib import pyplot as plt
gcloud_trian_path_data  = "https://storage.googleapis.com/uga-dsp/project4/data/"
gcloud_trian_path_masks = "https://storage.googleapis.com/uga-dsp/project4/masks/"

#downloads data to local directory at data/filename thanks to wget library
def download_images(path,http_url):
	f = open(path)
	for i in f:
		fetch_url = http_url+i.strip();
		print("Fetching:"+fetch_url)
		filename = wget.download(fetch_url+".tar")
		tar = tarfile.open(filename, "r:tar")
		tar.extractall()
		tar.close()
		break;

#converts the images present in data/file/* to numpy array  thanks to url library
def load_images(preifx):
	temp = list();
	dataset = glob.glob(preifx)
	for i in dataset:
		entries = list();
		files = glob.glob(i+"/*")
		for file in files:
			entries.append(cv2.imread(file,0))
		temp.append(entries)
		break;
	return np.array(temp)

def load_masks(path,http_url):
	entries = list();
	f = open(path)
	for i in f:
		fetch_url = http_url+i.strip();
		print("Fetching:"+fetch_url)
		filename = wget.download(fetch_url+".png")
		entries.append(cv2.imread(filename,1))
		print("called")
		break;
	return np.array(entries)


print("downloading training dataset")
download_images(sys.argv[1],gcloud_trian_path_data)
train_set = load_images("data/*");
shutil.rmtree("data")
print("downloading test set")
download_images(sys.argv[2],gcloud_trian_path_data)
test_set = load_images("data/*");
print("downloading masks")
mask_set = load_masks(sys.argv[1],gcloud_trian_path_masks)[0,:,:,1]
#train_set= np.std(train_set,axis=0)
np.save('train_set.npy',train_set)
np.save('mask_set.npy',mask_set)
np.save('test_set.npy',test_set)

