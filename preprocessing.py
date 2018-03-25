import sys;
#https://docs.python.org/2/library/tarfile.html
import tarfile
import shutil
import glob
import numpy as np
import cv2
import urllib.request

gcloud_trian_path_data  = "https://storage.googleapis.com/uga-dsp/project4/data/"
gcloud_trian_path_masks = "https://storage.googleapis.com/uga-dsp/project4/masks/"

train_path = "train_set.npy"
mask_path = "masks_set.npy"
test_path = "test_set.npy"


def download_images(path,http_url):

	"""Downloads the data specified by the url and extracts it and places it under data/* directory"""
	f = open(path)
	for i in f:
		fetch_url = http_url + i.strip();
		print("Fetching:"+fetch_url)
		urllib.request.urlretrieve(fetch_url + ".tar", filename = "temp.tar")
		tar = tarfile.open("temp.tar", "r:tar")
		tar.extractall()
		tar.close()
		count = count + 1


def load_images(preifx):
	"""loads images from data/* directory downloaded by download_images and creates a numpy array for further processing"""

	temp = list();
	dataset = glob.glob(preifx)
	dataset.sort();
	for i in dataset:
		entries = list();
		files = glob.glob(i + "/*")
		for file in files:
			entries.append(cv2.imread(file, 0))
		temp.append(entries)
		count = count + 1
	return np.array(temp)

def load_masks(path,http_url):
	"""downloads the mask for a particular has and creats a numpy array"""
	entries = list();
	f = open(path)
	for i in f:
		fetch_url = http_url+i.strip();
		print("Fetching:" + fetch_url)
		filename = urllib.request.urlretrieve(fetch_url + ".png", filename = i + ".png")
	

def load_mask_images():
	entries = glob.glob("*.png")
	print(entries)
	entries.sort()
	temp = list();
	for i in entries:
		temp.append(cv2.imread(i, 0))
	return np.array(temp)




print("downloading training dataset")
download_images(sys.argv[1], gcloud_trian_path_data)
train_set = load_images("data/*")
shutil.rmtree("data")

print("downloading test set")
download_images(sys.argv[2], gcloud_trian_path_data)
test_set = download_images(sys.argv[2], gcloud_trian_path_data)
test_set = load_images("data/*")

shutil.rmtree("data")
print("downloading masks")
mask_set = load_masks(sys.argv[1], gcloud_trian_path_masks)
mask_set = load_mask_images()


#train_set= np.std(train_set,axis=0)
np.save(train_path, train_set)
np.save(mask_path, mask_set)
np.save(test_path, test_set)

