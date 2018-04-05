import numpy as np
import tarfile
import cv2
import os
import sys



#https://stackoverflow.com/questions/2239655/how-can-files-be-added-to-a-tarfile-with-python-without-adding-the-directory-hi
#https://docs.python.org/2.4/lib/tar-examples.html

class TarManager:

	def __init__(self,filename,dataset,tar_file_name="submission"):
		self.filename = filename;
		self.dataset = dataset.astype(int)
		self.tar = self.intialize(tar_file_name)
		self.tar_file_name = tar_file_name

	def intialize(self,tar_file_name="submission"):

		tar = tarfile.open(tar_file_name+".tar.gz", "w:gz")
		return tar
	def close_tar(self):
		self.tar.close();

	def add_to_tar(self,name,entry,extension=".png"):
		entry[entry==1] = 2
		cv2.imwrite(name+extension,entry)
		self.tar.addfile(tarfile.TarInfo(name+extension),open(name+extension))

	def generate_tar(self):
		for i in range(0,self.dataset.shape[0]):
			self.add_to_tar(self.filename[i],self.dataset[i])


"""
#Sample execution
data = np.zeros((2,240,240,1))
entry = ["data","data2"]



maanger = TarManager(entry,data)
maanger.generate_tar();
"""



