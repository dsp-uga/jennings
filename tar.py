import numpy as np
import tarfile
import cv2
import os
import sys



#https://stackoverflow.com/questions/2239655/how-can-files-be-added-to-a-tarfile-with-python-without-adding-the-directory-hi
#https://docs.python.org/2.4/lib/tar-examples.html

def generate_tar(dataset,filename_list,tar_file_name,extension=".png"):
	tar = tarfile.open(tar_file_name+".tar.gz", "w:gz")
	for i in range(0,dataset.shape[0]):
		entry = dataset[i]
		entry[entry==1] = 2
		print(dataset[i])
		cv2.imwrite(filename_list[i]+extension,entry)
		tar.addfile(tarfile.TarInfo(filename_list[i]+extension),open(filename_list[i]+extension))
		os.remove(filename_list[i]+extension)
	tar.close();

if __name__== "__main__":
	data = np.zeros((2,240,240,1))
	entry = ["data","data2"]
	generate_tar(data,entry,"submission")

