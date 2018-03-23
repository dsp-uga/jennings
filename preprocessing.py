import os;
import sys;

gcloud_trian_path  = "https://storage.googleapis.com/uga-dsp/project4/data/"
f = open(sys.argv[1])
for i in f:
	print(gcloud_trian_path+i.strip())