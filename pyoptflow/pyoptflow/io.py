from pathlib import Path
import glob
def getimgfiles(stem):
    print(stem)
    flist = glob.glob(stem+"/*.png")
    flist.sort()

    return flist,".png"
