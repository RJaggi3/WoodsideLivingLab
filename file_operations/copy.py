import shutil
import os 
from nptdms import TdmsFile

desination = r"decimated_file"
src = r"tdms_files\\202305282340_SHM-6.tdms"

def copy(src,desc):
    path = shutil.copy2(src,desc) 
    tdms_file = TdmsFile.read(path)
