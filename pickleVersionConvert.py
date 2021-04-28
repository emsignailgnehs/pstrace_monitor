import pickle
import compress_pickle
import pickle5
import gzip
"""
load the compressed pickle file and decompress,
then use pickle5 to convert it back to lower verison.
from version 5 to version 4.
"""


f =  r"C:\Users\hui\Desktop\20210427_CapCadTraining.picklez"


tosave =  r"C:\Users\hui\Desktop\saved.picklez"



with open(f,'rb') as p:
    data = p.read()

dec = gzip.decompress(data)
data = pickle5.loads(dec)
 
 
with open(tosave,'wb') as p:
    compress_pickle.dump(data,p,compression='gzip')

