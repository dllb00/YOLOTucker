import numpy as np
import tensorly as tl
from scipy.misc import face, imresize
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from math import ceil



# first 4 entries are not weight parameters, they're metadata
metadata = np.fromfile("tiny-yolo-voc.weights", count=4, dtype=np.int32)
print "Metadata: ", metadata



#prints 10 first weights from 5th entry, they are weight parameters written in float32
data = np.fromfile("tiny-yolo-voc.weights", dtype=np.int32, count=10) 
print "Data: ", data



#random data
random_state = 12345
image = tl.tensor(imresize(face(), 0.3), dtype='float64')

#Rank of the Tucker decomposition
tucker_rank = [100, 100, 2]


#tucker decomposition with random data, we should change this to use our own weights
core, tucker_factors = tucker(image, ranks=tucker_rank, init='random',
                                tol=10e-5, random_state=random_state)
tucker_reconstruction = tl.tucker_to_tensor(core, tucker_factors)


print "Core: ", core
print "tucker_factors: ", tucker_factors 

