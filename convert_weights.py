import numpy as np
import tensorly as tl
from scipy.misc import face, imresize
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from math import ceil


metadata = np.fromfile("tiny-yolo-voc.weights", count=4, dtype=np.int32)
print "Metadata: ", metadata

W = 10 #width
H = 10 #height
D = 10 #depth, number of featuremaps
input_tensor = np.zeros((W,H,D)) #input_tensor with random numbers
for w in xrange(W):
	for h in xrange(H):
		for d in xrange(D):
			#temp = np.fromfile("tiny-yolo-voc.weights", dtype=np.int32, count=1) 	
			input_tensor[w,h,d] = (w + h + d) % 100
print "Input tensor shape: ", input_tensor.shape

#rank of the Tucker decomposition
tucker_rank = [100, 100, 2] # what rank you want to get after tucker


#creates the core and factor matrices
core, tucker_factors = tucker(input_tensor, ranks=tucker_rank)
tucker_reconstruction = tl.tucker_to_tensor(core, tucker_factors)

print "Core size: ", core.shape
#print "Core: ", core
#print "tucker_factors: ", tucker_factors 

