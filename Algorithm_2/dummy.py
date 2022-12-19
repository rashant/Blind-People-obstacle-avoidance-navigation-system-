lis=[[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,1,1],[1,0,0,0,1]]
import numpy as np
lis=np.array(lis)
r=lis[0:5,3:5]
l=lis[0:5,0:3]



print(np.count_nonzero(l),np.count_nonzero(r))

