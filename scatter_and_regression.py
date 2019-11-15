import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import sys
import math
import scipy.stats
from scipy.stats import gamma
import matplotlib.pyplot as plt

fsl = nib.load('zstat_fsl.nii.gz')
zstat_fsl=fsl.get_data()

mycode=nib.load('zstat_mycode.nii.gz')
zstat=mycode.get_data()

#scatter plot of both zstat values                    
plt.scatter(zstat,zstat_fsl)
plt.show()

# regression of zstat values
#estimating coefficients

n=np.size(zstat)
mean_x=np.mean(zstat)
mean_y=np.mean(zstat_fsl)

# cross deviation and deviation about x

ss_xy=np.sum(zstat_fsl*zstat - n*mean_y*mean_x)
ss_xx=np.sum(zstat*zstat - n*mean_x*mean_x)

#coefficients

a=ss_xy/ss_xx
b=mean_y - a*mean_x

print('a: ',a,' b: ',b) 