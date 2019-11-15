import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import sys
import math


original_image = nib.load(sys.argv[1]+'.nii')
data = original_image.get_data()
data1=data #for new image
print(data.shape)


# range of x and y coordinates of voxels
xdim=data.shape[0]
ydim=data.shape[1]
zdim=data.shape[2]
vdim=data.shape[3]

#TR and FWHM
FWHM= sys.argv[2] 
FWHM=float(FWHM)

#calculate sigmas for x,y,zdim

sx=FWHM/(2.35*3)
sy=FWHM/(2.35*3)
sz=FWHM/(2.35*3.33)

# compute gaussian kernel
k=np.zeros((6,6,6))

#reduce kernel size to get error less than 0.001. So kernel size is from 3 in both directions so 6*6*6
for x in range(-3,3):
    for y in range(-3,3):
        for z in range(-3,3):
            s=1 / np.sqrt(2 * np.pi) 
            k[x,y,z]=s* np.exp(-((x ** 2 / sx ** 2)+(y ** 2 / sy ** 2)+(z** 2 / sz ** 2)))

# convolution with gaussian kernel

for v in range(0,vdim):
    t=np.pad(data[:,:,:,v],((3,2),(3,2),(3,2)), 'constant') #padding edges 
    for xc in range(3,xdim):
        for yc in range(3,ydim):
            for zc in range(3,zdim):
                sum=0
                i=t[xc-3:xc+3,yc-3:yc+3,zc-3:zc+3] #extracting 6*6*6 matrix for convolution
                p=i*k
                sum=np.sum(p)
                data1[xc,yc,zc,v]=sum
            
new_image=nib.Nifti1Image(data1, original_image.affine, original_image.header) # saving the new image data as nifti file
nib.save(new_image,sys.argv[3]+'.nii.gz')

