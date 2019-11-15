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
tdim=data.shape[3]

#TR and cutoff_time 
TR = sys.argv[2] #ms
TR=float(TR)
ctime=int(sys.argv[3] )


#fft and then scale the cutoff time to TR and filter low frequency values
for x in range(0,xdim):
    for y in range(0,ydim):
        for z in range(0,zdim):
            time_course=data[x,y,z,:]
            time_course_fft=np.fft.fft(time_course)
            t=time_course_fft
           
            #filter cutoff
            
            low_values= t < (1/ctime)*TR
            t[low_values]=0
         
            time_course_new=np.fft.ifft(t)
            
            data[x,y,z:]=time_course_new
            
new_image=nib.Nifti1Image(data, original_image.affine, original_image.header) # saving the new image data as nifti file
nib.save(new_image,sys.argv[4]+'.nii.gz')



