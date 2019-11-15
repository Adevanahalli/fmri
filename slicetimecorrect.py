import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import sys


original_image = nib.load(sys.argv[1]+'.nii')
data = original_image.get_data()
data1=data #for new image
#print(data.shape)

# Read slice aquisiton file

acq_time=np.loadtxt(sys.argv[4]+'.txt',delimiter="\n")

# range of x and y coordinates of voxels
xdim=data.shape[0]
ydim=data.shape[1]

#TR and target time
TR = sys.argv[2] #ms
TR=float(TR)
target_time=int(sys.argv[3] )

nslices = data.shape[2] #number of slices
#print(nslices)

nvols= data.shape[-1] #number of volumes

vol_num = np.arange(data.shape[-1]) #array for creating volume acquisition time of every slice
vol_num = np.array(vol_num, dtype=np.float)
#print(vol_num)
vol_start_times= vol_num * TR

#target slice

target_slice= vol_start_times+target_time


    

for slice in range(0,nslices):
        slice_n=vol_start_times+acq_time[slice]      
        # get time series for each voxel and interpolate for each slice ( across all volumes for target time)
        for x in range(0,xdim) : 
            for y in range(0,ydim) :
                for t in range(1,nvols-1) :   # first and last volumes are constant
                        time_series_slice = data[x, y, slice, :]
                        target_x=float(target_slice[t])
                        x0=float(slice_n[t-1])
                        x1=float(slice_n[t])
                        y0=float(time_series_slice[t-1])
                        y1=float(time_series_slice[t])
                        #  linear interpolation formula
                        m=(y1 - y0) / (x1 - x0)
                        target_y = y0 + (target_x- x0) * m
                        data1[x,y,slice,t]=target_y #setting new value to  new image data
                        
new_image=nib.Nifti1Image(data1, original_image.affine, original_image.header) # saving the new image data as nifti file
nib.save(new_image,sys.argv[5]+'.nii.gz')



