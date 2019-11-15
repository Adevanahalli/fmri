from nifti import *
import sys

filename=sys.argv[1]
x=sys.argv[2]
y=sys.argv[3]
z=sys.argv[4]

nim = NiftiImage(filename)

x=int(x)
y=int(y)
z=int(z)

xdim,ydim,zdim=nim.getVolumeExtent()

if x<xdim and y<ydim and z<zdim :

    t= nim.data[:,z,y,x]

    for i in t:
        print i,
else :
    print ('voxel coordinates out of range')
