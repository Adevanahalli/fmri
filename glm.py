import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import sys
import math
import scipy.stats
from scipy.stats import gamma
import matplotlib.pyplot as plt

image = nib.load(sys.argv[1]+'.nii.gz')
data =image.get_data()

print(data.shape)


# rdimensions of image
xdim=data.shape[0]
ydim=data.shape[1]
zdim=data.shape[2]
vols=data.shape[3]
tr=2.2
data2=np.zeros((xdim,ydim,zdim))
data2=np.zeros((xdim,ydim,zdim))
no_tasks=5


def x_signal(task_fname, tr, vols):
    task = np.loadtxt(task_fname)
    # Check that the file is plausibly a task file
    if task.ndim != 2 or task.shape[1] != 3:
        raise ValueError("Is {0} really a task file?", task_fname)
    # Convert onset, duration seconds to TRs
    task[:, :2] = task[:, :2] / tr
    ons_dur = task[:, :2]
    # Neural time course from onset, duration, amplitude for each event
    time_course = np.zeros(vols)
    for onset, duration, amplitude in task:
        # Make onset and duration integers
        onset = int(round(onset))
        duration = int(round(duration))
        time_course[onset:onset + duration] = amplitude
    return time_course
    
def hrf(times):  #double gamma hrf
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6
    
tr_times = np.arange(0, 30, tr)
hrf_signal = hrf(tr_times)
l=len(hrf_signal)-1
#print(l)

x=np.zeros((no_tasks,vols))
b=np.zeros((5,1))
y=np.zeros((vols,1))

x0=x_signal('/home/archana/assignments/assignment-6/subdata/covariates/left_f',tr,vols)
x1=x_signal('/home/archana/assignments/assignment-6/subdata/covariates/left_h',tr,vols)
x2=x_signal('/home/archana/assignments/assignment-6/subdata/covariates/right_f',tr,vols)
x3=x_signal('/home/archana/assignments/assignment-6/subdata/covariates/right_h',tr,vols)
x4=x_signal('/home/archana/assignments/assignment-6/subdata/covariates/tongue',tr,vols)



x=[x0,x1,x2,x3,x4] # x array
x=np.asarray(x)


for c in range(0,5):
    x[c]=np.convolve(x[c], hrf_signal) [:-l]# convolve with hrf signal
    #x[c]=x[c][:-l] #removing length of hrf signal convolved

x=np.transpose(x) # x is supposed to be vols * 5 dimension

# contrast for left_foot covariates/left_f 

c_t=[1,0,0,0,0]
c_t=np.asarray(c_t)
e = np.zeros(y.shape)
model = np.zeros(y.shape)
 
for i in range(0,xdim):
   for j in range(0,ydim):
        for k in range(0,zdim):
            y=data[i,j,k,:]
            tmp= np.linalg.inv(x.transpose().dot(x))
            tmp1= tmp.dot(x.transpose())
            b=tmp1.dot(y) # estimating beta vector
            model=x.dot(b)
            e=y-model
            se=np.std(e)
            tmp2=c_t.dot(tmp)
            tmp2=tmp2.dot(c_t.transpose())
            if se*math.sqrt(tmp2)==0:
                tstat=0
            else:
                tstat=c_t.dot(b)/(se*math.sqrt(tmp2))
           
           # zstat=(y1-mean)/sd*math.sqrt(vols)
            #print(i,j,k)
            #print(zstat)
            data2[i,j,k]=tstat
            #print(data2[i,j,k,:])
            
            
              
                    
mean=np.mean(data2)
std=np.std(data2)
data2=(data2-mean)/std
zstat=data2


p_values= scipy.stats.norm.sf(zstat)
#print(p_values)
p_values[p_values <0.05]=1
p_values[p_values >0.05]=0
data1=p_values
new_image=nib.Nifti1Image(data1,image.affine,image.header) # saving the new image data with activated voxels

zstat_image=nib.Nifti1Image(data2,image.affine,image.header) # saving the zstat image

nib.save(new_image,'activated2.nii.gz')
nib.save(zstat_image,'zstat2.nii.gz')  





