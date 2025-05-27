import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

#file path
dir_str = r'C:\Users\mrste\Documents\Work Folder\University files\Ph.D Files\LiF\Data'

#where to save files
output_dir_str =  r'C:\Users\mrste\Documents\Work Folder\University files\Ph.D Files\LiF\Spectroscopy Output'

exp_name = 'LiF mKSpec_run2'

#hdf5 file containing power scan data
f = h5py.File(dir_str+'/'+exp_name+'.hdf5','r')

#number of traces per sweep (i.e number of fcent's)
N_traces = 13

mode_i = 14
dset=f[str(mode_i)]
npoints = int(dset.attrs['npoints'])

npsavefolder = r'C:\Users\mrste\Documents\Work Folder\University files\Ph.D Files\LiF\Data\numpy data'

for mode_no in range(1,N_traces):
    #initalize an array of correct dimensions

    dimension=np.zeros(N_traces)

    for i in range(N_traces):
        for trace in f.keys():
            j = int(trace)
            if ((j-i)%N_traces)==0: #window 1
                dimension[i]+=1
    
    mag = np.zeros((int(min(dimension)),npoints))
    Xn = []
    for j in range(mag.shape[0]):
            trace = str(mode_no + j*N_traces)
            dset = f[trace]
            span = int(dset.attrs['fspan'])
            fcent = int(dset.attrs['fcent'])
            f_start = fcent-span//2
            f_end = fcent+span//2
            npoints = int(dset.attrs['npoints'])
            fn=np.linspace(f_start,f_end,npoints)
            mag_data = np.sqrt(dset[:,0]**2+dset[:,1]**2)

            X = float(dset.attrs['b'])
            Y = fn
            Z = np.log10(mag_data)
            Xn = np.append(Xn,X)
            mag[j] = Z

    np.save(npsavefolder + '/s21dbdata_m' + str(mode_no) + '.npy', mag)
    np.save(npsavefolder + '/bdata_m' + str(mode_no) + '.npy', Y)
    np.save(npsavefolder + '/fdata_m' + str(mode_no) + '.npy', Xn)