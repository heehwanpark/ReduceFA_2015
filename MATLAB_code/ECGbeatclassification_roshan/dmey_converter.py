import h5py
import pywt
import numpy as np

def extractFeatureFromWave(wave):
    cA, cD = pywt.dwt(wave, 'dmey')
    counter = 1
    while counter <= 4:
        print cA.shape
        cA, cD = pywt.dwt(cA, 'dmey')
        counter = counter + 1
    return cA, cD

h5file = h5py.File('C:\Users\heehwan\workspace\Data\MIT_BIH\wholepeaks.h5','r')
peaks = h5file['/peaks']
peak_set = np.array(peaks)
h5file.close()

peak_set = np.transpose(peak_set)

qrs_num = peak_set.shape[0]

input = peak_set[1,:]
cA_4, cD_4, cD_3, cD_2, cD_1 = pywt.wavedec(input, 'dmey', level=4)
print input.shape
print cA_4.shape
print cD_4.shape
print cD_3.shape
print cD_2.shape
print cD_1.shape
# cA, cD = extractFeatureFromWave(peak_set[1,:])

# feature_array = np.zeros((peak_set.shape[0], 130))
# for i in xrange(qrs_num):
#     cA, cD = extractFeatureFromWave(peak_set[i,:])
#     raw_feature = np.append(cA, cD)
#     feature_array[i,:] = raw_feature
#
# print(feature_array[qrs_num-1,:])
#
# resultfile = h5py.File('C:\Users\heehwan\workspace\Data\MIT_BIH\wholefeatures.h5','w')
# resultfile.create_dataset('features', data=feature_array)
# resultfile.close()
