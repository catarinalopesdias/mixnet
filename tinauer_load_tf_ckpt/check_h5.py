import h5py

f = h5py.File('test.weights.h5', mode='r')
print(f['layers/conv3d_transpose/vars/0'])
