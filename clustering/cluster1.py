import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import affine
import h5py
import argparse
import multiprocess
from multiprocess import Pool
from scipy import signal

parser = argparse.ArgumentParser()
parser.add_argument('--readfile', help = 'Of the format fft.h5')
parser.add_argument('--savefile')
parser.add_argument('--ntrajs')
# parser.add_argument('--savefile')

args = parser.parse_args()

FFT_WINDOW = 8

direc = '/projects/kumar-lab/mehtav/fft/'
# direc = ''

hfw = h5py.File(direc + args.savefile + '_fftfeat.h5', 'w')

hf = h5py.File(direc + args.readfile, 'r')
key_list = [key for key in hf.keys()]
print("#Trajs available: ", len(key_list))
assert int(args.ntrajs)<= len(key_list)
key_list = np.random.choice(key_list, int(args.ntrajs), replace = False)
fft_features = []
count = 0
for traj in range(len(key_list)):
	print(count, flush = True)
	count = count+1
	key = key_list[traj]
	traj_key = hf.get(key+ '/traj_key')[()] 
	pointtraj = hf.get(key+ '/points')[()]   # 240,7,2
	conftraj = hf.get(key+ '/confidence')[()]	
	angletraj = hf.get(key+ '/angles')[()]
	lengthtraj = hf.get(key+ '/lengths')[()]
	rangletraj = hf.get(key+ '/relative_angles')[()]
	veltraj = hf.get(key+ '/velocity')[()]
	# print(angletraj.shape)

	def job(time_series):
		sp = np.fft.fft(time_series)
		return (sp.real**2 + sp.imag**2).tolist()

	# def job(frame):
	# 	local_feat = []		# [14*16=224 dimensional for each frame]
	# 	for rangle in range(rangletraj.shape[1]):
	# 		time_series = rangletraj[frame-FFT_WINDOW: frame+FFT_WINDOW, rangle]
	# 		sp = np.fft.fft(time_series)
	# 		# print("TS", time_series.shape)
	# 		local_feat = local_feat + (sp.real**2 + sp.imag**2).tolist()
	# 	return local_feat

	feat = []
	p = Pool(processes = pointtraj.shape[0] - 2*FFT_WINDOW)
	framedata = [[rangletraj[frame-FFT_WINDOW: frame+FFT_WINDOW, rangle] for rangle in range(rangletraj.shape[1])] for frame in range(FFT_WINDOW, pointtraj.shape[0]-FFT_WINDOW)]
	feat = (p.map(job, framedata))
	p.close()


	# for frame in range(FFT_WINDOW, pointtraj.shape[0]-FFT_WINDOW):

	# 	# print(count, frame)
	# 	p = Pool(processes = rangletraj.shape[1])
	# 	local_feat = p.map(job, [rangletraj[frame-FFT_WINDOW: frame+FFT_WINDOW, rangle] for rangle in range(rangletraj.shape[1])])
	# 	p.close()

	# 	# local_feat = []		# [14*16=224 dimensional for each frame]
	# 	# for rangle in range(rangletraj.shape[1]):
	# 	# 	time_series = rangletraj[frame-FFT_WINDOW: frame+FFT_WINDOW, rangle]
	# 	# 	sp = np.fft.fft(time_series)
	# 	# 	# print("TS", time_series.shape)
	# 	# 	local_feat = local_feat + (sp.real**2 + sp.imag**2).tolist()
			
	# 		# f, Pxx_den = signal.periodogram(time_series)
	# 		# plt.semilogy(f, Pxx_den)
	# 		# print(np.fft.fftfreq(time_series.shape[-1]))

	# 		# x = np.fft.fftfreq(time_series.shape[-1])
	# 		# y = (sp.real**2 + sp.imag**2)
	# 		# y = y[x.argsort()]
	# 		# x.sort()
	# 		# x = x*30
	# 		# # print(x)
	# 		# # print(y)
	# 		# plt.plot(x, y)
	# 		# plt.scatter(x, y)
	# 		# plt.ylim((0, 0.15))
	# 		# plt.grid()
	# 		# plt.xlabel("Frequency (Hz)")
	# 		# plt.ylabel("Power")
	# 		# # plt.plot(time_series)
	# 		# plt.show()
	# 	# print(len(local_feat))
	# 	feat.append(local_feat)

	g = hfw.create_group(key)
	print(traj_key)
	g.create_dataset('traj_key', data = traj_key)
	g.create_dataset('points', data=pointtraj)
	g.create_dataset('confidence', data=conftraj)
	g.create_dataset('angles', data=angletraj)
	g.create_dataset('rangles', data=rangletraj)
	g.create_dataset('fft_features', data=feat)
	g.create_dataset('velocity', data=veltraj)

hfw.close()


