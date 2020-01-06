#!/usr/bin/env python
import numpy as np
import copy
import os
import pickle
	
CENTER_SPINE_INDEX = 6
filelist = os.listdir('/projects/kumar-lab/mehtav/normalised_vd')

count = 0
for file in filelist[550:]:
	data = []
	with open('/projects/kumar-lab/mehtav/normalised_vd/'+file, 'rb') as f:
		f.seek(0)
		data = pickle.load(f)

	# data = [points/conf/vel, traj_number]
	# Each traj has shape (nrow x 12 x 2) or (nrow by 12)	
	print(count)
	count = count+1
	for t in range(len(data[1])):
		p_traj = data[0][t]
		c_traj = data[1][t]
		v_traj = data[2][t]
		op_traj = []
		for i in range(v_traj.shape[0]):
			s = np.ravel(np.delete(p_traj[i], CENTER_SPINE_INDEX, 0)) # Removing center spine, as it is always at the origin
			a = np.ravel(np.delete(v_traj[i], CENTER_SPINE_INDEX, 0)) # Removing center spine, as it is always at rest	
			s = np.delete(s, 17, 0) # Removing x coordinate of base tail
			a = np.delete(a, 17, 0) # Removing x coordinate of base tail
			op_traj.append((s, a))
		with open('/projects/kumar-lab/mehtav/sa_traj/' + file[:-23] + '_' + str(t) + '_sa.pkl', 'wb') as f:
			pickle.dump(op_traj, f)


