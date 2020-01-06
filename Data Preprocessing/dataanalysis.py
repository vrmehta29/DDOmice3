import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
CONFIDENCE_THRESHOLD = 0.3

filelist = os.listdir('/projects/kumar-lab/mehtav/dataset/dataset')

# filelist = filelist[:2]

def checkMinConf(confarray):
	for conf in confarray:
		if conf<=CONFIDENCE_THRESHOLD:
			return False
	return True

count=0
for filename in filelist:
	pointsdata = []
	confdata = []
	print(count)
	count=count+1
	f = h5py.File('/projects/kumar-lab/mehtav/dataset/dataset/'+filename, 'r')
	group = f['poseest']
	pointtraj = group['points'].value
	conftraj = group['confidence'].value
	temppointsdata = []
	tempconfdata = []
	for i in range(conftraj.shape[0]):
		valid = checkMinConf(conftraj[i])
		if valid:
			temppointsdata.append(pointtraj[i])
			tempconfdata.append(conftraj[i])
		if not valid and len(temppointsdata)>50:
			pointsdata.append(np.array(temppointsdata))
			confdata.append(np.array(tempconfdata))
		if not valid:
			temppointsdata = []
			tempconfdata = []

	# velocitydata = []
	PX_TO_CM = 19.5 * 2.54 / 400
	velocityvectordata = []

	for i in range(len(pointsdata)):
		# print(i)
		points = pointsdata[i].copy()
		conf = confdata[i].copy()
		# vel = points[:-1, :, 0] # Removing the last point as velocity cannot be calculated for it, and reducing the vector component
		velvector = points[:-1, :, :]
		# Iterating over time
		for j in range(points.shape[0]-1):
			# Iterating over body parts
			for k in range(points.shape[1]):
				currxcoord = int(points[j][k][0])
				nextxcoord = int(points[j+1][k][0])
				currycoord = int(points[j][k][1])
				nextycoord = int(points[j+1][k][1])
				xdiff = abs(nextxcoord - currxcoord)*PX_TO_CM
				ydiff = abs(nextycoord - currycoord)*PX_TO_CM
				# The units for 
				xvel = 30*xdiff		# Multiplying with 30 because the video is 30 fps
				yvel = 30*ydiff		# Multiplying with 30 because the video is 30 fps
				# vel[j,k] = np.sqrt(np.power(xvel,2) + np.power(yvel,2) )
				velvector[j,k,0] = xvel
				velvector[j,k,1] = yvel
		# velocitydata.append(vel)
		velocityvectordata.append(velvector)
	
	datalist = [pointsdata, confdata, velocityvectordata]
	import pickle
	with open('/projects/kumar-lab/mehtav/velocitydata/' + filename[:-3] + '_velocityvectordata.pkl', 'wb') as f:
		pickle.dump(velocityvectordata, f)

# with open('/projects/kumar-lab/mehtav/velocityvectordata.pkl', 'rb') as f:
# 	velocityvectordata = pickle.load(f)


#Plot confidence scores
# plt.figure()
# plt.plot(confdata[1][:,0])
# plt.xlabel('Time')
# plt.ylabel('Confidence')
# plt.savefig('/home/c-mehtav/fig2.png')

