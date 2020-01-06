import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import affine
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--savefile')
args = parser.parse_args()

CONFIDENCE_THRESHOLD = 0.3

filelist = os.listdir('/projects/kumar-lab/mehtav/dataset/dataset')
filelist.sort()
filelist = filelist[1200:]

hf = h5py.File('/projects/kumar-lab/mehtav/final_data/'+ args.savefile + '_globalvvd.h5', 'w')

def checkMinConf(confarray):
	for conf in confarray:
		if conf<=CONFIDENCE_THRESHOLD:
			return False
	return True

# CENTER_SPINE_INDEX = 6
# BASE_TAIL_INDEX = 9
# BASE_NECK_INDEX = 3
NOSE = 0
LEFT_EAR = 1
RIGHT_EAR = 2
BASE_NECK_INDEX = 3
LEFT_FRONT_PAW = 4
RIGHT_FRONT_PAW = 5
CENTER_SPINE_INDEX = 6
LEFT_REAR_PAW = 7
RIGHT_REAR_PAW = 8
BASE_TAIL_INDEX = 9
MID_TAIL = 10
TIP_TAIL = 11	

# Taking moving average
def smoothen_traj(stride_points, window = 3):

	for frame_index in range(stride_points.shape[0]):
		if frame_index< window:
			continue
		stride_points[frame_index] = stride_points[frame_index-window: frame_index+ window].mean(axis= 0)
	return stride_points[window:]


def _normalize_points(stride_points):

	"""
	A normalization method that uses the stride's displacement
	vector and the animals body length in order to perform
	normalization
	"""

	frame_count, point_count, dim_count = stride_points.shape
	assert frame_count >= 2, 'cannot interpolate stride with fewer than two frames'
	assert point_count == 12, 'twelve points expected'
	assert dim_count == 2, '2D points expected'

	trans_stride_points = np.empty_like(stride_points.copy()).astype(np.float32)
	trans_stride_points_2 = np.empty((stride_points.shape[0], 7 , stride_points.shape[2]))

	for frame_index in range(frame_count):

		# center_spine_x = int(stride_points[frame_index, CENTER_SPINE_INDEX, 0].copy())
		# center_spine_y = int(stride_points[frame_index, CENTER_SPINE_INDEX, 1].copy())
		# base_tail_x = int(stride_points[frame_index, BASE_TAIL_INDEX, 0].copy())
		# base_tail_y = int(stride_points[frame_index, BASE_TAIL_INDEX, 1].copy())

		# x_diff = base_tail_x - center_spine_x
		# y_diff = base_tail_y - center_spine_y

		# the displacement vector is used to calculate a stride_theta
		# which we will use later to normalize rotation
		# stride_theta = math.atan2(y_diff, x_diff)

		# rot_mat = affine.Affine.rotation(-math.degrees(stride_theta))
		# scale_mat = affine.Affine.scale(CM_PER_PIXEL / body_len_cm, CM_PER_PIXEL / body_len_cm)

		# calculate the transformation for this frame, shifting the origin to center spine
		curr_offset_x = -float(stride_points[frame_index, BASE_TAIL_INDEX, 0])
		curr_offset_y = -float(stride_points[frame_index, BASE_TAIL_INDEX, 1])
		translate_mat = affine.Affine.translation(curr_offset_x, curr_offset_y)

		transform_mat = translate_mat

		# apply the transformation to each frame
		for point_index in range(12):
			curr_pt_xy = np.array([float(stride_points[frame_index, point_index, 0]), float(stride_points[frame_index, point_index, 1])])
			trans_stride_points[frame_index, point_index] = transform_mat * curr_pt_xy

		x = trans_stride_points[frame_index]
		# y = trans_stride_points[frame_index][:, 1]
		coords = np.array([[(x[RIGHT_EAR]+ x[LEFT_EAR]+ x[NOSE])/3, x[BASE_NECK_INDEX], x[CENTER_SPINE_INDEX], x[RIGHT_REAR_PAW], x[LEFT_REAR_PAW], x[MID_TAIL], x[TIP_TAIL]  ]])
		trans_stride_points_2[frame_index] = coords

	return trans_stride_points_2


count=0
for filename in filelist:

	pointsdata = []
	confdata = []
	print(count)
	count=count+1
	f = h5py.File('/projects/kumar-lab/mehtav/dataset/dataset/'+filename, 'r')
	group = f['poseest']
	pointtraj = group['points'][()]
	conftraj = group['confidence'][()]
	temppointsdata = []
	tempconfdata = []
	for i in range(conftraj.shape[0]):
		valid = checkMinConf(conftraj[i])
		if valid:
			temppointsdata.append(pointtraj[i])
			tempconfdata.append(conftraj[i])
		if valid and len(temppointsdata)>=243:
			pointsdata.append(np.array(temppointsdata))
			confdata.append(np.array(tempconfdata))
			temppointsdata = []
			tempconfdata = []
		if not valid and len(temppointsdata)>=243:
			pointsdata.append(np.array(temppointsdata))
			confdata.append(np.array(tempconfdata))
		if not valid:
			temppointsdata = []
			tempconfdata = []

	if len(pointsdata)!=0:
		for i in range(len(pointsdata)):
			pointsdata[i] = _normalize_points(pointsdata[i].copy())
			pointsdata[i] = smoothen_traj(pointsdata[i].copy())

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
				currxcoord = float(points[j][k][0].copy())
				nextxcoord = float(points[j+1][k][0].copy())
				currycoord = float(points[j][k][1].copy())
				nextycoord = float(points[j+1][k][1].copy())
				xdiff = (nextxcoord - currxcoord)*PX_TO_CM
				ydiff = (nextycoord - currycoord)*PX_TO_CM
				# The units for 
				xvel = 30*xdiff		# Multiplying with 30 because the video is 30 fps
				yvel = 30*ydiff		# Multiplying with 30 because the video is 30 fps
				# vel[j,k] = np.sqrt(np.power(xvel,2) + np.power(yvel,2) )
				velvector[j,k,0] = xvel
				velvector[j,k,1] = yvel
		# velocitydata.append(vel)
		velocityvectordata.append(velvector)
	
	datalist = [pointsdata, confdata, velocityvectordata]
	
	for i in range(len(pointsdata)):
		g = hf.create_group(filename[:-3] + '_' + str(i), 'w')
		# hf = h5py.File('/gpfs/ctgs0/home/c-mehtav/DDOmice2/temp/'+ filename[:-3] + str(i) + '_vvd.h5', 'w')
		g.create_dataset('points', data=pointsdata[i])
		g.create_dataset('confidence', data=confdata[i])
		g.create_dataset('velocity', data=velocityvectordata[i])

hf.close()

	# import pickle
	# with open('/projects/kumar-lab/mehtav/normalised_vd_tail_2/' + filename[:-3] + '_vvd.pkl', 'wb') as f:
	# 	pickle.dump(datalist, f)

# with open('/projects/kumar-lab/mehtav/velocityvectordata.pkl', 'rb') as f:
# 	velocityvectordata = pickle.load(f)


#Plot confidence scores
# plt.figure()
# plt.plot(confdata[1][:,0])
# plt.xlabel('Time')
# plt.ylabel('Confidence')
# plt.savefig('/home/c-mehtav/fig2.png')

