import h5py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
import affine
import h5py
CONFIDENCE_THRESHOLD = 0.3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--savefile')
parser.add_argument('--length_traj')
args = parser.parse_args()
l_traj = int(args.length_traj)

hf = h5py.File('/projects/kumar-lab/mehtav/fft/'+ args.savefile + '_fft.h5', 'w')

filelist = os.listdir('/projects/kumar-lab/mehtav/dataset/dataset')
filelist.sort()
# filelist = filelist

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

	trans_stride_points = np.empty_like(stride_points.copy()).astype(np.int64)
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
		curr_offset_x = -int(stride_points[frame_index, BASE_TAIL_INDEX, 0])
		curr_offset_y = -int(stride_points[frame_index, BASE_TAIL_INDEX, 1])
		translate_mat = affine.Affine.translation(curr_offset_x, curr_offset_y)

		transform_mat = translate_mat

		# apply the transformation to each frame
		for point_index in range(12):
			curr_pt_xy = np.array([int(stride_points[frame_index, point_index, 0]), int(stride_points[frame_index, point_index, 1])])
			trans_stride_points[frame_index, point_index] = transform_mat * curr_pt_xy

	return trans_stride_points


count=0
angledata = []
meta = pd.read_csv("/projects/kumar-lab/MergedMetaList_2019-12-23.tsv", delimiter="\t")[["NetworkFilename", "Strain"]]

filelistdf = pd.DataFrame({"filename": [], "filename_orig": []})
for i, filename in enumerate(filelist):
	filename_mod = filename.replace("+", "/")[:-15] + ".avi"
	filelistdf.loc[i] = ([filename_mod, filename])

filelist_strain = filelistdf.merge(meta, left_on = "filename", right_on = "NetworkFilename")
strains = filelist_strain["Strain"].unique()

filelist_sampled = []
N_FROM_EACH_CLUSTER = 10
for strain in strains:
	df = filelist_strain[filelist_strain["Strain"] == strain]
	filelist_sampled = filelist_sampled + df['filename_orig'].sample(n= N_FROM_EACH_CLUSTER, random_state=0).tolist()


videodf = pd.DataFrame({"traj_key": [], "file": [], "start_frame": [], "duration": []})
traj_key = 0 	#Will start from 1 as it i updated before writing to df later

for filename in filelist_sampled:

	videofile = filename.replace("+", "/")[:-15] + ".avi"
	pointsdata = []
	confdata = []
	traj_key_list = []
	print(count)
	count=count+1

	# Sample a strain from the list of strains

	f = h5py.File('/projects/kumar-lab/mehtav/dataset/dataset/'+ filename, 'r')
	
	group = f['poseest']
	pointtraj = group['points'][()]
	conftraj = group['confidence'][()]
	temppointsdata = []
	tempconfdata = []
	start_pointer = 0
	for i in range(conftraj.shape[0]):
		valid = checkMinConf(conftraj[i])
		if valid:
			temppointsdata.append(pointtraj[i])
			tempconfdata.append(conftraj[i])
		if len(temppointsdata)>=l_traj+3:
			pointsdata.append(np.array(temppointsdata))
			confdata.append(np.array(tempconfdata))
			traj_key = traj_key+1
			traj_key_list.append(traj_key)
			print(videodf)
			videodf = videodf.append(pd.DataFrame({"traj_key": [traj_key],"file": [videofile],"start_frame": [start_pointer],"duration": [l_traj]}))
			# videometadata.append([videofile, start_pointer, l_traj])
			temppointsdata = []
			tempconfdata = []
			start_pointer = i+1
		if not valid:
			temppointsdata = []
			tempconfdata = []
			start_pointer = i+1

	# velocitydata = []
	PX_TO_CM = 19.5 * 2.54 / 400

	for i in range(len(pointsdata)):
		
		# print(count, i)
		pointsdata[i] = _normalize_points(pointsdata[i].copy())
		pointsdata[i] = smoothen_traj(pointsdata[i].copy())
		points = pointsdata[i].copy()
		conf = confdata[i].copy()

		frame_angle_data = []
		relative_angles_data = []
		length_data = []
		veldata = []
		# Iterating over time
		for j in range(points.shape[0]-1):

			# Lists of tuples having (x,y) for all 12 points
			curr_points = []

			velocities = []
			for k in range(points.shape[1]):
				currxcoord = float(points[j][k][0])
				currycoord = float(points[j][k][1])
				curr_points.append([currxcoord, currycoord])

				nextxcoord = float(points[j+1][k][0].copy())
				nextycoord = float(points[j+1][k][1].copy())
				xdiff = (nextxcoord - currxcoord)*PX_TO_CM
				ydiff = (nextycoord - currycoord)*PX_TO_CM
				# The units for 
				xvel = 30*xdiff		# Multiplying with 30 because the video is 30 fps
				yvel = 30*ydiff		# Multiplying with 30 because the video is 30 fps
				# vel[j,k] = np.sqrt(np.power(xvel,2) + np.power(yvel,2) )
				velocities.append((xvel, yvel))
			veldata.append(velocities)

			links = []
			links.append( [[0, 0], curr_points[CENTER_SPINE_INDEX]] )
			links.append( [curr_points[CENTER_SPINE_INDEX],  curr_points[BASE_NECK_INDEX]])
			links.append( [curr_points[BASE_NECK_INDEX], curr_points[ int((NOSE + LEFT_EAR + RIGHT_EAR)/3) ]] )
			links.append( [[0, 0], curr_points[LEFT_REAR_PAW]])
			links.append( [[0, 0], curr_points[RIGHT_REAR_PAW]])
			links.append( [[0, 0], curr_points[MID_TAIL]])
			links.append( [curr_points[MID_TAIL], curr_points[TIP_TAIL]])

			ind_map = {'x':0, 'y':1}
			thetas = []
			lengths = []
			for link in links:
				thetas.append(math.atan2(link[0][ind_map['y']] - link[1][ind_map['y']], link[0][ind_map['x']] - link[1][ind_map['x']]))
				lengths.append( np.sqrt((link[0][ind_map['y']] - link[1][ind_map['y']])**2 + (link[0][ind_map['x']] - link[1][ind_map['x']])**2))

			r_thetas = []
			r_thetas.append(np.pi + thetas[1] - thetas[0])
			r_thetas.append(np.pi + thetas[2] - thetas[1])
			r_thetas.append(np.pi + thetas[3] - thetas[0])
			r_thetas.append(np.pi + thetas[4] - thetas[0])
			r_thetas.append(np.pi + thetas[5] - thetas[0])
			r_thetas.append(np.pi + thetas[6] - thetas[5])

			frame_angle_data.append(thetas)
			length_data.append(lengths)
			relative_angles_data.append(r_thetas)

		# angledata.append(frame_angle_data)
			
		g = hf.create_group(filename[:-3] + "_" + str(i))
		g.create_dataset('traj_key', data=traj_key_list[i])
		g.create_dataset('points', data=pointsdata[i])
		g.create_dataset('confidence', data=confdata[i])
		g.create_dataset('angles', data=frame_angle_data)
		g.create_dataset('lengths', data=length_data)
		g.create_dataset('relative_angles', data=relative_angles_data)
		g.create_dataset('velocity', data=veldata)

videodf.to_csv('/projects/kumar-lab/mehtav/fft/'+ args.savefile + 'video.csv')
hf.close()

