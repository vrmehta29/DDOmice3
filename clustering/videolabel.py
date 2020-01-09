import numpy as np
import pickle
import h5py
import os
import numpy as np
import pandas as pd
import argparse
import pickle
from ffmpy import FFmpeg

FFMPEG_BIN = "/home/vkumar/Vinit/ffmpeg-3.3.3-64bit-static/ffmpeg"
# FFmpeg = FFmpeg()

parser = argparse.ArgumentParser()
parser.add_argument('--videofile', help = "eg. 7video.csv")
parser.add_argument('--videolabel', help = "eg. 7videolabels.csv")
parser.add_argument('--videodir', help = "directory having the actual videos")
parser.add_argument('--n_trajs')
parser.add_argument('--savedir', help = "directory to save the labled videos")
args = parser.parse_args()

videodata = pd.read_csv(args.videofile)
videolabel = pd.read_csv(args.videolabel)
videodf = videodata.merge(videolabel, left_on = "traj_key", right_on = "traj_key")
videodf = videodf.sample(n = int(args.n_trajs), random_state = 0)

labeled_metadata = pd.DataFrame({"file": [], "label": []})

for index, row in videodf.iterrows():
	filename = row['file']
	print(filename)
	fullfilename = os.path.join(args.videodir, row['file'])
	assert filename.endswith(".avi"), "Invalid video format, must be avi" 
	newfilename = os.path.join(args.savedir, filename.replace('/', '+'))
	

	# os.system("ffmpeg -i %s -ss $((%d+8)) -t %d -c:v mpeg4 -q 0 -vsync drop %s" % (fullfilename, row['start_frame']+8, row['duration']-16, newfilename))

	label_list = row['label'].split(",")[:-1]
	start_frame = row['start_frame']
	duration = 1
	for i in range(len(label_list)-1):
		if label_list[i+1]== label_list[i]:
			duration = duration+1
			continue
		else:
			newfilename = os.path.join(args.savedir, filename.replace('/', '+'))
			newfilename = newfilename[:-4] + "__%s.avi" % str(start_frame)
			if duration>= 10:
				# Cutting the video
				ff = FFmpeg(
					executable = FFMPEG_BIN,
				    inputs={fullfilename: "-y"},
			    	# outputs={newfilename: "-ss $((%d+8)) -t %d -c:v mpeg4 -q 0 -vsync drop" % (row['start_frame']+8, row['duration']-16)}
			    	outputs={newfilename: "-vf 'select=gte(n\,%d+8),setpts=PTS-STARTPTS' -vframes %d -c:v mpeg4 -q 0 -reset_timestamps 1" % (start_frame, duration)}
			    	)
				# Labeling the video
				ff2 = FFmpeg(
					executable = FFMPEG_BIN,
				    inputs={newfilename: "-y"},
			    	# outputs={newfilename: "-ss $((%d+8)) -t %d -c:v mpeg4 -q 0 -vsync drop" % (row['start_frame']+8, row['duration']-16)}
			    	outputs={newfilename[:-4] + "_labeled.avi": "-vf drawtext='fontfile=/home/vkumar/Vinit/arial.ttf:text=%s' -c:v mpeg4 -q 0" % label_list[i]}
			    	) 
				labeled_metadata= labeled_metadata.append(pd.DataFrame({"file": [newfilename[:-4] + "_labeled.avi"], "label": [label_list[i]]}))
				print(ff.cmd)
				ff.run(stdout=None, stderr=None)
				ff2.run()
			start_frame = row['start_frame']+ i+1
			duration = 1

# print(labeled_metadata)
for label in labeled_metadata['label'].unique():
	f= open(os.path.join(args.savedir, label+ ".txt"),"w+")
	for index, row in labeled_metadata.iterrows():
		if row['label'] == label:
			f.write("file %s\n" % row['file'])
	f.close()
	newfilename = os.path.join(args.savedir, label + ".avi")
	ff = FFmpeg(
					executable = FFMPEG_BIN,
				    inputs={os.path.join(args.savedir, label+ ".txt"): "-y -f concat -safe 0"},
			    	# outputs={newfilename: "-ss $((%d+8)) -t %d -c:v mpeg4 -q 0 -vsync drop" % (row['start_frame']+8, row['duration']-16)}
			    	outputs={newfilename: "-c copy"}
			    	)
	print(ff.cmd)
	ff.run()

 # ffmpeg -i /home/vkumar/Desktop/LabShareFolder/LL3-B2B/2016-01-12_BG/LL3-1_C3HeB.avi -vf 'select=gte(n\,20452+8),setpts=PTS-STARTPTS' -vframes 240 -c:v mpeg4 -q 0 -reset_timestamps 1 /home/vkumar/Vinit/labeled_videos/LL3-B2B+2016-01-12_BG+LL3-1_C3HeB_labeled.avi
