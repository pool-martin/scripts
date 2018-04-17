#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  get_dimensions.py
#  
#  Copyright 2018 Joao Paulo Martin <joao.paulo.pmartin@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; 
#  either version 2 of the License, or (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
#  PURPOSE.  See the GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth 
#  Floor, Boston, MA 02110-1301, USA.
#  
#  

''' 
generate_sliding_test_clips. 
- Authors: Joao Paulo Martin (joao.paulo.pmartin@gmail.com) 
''' 

dataRoot = '/data/torch/ltc/datasets/2kporn/';
targetTestDir = 'validation'
import os, sys, time, argparse
import math
from subprocess import call

def load_args():
	ap = argparse.ArgumentParser(description='Create test split.')
	ap.add_argument('-s', '--split_number',
					dest='split_number',
					help='split to be created.',
					type=int, required=False, default=1)
	ap.add_argument('-w', '--window',
					dest='w',
					help='Window to use.',
					type=int, required=False, default=80)
	ap.add_argument('-skip', '--skip',
					dest='skip',
					help='skip to use.',
					type=int, required=False, default=99)
	ap.add_argument('-st', '--split-type',
					dest='split_type',
					help='split type.',
					type=str, required=False, default="normal")
	args = ap.parse_args()

	
	print args
	
	return args

def get_dir_content(folder):
	video_patches = []
	for file in os.listdir(os.path.join(folder)):
		video_patches.append(file)
	return video_patches

def create_split(args):
    split = args.split_number
    W = args.w
    skip = args.skip
    testDir = os.path.join(dataRoot, 'splits', 'split' + str(split), 'validation');
    #testDir = os.path.join(dataRoot, 'rgb', 'jpg');
    # Loop over possible window sizes
    targetDir = os.path.join(dataRoot, 'splits', 'split' + str(split), targetTestDir + '_'+ str(W) + '_' + str(skip))
    
    classes = ['NonPorn','Porn']
    
    C = len(classes);
    
    for video_class in classes:
        print ('Class: ' + video_class);
        command = "mkdir -p " + os.path.join(targetDir, video_class)
        print command
        call(command, shell=True)
        videos = get_dir_content(os.path.join(testDir, video_class))
        for video in videos:
            print ('Class :' + video_class + ' - Video :' + video);
            #frames = get_dir_content(os.path.join(dataRoot, 'rgb', 'jpg', video));
            #totalDuration = len(frames); # note: nFlow = nRGB -1
            #"vPorn000157_00310_27197.mp4"
            video_splited = video.split('_')
            #print(splited)
            begin = int(video_splited[1])
            end   = int(video_splited[2].split('.')[0])
            extension = video_splited[2].split('.')[1]
            totalDuration = end - begin + 1
            
            nClips = int(math.ceil((totalDuration - W)/float(skip)) + 1)
            if(totalDuration < W):
                nClips = 1;
                command = "touch ''%s''" % (os.path.join(targetDir, video_class, video));
                print command
                call(command, shell=True)
            else:
                if(args.split_type == "normal"):
                    for tt in range(0,nClips):
                        n_begin = begin + tt * skip
                        n_end = n_begin + W -1
                        if n_end > end:
                            n_end = end
                            n_begin = n_end - W + 1
                        command = "touch ''%s''" % (os.path.join(targetDir, video_class, video_splited[0] + '_' + "%.05d" % (n_begin) + '_' + "%.05d" % (n_end) + '.' + extension));
                        print command
                        call(command, shell=True)
                elif(args.split_type == "minimal"):
                    command = "touch ''%s''" % (os.path.join(targetDir, video_class, video_splited[0] + '_' + "%.05d" % (begin) + '_' + "%.05d" % (begin + W - 1) + '.' + extension));
                    print command
                    call(command, shell=True)

def main():
    args = load_args()

    print '> Create test splits from videos -', time.asctime( time.localtime(time.time()) )

    create_split(args)

    print '\n> Create test splits  from videos done -', time.asctime( time.localtime(time.time()) )

    return 0


if __name__ == '__main__':
    main()
