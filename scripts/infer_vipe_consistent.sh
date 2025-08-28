#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES=7

# Configuration variables
BASE_PATH="/home/nas4_user/kinamkim/DATA/Ego4D/data"
SEQUENCE="cmu_bike01_2"
CAMERA="cam02"
VIDEO_PATH="${BASE_PATH}/${SEQUENCE}/frame_aligned_videos/downscaled/448/${CAMERA}.mp4"
NUM_FRAMES=300

echo "Processing video: $VIDEO_PATH"
echo "Number of frames: $NUM_FRAMES"

# Run ViPE inference
vipe infer "$VIDEO_PATH" \
    --num_frame $NUM_FRAMES \
    --assume_fixed_camera_pose

# vipe infer /home/nas4_user/taewoongkang/repos_4_students/dohyeon/Exo-to-Ego/Geo4D/data/demo/drift-turn.mp4\
#     --assume_fixed_camera_pose