# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import click
import hydra

from vipe import get_config_path, make_pipeline
from vipe.streams.base import ProcessedVideoStream
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.utils.logging import configure_logging
from vipe.utils.viser import run_viser


@click.command()
@click.argument("video", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory (default: current directory)",
    default=Path.cwd() / "vipe_results",
)
@click.option("--pipeline", "-p", default="default", help="Pipeline configuration to use (default: 'default')")
@click.option("--visualize", "-v", is_flag=True, help="Enable visualization of intermediate results")
@click.option("--start_frame", type=int, default=0, help="Starting frame number (default: 0)")
@click.option("--end_frame", type=int, default=None, help="Ending frame number (inclusive, default: process all frames)")
@click.option("--assume_fixed_camera_pose", is_flag=True, help="Assume camera pose is fixed throughout the video (skips SLAM pose estimation)")
def infer(video: Path, output: Path, pipeline: str, visualize: bool, start_frame: int, end_frame: int, assume_fixed_camera_pose: bool):
    """Run inference on a video file."""

    logger = configure_logging()

    # Create output directory based on video name
    video_name = video.stem  # Get filename without extension
    
    # Extract dataset directory from video path for better organization
    university_names = ["cmu", "fair", "georgiatech", "iiith", "indiana", "minnesota", "nus", "sfu", "unc", "uniandes", "upenn", "utokyo"]
    dataset_dir = None
    
    for parent in video.parents:
        parent_name_lower = parent.name.lower()
        for university in university_names:
            if university in parent_name_lower:
                dataset_dir = parent.name
                break
        if dataset_dir:
            break
    
    if dataset_dir:
        video_output_path = output / dataset_dir / video_name
    else:
        video_output_path = output / video_name
    
    overrides = [f"pipeline={pipeline}", f"pipeline.output.path={video_output_path}", "pipeline.output.save_artifacts=true"]
    if visualize:
        overrides.append("pipeline.output.save_viz=true")
        overrides.append("pipeline.slam.visualize=true")
    else:
        overrides.append("pipeline.output.save_viz=false")
    
    if assume_fixed_camera_pose:
        overrides.append("pipeline.assume_fixed_camera_pose=true")
        logger.info("Fixed camera pose mode enabled - SLAM pose estimation will be skipped")

    with hydra.initialize_config_dir(config_dir=str(get_config_path()), version_base=None):
        args = hydra.compose("default", overrides=overrides)

    logger.info(f"Processing {video}...")
    logger.info(f"Output will be saved to: {video_output_path}")
    vipe_pipeline = make_pipeline(args.pipeline)

    # Some input videos can be malformed, so we need to cache the videos to obtain correct number of frames.
    # Apply frame range if specified
    if end_frame is not None:
        seek_range = range(start_frame, end_frame + 1)  # +1 to make end_frame inclusive
        video_stream = ProcessedVideoStream(RawMp4Stream(video, seek_range=seek_range), []).cache(desc="Reading video stream")
        logger.info(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)")
    elif start_frame > 0:
        # If only start_frame is specified, process from start_frame to end
        video_stream = ProcessedVideoStream(RawMp4Stream(video), []).cache(desc="Reading video stream")
        total_frames = len(video_stream)
        seek_range = range(start_frame, total_frames)
        video_stream = ProcessedVideoStream(RawMp4Stream(video, seek_range=seek_range), []).cache(desc="Reading video stream")
        logger.info(f"Processing frames {start_frame} to {total_frames-1} ({total_frames - start_frame} frames)")
    else:
        video_stream = ProcessedVideoStream(RawMp4Stream(video), []).cache(desc="Reading video stream")
        logger.info(f"Processing all {len(video_stream)} frames (0 to {len(video_stream)-1})")

    vipe_pipeline.run(video_stream)
    logger.info("Finished")


@click.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path), default=Path.cwd() / "vipe_results")
@click.option("--port", "-p", default=20540, type=int, help="Port for the visualization server (default: 20540)")
@click.option("--use_mean_bg", is_flag=True, help="Use robust statistical mean background instead of standard background")
def visualize(data_path: Path, port: int, use_mean_bg: bool):
    run_viser(data_path, port, use_mean_bg)


@click.group()
@click.version_option()
def main():
    """NVIDIA Video Pose Engine (ViPE) CLI"""
    pass


# Add subcommands
main.add_command(infer)
main.add_command(visualize)


if __name__ == "__main__":
    main()
