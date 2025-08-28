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
@click.option("--num_frame", "-n", type=int, default=None, help="Number of frames to process from the beginning of the video")
@click.option("--assume_fixed_camera_pose", is_flag=True, help="Assume camera pose is fixed throughout the video (skips SLAM pose estimation)")
def infer(video: Path, output: Path, pipeline: str, visualize: bool, num_frame: int, assume_fixed_camera_pose: bool):
    """Run inference on a video file."""

    logger = configure_logging()

    # Create output directory based on video name
    video_name = video.stem  # Get filename without extension
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
    # Apply frame limit if specified
    if num_frame is not None:
        seek_range = range(0, num_frame)
        video_stream = ProcessedVideoStream(RawMp4Stream(video, seek_range=seek_range), []).cache(desc="Reading video stream")
        logger.info(f"Processing only first {num_frame} frames")
    else:
        video_stream = ProcessedVideoStream(RawMp4Stream(video), []).cache(desc="Reading video stream")
        logger.info(f"Processing all {len(video_stream)} frames")

    vipe_pipeline.run(video_stream)
    logger.info("Finished")


@click.command()
@click.argument("data_path", type=click.Path(exists=True, path_type=Path), default=Path.cwd() / "vipe_results")
@click.option("--port", "-p", default=20540, type=int, help="Port for the visualization server (default: 20540)")
def visualize(data_path: Path, port: int):
    run_viser(data_path, port)


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
