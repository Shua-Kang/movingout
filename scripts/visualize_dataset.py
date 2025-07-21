#!/usr/bin/env python3
"""
Dataset to Video Converter

This script loads trajectories from a Hugging Face dataset and either:
1. Generates an MP4 video file (if mode is 'video')
2. Displays the trajectory in real-time (if mode is 'human')

Usage:
    python dataset_to_video.py <hugging_face_dataset_name> <map_name> <traj_id> <video_or_human>

Arguments:
    hugging_face_dataset_name: Name of the Hugging Face dataset to load
    map_name: Name of the map for the environment
    traj_id: Trajectory ID to render
    video_or_human: Either 'video' to save MP4 or 'human' to display directly
"""

import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np
from datasets import load_dataset
from moving_out.benchmarks.moving_out import MovingOutEnv


def collect_rgb_frames(env: MovingOutEnv, trajectory_data: List) -> List[np.ndarray]:
    """
    Collect RGB frames from trajectory data.
    
    Args:
        env: MovingOut environment instance
        trajectory_data: List of (step_id, state, action) tuples
        
    Returns:
        List of RGB frames as numpy arrays
    """
    frames = []
    
    for i, state, _ in trajectory_data:
        # Update environment state
        state_dict = {"states": state}
        env.update_env_by_given_state(state_dict)
        
        # Render and collect frame
        frame = env.render("rgb_array")
        if frame is not None:
            frames.append(frame)
    
    return frames


def save_frames_as_mp4(frames: List[np.ndarray], output_path: str, fps: int = 30) -> None:
    """
    Save RGB frames as MP4 video.
    
    Args:
        frames: List of RGB frames
        output_path: Output video file path
        fps: Frames per second for the video
    """
    if not frames:
        print("No frames to save!")
        return
    
    # Get frame dimensions
    height, width, channels = frames[0].shape
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
        
        print(f"Video saved successfully: {output_path}")
        
    except Exception as e:
        print(f"Error saving video: {e}")
    finally:
        out.release()


def display_trajectory_human(env: MovingOutEnv, trajectory_data: List, delay: float = 0.1) -> None:
    """
    Display trajectory in human-readable format with real-time rendering.
    
    Args:
        env: MovingOut environment instance
        trajectory_data: List of (step_id, state, action) tuples
        delay: Delay between frames in seconds
    """
    print("Displaying trajectory in human mode. Press 'q' to quit early.")
    
    for i, state, _ in trajectory_data:
        # Update environment state
        state_dict = {"states": state}
        env.update_env_by_given_state(state_dict)
        
        # Render in human mode
        env.render("human")
        
        # Add delay to control playback speed
        time.sleep(delay)
        
        # Check for quit signal (this is environment-dependent)
        # You might need to modify this based on your environment's event handling


def main():
    """Main function to parse arguments and execute the appropriate rendering mode."""
    parser = argparse.ArgumentParser(description="Convert dataset trajectories to video or display them")
    parser.add_argument("--hugging_face_dataset_name","-f", type=str, 
                       help="Name of the Hugging Face dataset to load")
    parser.add_argument("--map_name","-m", type=str, 
                       help="Name of the map for the environment")
    parser.add_argument("--traj_id","-t", type=int, 
                       help="Trajectory ID to render")
    parser.add_argument("--video_or_human","-v", type=str, choices=['video', 'human'],
                       help="Rendering mode: 'video' to save MP4 or 'human' to display")
    parser.add_argument("--fps", type=int, default=10,
                       help="Frames per second for video output (default: 30)")
    parser.add_argument("--delay", type=float, default=0.03,
                       help="Delay between frames in human mode (default: 0.1s)")
    
    args = parser.parse_args()
    
    try:
        # Load dataset
        print(f"Loading dataset: {args.hugging_face_dataset_name}")
        ds = load_dataset(args.hugging_face_dataset_name)
        
        # Search for matching trajectory
        trajectory_entry = None
        for entry in ds["all_data"]:
            if entry["map_name"] == args.map_name and entry["trajectory_id"] == args.traj_id:
                trajectory_entry = entry
                break
        
        if trajectory_entry is None:
            print(f"Error: No trajectory found with map_name='{args.map_name}' and trajectory_id={args.traj_id}")
            print("Available combinations:")
            # Show some examples
            seen_combinations = set()
            for entry in ds["all_data"][:10]:  # Show first 10 combinations
                combo = (entry["map_name"], entry["trajectory_id"])
                if combo not in seen_combinations:
                    print(f"  map_name='{entry['map_name']}', trajectory_id={entry['trajectory_id']}")
                    seen_combinations.add(combo)
            if len(ds["all_data"]) > 10:
                print(f"  ... and {len(ds['all_data']) - 10} more entries")
            return
        
        actual_map_name = args.map_name
        
        # Parse trajectory data
        trajectory_data = json.loads(trajectory_entry["steps_data"])
        print(f"Loaded trajectory {args.traj_id} with {len(trajectory_data)} steps on map '{actual_map_name}'")
        
        # Create environment
        env = MovingOutEnv(map_name=actual_map_name)
        
        if args.video_or_human == "video":
            # Generate video
            print("Collecting frames for video generation...")
            frames = collect_rgb_frames(env, trajectory_data)
            
            if frames:
                # Create output filename
                video_filename = f"{args.hugging_face_dataset_name.replace('/', '_')}_{actual_map_name}_{args.traj_id}.mp4"
                video_path = os.path.join(os.getcwd(), video_filename)
                
                print(f"Saving video to: {video_path}")
                save_frames_as_mp4(frames, video_path, args.fps)
            else:
                print("No frames collected. Check your trajectory data.")
                
        elif args.video_or_human == "human":
            # Display in human mode
            print("Displaying trajectory in human mode...")
            display_trajectory_human(env, trajectory_data, args.delay)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 