import json
import os
import copy
import argparse
from tqdm import tqdm

def process_a2i_data(
    input_dir: str,
    his: int,
    task_name_for_output: str,
    resolution: int,
    output_dir: str
):
    """
    Processes Libero robot trajectory data to create action-to-image (a2i) conversational datasets.

    This script automatically discovers task folders within the `input_dir`. The goal is to
    predict the next image given a history of images and actions.

    Expected Directory Structure:
    input_dir/
    ├── TASK_NAME_1/
    │   ├── trajectory_0/
    │   │   ├── abs_action/
    │   │   │   ├── action_0/
    │   │   │   │   └── 0.npy
    │   │   │   └── action_1/
    │   │   │       └── 0.npy
    │   │   └── wrist_image/
    │   │       ├── image_0.png
    │   │       └── image_1.png
    │   └── ...
    └── ...

    Args:
        input_dir (str): The directory containing task subdirectories.
        his (int): The number of historical image and action frames to include.
        task_name_for_output (str): A string for the output JSON file name to identify the build.
        resolution (int): The image resolution, used in the output file name.
        output_dir (str): The directory where the generated JSON dataset will be saved.
    """
    train_convs = []
    total_traj_count = 0

    print(f"Processing data for Action-to-Image (a2i) task.")
    print(f"Scanning for task directories in: {input_dir}")

    # --- Automatic Task Discovery ---
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    task_paths = [os.path.join(input_dir, d) for d in sorted(os.listdir(input_dir)) if os.path.isdir(os.path.join(input_dir, d))]
    
    if not task_paths:
        print(f"Error: No task subdirectories found in '{input_dir}'.")
        return

    print(f"Found {len(task_paths)} task(s).")
    print("-" * 30)
    print(f"Historical frames (his): {his}")
    print(f"Output label: {task_name_for_output}")
    print(f"Resolution: {resolution}")
    print(f"Output directory: {output_dir}")
    print("-" * 30)

    # --- Main Processing Loop ---
    for task_path in tqdm(task_paths, desc="Processing Tasks"):
        task_name_readable = os.path.basename(task_path).replace('_', ' ')
        
        if not os.path.isdir(task_path):
            continue

        trj_list = sorted(os.listdir(task_path))
        
        for trj in tqdm(trj_list, desc=f"  - Trajectories for '{task_name_readable}'", leave=False):
            trj_path = os.path.join(task_path, trj)
            
            if not os.path.isdir(trj_path):
                continue
                
            action_base_path = os.path.join(trj_path, 'abs_action')
            imgs_path = os.path.join(trj_path, 'wrist_image')
            # wrist_image is not used in this script but path could be defined here if needed
            # imgs_path_w = os.path.join(trj_path, 'wrist_image')
            
            if not os.path.exists(action_base_path) or not os.path.exists(imgs_path):
                print(f"    Warning: Missing 'abs_action' or 'wrist_image' in {trj_path}. Skipping.")
                continue

            img_list = []
            action_list = []
            
            try:
                # Robustly find common indices
                action_dirs_raw = [d for d in os.listdir(action_base_path) if d.startswith('action_')]
                img_files_raw = [f for f in os.listdir(imgs_path) if f.startswith('image_') and f.endswith('.png')]
                
                action_indices = sorted([int(d.split('_')[1]) for d in action_dirs_raw])
                img_indices = sorted([int(f.split('_')[1].split('.')[0]) for f in img_files_raw])
                
                common_indices = sorted(list(set(action_indices) & set(img_indices)))
            except (ValueError, IndexError) as e:
                print(f"    Warning: Could not parse indices in {trj_path}. Error: {e}. Skipping.")
                continue
                
            for idx in common_indices:
                # NOTE: This script specifically looks for '0.npy' inside the action folder
                action_file = os.path.join(action_base_path, f"action_{idx}", "0.npy")
                img_file = os.path.join(imgs_path, f"image_{idx}.png")
                
                if os.path.exists(action_file) and os.path.exists(img_file):
                    img_list.append(img_file)
                    action_list.append(action_file)
            
            # A conversation requires at least one history step and one future step.
            # Total length must be at least 2.
            if len(img_list) < 2:
                continue

            # Generate conversations for each valid step
            # Loop up to the second to last element to ensure there is a future frame for prediction
            for j in range(len(action_list) - 1):
                history_start_idx = max(0, j - his + 1)
                
                img_c_h = copy.deepcopy(img_list[history_start_idx : j + 1])
                action_c = copy.deepcopy(action_list[history_start_idx : j + 1])
                img_c_f = copy.deepcopy(img_list[j + 1 : j + 2]) # The target image

                conv = {
                    "conversations": [
                        {
                            "from": "human",
                            # This prompt is generic as per the original script
                            "value": "Generate the next image based on the provided sequence of historical images and corresponding actions." + "<|image|><|action|>" * len(img_c_h)
                        },
                        {
                            "from": "gpt",
                            "value": "<|image|>"
                        }
                    ],
                    "image": img_c_h + img_c_f,
                    "action": action_c
                }
                train_convs.append(conv)
            
            if len(img_list) > 1:
                total_traj_count += 1

    # --- Save Dataset ---
    print("-" * 30)
    print("Saving dataset...")

    # Construct output filename as per the original script's format
    output_filename = f'libero_{task_name_for_output}_his_{his}_train_a2i_{resolution}_abs_wrist_all_data.json'
    output_path = os.path.join(output_dir, output_filename)

    print(f"Saving training data to: {output_path}")
    with open(output_path, 'w') as f:
        # Using indent=2 for smaller file size, but indent=4 is fine too.
        json.dump(train_convs, f, indent=2)

    print("\n--- Dataset Generation Summary ---")
    print(f"Total trajectories processed: {total_traj_count}")
    print(f"Total conversations generated: {len(train_convs)}")
    print("---------------------")


def main():
    parser = argparse.ArgumentParser(
        description="Process Libero robot trajectory data from a directory of tasks to create Action-to-Image (a2i) datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input_dir', '-i', type=str, required=True,
        help='The input directory containing task subdirectories. Each subdirectory is treated as a task.'
    )
    parser.add_argument(
        '--his', '-H', type=int, default=1,
        help='The number of historical image/action frames to include in each conversation.'
    )
    parser.add_argument(
        '--task_name', '-T', type=str, default='multi_task_a2i',
        help="A string used in the output JSON file names to identify this dataset build."
    )
    parser.add_argument(
        '--resolution', '-R', type=int, default=512,
        help='The image resolution, used in the output JSON file names.'
    )
    parser.add_argument(
        '--output_dir', '-o', type=str, default='./convs_output',
        help='Directory where the generated JSON dataset files will be saved.'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Call the processing function with parsed arguments
    process_a2i_data(
        input_dir=args.input_dir,
        his=args.his,
        task_name_for_output=args.task_name,
        resolution=args.resolution,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
