import json
import os
import math
import copy
import argparse
from tqdm import tqdm

def process_libero_data(
    input_dir: str,
    his: int,
    task_name_for_output: str,
    resolution: int,
    len_action: int,
    output_dir: str
):
    """
    Processes Libero robot trajectory data from a specified input directory to create conversational datasets.

    This script automatically discovers task folders within the `input_dir`. Each subdirectory
    is treated as a separate task. The name of the subdirectory is used as the task's description.

    Expected Directory Structure:
    input_dir/
    ├── TASK_NAME_1/
    │   ├── trajectory_0/
    │   │   ├── abs_action/
    │   │   ├── front_image/
    │   │   ├── wrist_image/
    │   │   └── state/
    │   └── trajectory_1/
    │       └── ...
    ├── TASK_NAME_2/
    │   ├── trajectory_0/
    │   │   └── ...
    │   └── ...
    └── ...

    This version includes state information (state_n.npy) alongside images.
    For each observation at timestep 'i', this script generates a sample
    to predict the action at timestep 'i', which consists of 20 sub-actions.

    Args:
        input_dir (str): The directory containing task subdirectories.
        his (int): The number of historical image frames to include in each conversation.
        task_name_for_output (str): A string used in the output JSON file names to
                                    identify the collection of tasks (e.g., 'multi_task', 'all_tasks').
        resolution (int): The image resolution, used in the output JSON file names.
        output_dir (str): The directory where the generated JSON dataset files will be saved.
    """
    # --- CONSTANTS ---
    ACTION_CHUNK_PREDICTION_HORIZON = 1
    SUB_ACTIONS_PER_CHUNK = len_action

    train_convs = []
    train_traj_count = 0

    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning for task directories in: {input_dir}")

    # --- Automatic Task Discovery ---
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found at '{input_dir}'")
        return

    # Get a list of all subdirectories in the input directory, assuming each is a task.
    task_paths = [os.path.join(input_dir, d) for d in sorted(os.listdir(input_dir)) if os.path.isdir(os.path.join(input_dir, d))]
    
    if not task_paths:
        print(f"Error: No task subdirectories found in '{inputdir}'. Please check the directory structure.")
        return

    print(f"Found {len(task_paths)} task(s) to process.")
    print("-" * 30)
    print(f"Historical frames (his): {his}")
    print(f"Action chunk prediction horizon: {ACTION_CHUNK_PREDICTION_HORIZON}")
    print(f"Sub-actions per chunk: {SUB_ACTIONS_PER_CHUNK}")
    print(f"Output label: {task_name_for_output}")
    print(f"Resolution: {resolution}")
    print(f"Output directory: {output_dir}")
    print("-" * 30)

    # --- Main Processing Loop ---
    for task_path in tqdm(task_paths, desc="Processing Tasks"):
        # The task name is the name of the subfolder, with underscores replaced by spaces.
        task_name_readable = os.path.basename(task_path).replace('_', ' ')
        
        if not os.path.isdir(task_path):
            continue

        trj_list = sorted(os.listdir(task_path))
        
        for trj in tqdm(trj_list, desc=f"  - Trajectories for '{task_name_readable}'", leave=False):
            trj_path = os.path.join(task_path, trj)
            
            if not os.path.isdir(trj_path):
                continue
                
            action_base_path = os.path.join(trj_path, 'abs_action')
            imgs_path = os.path.join(trj_path, 'front_image')
            imgs_path_w = os.path.join(trj_path, 'wrist_image')
            state_path = os.path.join(trj_path, 'state')
            
            if not all(os.path.exists(p) for p in [action_base_path, imgs_path, imgs_path_w, state_path]):
                print(f"    Warning: Missing required data directories in {trj_path}. Skipping.")
                continue
            
            try:
                action_dirs_raw = [d for d in os.listdir(action_base_path) if d.startswith('action_') and os.path.isdir(os.path.join(action_base_path, d))]
                img_files_raw = [f for f in os.listdir(imgs_path) if f.startswith('image_') and f.endswith('.png')]
                state_files_raw = [f for f in os.listdir(state_path) if f.startswith('state_') and f.endswith('.npy')]

                action_indices = sorted([int(d.split('_')[1]) for d in action_dirs_raw])
                img_indices = sorted([int(f.split('_')[1].split('.')[0]) for f in img_files_raw])
                state_indices = sorted([int(f.split('_')[1].split('.')[0]) for f in state_files_raw])
            except (ValueError, IndexError) as e:
                print(f"    Warning: Could not parse file/dir indices in {trj_path}. Error: {e}. Skipping.")
                continue

            common_indices = sorted(list(set(action_indices) & set(img_indices) & set(state_indices)))
            if not common_indices:
                print(f"    Warning: No common indices found for all data types in {trj_path}. Skipping.")
                continue

            img_list, img_list_w, action_list, state_list = [], [], [], []

            for idx in common_indices:
                img_file = os.path.join(imgs_path, f"image_{idx}.png")
                img_file_w = os.path.join(imgs_path_w, f"image_{idx}.png")
                action_dir = os.path.join(action_base_path, f"action_{idx}")
                state_file = os.path.join(state_path, f"state_{idx}.npy")
                
                if os.path.exists(img_file) and os.path.exists(img_file_w) and os.path.isdir(action_dir) and os.path.exists(state_file):
                    try:
                        sub_action_files_raw = [f for f in os.listdir(action_dir) if f.endswith('.npy')]
                        sub_action_files_sorted = sorted(sub_action_files_raw, key=lambda f: int(os.path.splitext(f)[0]))
                        
                        if len(sub_action_files_sorted) == SUB_ACTIONS_PER_CHUNK:
                            sub_action_paths = [os.path.join(action_dir, f) for f in sub_action_files_sorted]
                            img_list.append(img_file)
                            img_list_w.append(img_file_w)
                            action_list.append(sub_action_paths)
                            state_list.append(state_file)
                        else:
                            # This warning can be noisy, so it's useful but can be commented out if needed.
                            # print(f"      Warning: Action dir {action_dir} has {len(sub_action_files_sorted)} files, expected {SUB_ACTIONS_PER_CHUNK}. Skipping index {idx}.")
                            pass
                    except (ValueError, FileNotFoundError) as e:
                         print(f"      Warning: Error processing action dir {action_dir}: {e}. Skipping index {idx}.")

            if not img_list or not action_list or not state_list:
                continue

            for j in range(len(action_list)):
                img_history_start_idx = max(0, j - his + 1)
                img_c = copy.deepcopy(img_list[img_history_start_idx : j + 1])
                img_c_w = copy.deepcopy(img_list_w[img_history_start_idx : j + 1])
                action_c = copy.deepcopy(action_list[j])
                state_c = copy.deepcopy(state_list[j])

                if len(action_c) != SUB_ACTIONS_PER_CHUNK:
                    continue

                conv = {
                    "conversations":[
                        {
                            "from": "human",
                            "value": f"What action should the robot take to {task_name_readable}?" + "<|state|>" + "<|image|>" * len(img_c) * 2
                        },
                        {
                            "from": "gpt",
                            "value": "<|action|>" * SUB_ACTIONS_PER_CHUNK
                        },
                    ],
                    "image": img_c + img_c_w,
                    "action": action_c,
                    "state": state_c
                }
                train_convs.append(conv)
            
            train_traj_count += 1
                
    print("-" * 30)
    print("Saving dataset...")

    train_output_path = os.path.join(output_dir, f'libero_{task_name_for_output}_his_{his}_train_img_state_abs_ck_{ACTION_CHUNK_PREDICTION_HORIZON}_{resolution}.json')
    
    with open(train_output_path, 'w') as f:
        json.dump(train_convs, f, indent=2)
    print(f"Saved train conversations to: {train_output_path}")

    print("\n--- Final Summary ---")
    print(f"Total trajectories processed: {train_traj_count}")
    print(f"Total conversations generated: {len(train_convs)}")
    print("---------------------")

def main():
    parser = argparse.ArgumentParser(
        description="Process Libero robot trajectory data from a directory of tasks to create conversational datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input_dir', '-i', type=str, required=True,
        help='The input directory containing task subdirectories. Each subdirectory is treated as a task.'
    )
    parser.add_argument(
        '--his', '-H', type=int, default=1,
        help='The number of historical image frames to include in each conversation (for observation history).'
    )
    parser.add_argument(
        '--task_name', '-T', type=str, default='multi_task',
        help="A string used in the output JSON file names to identify this dataset build (e.g., 'multi_task', 'set_A')."
    )
    parser.add_argument(
        '--resolution', '-R', type=int, default=256,
        help='The image resolution, used in the output JSON file names (e.g., 224, 512).'
    )
    parser.add_argument(
        '--len_action', type=int, default=20,
        help='Action chunk size'
    )
    parser.add_argument(
        '--output_dir', '-o', type=str, default='./convs_output',
        help='The directory where the generated JSON dataset files will be saved. Will be created if it does not exist.'
    )

    args = parser.parse_args()

    process_libero_data(
        input_dir=args.input_dir,
        his=args.his,
        task_name_for_output=args.task_name,
        resolution=args.resolution,
        len_action=args.len_action,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
