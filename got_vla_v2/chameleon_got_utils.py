"""
chameleon_got_utils.py
----------------------
Thin adapter around the existing Chameleon_utils functions in
rynnvla-002/libero_util/Chameleon_utils.py.

Adds:
  - do_sample / temperature parameters (for diverse candidate generation)
  - A unified get_action_for_got() entry point used by got_pipeline.py

No existing files are modified. Import this alongside the original utils.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import GenerationConfig


def parse_his_type(his_type: str) -> dict:
    """
    Replicates the parse_his_type logic from Chameleon_utils.py.
    Kept here so got_pipeline.py has no circular import on the original file.
    """
    try:
        parts = his_type.split("_")
        if len(parts) < 5 or parts[0] != "his" or parts[-1] != "state":
            return {"his": 1, "views": ["front", "wrist"], "with_state": True}
        his = int(parts[1])
        state_prefix = parts[-2]
        with_state = state_prefix == "w"
        views = parts[2:-2]
        return {"his": his, "views": views, "with_state": with_state}
    except Exception:
        return {"his": 1, "views": ["front", "wrist"], "with_state": True}


def _build_image_list(
    his_type_dict: dict,
    his_img: List[Image.Image],
    his_wrist_img: List[Image.Image],
    cur_img: Image.Image,
    cur_wrist_img: Optional[Image.Image],
) -> List[Image.Image]:
    """Build the image context list matching the original Chameleon_utils logic."""
    his_n = his_type_dict["his"]
    n_views = len(his_type_dict["views"])

    if his_n == 2 and n_views == 3:
        img_c = his_img[-1:] + his_wrist_img[-1:] + [cur_img]
        if cur_wrist_img is not None:
            img_c.append(cur_wrist_img)
    elif his_n == 1 and n_views == 3:
        img_c = [cur_img]
        if cur_wrist_img is not None:
            img_c.append(cur_wrist_img)
    elif his_n == 2 and n_views == 2:
        img_c = his_img[-1:] + [cur_img]
    else:
        img_c = [cur_img]

    return img_c


def get_action_for_got(
    model,
    cur_img: Image.Image,
    cur_wrist_img: Optional[Image.Image],
    task_description: str,
    item_processor,
    his_img: List[Image.Image],
    his_wrist_img: List[Image.Image],
    cur_state: np.ndarray,
    his_action: List[np.ndarray],
    his_type: str,
    action_steps: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> list:
    """
    Unified action-generation entry point for GoT.

    Mirrors get_action_Chameleon_dis_awm_ck_discrete_action from
    Chameleon_utils.py but adds do_sample / temperature for diverse
    candidate generation (needed for k > 1 in GoT Generate step).

    Returns:
        list of torch.Tensor, each shape (7,) — raw discrete action tokens
        (same format as the original function's return value).
        Returns [] on failure.
    """
    if device is None:
        device = next(model.parameters()).device

    try:
        his_type_dict = parse_his_type(his_type)
        img_c = _build_image_list(
            his_type_dict, his_img, his_wrist_img, cur_img, cur_wrist_img
        )

        # Build prompt (same as original Chameleon_utils)
        if his_type_dict["with_state"]:
            human_val = (
                f"What action should the robot take to {task_description}?"
                + "<|state|>" * 1
                + "<|image|>" * len(img_c)
            )
        else:
            human_val = (
                f"What action should the robot take to {task_description}?"
                + "<|image|>" * len(img_c)
            )

        conv = {
            "conversations": [{"from": "human", "value": human_val}],
            "image": img_c,
            "action": [],
        }
        if his_type_dict["with_state"]:
            conv["state"] = cur_state

        tokens = item_processor.process_item(conv, training_mode=False)

        # Generation config — temperature / top_k only active when do_sample=True
        # Temperature만 사용, top_k 제거
        # - do_sample=False (i=0): greedy, exploitation
        # - do_sample=True  (i>0): temperature로 다양성 조절
        # - temperature 범위 0~1.5, top_k는 temperature와 중복이므로 제거
        generation_config = GenerationConfig(
            max_new_tokens=action_steps * 12,
            max_length=model.config.max_position_embeddings,
            temperature=temperature if do_sample else 1.0,
            top_k=None,
            top_p=0.95 if do_sample else None,
            do_sample=do_sample,
            eos_token_id=[8710],
        )

        input_ids = torch.tensor(
            tokens, dtype=torch.int64, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            dis_action = model.generate_dis_ma(input_ids, generation_config)

        return dis_action  # list of tensors

    except Exception as e:
        print(f"[chameleon_got_utils] get_action_for_got failed: {e}")
        return []
