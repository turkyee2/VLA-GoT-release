import torch
import numpy as np
from PIL import Image
from transformers import GenerationConfig


def world_model_score(candidates, current_img, wrist_img, world_model, item_processor, device, verbose=False, **kwargs):
    saved_init = world_model.init_input_ids
    world_model.init_input_ids = None

    for i, cand in enumerate(candidates):
        conv = {
            "conversations": [{"from": "human",
                "value": "Generate the image based on the current image and the action.<|image|><|image|><|action|>"}],
            "image": [current_img, wrist_img],
            "action": [cand.actions[0]],
        }
        try:
            tokens = item_processor.process_item(conv, training_mode=False)
            input_ids = torch.tensor(tokens, dtype=torch.int64, device=device).unsqueeze(0)

            # 2장 생성을 위해 넉넉하게
            generation_config = GenerationConfig(
                max_new_tokens=700,
                max_length=world_model.config.max_position_embeddings,
                temperature=1.0, top_k=None, do_sample=False, eos_token_id=[8710],
            )
            with torch.no_grad():
                g_image_tokens = world_model.generate_img(input_ids, generation_config)

            tokens_sequence = g_image_tokens[0]
            start_indices = torch.where(tokens_sequence == 8197)[0]
            end_indices = torch.where(tokens_sequence == 8196)[0]

            # 원본과 동일: 2장 이상 생성됐을 때만 사용
            if len(start_indices) >= 2 and len(end_indices) >= 2:
                try:
                    # front 이미지만 score에 사용
                    front_tokens = tokens_sequence[start_indices[0]:end_indices[0]+1].cpu().tolist()
                    pred_img = item_processor.decode_image(front_tokens)
                    arr_cur = np.array(current_img.convert("RGB")).astype(np.float32)
                    arr_pred = np.array(pred_img.convert("RGB")).astype(np.float32)
                    cand.score = float(np.mean(np.abs(arr_cur - arr_pred)) / 255.0)
                    if verbose:
                        print(f"  [WM Score] 후보{i+1}: score={cand.score:.4f}")
                except Exception as de:
                    if verbose:
                        print(f"  [WM Score] 후보{i+1}: decode 실패: {de}")
                    cand.score = 0.0
            elif len(start_indices) >= 1 and len(end_indices) >= 1:
                # 1장만 생성된 경우도 시도
                try:
                    front_tokens = tokens_sequence[start_indices[0]:end_indices[0]+1].cpu().tolist()
                    pred_img = item_processor.decode_image(front_tokens)
                    arr_cur = np.array(current_img.convert("RGB")).astype(np.float32)
                    arr_pred = np.array(pred_img.convert("RGB")).astype(np.float32)
                    cand.score = float(np.mean(np.abs(arr_cur - arr_pred)) / 255.0)
                    if verbose:
                        print(f"  [WM Score] 후보{i+1}: score={cand.score:.4f} (1장)")
                except Exception as de:
                    if verbose:
                        print(f"  [WM Score] 후보{i+1}: decode 실패: {de}")
                    cand.score = 0.0
            else:
                cand.score = -1.0

        except Exception as e:
            if verbose:
                print(f"  [WM Score] 후보{i+1} 예외: {e}")
            cand.score = -float("inf")

    if verbose:
        scores = [f"{c.score:.4f}" for c in candidates]
        print(f"  [WM Score] 점수: {scores}")

    world_model.init_input_ids = saved_init
    return candidates
