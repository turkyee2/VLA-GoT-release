import sys, os
os.chdir('/home/gpu_06/rynnvla-002')
sys.path.insert(0, '/home/gpu_06/LIBERO')
sys.path.insert(0, '/home/gpu_06/rynnvla-002')
sys.path.insert(0, '/home/gpu_06')

import torch
import numpy as np
from PIL import Image
from transformers import GenerationConfig
from data.pre_tokenize_action_state import ItemProcessor
from model import ChameleonXLLMXForConditionalGeneration_ck_action_head

device = torch.device('cuda:0')
model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
    './ckpts/Action_World_model_512/libero_spatial',
    torch_dtype=torch.bfloat16,
    device_map='cuda:0',
)
model.eval()

item_processor = ItemProcessor(
    target_size=256,
    tokenizer='./ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af'
)

cur_img = Image.new("RGB", (256, 256), color=(128, 64, 32))

# 이미지 1장만, 액션 없이
conv = {
    "conversations": [{
        "from": "human",
        "value": "Generate the image based on the current image and the action." + "<|image|>"
    }],
    "image": [cur_img],
    "action": [],
}

tokens = item_processor.process_item(conv, training_mode=False)
print(f"입력 토큰 수: {len(tokens)}")

generation_config = GenerationConfig(
    max_new_tokens=3000,
    max_length=model.config.max_position_embeddings,
    temperature=1,
    top_k=None,
    do_sample=False,
    eos_token_id=[8710],
)
input_ids = torch.tensor(tokens, dtype=torch.int64, device=device).unsqueeze(0)
with torch.no_grad():
    g_image_tokens = model.generate_img(input_ids, generation_config)

tokens_sequence = g_image_tokens[0]
start_indices = torch.where(tokens_sequence == 8197)[0]
end_indices = torch.where(tokens_sequence == 8196)[0]
print(f"START: {len(start_indices)}개, END: {len(end_indices)}개")
print(f"첫 20개: {tokens_sequence[:20].tolist()}")

if len(start_indices) >= 1 and len(end_indices) >= 1:
    front_tokens = tokens_sequence[start_indices[0]:end_indices[0]+1]
    img = item_processor.decode_image(front_tokens.cpu().tolist())
    if img:
        os.makedirs('/home/gpu_06/results', exist_ok=True)
        img.save('/home/gpu_06/results/test_wm_output.png')
        print("성공! 저장됨: ~/results/test_wm_output.png")
else:
    print("실패")
