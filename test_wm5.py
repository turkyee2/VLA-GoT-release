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
    max_position_embeddings=8192,
    mask_image_logits=False,
    dropout=0.0,
    z_loss_weight=0.0,
    torch_dtype=torch.bfloat16,
    device_map='cuda:0',
)
model.eval()

item_processor = ItemProcessor(
    target_size=256,
    tokenizer='./ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af'
)

cur_img = Image.new("RGB", (256, 256), color=(128, 64, 32))
wrist_img = Image.new("RGB", (256, 256), color=(64, 128, 32))

conv_infer = {
    "conversations": [
        {
            "from": "human",
            "value": "Generate the next image based on the provided sequence of historical images and corresponding actions.<|image|><|image|><|action|>"
        },
    ],
    "image": [cur_img, wrist_img],
    "action": [np.zeros(7)],
}

conv_train = {
    "conversations": [
        {
            "from": "human",
            "value": "Generate the next image based on the provided sequence of historical images and corresponding actions.<|image|><|image|><|action|>"
        },
        {
            "from": "gpt",
            "value": "<|image|><|image|>"
        }
    ],
    "image": [cur_img, wrist_img, cur_img, wrist_img],
    "action": [np.zeros(7)],
}
conv = conv_infer

# training_mode=True로 전체 토큰 생성 후 gpt 시작 토큰까지만 잘라서 입력
tokens_full = item_processor.process_item(conv_train, training_mode=True)
tokens_infer = item_processor.process_item(conv_infer, training_mode=False)
print(f"추론 입력 토큰 수: {len(tokens_infer)}")
print(f"학습 전체 토큰 수: {len(tokens_full)}")

import numpy as np
tok_arr = np.array(tokens_full)
img_starts = np.where(tok_arr == 8197)[0]
print(f"학습 토큰에서 8197 위치: {img_starts.tolist()[:5]}")

# gpt 응답 시작 직전까지 (3번째 8197 앞까지)
if len(img_starts) >= 3:
    cut_point = img_starts[2]
    tokens = tokens_full[:cut_point].tolist()
    print(f"잘린 입력 토큰 수: {len(tokens)}")
else:
    tokens = tokens_infer
    print("fallback to infer tokens")

generation_config = GenerationConfig(
    max_new_tokens=7000,
    max_length=8192,
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
print(f"생성된 토큰 수: {len(tokens_sequence)}")
print(f"START: {len(start_indices)}개 위치: {start_indices.tolist()}")
print(f"첫 30개: {tokens_sequence[:30].tolist()}")
print(f"마지막 30개: {tokens_sequence[-30:].tolist()}")
print(f"END: {len(end_indices)}개 위치: {end_indices.tolist()}")

if len(start_indices) >= 1 and len(end_indices) >= 1:
    size = end_indices[0] - start_indices[0] + 1
    print(f"첫 번째 이미지 토큰 수: {size} (필요: 1058)")
    if size == 1058:
        front_tokens = tokens_sequence[start_indices[0]:end_indices[0]+1]
        img = item_processor.decode_image(front_tokens.cpu().tolist())
        if img:
            img.save('/home/gpu_06/results/test_wm5.png')
            print("성공!")
    else:
        print(f"토큰 수 불일치: {size} != 1058")
