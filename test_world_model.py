import sys
sys.path.insert(0, '/home/gpu_06/LIBERO')
sys.path.insert(0, '/home/gpu_06/rynnvla-002')
sys.path.insert(0, '/home/gpu_06')

import torch
import numpy as np
from PIL import Image
from data.pre_tokenize_action_state import ItemProcessor
from model import ChameleonXLLMXForConditionalGeneration_ck_action_head
from libero_util.Chameleon_utils import get_action_Chameleon_dis_awm_g_video_wrist

device = torch.device('cuda:0')

print("모델 로드 중...")
model = ChameleonXLLMXForConditionalGeneration_ck_action_head.from_pretrained(
    './ckpts/Action_World_model_512/libero_spatial',
    torch_dtype=torch.bfloat16,
    device_map='cuda:0',
)
model.eval()
print(f"World Model 로드 완료, VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")

item_processor = ItemProcessor(
    target_size=256,
    tokenizer='./ckpts/models--Alpha-VLLM--Lumina-mGPT-7B-768/snapshots/9624463a82ea5ce814af9b561dcd08a31082c3af'
)

# 더미 입력
cur_img = Image.new("RGB", (256, 256), color=(128, 64, 32))
wrist_img = Image.new("RGB", (256, 256), color=(64, 128, 32))
his_action = [np.zeros(7)]
his_type = "his_2_third_view_wrist_w_state"

print("이미지 생성 중...")
front_img, wrist_img_out = get_action_Chameleon_dis_awm_g_video_wrist(
    model, "pick up the black bowl",
    item_processor,
    [cur_img], [wrist_img],
    his_action, his_type
)

if front_img is not None:
    front_img.save("/home/gpu_06/results/test_wm_front.png")
    print("성공! 이미지 저장됨: ~/results/test_wm_front.png")
else:
    print("실패: 이미지 생성 안 됨")
