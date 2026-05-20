import os, re

TARGET_FILE = "/workspace/rynnvla-002/eval_solver_libero_got_v2.py"

with open(TARGET_FILE, "r", encoding="utf-8") as f:
    code = f.read()

code = code.replace("\r\n", "\n")
patched_count = 0

# 1. 중복 return model, None 제거
dup_return_pattern = r'(return model,\s*None\s*)\n\s*return model,\s*None'
if re.search(dup_return_pattern, code):
    code = re.sub(dup_return_pattern, r'\1', code)
    print("✅ 중복 return 제거 완료")
    patched_count += 1
else:
    print("ℹ️ 중복 return 이미 정리됨")

# 2. got 모드 t 이중누적 방지
old_queue_check = "                        if not action_queue:"
new_queue_check = """                        # got 모드는 plan_and_execute가 직접 실행하므로 큐 체크 건너뜀 (t 이중누적 방지)
                        if args.mode == "got":
                            continue

                        if not action_queue:"""

if old_queue_check in code:
    code = code.replace(old_queue_check, new_queue_check)
    print("✅ got 모드 t 이중누적 방지 삽입 완료")
    patched_count += 1
else:
    print("ℹ️ 이미 적용됨 또는 탐색 실패")

if patched_count > 0:
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(code)
    print("💾 저장 완료")

print("완료!")
