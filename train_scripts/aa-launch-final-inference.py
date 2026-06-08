import os
import subprocess

# total_gen_num = 8236
# step_num1 = 200
# step_num2 = 200
# size = 1024
# seed = 0
# t2mn_path = "/fsx_laion/alvin/sd4human/a-ranstart-body-sdv20-v-nd-flaw-avg-copy1-glc-resume288k-512-ft1024/checkpoint-340000"
# controlnet_model_name_or_path = "/fsx_laion/alvin/sd4human/ctrl-sdxl10-eps-glc-composer-bmn-sum-1024-ft1024/checkpoint-75000"
# save_path = "/fsx_laion/alvin/Dataset/evaluation/1024-1024-size1024"

# for i in range(8):
#     start = i * (total_gen_num // 8)
#     end = (i + 1) * (total_gen_num // 8)
#     if i == 7:
#         end = total_gen_num
#     os.system(f'CUDA_VISIBLE_DEVICES={i} python aa-final-inference.py --step_num1={step_num1} --step_num2={step_num2} --size={size} --t2mn_path={t2mn_path} --controlnet_model_name_or_path={controlnet_model_name_or_path} --seed={seed} --inference_folder_name={save_path} --start={start} --end={end} &')
# os.system('wait')


# total_gen_num = 8236
# step_num1 = 200
# step_num2 = 200
# size = 2048
# seed = 0
# t2mn_path = "/fsx_laion/alvin/sd4human/a-ranstart-body-sdv20-v-nd-flaw-avg-copy1-glc-resume288k-512-ft1024/checkpoint-340000"
# controlnet_model_name_or_path = "/fsx_laion/alvin/sd4human/ctrl-sdxl10-eps-glc-composer-bmn-sum-1024-ft1024/checkpoint-75000"
# save_path = "/fsx_laion/alvin/Dataset/evaluation/1024-1024-size2048"

# for i in range(8):
#     start = i * (total_gen_num // 8)
#     end = (i + 1) * (total_gen_num // 8)
#     if i == 7:
#         end = total_gen_num
#     os.system(f'CUDA_VISIBLE_DEVICES={i} python aa-final-inference.py --step_num1={step_num1} --step_num2={step_num2} --size={size} --t2mn_path={t2mn_path} --controlnet_model_name_or_path={controlnet_model_name_or_path} --seed={seed} --inference_folder_name={save_path} --start={start} --end={end} &')
# os.system('wait')

# commands = []

# total_gen_num = 8236
# step_num1 = 200
# step_num2 = 200
# size = 1024
# seed = 0
# t2mn_path = "/fsx_laion/alvin/sd4human/a-ranstart-body-sdv20-v-nd-flaw-avg-copy1-glc-resume136k-512/checkpoint-280000"
# controlnet_model_name_or_path = "/fsx_laion/alvin/sd4human/ctrl-sdxl10-eps-glc-composer-bmn-sum-1024-ft1024/checkpoint-75000"
# save_path = "/fsx_laion/alvin/Dataset/evaluation/512-1024-size1024"

# for i in range(8):
#     start = i * (total_gen_num // 8)
#     end = (i + 1) * (total_gen_num // 8)
#     if i == 7:
#         end = total_gen_num
#     # os.system(f'CUDA_VISIBLE_DEVICES={i} python aa-final-inference.py --step_num1={step_num1} --step_num2={step_num2} --size={size} --t2mn_path={t2mn_path} --controlnet_model_name_or_path={controlnet_model_name_or_path} --seed={seed} --inference_folder_name={save_path} --start={start} --end={end} &')
#     commands.append(f'CUDA_VISIBLE_DEVICES={i} python aa-final-inference.py --step_num1={step_num1} --step_num2={step_num2} --size={size} --t2mn_path={t2mn_path} --controlnet_model_name_or_path={controlnet_model_name_or_path} --seed={seed} --inference_folder_name={save_path} --start={start} --end={end} &')

# for cmd in commands:
#     subprocess.Popen(cmd, shell=True)

# # Wait for all background processes to finish
# subprocess.call("wait", shell=True)

# commands = []
# background_processes = []

# total_gen_num = 8236
# step_num1 = 200
# step_num2 = 200
# size = 2048
# seed = 0
# t2mn_path = "/fsx_laion/alvin/sd4human/a-ranstart-body-sdv20-v-nd-flaw-avg-copy1-glc-resume136k-512/checkpoint-280000"
# controlnet_model_name_or_path = "/fsx_laion/alvin/sd4human/ctrl-sdxl10-eps-glc-composer-bmn-sum-1024-ft1024/checkpoint-75000"
# save_path = "/fsx_laion/alvin/Dataset/evaluation/512-1024-size2048"

# for i in range(8):
#     start = i * (total_gen_num // 8)
#     end = (i + 1) * (total_gen_num // 8)
#     if i == 7:
#         end = total_gen_num
#     # os.system(f'CUDA_VISIBLE_DEVICES={i} python aa-final-inference.py --step_num1={step_num1} --step_num2={step_num2} --size={size} --t2mn_path={t2mn_path} --controlnet_model_name_or_path={controlnet_model_name_or_path} --seed={seed} --inference_folder_name={save_path} --start={start} --end={end} &')
#     commands.append(f'CUDA_VISIBLE_DEVICES={i} python aa-final-inference.py --step_num1={step_num1} --step_num2={step_num2} --size={size} --t2mn_path={t2mn_path} --controlnet_model_name_or_path={controlnet_model_name_or_path} --seed={seed} --inference_folder_name={save_path} --start={start} --end={end} &')

# for cmd in commands:
#     process = subprocess.Popen(cmd, shell=True)
#     background_processes.append(process)

# # Wait for all background processes to finish
# for process in background_processes:
#     process.wait()
    

commands = []
background_processes = []

total_gen_num = 8236
step_num1 = 200
step_num2 = 200
size = 2048
seed = 0
t2mn_path = "/fsx_laion/alvin/sd4human/a-ranstart-body-sdv20-v-nd-flaw-avg-copy1-glc-resume288k-512-ft1024/checkpoint-340000"
controlnet_model_name_or_path = "/fsx_laion/alvin/sd4human/ctrl-sdxl10-eps-glc-composer-bmn-sum-1024/checkpoint-75000"
save_path = "/fsx_laion/alvin/Dataset/evaluation/1024-512-size2048"

for i in range(8):
    start = i * (total_gen_num // 8)
    end = (i + 1) * (total_gen_num // 8)
    if i == 7:
        end = total_gen_num
    # os.system(f'CUDA_VISIBLE_DEVICES={i} python aa-final-inference.py --step_num1={step_num1} --step_num2={step_num2} --size={size} --t2mn_path={t2mn_path} --controlnet_model_name_or_path={controlnet_model_name_or_path} --seed={seed} --inference_folder_name={save_path} --start={start} --end={end} &')
    commands.append(f'CUDA_VISIBLE_DEVICES={i} python aa-final-inference.py --step_num1={step_num1} --step_num2={step_num2} --size={size} --t2mn_path={t2mn_path} --controlnet_model_name_or_path={controlnet_model_name_or_path} --seed={seed} --inference_folder_name={save_path} --start={start} --end={end} &')

for cmd in commands:
    process = subprocess.Popen(cmd, shell=True)
    background_processes.append(process)

# Wait for all background processes to finish
for process in background_processes:
    process.wait()
