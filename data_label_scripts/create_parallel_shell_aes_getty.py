# openpose
import os
# [0, 32400]
for i in range(0, 32000, 1600):
    start_list = [i + 50 * j for j in range(33)]
    file_name = f"aes-getty-label-{start_list[0]}-{start_list[-1]}.sh"
    with open(f"shells/{file_name}", "w") as file:
        lines = [
f"CUDA_VISIBLE_DEVICES=0 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[0]} --end={start_list[1]} &",
f"CUDA_VISIBLE_DEVICES=1 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[1]} --end={start_list[2]} &",
f"CUDA_VISIBLE_DEVICES=2 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[2]} --end={start_list[3]} &",
f"CUDA_VISIBLE_DEVICES=3 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[3]} --end={start_list[4]} &",
f"CUDA_VISIBLE_DEVICES=4 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[4]} --end={start_list[5]} &",
f"CUDA_VISIBLE_DEVICES=5 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[5]} --end={start_list[6]} &",
f"CUDA_VISIBLE_DEVICES=6 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[6]} --end={start_list[7]} &",
f"CUDA_VISIBLE_DEVICES=7 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[7]} --end={start_list[8]} &",
f"CUDA_VISIBLE_DEVICES=0 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[8]} --end={start_list[9]} &",
f"CUDA_VISIBLE_DEVICES=1 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[9]} --end={start_list[10]} &",
f"CUDA_VISIBLE_DEVICES=2 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[10]} --end={start_list[11]} &",
f"CUDA_VISIBLE_DEVICES=3 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[11]} --end={start_list[12]} &",
f"CUDA_VISIBLE_DEVICES=4 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[12]} --end={start_list[13]} &",
f"CUDA_VISIBLE_DEVICES=5 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[13]} --end={start_list[14]} &",
f"CUDA_VISIBLE_DEVICES=6 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[14]} --end={start_list[15]} &",
f"CUDA_VISIBLE_DEVICES=7 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[15]} --end={start_list[16]} &",
f"CUDA_VISIBLE_DEVICES=0 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[16]} --end={start_list[17]} &",
f"CUDA_VISIBLE_DEVICES=1 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[17]} --end={start_list[18]} &",
f"CUDA_VISIBLE_DEVICES=2 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[18]} --end={start_list[19]} &",
f"CUDA_VISIBLE_DEVICES=3 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[19]} --end={start_list[20]} &",
f"CUDA_VISIBLE_DEVICES=4 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[20]} --end={start_list[21]} &",
f"CUDA_VISIBLE_DEVICES=5 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[21]} --end={start_list[22]} &",
f"CUDA_VISIBLE_DEVICES=6 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[22]} --end={start_list[23]} &",
f"CUDA_VISIBLE_DEVICES=7 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[23]} --end={start_list[24]} &",
f"CUDA_VISIBLE_DEVICES=0 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[24]} --end={start_list[25]} &",
f"CUDA_VISIBLE_DEVICES=1 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[25]} --end={start_list[26]} &",
f"CUDA_VISIBLE_DEVICES=2 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[26]} --end={start_list[27]} &",
f"CUDA_VISIBLE_DEVICES=3 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[27]} --end={start_list[28]} &",
f"CUDA_VISIBLE_DEVICES=4 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[28]} --end={start_list[29]} &",
f"CUDA_VISIBLE_DEVICES=5 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[29]} --end={start_list[30]} &",
f"CUDA_VISIBLE_DEVICES=6 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[30]} --end={start_list[31]} &",
f"CUDA_VISIBLE_DEVICES=7 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[31]} --end={start_list[32]} &",
"wait"
        ]
        file.write("\n".join(lines))
        file.write("\n")
        
# [32400, 32987]
file_name = f"aes-getty-label-32000-32987.sh"
start_list = [32000 + 42 * j for j in range(24)]
with open(f"shells/{file_name}", "w") as file:
    lines = [
f"CUDA_VISIBLE_DEVICES=0 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[0]} --end={start_list[1]} &",
f"CUDA_VISIBLE_DEVICES=1 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[1]} --end={start_list[2]} &",
f"CUDA_VISIBLE_DEVICES=2 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[2]} --end={start_list[3]} &",
f"CUDA_VISIBLE_DEVICES=3 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[3]} --end={start_list[4]} &",
f"CUDA_VISIBLE_DEVICES=4 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[4]} --end={start_list[5]} &",
f"CUDA_VISIBLE_DEVICES=5 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[5]} --end={start_list[6]} &",
f"CUDA_VISIBLE_DEVICES=6 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[6]} --end={start_list[7]} &",
f"CUDA_VISIBLE_DEVICES=7 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[7]} --end={start_list[8]} &",
f"CUDA_VISIBLE_DEVICES=0 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[8]} --end={start_list[9]} &",
f"CUDA_VISIBLE_DEVICES=1 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[9]} --end={start_list[10]} &",
f"CUDA_VISIBLE_DEVICES=2 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[10]} --end={start_list[11]} &",
f"CUDA_VISIBLE_DEVICES=3 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[11]} --end={start_list[12]} &",
f"CUDA_VISIBLE_DEVICES=4 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[12]} --end={start_list[13]} &",
f"CUDA_VISIBLE_DEVICES=5 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[13]} --end={start_list[14]} &",
f"CUDA_VISIBLE_DEVICES=6 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[14]} --end={start_list[15]} &",
f"CUDA_VISIBLE_DEVICES=7 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[15]} --end={start_list[16]} &",
f"CUDA_VISIBLE_DEVICES=0 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[16]} --end={start_list[17]} &",
f"CUDA_VISIBLE_DEVICES=1 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[17]} --end={start_list[18]} &",
f"CUDA_VISIBLE_DEVICES=2 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[18]} --end={start_list[19]} &",
f"CUDA_VISIBLE_DEVICES=3 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[19]} --end={start_list[20]} &",
f"CUDA_VISIBLE_DEVICES=4 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[20]} --end={start_list[21]} &",
f"CUDA_VISIBLE_DEVICES=5 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[21]} --end={start_list[22]} &",
f"CUDA_VISIBLE_DEVICES=6 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[22]} --end={start_list[23]} &",
f"CUDA_VISIBLE_DEVICES=7 python aes-watermark_getty.py --data-path /fsx_laion/alvin/Dataset/getty_human_midas --start={start_list[23]} --end=32987 &",
"wait"
    ]
    file.write("\n".join(lines))
    file.write("\n")