import os

# l1 = [x for x in sorted(os.listdir('/fsx_laion/alvin/Dataset/getty_images_webdataset_human')) if x.endswith('.tar')]
# l2 = [x for x in sorted(os.listdir('/fsx_laion/alvin/Dataset/getty_human_merge')) if x.endswith('.tar')]

# l1 = [x for x in sorted(os.listdir('/fsx_laion/alvin/Dataset/COYO-700M-512-min-image-size200_human')) if x.endswith('.tar')]
# l2 = [x for x in sorted(os.listdir('/fsx_laion/alvin/Dataset/coyo_human_structural')) if x.endswith('.tar')]

l1 = [x for x in sorted(os.listdir('/fsx_laion/del22/fake_data_sdxl_human_eular')) if x.endswith('.tar')]
l2 = [x for x in sorted(os.listdir('/fsx_laion/alvin/Dataset/fake_sdxl_human_eular')) if x.endswith('.tar')]
# numbers = []
# with open("/fsx_laion/alvin/sd4human/mmpose_coyo_miss.txt", "r") as file:
#     for line in file:
#         numbers.append(int(line.strip()))
# numbers = numbers[:7360]
# l1 = [x for x in sorted(os.listdir('/fsx_laion/alvin/Dataset/COYO-700M-512-min-image-size200_human')) if x.endswith('.tar')]
# l2 = [x for x in sorted(os.listdir('/fsx_laion/alvin/Dataset/coyo_human_depnorm')) if x.endswith('.tar')]
# l1 = l1[129600:131200]
# l1 = l1[25600:27200]
# l1 = l1[14800:15200]
l = []
for i, x in enumerate(l1):
    if x not in l2:
        l.append(i)
print(len(l))
print(l)
# for num in l:
#     if num % 50 == 0:
#         print(num)
# numbers = []
# with open("/fsx_laion/alvin/sd4human/ldmk_laion_miss.txt", "r") as file:
#     for line in file:
#         numbers.append(int(line.strip()))
# numbers = numbers[:1200]    
# # print(numbers[42], )        
# l = []
# # # c = []
# for i, x in enumerate(numbers):
#     if l1[x] not in l2:
#         l.append(i)
# #         # c.append(x)
# sup = []
# for num in l:
#     if (num - 1) not in l and (num + 1) in l:
#         sup.append(num - 1)
# l = l + sup

# # # print(len(l))
# # # l += [80483, 79685, 81282]
# # print(l)
# miss = [numbers[x] for x in l]
# # print(c)
# with open("mmpose-depth-normal-fake3-miss.txt", "w") as file:
#     for num in l:
#         file.write(str(num) + "\n")

# sub = []
# with open("/fsx_laion/alvin/sd4human/mmpose-getty-miss.txt", "r") as file:
#     for line in file:
#         sub.append(int(line.strip()))
# sub = sub[800:]
# res = []
# for y in (sub):
#     if y in l:
#         res.append(y)
# print(len(res))