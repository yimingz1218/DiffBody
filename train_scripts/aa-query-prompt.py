import os
import json

input = json.load(open('/fsx_laion/alvin/Dataset/coco/512x512-img_text_pose_val2014-human.json'))
keys_list = list(input.keys())
keys_list = [int(x) for x in keys_list]
keys_list.sort()

# keys_list = [567812, 566159, 565957, 564868, 564572, 564612, 564404, 563870, 563775, 559821, 559185, 557564, 557562, 555686, 555396, 555361, 552744, 551737, 550117, 549683, 546222, 546160, 544655, 543065, 542782, 538344, 526778, 526570, 522427]
keys_list = [147128]
# key = int(6658)

for key in keys_list:
    contents = input[str(key)]
    string_list = contents['captions']
    text = max(string_list, key=len)
    # if "Little girl in a chair gazing at three desserts displayed on a table" in text:
    #     print(key)
    #     break
    print(text)
    # print(string_list)

# text_list = []

# for i, key in enumerate(keys_list):
#     # image_id = key
#     content = input[f"{key}"]
#     # self.depth.append(f'/fsx_laion/alvin/Dataset/coco/512x512-val2014-human-structure/depth/COCO_val2014_{int(image_id):012d}.jpg')
#     # self.normal.append(f'/fsx_laion/alvin/Dataset/coco/512x512-val2014-human-structure/normal/COCO_val2014_{int(image_id):012d}.jpg')
#     # self.midas_depth.append(f'/fsx_laion/alvin/Dataset/coco/512x512-val2014-human-structure/midas_depth/COCO_val2014_{int(image_id):012d}.jpg')
#     text_list.append(max(content['captions'], key=len).strip('\n'))
    
# with open("ms-coco-2014-humanval-prompts.txt", "w") as file:
#     for text in text_list:
#         if text != "" and text != " ":
#             file.write(text + "\n")