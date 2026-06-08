import webdataset as wds
import numpy as np
import PIL
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import io
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import os
from tqdm import tqdm
import cv2
import math
import argparse

def imshow_keypoints(img,
                     pose_result,
                     skeleton=None,
                     kpt_score_thr=0.3,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=4,
                     thickness=1,
                     show_keypoint_weight=False,
                     height=None,
                     width=None):
    """Draw keypoints and links on an image.

    Args:
            img (str or Tensor): The image to draw poses on. If an image array
                is given, id will be modified in-place.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_link_color (np.array[Mx3]): Color of M links. If None, the
                links will not be drawn.
            thickness (int): Thickness of lines.
    """

    # img = mmcv.imread(img)
    # img_h, img_w, _ = img.shape
    if img is None:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img_h, img_w = height, width
    else:
        img_h, img_w, _ = img.shape

    for kpts in pose_result:

        kpts = np.array(kpts, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)
            for kid, kpt in enumerate(kpts):
                if kid in [17, 18, 19, 20, 21, 22]:
                    continue
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]
                if kpt_score > kpt_score_thr:
                    color = tuple(int(c) for c in pose_kpt_color[kid])
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        cv2.circle(img_copy, (int(x_coord), int(y_coord)),
                                   radius, color, -1)
                        transparency = max(0, min(1, kpt_score))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                    else:
                        cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                                   color, -1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)
            for sk_id, sk in enumerate(skeleton):
                if sk[0] in [17, 18, 19, 20, 21, 22] or sk[1] in [17, 18, 19, 20, 21, 22]:
                    continue
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                # if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                #         and pos1[1] < img_h and pos2[0] > 0 and pos2[0] < img_w
                #         and pos2[1] > 0 and pos2[1] < img_h
                #         and kpts[sk[0], 2] > kpt_score_thr
                #         and kpts[sk[1], 2] > kpt_score_thr):
                if (kpts[sk[0], 2] > kpt_score_thr
                        and kpts[sk[1], 2] > kpt_score_thr):
                    color = tuple(int(c) for c in pose_link_color[sk_id])
                    if show_keypoint_weight:
                        img_copy = img.copy()
                        X = (pos1[0], pos2[0])
                        Y = (pos1[1], pos2[1])
                        mX = np.mean(X)
                        mY = np.mean(Y)
                        length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                        angle = math.degrees(
                            math.atan2(Y[0] - Y[1], X[0] - X[1]))
                        stickwidth = thickness
                        polygon = cv2.ellipse2Poly(
                            (int(mX), int(mY)),
                            (int(length / 2), int(stickwidth)), int(angle), 0,
                            360, 1)
                        cv2.fillConvexPoly(img_copy, polygon, color)
                        transparency = max(
                            0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                        cv2.addWeighted(
                            img_copy,
                            transparency,
                            img,
                            1 - transparency,
                            0,
                            dst=img)
                    else:
                        cv2.line(img, pos1, pos2, color, thickness=thickness)

    return img

def draw_whole_body_skeleton(
    img,
    pose,
    radius=4,
    thickness=1,
    kpt_score_thr=0.3,
    height=None,
    width=None,
    ):
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255], [0, 0, 0]])
    
    # below are for the whole body keypoints
    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2],
                        [1, 3], [2, 4], [3, 5], [4, 6], 
                        [15, 17], [15, 18],
                        [15, 19], [16, 20], [16, 21], [16, 22], 
                        [91, 92],
                        [92, 93], [93, 94], [94, 95], [91, 96], [96, 97],
                        [97, 98], [98, 99], [91, 100], [100, 101], [101, 102],
                        [102, 103], [91, 104], [104, 105], [105, 106],
                        [106, 107], [91, 108], [108, 109], [109, 110],
                        [110, 111], [112, 113], [113, 114], [114, 115],
                        [115, 116], [112, 117], [117, 118], [118, 119],
                        [119, 120], [112, 121], [121, 122], [122, 123],
                        [123, 124], [112, 125], [125, 126], [126, 127],
                        [127, 128], [112, 129], [129, 130], [130, 131],
                        [131, 132]]

    # pose_link_color = palette[[
    #     0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
    # ] + [16, 16, 16, 16, 16, 16] + [
    #     0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
    #     16
    # ] + [
    #     0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
    #     16
    # ]]
    pose_link_color = palette[[
        0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
    ] + [20] * 6 + [
        0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
        16
    ] + [
        0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
        16
    ]]
    pose_kpt_color = palette[
        [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0] +
        [20, 20, 20, 20, 20, 20] + [19] * (68 + 42)]
    
    draw = imshow_keypoints(img, pose, skeleton, 
                     kpt_score_thr=0.3,
                     pose_kpt_color=pose_kpt_color,
                     pose_link_color=pose_link_color,
                     radius=radius,
                     thickness=thickness,
                     show_keypoint_weight=True,
                     height=height,
                     width=width)
    return draw

def draw_body_skeleton(
    img,
    pose,
    radius=4,
    thickness=1,
    kpt_score_thr=0.3,
    height=None,
    width=None,
    ):
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])
    
    # below are for the body keypoints
    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                        [3, 5], [4, 6]]

    pose_link_color = palette[[
        0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
    ]]
    pose_kpt_color = palette[[
        16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
    ]]
    draw = imshow_keypoints(img, pose, skeleton, 
                     kpt_score_thr=kpt_score_thr,
                     pose_kpt_color=pose_kpt_color,
                     pose_link_color=pose_link_color,
                     radius=radius,
                     thickness=thickness,
                     show_keypoint_weight=True,
                     height=height,
                     width=width)
    return draw

def draw_face_skeleton(
    img,
    pose,
    radius=4,
    thickness=1,
    kpt_score_thr=0.3,
    height=None,
    width=None,
    ):
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])
    
    # below are for the face keypoints
    skeleton = []

    pose_link_color = palette[[]]
    pose_kpt_color = palette[[19] * 68]
    kpt_score_thr = 0
    
    draw = imshow_keypoints(img, pose, skeleton, 
                     kpt_score_thr=kpt_score_thr,
                     pose_kpt_color=pose_kpt_color,
                     pose_link_color=pose_link_color,
                     radius=radius,
                     thickness=thickness,
                     show_keypoint_weight=True,
                     height=height,
                     width=width)
    return draw

def draw_hand_skeleton(
    img,
    pose,
    radius=4,
    thickness=1,
    kpt_score_thr=0.3,
    height=None,
    width=None,
    ):
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255],
                            [255, 0, 0], [255, 255, 255]])
    
    # hand option 1
    skeleton = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7],
                        [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13],
                        [13, 14], [14, 15], [15, 16], [0, 17], [17, 18],
                        [18, 19], [19, 20]]

    pose_link_color = palette[[
        0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
        16
    ]]
    pose_kpt_color = palette[[
        0, 0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16,
        16, 16
    ]]
    
    # # hand option 2
    # skeleton = [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [8, 9],
    #                     [9, 10], [10, 11], [12, 13], [13, 14], [14, 15],
    #                     [16, 17], [17, 18], [18, 19], [3, 20], [7, 20],
    #                     [11, 20], [15, 20], [19, 20]]

    # pose_link_color = palette[[
    #     0, 0, 0, 4, 4, 4, 8, 8, 8, 12, 12, 12, 16, 16, 16, 0, 4, 8, 12,
    #     16
    # ]]
    # pose_kpt_color = palette[[
    #     0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12, 16, 16, 16,
    #     16, 0
    # ]]
    
    draw = imshow_keypoints(img, pose, skeleton, 
                     kpt_score_thr=kpt_score_thr,
                     pose_kpt_color=pose_kpt_color,
                     pose_link_color=pose_link_color,
                     radius=radius,
                     thickness=thickness,
                     show_keypoint_weight=True,
                     height=height,
                     width=width)
    return draw


def parse():
    parser = argparse.ArgumentParser(description='PyTorch DDP Training')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    args = parser.parse_args()
    return args


args = parse()

import json


# pipeline = [wds.SimpleShardList(['/fsx_laion/alvin/Dataset/getty_human_structural/00016fdf-75da-48f1-8849-7b7500ddc4f9.tar']), \
#     wds.tarfile_to_samples(), \
#     wds.to_tuple("jpg", "txt", "__key__", "__url__", "json", "body_kp", "body_kpconf", "face_kp", "face_kpconf", "hand_kp", "hand_kpconf"), \
#     wds.batched(8, partial=True)
# ]

# dataset = wds.DataPipeline(*pipeline)

# dataset = wds.WebDataset(['/fsx_laion/alvin/Dataset/getty_human_structural/00016fdf-75da-48f1-8849-7b7500ddc4f9.tar'])
# dataset = wds.WebDataset(['/fsx_laion/alvin/Dataset/getty_human_depnorm/00016fdf-75da-48f1-8849-7b7500ddc4f9.tar'])
# all_tar_list = ['/fsx_laion/alvin/Dataset/getty_human_merge/00016fdf-75da-48f1-8849-7b7500ddc4f9.tar']
# depnorm: [29549, 39097, 38347, 37547, 37847, 37897, 38597]
# mmpose: [51594, 53190, 53290, 50589, 63187, 62887, 49190, 61687, 61987, 58084]
# numbers = [51676, 52227, 51275, 51375, 51475, 51527, 51594, 65811, 65911, 53190, 53290, 50278, 50525, 50575, 50589, 54171, 54471, 42797, 42897, 42947, 61511, 64711, 50725, 40546, 55866, 41096, 45144, 63187, 63214, 63364, 62887, 42197, 56816, 57066, 64161, 64361, 64561, 49190, 48228, 61687, 61987, 44994, 55316, 58084, 57516, 41146, 41346]
# all_tar_list = ['/fsx_laion/alvin/Dataset/coyo_human_depnorm/38597.tar']
# numbers = [0]
# numbers = [123397, 123408, 123425, 123435, 123453, 123552, 123627, 123635, 123646, 123663, 123689, 53150, 53551, 20534, 40226, 16354, 49016, 55050, 12597, 89741, 115331, 67334, 43916, 44116, 41625, 48516, 68034, 129788, 129879, 56713, 131060, 137142, 107060, 90343, 27721, 58313, 58713, 128242, 29621, 29720, 78432, 110245, 34919, 35318, 35418, 57514, 26421, 26520, 25021, 80334, 60910, 46616, 103764, 103806, 103941, 104031, 104050, 10407, 104213, 104222, 104265, 104278, \
# 104288, 104377, 135506, 104099, 2041, 2178, 2567, 2575, 2588, 2595, 2715, 2746, 2758, 102856, 102947, 91, 191, 642, 742, 942, 98831, 98882, 100538, 100901, 9891, 8391, 8892, 6964, 7191, 7365]
# numbers = [10592, 10992, 13142, 13242, 14141, 14187, 14336, 14435, 14456, 14477, 14635, 14645, 14692, 14706, 14717, 14815, 15391, 17767, 21270, 26865, 35737, 35749, 35768, 35779, 35799, 35908, 35990, 35999, 36011, 36030, 36058, 38567, 41067, 42767, 42867, 44167, 49057, 50857, 53589, 57769, 62256, 63656, 63755, 64956, 66856, 66955, 72154, 72553, 72653, 77461, 78860, 81151, 81351, 83851, 85751, 86251, 90385, 90786, 92285, 93948, 94749, 95548, 95948, 98145, 104569, 105269, 115667, 117569, 126976, 127578, 136066, 136117]
all_tar_list = [os.path.join('/fsx_laion/alvin/Dataset/getty_human_1024x1024', x) for x in sorted(os.listdir('/fsx_laion/alvin/Dataset/getty_human_1024x1024')) if x.endswith('.tar')]
# all_tar_list = all_tar_list[args.start : args.end]
# all_tar_list = ["/fsx_laion/alvin/Dataset/getty_human_aes/1a90356c-989f-4d0b-b7ef-012a59d2644e.tar"]
# all_tar_list = ['/fsx_laion/alvin/Dataset/fake_sdxl_human/2cd62efa6-717b-458a-a816-a56283f7b185.tar']
fail_list = []
sample_count = 0
# os.makedirs("/fsx_laion/alvin/visualization/pose-vis-whole", exist_ok=True)
# os.makedirs("/fsx_laion/alvin/visualization/ori", exist_ok=True)
# os.makedirs("/fsx_laion/alvin/visualization/inpaint-verify", exist_ok=True)
cnt = 0
for ii, name in tqdm(enumerate(all_tar_list)):
    # dataset = wds.WebDataset(['/fsx_laion/alvin/Dataset/getty_human_structural/3a00453a-497c-4bcd-bb70-b7c9ad6d9a4f.tar'])
    # try:
    for k in range(1):
        tar_name = name.split('/')[-1]
        dataset1 = wds.WebDataset(os.path.join("/fsx_laion/alvin/Dataset/getty_human_1024x1024", tar_name))
        for i, sample in enumerate(dataset1):
            print(sample["blip_humanft"])
            exit(0)
                # if 'jpg' in sample.keys() and 'new_body_kp' in sample.keys() and 'midas_depth' in sample.keys() and 'omni_normal' in sample.keys():          
                #     cnt += 1
                # else:
                #     fail_list.append(tar_name)
        # print(cnt)
        #     print((sample['json'].decode('utf-8')).keys())
        #     exit(0)
        #     print(i)
        #     print(sample["blip_humanft"])
        #     print(sample["blip"])
        #     print(sample["txt"])
        #     with io.BytesIO(sample["jpg"]) as stream:
        #         img = PIL.Image.open(stream)
        #         img.load()
        #         img = img.convert("RGB")
        #         # print(img.size)
        #         img.save(f"/fsx_laion/alvin/visualization/ori/{i}.png")
        #     if i >= 15:
        #         exit(0)
            # print(sample[])
            # exit(0)
        # dataset2 = wds.WebDataset(os.path.join("/fsx_laion/alvin/Dataset/getty_human_blipft", tar_name))
        # # # dataset3 = wds.WebDataset(os.path.join("/fsx_laion/alvin/Dataset/coyo_human_ldmk/", tar_name))
        # n1=n2=0
        # # # dataset2 = wds.WebDataset(os.path.join("/fsx_laion/alvin/Dataset/laion_human_final/", tar_name))
        # # # dataset = wds.WebDataset([name])
        # # # sample_count = 0
        # # # i = 0
        # # # try:
        # for i, sample in enumerate(dataset1):
        #     n1+=1
        # for i, sample in enumerate(dataset2):
        #     n2+=1
        # # # for i, sample in enumerate(dataset3):
        # # #     n3+=1
        # if not (n1 == n2):
        #     fail_list.append(ii + args.start)
        # except:
        #     print(n1, n2)
        #     fail_list.append(ii + args.start)
            # i += 1
            # print(sample.keys())
            # break
            # # if i == 0:
            # x = sample["inpaint"]
            # x = sample["location"]
            # x = sample["ldmk"]
            # x = sample["omni_depth"]
            # break
            # type(sample["omni_normal"])
            # type(sample["omni_depth"])
            # type(sample["blip"])
            # type(sample["__key__"])
            # type(sample["__url__"])
            # type(sample["jpg"])
            # type(sample["json"])
            # type(sample["txt"])
            # type(sample["body_kp"])
            # type(sample["body_kpconf"])
            # type(sample["face_kp"])
            # type(sample["face_kpconf"])
            # type(sample["hand_kp"])
            # type(sample["hand_kpconf"])
            # break
            # assert type(sample["__key__"]) is str
            # assert type(sample["__url__"]) is str
            # assert type(sample["jpg"]) is byte
            # assert "json" in sample.keys()
            # assert "txt" in sample.keys()
            # assert "__key__" in sample.keys()
            
            # if i <= 32:
            #     # print(sample.keys())
            #     string_data = sample["json"].decode('utf-8')
            #     string_data = json.loads(string_data)
            #     print(string_data.keys())
            #     print(string_data)
            #     exit(0)
                # print(string_data["height"], string_data["width"])
                # # print(type(sample["__key__"]))
                # print(sample["__key__"])
                # # # print(type(sample["__url__"]))
                # print(sample["__url__"])
                # print("*******************************")
                # print(type(sample["location"]))
                # print(np.frombuffer(sample["location"], dtype=np.float32).shape)
                # print(np.frombuffer(sample["location"], dtype=np.float32))
                # location = np.frombuffer(sample["location"], dtype=np.float32)
                # print(location[0])
                # print(location[1])
                # print(location[2])
                # print(location[3])
                # print(location[4])
                # with io.BytesIO(sample["midas_depth"]) as stream:
                #     img = PIL.Image.open(stream)
                #     img.load()
                #     img = img.convert("RGB")
                # print(img.size)
                # img.save(f'/fsx_laion/alvin/visualization/midas/depth-tar-{i}.png')
                # with io.BytesIO(sample["jpg"]) as stream:
                #     img = PIL.Image.open(stream)
                #     img.load()
                #     img = img.convert("RGB")
                # img.save(f"/fsx_laion/alvin/visualization/ori-vis/{i}.png")
                # print(type(sample["jpg"]))
                # print(type(sample["json"]))
                # print(type(sample["txt"]))
                # print(sample["txt"].decode("utf-8"))
                # # print("*******************************")
                # # print(type(sample["blip"]))
                # print(type(sample["blip"].decode("utf-8")))
                # print(sample["blip"].decode("utf-8"))
                # # print("*******************************")
                # # print(type(sample["body_kp"]))
                # # print(type(sample["body_kpconf"]))
                # # # type(sample["body_kp"])
                # # # type(sample["body_kpconf"])
                # print(np.frombuffer(sample["location"], dtype=np.float32))
                # print(np.frombuffer(sample["ldmk"], dtype=np.float32).reshape(-1, 68, 3))
            
                # # print(np.frombuffer(sample["body_kpconf"], dtype=np.float32).shape)
                # print(np.frombuffer(sample["new_body_bbox"], dtype=np.float32).reshape(-1, 4).shape)
                # print(np.frombuffer(sample["new_body_bbox_score"], dtype=np.float32).shape)
                
                # print(np.frombuffer(sample["new_body_kp"], dtype=np.float32).reshape(-1, 17, 2))
                # print(np.frombuffer(sample["new_body_kp"], dtype=np.float32).reshape(-1, 17, 2).shape)
                # print(np.frombuffer(sample["new_body_kp_score"], dtype=np.float32).reshape(-1, 17).shape)
                
                # print(np.frombuffer(sample["new_face_bbox"], dtype=np.float32).reshape(-1, 4).shape)
                # print(np.frombuffer(sample["new_face_bbox_score"], dtype=np.float32).shape)
                
                # print(np.frombuffer(sample["new_face_kp"], dtype=np.float32).reshape(-1, 68, 2).shape)
                # print(np.frombuffer(sample["new_face_kp_score"], dtype=np.float32).reshape(-1, 68).shape)
                
                # print(np.frombuffer(sample["new_hand_bbox"], dtype=np.float32).reshape(-1, 4).shape)
                # print(np.frombuffer(sample["new_hand_bbox_score"], dtype=np.float32).shape)
                
                # print(np.frombuffer(sample["new_hand_kp"], dtype=np.float32).reshape(-1, 21, 2).shape)
                # print(np.frombuffer(sample["new_hand_kp_score"], dtype=np.float32).reshape(-1, 21).shape)
                
                # print(np.frombuffer(sample["new_wholebody_bbox"], dtype=np.float32).reshape(-1, 4).shape)
                # print(np.frombuffer(sample["new_wholebody_bbox_score"], dtype=np.float32).shape)
                
                # print(np.frombuffer(sample["new_wholebody_kp"], dtype=np.float32).reshape(-1, 133, 2).shape)
                # print(np.frombuffer(sample["new_wholebody_kp_score"], dtype=np.float32).reshape(-1, 133).shape)
                
                # print(np.frombuffer(sample["new_i_body_bbox"], dtype=np.float32).reshape(-1, 4).shape)
                # print(np.frombuffer(sample["new_i_body_bbox_score"], dtype=np.float32).shape)
                
                # print(np.frombuffer(sample["new_i_body_kp"], dtype=np.float32).reshape(-1, 17, 2))
                # print(np.frombuffer(sample["new_i_body_kp"], dtype=np.float32).reshape(-1, 17, 2).shape)
                # print(np.frombuffer(sample["new_i_body_kp_score"], dtype=np.float32).reshape(-1, 17).shape)
                
                # print(np.frombuffer(sample["new_i_face_bbox"], dtype=np.float32).reshape(-1, 4).shape)
                # print(np.frombuffer(sample["new_i_face_bbox_score"], dtype=np.float32).shape)
                
                # print(np.frombuffer(sample["new_i_face_kp"], dtype=np.float32).reshape(-1, 68, 2).shape)
                # print(np.frombuffer(sample["new_i_face_kp_score"], dtype=np.float32).reshape(-1, 68).shape)
                
                # print(np.frombuffer(sample["new_i_hand_bbox"], dtype=np.float32).reshape(-1, 4).shape)
                # print(np.frombuffer(sample["new_i_hand_bbox_score"], dtype=np.float32).shape)
                
                # print(np.frombuffer(sample["new_i_hand_kp"], dtype=np.float32).reshape(-1, 21, 2).shape)
                # print(np.frombuffer(sample["new_i_hand_kp_score"], dtype=np.float32).reshape(-1, 21).shape)
                
                # print(np.frombuffer(sample["new_i_wholebody_bbox"], dtype=np.float32).reshape(-1, 4).shape)
                # print(np.frombuffer(sample["new_i_wholebody_bbox_score"], dtype=np.float32).shape)
                
                # print(np.frombuffer(sample["new_i_wholebody_kp"], dtype=np.float32).reshape(-1, 133, 2).shape)
                # print(np.frombuffer(sample["new_i_wholebody_kp_score"], dtype=np.float32).reshape(-1, 133).shape)
                
                # print(np.frombuffer(sample["body_kpconf"], dtype=np.float32).shape)
                
                
                # body_kp = np.frombuffer(sample["body_kp"], dtype=np.float32).reshape(17, 2)
                # body_kpconf = np.frombuffer(sample["body_kpconf"], dtype=np.float32)
                # body_all = np.concatenate([body_kp, body_kpconf[:, np.newaxis]], axis=1)
                # body_all = body_all[np.newaxis, ...]
                
                # body_kp = np.frombuffer(sample2["new_body_kp"], dtype=np.float32).reshape(-1, 17, 2)
                # body_kpconf = np.frombuffer(sample2["new_body_kp_score"], dtype=np.float32).reshape(-1, 17)
                # body_all = np.concatenate([body_kp, body_kpconf[..., np.newaxis]], axis=2)
                # # with io.BytesIO(sample["jpg"]) as stream:
                # #     img = PIL.Image.open(stream)
                # #     img.load()
                # #     img = img.convert("RGB")
                
                # body_draw = draw_body_skeleton(
                #     img=np.array(img),
                #     pose=body_all,
                #     radius=4, 
                #     thickness=1
                # )
                # body_draw = Image.fromarray(body_draw)
                
                # body_draw.save(f"/fsx_laion/alvin/visualization/pose-vis3/{i}.png")
                
                # body_draw_blank = draw_body_skeleton(
                #     img=None,
                #     pose=body_all,
                #     radius=20, 
                #     thickness=10,
                #     height=img.height,
                #     width=img.width
                # )
                # body_draw_blank = Image.fromarray(body_draw_blank)
                
                # body_draw_blank.save(f"/fsx_laion/alvin/visualization/pose-vis/{i}-blank.png")
                
                
                # face_kp = np.frombuffer(sample["face_kp"], dtype=np.float32).reshape(-1, 98, 2)
                # face_kpconf = np.frombuffer(sample["face_kpconf"], dtype=np.float32).reshape(-1, 98)   
                # face_all = np.concatenate([face_kp, face_kpconf[..., np.newaxis]], axis=2)
                
                # face_draw = draw_face_skeleton(
                #     img=np.array(img),
                #     # img=None,
                #     pose=face_all,
                #     radius=6, 
                #     thickness=3,
                #     height=img.height,
                #     width=img.width
                # )
                # face_draw = Image.fromarray(face_draw)
                # face_draw = transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR)(face_draw)
                
                # face_draw.save(f"/fsx_laion/alvin/visualization/pose-vis/{i}-face.png")

                # hand_kp = np.frombuffer(sample["hand_kp"], dtype=np.float32).reshape(-1, 21, 2)
                # hand_kpconf = np.frombuffer(sample["hand_kpconf"], dtype=np.float32).reshape(-1, 21)   
                # hand_all = np.concatenate([hand_kp, hand_kpconf[..., np.newaxis]], axis=2)
                
                # hand_draw = draw_hand_skeleton(
                #     img=np.array(img),
                #     # img=None,
                #     pose=hand_all,
                #     radius=12, 
                #     thickness=6,
                #     height=img.height,
                #     width=img.width
                # )
                # hand_draw = Image.fromarray(hand_draw)
                # hand_draw = transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR)(hand_draw)
                
                # hand_draw.save(f"/fsx_laion/alvin/visualization/pose-vis/{i}-hand.png")
                
                # print(type(sample["face_kp"]))
                # print(type(sample["face_kpconf"]))
                # # type(sample["face_kp"])
                # # type(sample["face_kpconf"])
                # print(np.frombuffer(sample["face_kp"], dtype=np.float32).reshape(-1, 98, 2).shape)
                # print(np.frombuffer(sample["face_kpconf"], dtype=np.float32).reshape(-1, 98).shape)
                # print(np.frombuffer(sample["face_kp"], dtype=np.float32).reshape(-1, 98, 2))
                # print(np.frombuffer(sample["face_kpconf"], dtype=np.float32).reshape(-1, 98))
                # print(type(sample["hand_kp"]))
                # print(type(sample["hand_kpconf"]))
                # # type(sample["hand_kp"])
                # # type(sample["hand_kpconf"])
                # print(np.frombuffer(sample["hand_kp"], dtype=np.float32).reshape(-1, 21, 2).shape)
                # print(np.frombuffer(sample["hand_kpconf"], dtype=np.float32).reshape(-1, 21).shape)
                # print(np.frombuffer(sample["hand_kp"], dtype=np.float32).reshape(-1, 21, 2))
                # print(np.frombuffer(sample["hand_kpconf"], dtype=np.float32).reshape(-1, 21))
                # print("*******************************")
                # # print(type(sample["depth"]))
                # # with io.BytesIO(sample["depth"]) as stream:
                # #     img = PIL.Image.open(stream)
                # #     img.load()
                # #     img = img.convert("RGB")
                # # print(img.size)
                # # img.save("/fsx_laion/alvin/visualization/depth.png")
                
                # with io.BytesIO(sample["jpg"]) as stream:
                #     img = PIL.Image.open(stream)
                #     img.load()
                #     img = img.convert("RGB")
                # print(img.size)
                # img.save(f"/fsx_laion/alvin/visualization/ori/{i}.png")
                
            # with io.BytesIO(sample["jpg"]) as stream:
            #     img = PIL.Image.open(stream)
            #     img.load()
            #     img = img.convert("RGB")
                
            # location = np.frombuffer(sample["location"], dtype=np.float32)
            # body_kp = np.frombuffer(sample["new_i_body_kp"], dtype=np.float32).reshape(-1, 17, 2)
            # x_coord = (body_kp[:, :, 0] - location[0]) / location[2] * location[7]
            # y_coord = (body_kp[:, :, 1] - location[1]) / location[3] * location[8]
            # body_kp = np.stack([x_coord, y_coord], axis=2)
            # # print(body_kp.shape)
            # body_kpconf = np.frombuffer(sample["new_i_body_kp_score"], dtype=np.float32).reshape(-1, 17)
            # body_all = np.concatenate([body_kp, body_kpconf[..., np.newaxis]], axis=2)
            # # with io.BytesIO(sample["jpg"]) as stream:
            # #     img = PIL.Image.open(stream)
            # #     img.load()
            # #     img = img.convert("RGB")
            
            # body_draw = draw_body_skeleton(
            #     img=np.array(img),
            #     pose=body_all,
            #     radius=4, 
            #     thickness=1
            # )
            
            # # # location = np.frombuffer(sample["location"], dtype=np.float32)
            # # body_kp = np.frombuffer(sample["new_i_body_kp"], dtype=np.float32).reshape(-1, 17, 2)
            # # x_coord = (body_kp[:, :, 0] - location[0]) / location[2] * location[7]
            # # y_coord = (body_kp[:, :, 1] - location[1]) / location[3] * location[8]
            # # body_kp = np.stack([x_coord, y_coord], axis=2)
            # # # print(body_kp.shape)
            # # body_kpconf = np.frombuffer(sample["new_i_body_kp_score"], dtype=np.float32).reshape(-1, 17)
            # # body_all = np.concatenate([body_kp, body_kpconf[..., np.newaxis]], axis=2)
            # # # with io.BytesIO(sample["jpg"]) as stream:
            # # #     img = PIL.Image.open(stream)
            # # #     img.load()
            # # #     img = img.convert("RGB")
            
            # # body_draw = draw_body_skeleton(
            # #     img=np.array(img),
            # #     pose=body_all,
            # #     radius=4, 
            # #     thickness=1
            # # )
            
            # # location = np.frombuffer(sample["location"], dtype=np.float32)
            # face_kp = np.frombuffer(sample["new_i_face_kp"], dtype=np.float32).reshape(-1, 68, 2)
            # x_coord = (face_kp[:, :, 0] - location[0]) / location[2] * location[7]
            # y_coord = (face_kp[:, :, 1] - location[1]) / location[3] * location[8]
            # face_kp = np.stack([x_coord, y_coord], axis=2)

            # face_kpconf = np.frombuffer(sample["new_i_face_kp_score"], dtype=np.float32).reshape(-1, 68)
            # face_all = np.concatenate([face_kp, face_kpconf[..., np.newaxis]], axis=2)
            # # with io.BytesIO(sample["jpg"]) as stream:
            # #     img = PIL.Image.open(stream)
            # #     img.load()
            # #     img = img.convert("RGB")
            
            # face_draw = draw_face_skeleton(
            #     img=np.array(body_draw),
            #     pose=face_all,
            #     radius=4, 
            #     thickness=1
            # )
            
            # hand_kp = np.frombuffer(sample["new_i_hand_kp"], dtype=np.float32).reshape(-1, 21, 2)
            # x_coord = (hand_kp[:, :, 0] - location[0]) / location[2] * location[7]
            # y_coord = (hand_kp[:, :, 1] - location[1]) / location[3] * location[8]
            # hand_kp = np.stack([x_coord, y_coord], axis=2)
            # hand_kpconf = np.frombuffer(sample["new_i_hand_kp_score"], dtype=np.float32).reshape(-1, 21)   
            # hand_all = np.concatenate([hand_kp, hand_kpconf[..., np.newaxis]], axis=2)
            
            # hand_draw = draw_hand_skeleton(
            #     img=np.array(face_draw),
            #     # img=None,
            #     pose=hand_all,
            #     radius=12, 
            #     thickness=6,
            #     height=img.height,
            #     width=img.width
            # )
            
            # hand_draw = Image.fromarray(hand_draw)
            
            # hand_draw.save(f"/fsx_laion/alvin/visualization/pose-vis-inpaint-combine/{i}.png")
            
            # location = np.frombuffer(sample["location"], dtype=np.float32)
            # whole_kp = np.frombuffer(sample["new_i_wholebody_kp"], dtype=np.float32).reshape(-1, 133, 2)
            # x_coord = (whole_kp[:, :, 0] - location[0]) / location[2] * location[7]
            # y_coord = (whole_kp[:, :, 1] - location[1]) / location[3] * location[8]
            # whole_kp = np.stack([x_coord, y_coord], axis=2)
            # whole_kpconf = np.frombuffer(sample["new_i_wholebody_kp_score"], dtype=np.float32).reshape(-1, 133)  
            # whole_all = np.concatenate([whole_kp, whole_kpconf[..., np.newaxis]], axis=2)
            
            # whole_kp = np.frombuffer(sample["new_wholebody_kp"], dtype=np.float32).reshape(-1, 133, 2)
            # whole_kpconf = np.frombuffer(sample["new_wholebody_kp_score"], dtype=np.float32).reshape(-1, 133)  
            # whole_all = np.concatenate([whole_kp, whole_kpconf[..., np.newaxis]], axis=2)
            
            # whole_draw = draw_whole_body_skeleton(
            #     img=np.array(img),
            #     # img=None,
            #     pose=whole_all,
            #     radius=8, 
            #     thickness=2,
            # )
            # whole_draw = Image.fromarray(whole_draw)
            # whole_draw.save(f"/fsx_laion/alvin/visualization/pose-vis-whole/{i}.png")
            # if i >= 50:
            #     exit(0)
                
            # print(img.size)
                # img.save(f"/fsx_laion/alvin/visualization/new/{i}-inpaint.png")
                
                # print(type(sample["omni_normal"]))
                # sample["omni_normal"]
                # sample["omni_depth"]
                # with io.BytesIO(sample["omni_normal"]) as stream:
                #     img = PIL.Image.open(stream)
                #     img.load()
                #     img = img.convert("RGB")
                # print(img.size)
                # img.save("/fsx_laion/alvin/visualization/new/omni_normal.png")
                # # print(type(sample["canny"]))
                # with io.BytesIO(sample["omni_depth"]) as stream:
                #     img = PIL.Image.open(stream)
                #     img.load()
                #     img = img.convert("RGB")
                # print(img.size)
                # img.save("/fsx_laion/alvin/visualization/new/omni_depth.png")
                
                # print(type(sample["blip"]))
                # print(type(sample["blip"].decode("utf-8")))
                # print(type(sample["omni_depth"]))
                # with io.BytesIO(sample["omni_depth"]) as stream:
                #     img = PIL.Image.open(stream)
                #     img.load()
                #     img = img.convert("RGB")
                # print(img.size)
                # img = np.array(img)
                # print(img[0, 0])
                # print(img[1024, 1024])
                # break
                # # # print(transforms.ToTensor()(img).shape)
                # # # plt.imsave(f"/fsx_laion/alvin/visualization/omni_depth.png", transforms.ToTensor()(img).squeeze(), cmap='viridis') 
                # img.save("/fsx_laion/alvin/visualization/omni_depth.png")
               
                # exit()
            # if "__url__" in sample.keys():
            # sample_count += 1
                # if i >= 50:
                #     exit(0)
    # except:
    #     # print(str(e))
    #     fail_list.append(tar_name)
print(fail_list)
print(cnt)
# print("Number of samples in dataset:", sample_count)
