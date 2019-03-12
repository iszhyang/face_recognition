# -*- coding: utf-8 -*-
import os
import pickle
import posixpath
import re

import numpy as np

import Faces

FACES_FOLDER = "./stdface/"
NUM_JITTERS = 1                                             # 获取图像编码时抖动次数


def image_files_in_folder(folder):
    # 统一使用unix的/分隔符
    return [posixpath.join(folder, f_) for f_ in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f_, flags=re.I)]


id_list = []
face_encodings = []

print("---开始读取文件---")
for directory in os.listdir(FACES_FOLDER):
    print("[ ]开始处理文件夹{}".format(directory))
    directory_path = posixpath.join(FACES_FOLDER, directory)

    for file in image_files_in_folder(directory_path):
        basename = os.path.splitext(os.path.basename(file))[0]
        print("[ ]正在处理图片{}".format(file.split('/')[-1]))
        image = Faces.load_image_file(file)
        top, right, bottom, left = 0, 255, 255, 0
        # top, right, bottom, left = Faces.get_only_face(image, min_face_area=1000, upsample=2)
        # print("[+]人脸位置", (top, right, bottom, left))
        
        encoding = Faces.face_encodings(image, [(top, right, bottom, left)], NUM_JITTERS)
        id_list.append(int(directory))
        face_encodings.append(encoding)

        
# 对face_encodings做整形处理 整理成(n, 128)的形式
length = len(id_list)
np_face_encodings = np.asarray(face_encodings)
np_face_encodings = np_face_encodings.reshape(length, -1)

# 结果以pickle的形式保存到本地
print("[+]检测结束，开始保存结果")
with open("./data/id_list.data", "wb") as f:
    pickle.dump(id_list, f)

with open("./data/face_encodings.data", "wb") as f:
    pickle.dump(np_face_encodings, f)
print("[+]保存成功")
