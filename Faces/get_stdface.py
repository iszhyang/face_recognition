import os
import posixpath
import time

import cv2

import Faces

ROOTDIR = "./known"
SAVEDIR = "./stdface"
DETECTION_DIR = "./detection"

error_path_list = []

if not os.path.exists(SAVEDIR):
    os.mkdir(SAVEDIR)

if not os.path.exists(DETECTION_DIR):
    os.mkdir(DETECTION_DIR)

for directory in os.listdir(ROOTDIR):
    save_path = posixpath.join(SAVEDIR, directory)
    detection_dir_path = posixpath.join(DETECTION_DIR, directory)
    directory_path = posixpath.join(ROOTDIR, directory)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if not os.path.exists(detection_dir_path):
        os.mkdir(detection_dir_path)

    for file in os.listdir(directory_path):
        image_path = posixpath.join(directory_path, file)
        print("[ ]正在处理{}".format(image_path))
        image = Faces.load_image_file(image_path)
        try:
            time_start = time.time()
            detection_save_path = posixpath.join(detection_dir_path, file)
            normalized_image = Faces.get_stdface(image, detection_save_path=detection_save_path)
            time_end = time.time()
            print("[t]花费时间{:.3f}s".format(time_end - time_start))

            basename = file.split('.')[0]
            new_path = posixpath.join(save_path, basename + "_256x256.jpg")

            Faces.save_image_file(new_path, normalized_image)
        except:
            error_path_list.append(image_path)
            print("[-]得到stdface的过程中出错")

# 将得到stdface过程中出错的图片路径出来
print("[ ]Error_file_path:{}".format(error_path_list))
