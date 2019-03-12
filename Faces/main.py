# -*- coding: utf-8 -*-
import os
import pickle
import posixpath
import re
import time

from PIL import Image, ImageDraw, ImageFont

import Faces

CHECK_DIR = "./unknown"
SAVE_DIR = "./result"
THRESHOLD = 0.52                            # 对相似人脸编码上欧氏距离的阈值
NUM_JITTERS = 1                             # 获取图像编码时抖动次数

# time_last = time.time()

# 从本地加载已知人脸的编码
with open("./data/face_encodings.data", "rb") as f:
    known_face_encodings = pickle.load(f)

# 从本地加载已知人脸所属的id(0~40)
with open("./data/id_list.data", "rb") as f:
    id_list = pickle.load(f)

# 从本地加载id到中文姓名的映射表
with open("./data/name_list.data", "rb") as f:
    name_list = pickle.load(f)

# time_next = time.time()
# print("[t]加载原始数据，花费时间{}s".format((time_next - time_last)))


def find_min_index(number_list):
    """
    返回ndarray数组中的最小值的下标
    """
    min_number = float("inf")
    min_index = 0
    index = 0
    for number in number_list:
        if number < min_number:
            min_number = number
            min_index = index
        index += 1
    return min_index


def test_image(image_path, known_face_encodings_, threshold=0.6):
    time_start = time.time()
    print("\n[f]当前检测的图片为:{}".format(image_path))
    image = Faces.load_image_file(image_path)
    stdface = Faces.get_stdface(image)

    # 获取人脸位置
    top, right, bottom, left = 0, 255, 255, 0
    
    # 获取人脸编码
    test_encoding = Faces.face_encodings(stdface, [(top, right, bottom, left)], 1)[0]

    distances = Faces.face_distance(known_face_encodings_, test_encoding)

    min_distance_id = find_min_index(distances)
    min_distance = distances[min_distance_id]
    similarity = Faces.get_similarity(min_distance)
    if min_distance > threshold:
        predict_result = "未知"
        flag = '[-]'
    else:
        predict_result = name_list[id_list[min_distance_id]]
        flag = '[+]'

    print(f"{flag}预测结果为：{predict_result:10}最近欧式距离{min_distance:.3f}    相似度{similarity:.2f}")

    # 绘制结果
    pil_image = Image.fromarray(stdface)
    draw = ImageDraw.Draw(pil_image)

    draw.rectangle(((left, bottom - 20), (right, bottom)), fill=(220, 133, 0), outline=(220, 133, 0))

    font = ImageFont.truetype("simhei.ttf", 15, encoding="utf-8")
    draw.text((left + 6, bottom - 18), predict_result, fill=(255, 255, 255, 255), font=font)

    time_end = time.time()
    print("[t]检测本张图片，花费时间{}s".format((time_end - time_start)))

    del draw
    pil_image.show()
    pil_image.save(posixpath.join("result", image_path.split("/")[-1]))
    

def main():
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    for file in os.listdir(CHECK_DIR):
        file_path = posixpath.join(CHECK_DIR, file)
        test_image(file_path, known_face_encodings, THRESHOLD)


if __name__ == "__main__":
    main()
