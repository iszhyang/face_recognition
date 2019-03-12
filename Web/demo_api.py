# -*- coding: utf-8 -*-
import os
import pickle
import time

import cv2
import numpy as np
import tensorflow as tf

import Faces
from model import deepnn, image_to_tensor

IMAGE_TO_SAVE = "./result"
THRESHOLD = 0.375  # 对相似人脸编码上欧氏距离的阈值
NUM_JITTERS = 0  # 获取图像编码时抖动次数

CKPT_PATH = "data/checkpoint"  # checkpoint路径
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
EMOTIONS_CH = ['生气', '厌烦', '恐惧', '开心', '伤心', '惊喜', '平淡']

print("[ ]本地加载人脸数据中...")

# 从本地加载已知人脸的编码
with open("data/face_encodings.data", "rb") as f:
    known_face_encodings = pickle.load(f)

# 从本地加载已知人脸所属的id(0到40)
with open("data/id_list.data", "rb") as f:
    id_list = pickle.load(f)

# 从本地加载id到中文姓名的映射表
with open("data/name_list.data", "rb") as f:
    name_list = pickle.load(f)

# 从本地加载id对应的图片路径
with open("data/image_file_list.data", "rb") as f:
    image_file_list = pickle.load(f)

print("[+]人脸数据加载成功")

print("[ ]从本地加载用于表情识别的模型中...")

# 为神经网络的输入占位
face_x = tf.placeholder(tf.float32, [None, 2304])
# 神经网络的输出
y_conv = deepnn(face_x)
# 输出经过softmax函数
probs = tf.nn.softmax(y_conv)

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
sess = tf.Session()

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("[+]用于表情识别的模型加载成功")
else:
    print("[-]用于表情识别的模型加载失败")
    exit(1)


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


# 人脸识别 返回值1为预测结果(姓名) 返回值2为本人其他照片的列表 返回值2为相似度
def face_recognition(image_path, detection_save_path, threshold=THRESHOLD):
    """
    给Web端使用的api接口
    :param image_path:需要识别的人脸路径
    :param detection_save_path:人脸检测结果保存的路径
    :param threshold:判定是同一个人的阈值
    :return:预测结果(str), 明星其他照片的路径(list), 相似度(float)
    """
    # 加载图片 以RGB通道的形式加载
    image = Faces.load_image_file(image_path)

    # 截取得到图片内的stdface，并将人脸检测的结果存放在detection_save_path中
    time_start = time.time()
    stdface = Faces.get_stdface(image, detection_save_path=detection_save_path)
    time_end = time.time()
    print("[ ]得到stdface花费时间{}".format(time_end - time_start))

    # 检测不到人脸时 返回空值
    if stdface is None:
        return None, None, 0.0

    # 人脸位置为stdface的边界
    top, right, bottom, left = 0, 255, 255, 0

    # 获取人脸编码
    time_start = time.time()
    test_encoding = Faces.face_encodings(stdface, [(top, right, bottom, left)], NUM_JITTERS)[0]
    time_end = time.time()
    print("[ ]得到encoding花费时间{}".format(time_end - time_start))

    # 与已知人脸编码库对比 得到与所有人脸比对的距离
    distances = Faces.face_distance(known_face_encodings, test_encoding)

    # 查找最小距离
    min_distance_id = find_min_index(distances)
    min_distance = distances[min_distance_id]

    # 根据距离拟合相似度
    similarity = Faces.get_similarity(min_distance)

    image_paths = []

    # 最近距离超出阈值 认为人脸库中无匹配项
    if min_distance > threshold:
        return "unknown", image_paths, 0.0

    # 根据本地数据 得到预测ID 预测结果
    predict_result_id = id_list[min_distance_id]
    predict_result = name_list[predict_result_id]

    for image_path in image_file_list[predict_result_id]:
        # basename = posixpath.basename(image_path)
        # 如果要在本机上展示的话
        # image = cv2.imread(image_path)
        # cv2.imshow(basename, image)
        image_paths.append(image_path)

    return predict_result, image_paths, similarity


def stdface_recognition(stdface, threshold=THRESHOLD):
    """
    直接将stdface与已知人脸数据库做比对
    :param stdface:  256×256的stdface
    :param threshold: 阈值
    :return: 预测结果(str), 明星其他照片的路径(list), 相似度(float)
    """

    # 人脸位置为stdface的边界
    top, right, bottom, left = 0, 255, 255, 0

    # 获取人脸编码
    time_start = time.time()
    test_encoding = Faces.face_encodings(stdface, [(top, right, bottom, left)], NUM_JITTERS)[0]
    time_end = time.time()
    print("[ ]得到encoding花费时间{}".format(time_end - time_start))

    # 与已知人脸编码库对比 得到与所有人脸比对的距离
    distances = Faces.face_distance(known_face_encodings, test_encoding)

    # 查找最小距离
    min_distance_id = find_min_index(distances)
    min_distance = distances[min_distance_id]

    # 根据距离拟合相似度
    similarity = Faces.get_similarity(min_distance)

    image_paths = []

    # 最近距离超出阈值 认为人脸库中无匹配项
    if min_distance > threshold:
        return "unknown", image_paths, 0.0

    # 根据本地数据 得到预测ID 预测结果
    predict_result_id = id_list[min_distance_id]
    predict_result = name_list[predict_result_id]

    for image_path in image_file_list[predict_result_id]:
        # basename = posixpath.basename(image_path)
        # 如果要在本机上展示的话
        # image = cv2.imread(image_path)
        # cv2.imshow(basename, image)
        image_paths.append(image_path)

    return predict_result, image_paths, similarity


def format_image(image):
    """
    将stdface经过format变为48*48*1的尺寸 方便神经网络的读入
    :param image: 需要format的图像
    :return: format过后的图像
    """

    # 将图像resize到48*48
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)

    # 如果是三通道(RGB)图像 则将图像转为灰色
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def expression_recognition(image):
    """
    识别图像中人物的表情
    :param image: 传入的图像
    :return: 返回值1为表情id(int) 返回值2位表情指数数组(ndarray)
    """
    stdface = Faces.get_stdface(image)

    if stdface is None:
        return None, None

    # 先对stdface进行format操作 再将其拉直变为一维张量
    tensor = image_to_tensor(format_image(stdface))
    # 将图片转为的一维张量输入到神经网络中得到输出
    result = sess.run(probs, feed_dict={face_x: tensor})

    # 表情识别结果为指数最大的那个
    expression_id = np.argmax(result[0])

    return expression_id, result[0]


def stdface_expression_recognition(stdface):
    """
    直接识别stdface中人物的表情
    :param stdface: 传入的stdface
    :return: 返回值1为表情id(int) 返回值2位表情指数数组(ndarray)
    """
    # 先对stdface进行format操作 再将其拉直变为一维张量
    tensor = image_to_tensor(format_image(stdface))
    # 将图片转为的一维张量输入到神经网络中得到输出
    result = sess.run(probs, feed_dict={face_x: tensor})

    # 表情识别结果为指数最大的那个
    expression_id = np.argmax(result[0])

    return expression_id, result[0]


def main():
    dir_path = 'static\images\known'
    for file in os.listdir(dir_path):
        time_start = time.time()

        file_path = os.path.join(dir_path, file)
        image = Faces.load_image_file(file_path)
        expression, expr_score_list = expression_recognition(image)
        print("[+]表情识别预测结果为\"{}\"".format(expression))
        for expr_name, expr_score in zip(EMOTIONS, expr_score_list):
            print('   {:>11}指数:{:3%}'.format(expr_name, expr_score))

        time_end = time.time()
        print("[t]本次识别花费{:.2f}s".format((time_end - time_start)))

        cv2.imshow('image', Faces.rgb2bgr(image))
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
