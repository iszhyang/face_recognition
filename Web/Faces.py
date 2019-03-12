# -*- coding: utf-8 -*-

from math import atan, cos, degrees, radians, sin

import cv2
import dlib
import face_recognition_models
import numpy as np
from PIL import Image

# 当检测不到人脸时对人脸进行旋转的 按照如下列表的角度逐个尝试
ANGLE_LIST = [0, -45, 45]

# 加载Dlib库中的人脸检测算子、ResNet模型
face_detector = dlib.get_frontal_face_detector()

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


class Vector(object):
    """
    定义2D平面的向量类
    """

    def __init__(self, x, y):
        """
        向量初始化
        :param x: X值
        :param y: Y值
        """
        self.x = x
        self.y = y

    def rotate(self, alpha):
        """
        将向量旋转一定的角度 无返回值
        :param alpha: 要旋转的角度
        :return: 无返回值
        """
        # print("cos:{:.2f}  sin:{:.2f}".format(cos(alpha), sin(alpha)))
        rad_alpha = radians(alpha)
        x_new = self.x * cos(rad_alpha) - self.y * sin(rad_alpha)
        y_new = self.y * cos(rad_alpha) + self.x * sin(rad_alpha)
        self.x = x_new
        self.y = y_new
        return

    def get_point(self, center_x, center_y):
        result_x = center_x + self.x
        result_y = center_y + self.y
        return [result_x, result_y]


def rect2css(rect):
    """
    将Dlib库中rect转换为(top, right, bottom, left)
    :param rect:Dlib库中的rect
    :return:(top, right, bottom, left)
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def css2rect(css):
    """
    将(top, right, bottom, left)转换为Dlib的rect对象
    :param css:(top, right, bottom, left)
    :return:Dlib中rect对象
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def trim_css_to_bounds(css, image_shape):
    """
    :return: 返回经过调整后的(top, right, bottom, left)
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def raw_face_locations(img, number_of_times_to_upsample=1):
    return face_detector(img, number_of_times_to_upsample)


def raw_face_landmarks(face_image, face_locations_=None, model="large"):
    if face_locations_ is None:
        face_locations_ = raw_face_locations(face_image)
    else:
        face_locations_ = [css2rect(face_location) for face_location in face_locations_]

    pose_predictor = pose_predictor_68_point

    if model == "small":
        pose_predictor = pose_predictor_5_point

    return [pose_predictor(face_image, face_location) for face_location in face_locations_]


def rgb2bgr(rgb_image):
    """
    (ndarray)将rgb通道转换为bgr通道
    """
    return rgb_image[..., ::-1]


def bgr2rgb(bgr_image):
    """
    (ndarray)将bgr通道转换为rgb通道
    """
    return bgr_image[..., ::-1]


def face_encodings(face_image, known_face_locations=None, num_jitters=0):
    """
    得到图像中的人脸128维编码

    :param face_image: RGB三通道图像(ndarray)
    :param known_face_locations: 已知人脸的位置 图像编码会与已知图像位置有相同序号
    :param num_jitters: 计算编码时的抖动次数 较高的抖动次数会获得更稳定的编码
    :return: 返回(n, 128)的ndarray数组 n为图片中人脸数量
    """
    raw_landmarks = raw_face_landmarks(face_image, known_face_locations, model="small")
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
            raw_landmark_set in raw_landmarks]


def face_locations(img, number_of_times_to_upsample=1):
    """
    返回图像中人脸的bounding boxes

    :param img: RGB三通道图像(ndarray)
    :param number_of_times_to_upsample: 对图像的上采样次数
    :return: 返回包含n个人脸位置(top, right, bottom, left)的列表
    """
    return [trim_css_to_bounds(rect2css(face), img.shape) for face in
            raw_face_locations(img, number_of_times_to_upsample)]


def face_distance(face_encodings_, face_to_compare):
    """
    求待检测的人脸编码和一组已知人脸编码的距离
    返回待检测编码和已知编码一一比较得到的距离列表
    :param face_encodings_:已知编码
    :param face_to_compare:待比较编码
    :return:待检测编码和已知编码一一比较得到的距离列表
    """
    if len(face_encodings_) == 0:
        return np.empty(0)

    # 欧氏距离
    return np.linalg.norm(face_encodings_ - face_to_compare, axis=1)


def load_image_file(file_location):
    """
    加载一张图片 返回一个RGB三通道的ndarray数组
    :param file_location: 文件位置
    """
    image = Image.open(file_location)
    image = image.convert('RGB')

    scale = image.size[1] / image.size[0]
    image = image.resize((400, int(400 * scale)), Image.ANTIALIAS)

    return np.array(image)


def save_image_file(file_location, image):
    """
    保存一张RGB通道图片
    :param file_location: 保存路径
    :param image: 图像矩阵
    :return:
    """
    bgr_image = rgb2bgr(image)
    cv2.imwrite(file_location, bgr_image)


# 定义旋转rotate函数
def image_rotate(image, angle, center=None, scale=1.0):
    """
    对图片进行旋转 正值时顺时针旋转 负值时逆时针旋转
    :param image:原图
    :param angle:旋转角度 正值为顺时针旋转 负值为逆时针旋转
    :param center:旋转中心 不指定时为图像中心
    :param scale:图像尺度变化比率
    :return:旋转过的图像
    """
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 使用opencv库中的仿射变换对图像进行旋转
    transform_matrix = cv2.getRotationMatrix2D(center, -1 * angle, scale)
    rotated = cv2.warpAffine(image, transform_matrix, (w, h))

    return rotated


def get_only_face(image, min_face_area=0, upsample=0, rotate=False):
    """
    获取图像中像素面积最大的人脸位置(比赛场景特殊 一张图片只有一个主体)
    :param image:图像
    :param min_face_area:最小像素面积
    :param upsample: 对图像上采样的次数
    :param rotate: 是否尝试对图像进行旋转
    :return: 返回像素面积最大的人脸的位置(top, right, bottom, left)
    :return: 如果rotate为True 返回(rotated_image, alpha, top, right, bottom, left) alpha为旋转角度
    :return: 检测不到人脸的时候top right bottom left的返回值都为-1
    """

    # 进行旋转时
    if rotate:
        rotated_image = image
        # 遍历旋转角度列表 尝试从不同角度对图像进行旋转
        for angle in ANGLE_LIST:
            if angle != 0:
                # 角度不为0时对图像进行旋转
                rotated_image = image_rotate(image, angle)

            # 获取图像中所有人脸的位置
            locations = face_locations(rotated_image, upsample)

            # 统计人脸个数，并得到像素面积最大的人脸的位置
            max_pixel_area = 0
            number_of_faces = len(locations)
            result_top, result_right, result_bottom, result_left = 0, 0, 0, 0

            for location in locations:
                top, right, bottom, left = location
                pixel_area = (bottom - top) * (right - left)
                # 忽略面积小于min_face_area的脸
                if pixel_area < min_face_area:
                    number_of_faces -= 1
                    continue
                # 更新像素面积最大值
                if pixel_area > max_pixel_area:
                    max_pixel_area = pixel_area
                    result_top, result_right, result_bottom, result_left = top, right, bottom, left

            # 检测到的人脸后 返回像素面积最大的人脸的位置 函数结束
            if number_of_faces > 0:
                if angle != 0:
                    print("[!]图像旋转{}°后检测到人脸".format(angle))
                return rotated_image, angle, result_top, result_right, result_bottom, result_left

        # 多个角度旋转后仍未检测到人脸 返回(-1, -1, -1, -1)
        print("[-]Warning: 多角度旋转后仍未检测到人脸")
        return image, 0, -1, -1, -1, -1

    # 不进行旋转时
    else:
        # 得到所有人脸的位置
        locations = face_locations(image, upsample)

        # 统计人脸个数，并得到像素面积最大的人脸的位置
        max_pixel_area = 0
        number_of_faces = len(locations)
        result_top, result_right, result_bottom, result_left = -1, -1, -1, -1

        for location in locations:
            top, right, bottom, left = location
            pixel_area = (bottom - top) * (right - left)
            # 忽略面积小于min_face_area的脸
            if pixel_area < min_face_area:
                number_of_faces -= 1
                continue
            # 更新像素面积最大值
            if pixel_area > max_pixel_area:
                max_pixel_area = pixel_area
                result_top, result_right, result_bottom, result_left = top, right, bottom, left

        return result_top, result_right, result_bottom, result_left


def get_angle(point_x, point_y):
    """
    获取像素空间下两点斜率的角度
    point_x在左边 point_y在右边的情况
    :param point_x: 左边的点
    :param point_y: 右边的点
    :return: 角度
    """
    if abs(point_y[0] - point_x[0]) < 0.000001:
        return 90
    return degrees(atan((point_x[1] - point_y[1]) / (point_y[0] - point_x[0])))


def get_stdface(image, detection_save_path=None, stdface_save_path=None):
    """
    输入一张RGB三通道图片 返回经过摆正并裁剪的256*256的人脸图像
    检测不到的时候返回空值None
    :param image: 原图
    :param detection_save_path: 如果需要将检测结果保存下来 请传入本值
    :param stdface_save_path: 如果需要将stdface结果保存下来 请传入本值
    :return: 返回stdface 检测不到时返回None
    """
    # 拷贝一份原图用于绘制带有角度的bounding box(后续过程可能对原图进行旋转)
    image_copy = image.copy()

    # alpha是图像第一次旋转的角度
    alpha = 0

    # 尝试不旋转、不进行上采样的情况下获取图片上的人脸
    top, right, bottom, left = get_only_face(image)

    # 无法检测到人脸时的自适应调整:尝试将上采样次数加大1 并允许对图像进行旋转操作
    if top == -1:
        image, alpha, top, right, bottom, left = get_only_face(image, upsample=1, rotate=True)

    # 检测不到人脸 报错并返回
    if top == -1:
        print("[-]检测不到人像")
        return None

    location = (top, right, bottom, left)

    # 根据人脸位置获取人脸的特征点的位置
    face_landmark = face_landmarks(image, [location])[0]

    # 根据左眼、右眼的位置对人脸进行校正
    left_eye = face_landmark['left_eye']
    right_eye = face_landmark['right_eye']

    # 眼睛部位六个对称点的倾斜角度
    angle_list = [0, 0, 0, 0, 0, 0]

    for i in range(6):
        if i < 4:
            angle_list[i] = get_angle(left_eye[i], right_eye[3 - i])
        elif i == 4:
            angle_list[i] = get_angle(left_eye[i], right_eye[5])
        elif i == 5:
            angle_list[i] = get_angle(left_eye[i], right_eye[4])

    # 六个对称点倾斜角度取平均
    beta = sum(angle_list) / 6.0

    # 根据beta对图像再次校正
    rotated_img = image_rotate(image, beta)

    # 对摆正的图片再次进行人脸检测 并将人像进行裁剪 此时不需要进行旋转
    top, right, bottom, left = get_only_face(rotated_img)

    # 由于旋转会一定比例缩小图像 需要进行上采样才能得到所有的脸 不进行旋转
    if top == -1:
        top, right, bottom, left = get_only_face(rotated_img, upsample=1)

    # 如果仍然检测不到人脸 返回空值
    if top == -1:
        return None

    cut_area = rotated_img[top:bottom, left:right]
    stdface = cv2.resize(cut_area, (256, 256), interpolation=cv2.INTER_CUBIC)

    # 保存stdface
    if stdface_save_path is not None:
        cv2.imwrite(stdface_save_path, rgb2bgr(stdface))
        print("[+]stdface保存成功：{}".format(stdface_save_path))

    # 保存人像检测结果
    if detection_save_path is not None:
        # 保存带有角度的人脸检测图像
        height, weight = image.shape[:2]  # 图像尺寸
        anti_angle = -1 * (alpha + beta)  # 反向旋转角度
        center_x, center_y = (weight // 2, height // 2)  # 旋转中心坐标

        # 图像中心到四个顶点的向量
        vector_left_top = Vector((left - center_x), (top - center_y))
        vector_right_top = Vector((right - center_x), (top - center_y))
        vector_left_bottom = Vector((left - center_x), (bottom - center_y))
        vector_right_bottom = Vector((right - center_x), (bottom - center_y))

        # 对向量进行旋转
        vector_left_top.rotate(anti_angle)
        vector_right_top.rotate(anti_angle)
        vector_left_bottom.rotate(anti_angle)
        vector_right_bottom.rotate(anti_angle)

        # 带有角度的bounding box的边缘点
        point_left_top = vector_left_top.get_point(center_x, center_y)
        point_right_top = vector_right_top.get_point(center_x, center_y)
        point_left_bottom = vector_left_bottom.get_point(center_x, center_y)
        point_right_bottom = vector_right_bottom.get_point(center_x, center_y)

        # 组建要绘制的点的序列
        points = np.array([point_left_top, point_right_top, point_right_bottom, point_left_bottom], np.int32)
        points = points.reshape((-1, 1, 2))

        # 在克隆出来的图像上进行绘制
        cv2.polylines(image_copy, [points], True, (15, 158, 254), 2, cv2.LINE_AA)

        save_image_file(detection_save_path, image_copy)

    return stdface


def face_landmarks(face_image, face_locations_=None, model="large"):
    """
    输入一张图片 返回包含人脸特征点的字典的集合
    人脸关键点的顺序 请查看landmark.png文件
    :return: A list of dicts of face feature locations (eyes, nose, etc)
    """
    landmarks = raw_face_landmarks(face_image, face_locations_, model)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    if model == 'large':
        return [{
            "chin": points[0:17],
            "left_eyebrow": points[17:22],
            "right_eyebrow": points[22:27],
            "nose_bridge": points[27:31],
            "nose_tip": points[31:36],
            "left_eye": points[36:42],
            "right_eye": points[42:48],
            "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
            "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [
                points[64]]
        } for points in landmarks_as_tuples]
    elif model == 'small':
        return [{
            "nose_tip": [points[4]],
            "left_eye": points[2:4],
            "right_eye": points[0:2],
        } for points in landmarks_as_tuples]


def get_similarity(distance):
    """
    自己拟合的从欧氏距离→相似度的函数
    """
    similarity = 99.0
    if distance <= 0.3:
        dy = distance * 13.33
        similarity -= dy
    elif distance <= 0.35:
        dx = distance - 0.3
        dy = 100 * dx
        similarity = 95.0 - dy
    else:
        dx = distance - 0.5
        dy = 160 * dx
        similarity = 90.0 - dy

    # 对相似度的限制
    similarity = min(99.0, similarity)
    similarity = max(31.4, similarity)
    print("[ ]欧式距离:{:.4f}  拟合后相似度:{:.4f}".format(distance, similarity))
    return similarity
