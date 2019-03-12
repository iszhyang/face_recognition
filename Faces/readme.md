# Faces人脸识别

## 简介

**一款基于Dlib、opencv开发的人脸识别程序，包含人脸检测、人脸校正、人脸识别三个模块**

- 人脸检测问题上，采用了传统**HOG+SVM**的方式，单次人脸检测仅需0.1s
- 针对人脸检测过程中部分人头偏移角度过大而检测不到人脸的问题，**加入自适应算法**，可以在检测不到人脸时自动尝试旋转几个角度再进行检测
- 人脸识别问题上，使用适用于人脸的**ResNet深度神经网络**来提取人脸特征，拥有99.37％的准确率
- 针对女明星妆容变化较大(不同年龄、不同化妆风格)的情况，采用**数据增强**的方法，通过爬虫自动爬取女明星本人更多的照片，扩大已知人像库，大幅度提高人脸识别的准确性
- 为了拥有更高的准确性，加入**人脸矫正**模块，根据眼睛的位置对人像进行摆正，提高编码的稳定性

人脸检测效果图如下

![](http://wx2.sinaimg.cn/mw1024/0060lm7Tly1fvkpablpvgj318f0mwb29.jpg)

经过人脸检测、人脸校正并裁剪后的stdface如图

![](http://wx1.sinaimg.cn/mw1024/0060lm7Tly1fvjo5ka6jyj30fr0ornar.jpg)

部分识别结果如图

![](http://wx1.sinaimg.cn/mw1024/0060lm7Tly1fvkdq1dyhwj318t0qxe81.jpg)

## Python环境

python 3.6

Dlib(一款包含图像处理、机器学习的强大的C++库，提供python接口)

opencv-python(用于图像处理，人脸校正、基本图像处理时用到)

face_recognition_models(开源模型，有用于提取人脸特征的ResNet模型)

## 程序结构说明

1. known目录下存放了每个id对应的照片
2. data目录下维护了一些信息(姓名到id的映射关系、已知数据库的编码等)
3. stdface目录下存放了每个id对应的stdface(256*256)
4. unknown目录下存放了待检测的人脸图像
5. `Faces.py`里面封装了程序常用的函数
6. `get_stdface.py`会得到stdface(256×256)标准人脸，且进行摆正处理
7. `get_encodings.py`会根据stdface得到人脸编码并保存在data目录中
8. `main.py`是用于人脸识别的主程序
9. `landmark.png`是人脸关键点的顺序