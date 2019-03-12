# Faces人脸识别

分为两个模块，Faces文件夹下存放人脸识别算法的代码，Web文件夹下存放网站搭建的代码

详情请查看各个模块下的readme文档

## 项目简介

### 核心算法

**一款基于Dlib、opencv开发的人脸识别程序，包含人脸检测、人脸校正、人脸识别、表情识别四个模块**

- 人脸检测问题上，初步采用了传统**HOG+SVM**的方式，单次人脸检测仅需0.1s
- 针对人脸检测过程中部分人头偏移角度过大而检测不到人脸的问题，**加入具有角度自适应性的旋转鲁棒算法**
- 人脸识别问题上，使用适用于人脸的**ResNet-34深度神经网络**来提取人脸特征，在公共数据集上拥有99.37％的准确率
- 针对女明星妆容变化较大(不同年龄、不同化妆风格)的情况，采用**数据增强**的方法，通过爬虫自动爬取女明星本人更多的照片，扩大已知人像库，尽可能消除女明星妆容变化较大引起的误差
- 为了拥有更高的准确性，**加入基于人脸关键点的人脸校正模块**。先将人脸校正、标准化后再送入深度神经网络中，可以得到更加稳定的编码，同等情况下可以提升3.6％的准确率
- 为了增加项目的可玩性，**加入了基于深度学习的表情识别模块**，快来体验一下！

人脸检测效果图如下

![](http://wx1.sinaimg.cn/mw1024/0060lm7Tly1fwmwo2kvg6j30tv0gv4qp.jpg)

经过人脸检测、人脸校正并裁剪后的stdface如图

![](http://wx1.sinaimg.cn/mw1024/0060lm7Tly1fvjo5ka6jyj30fr0ornar.jpg)

人脸识别结果

![](http://wx4.sinaimg.cn/mw1024/0060lm7Tly1fwmwsyyuesj311w0v91kx.jpg)

表情识别结果

![](http://wx1.sinaimg.cn/mw1024/0060lm7Tly1fwmwqmsob9j314c0v9tws.jpg)

### Web部分

**一个基于flask框架搭建的包含人脸图库、人脸识别的轻量级网站**

- 为了与python程序高度耦合，**采用python轻量级框架flask**，部署在服务器的127.0.0.1:5000上
- 针对每次调用程序都要花费大量时间重新加载模型的问题，**将算法使用到的模型常驻内存中**，单次调用人脸识别程序仅需0.3s

网站主要页面如图

- 首页：

  ![](http://wx1.sinaimg.cn/mw1024/0060lm7Tly1fvn8xhhmjdj31hc0rcqnu.jpg)

- 人像图库

  ![](http://wx4.sinaimg.cn/mw1024/0060lm7Tly1fvn9075ykmj31hc0rch7k.jpg)

- 人脸识别（未上传）

  ![](http://wx4.sinaimg.cn/mw1024/0060lm7Tly1fwmwsh1islj310y0v9q9d.jpg)

- 人脸识别结果

  ![](http://wx4.sinaimg.cn/mw1024/0060lm7Tly1fwmwsyyuesj311w0v91kx.jpg)

- 表情识别结果

  ![](http://wx1.sinaimg.cn/mw1024/0060lm7Tly1fwmwqmsob9j314c0v9tws.jpg)

- 关于我们

  ![](http://wx4.sinaimg.cn/mw1024/0060lm7Tly1fvn96hmvrxj31hc0rcdkn.jpg)

- API调用耗时

  ![](http://wx3.sinaimg.cn/mw1024/0060lm7Tly1fwmwvguw9rj30yt0lwju5.jpg)