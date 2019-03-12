# Faces人脸识别-Web模块

## 简介

**一个基于flask框架搭建的包含人脸图库、人脸识别的轻量级网站**

- 为了与python程序高度耦合，**采用python轻量级框架flask**，部署在服务器的127.0.0.1:5000上
- 针对每次调用程序都要花费大量时间重新加载模型的问题，**将算法使用到的模型常驻内存中**，单次调用人脸识别程序仅需0.1~0.5s

网站主要页面如图

- 首页：

  ![](http://wx1.sinaimg.cn/mw1024/0060lm7Tly1fvn8xhhmjdj31hc0rcqnu.jpg)

- 人像图库

  ![](http://wx4.sinaimg.cn/mw1024/0060lm7Tly1fvn9075ykmj31hc0rch7k.jpg)

- 人脸识别

  ![](http://wx2.sinaimg.cn/mw1024/0060lm7Tly1fvpauzkcpgj31hc0s34a9.jpg)

- 关于我们

  ![](http://wx4.sinaimg.cn/mw1024/0060lm7Tly1fvn96hmvrxj31hc0rcdkn.jpg)