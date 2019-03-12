import json
import os
import pickle
import posixpath
import time

from flask import Flask, render_template, request, url_for
from werkzeug import secure_filename

import demo_api
import Faces
import utils

# 用户上传图片的保存路径
SAVE_DIR = "static/images/unknown"
# 已知人像库的路径
FACELIB_DIR = "static/images/known"
# 允许用户上传的文件格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# 表情库位置
EMOJI_FOLDER_PATH = "static/images/emoji"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/Faces/unknown'
app.config['MAX_CONTEN T_LENGTH'] = 5 * 1024 * 1024


# 用户访问根目录时的路由行为
@app.route('/')
def index():
    return render_template('index.html')


def get_current_time():
    return str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime(time.time())))


# 用户访问人脸库时的路由行为
@app.route('/facelib')
def facelib():
    with open("data/known_image_info_list.data", "rb") as f:
        image_infos = pickle.load(f)

    return render_template('facelib.html', image_infos=image_infos)


# 用户访问人脸识别模块时的路由行为
@app.route('/facercg', methods=['GET', 'POST'])
def facercg():
    # 如果用户提交POST表单
    if request.method == 'POST':

        # 记录时间开销
        timecosts = []

        # 用户提出POST开始计时
        time_post_start = time.time()
        print("[ ]开始处理POST请求")

        # POST表单中没有文件时 提示用户先选择图片
        if 'file' not in request.files:
            return render_template('facercg.html', unknown_show=True,
                                   similarity=0, result="unknown",
                                   message="请先选择图片")

        # 读取表单内的文件
        file = request.files['file']

        # 文件合法
        if file and allowed_file(file.filename):
            # 使用安全文件名以避免中文等字符的出现
            file_type = secure_filename(file.filename).split('.')[-1]
            file_name = get_current_time() + '.' + file_type
            print(file_name)

            #  用户上传图片的保存路径
            image_path = posixpath.join(SAVE_DIR, file_name)

            # 人脸检测结果图片的文件名
            detection_save_name = file_name.split('.')[0] + '_de.jpg'

            # 人脸检测结果图片的保存路径
            detection_save_path = posixpath.join(SAVE_DIR, detection_save_name)

            # 文件名重复时更新文件名
            cnt = 1
            while os.path.exists(image_path):
                basename, filetype = file_name.split('.')
                file_name = basename + '_' + str(cnt) + '.' + filetype
                image_path = posixpath.join(SAVE_DIR, file_name)
            file.save(image_path)
            print("[+]保存成功! 保存路径{}".format(image_path))

            # 传输结束 记录传输时间
            time_trans_end = time.time()
            trans_timecost = time_trans_end - time_post_start
            timecosts.append(trans_timecost)

            image = Faces.load_image_file(image_path)

            # 调用人脸检测程序 得到经过角度校正的尺寸为256*256的标准人脸
            stdface = Faces.get_stdface(image, detection_save_path)
            # 人脸检测调用结束 记录人脸检测花费的时间
            time_detection_end = time.time()
            detection_timecost = time_detection_end - time_trans_end
            timecosts.append(detection_timecost)

            # 未能检测到人脸时
            if stdface is None:
                print(f"[-]未检测出人脸 接受文件+检测共花费"
                      f"{(time_detection_end - time_post_start):.1f}s "
                      f"其中人脸检测花费{detection_timecost:.1f}s")
                return render_template('facercg.html', image_paths=None,
                                       left_photo_path=image_path,
                                       similarity=0.0, result=None,
                                       message="未能识别出人脸")

            # 调用人脸识别程序 得到人脸识别结果
            result, image_paths, similarity = demo_api.stdface_recognition(stdface)
            # 调整相似度的格式
            format_similarity = str(similarity)[:5] + "%"
            # 人脸识别结束 记录人脸识别花费的时间
            time_recognition_end = time.time()
            recognition_timecost = time_recognition_end - time_detection_end
            timecosts.append(recognition_timecost)

            # 调用表情识别API
            expression_id, expr_score_list = demo_api.stdface_expression_recognition(stdface)
            expression_result = demo_api.EMOTIONS[expression_id]
            expression_result_ch = demo_api.EMOTIONS_CH[expression_id]

            # 构造表情识别结果
            expression_data = []
            for expr_name, expr_score in zip(demo_api.EMOTIONS_CH, expr_score_list):
                temp_dict = {'name': expr_name, 'y': expr_score * 100.0}
                expression_data.append(temp_dict)

            # 表情图片地址
            emoji_path = posixpath.join(EMOJI_FOLDER_PATH, expression_result + '.png')

            # 表情识别结束 记录表情识别花费的时间
            time_expression_end = time.time()
            expression_timecost = time_expression_end - time_recognition_end
            timecosts.append(expression_timecost)

            # 将本次识别花费的时间保存到本地
            utils.push_timecost(file_name, timecosts)

            # 检测出人脸但数据库中无匹配项时
            if result == "unknown":
                print(f"[-]库中无匹配项 接受文件+识别共花费"
                      f"{(time_recognition_end - time_post_start):.1f}s "
                      f"其中人脸识别花费{(time_recognition_end-time_trans_end):.1f}s")
                return render_template('facercg.html', image_paths=image_paths, left_photo_path=image_path,
                                       similarity=format_similarity, result=result, message="数据库中无匹配人脸",
                                       expression_data=expression_data, emoji_path=emoji_path,
                                       expression_result=expression_result_ch)
            # 识别成功
            elif result is not None:
                print(f"[+]识别成功 接受文件+识别共花费"
                      f"{(time_recognition_end - time_post_start):.1f}s "
                      f"其中人脸识别花费{(time_recognition_end-time_trans_end):.1f}s")
                return render_template('facercg.html', image_paths=image_paths, left_photo_path=detection_save_path,
                                       similarity=format_similarity, result=result, message="识别成功",
                                       expression_data=expression_data, emoji_path=emoji_path,
                                       expression_result=expression_result_ch)
        # 文件不合法
        else:
            print("[-]图片上传有误")
            left_photo_path = url_for('static', filename='images/loading.gif')
            return render_template('facercg.html', left_photo_path=left_photo_path,
                                   similarity=0, result="unknown", message="图片上传有误")

    # 用户向本页面发出GET请求 返回该页面内容
    else:
        return render_template('facercg.html', unknown_show=True, similarity=0, result="unknown", message=None)


# 用户访问"关于我们"页面的路由行为
@app.route('/about')
def about():
    return render_template('about.html')


# 用户访问"API调用时间统计"页面时的路由行为
@app.route('/timecost')
def timecost():
    with open(utils.FILENAMES_PATH, encoding='utf-8') as filenames_f, \
            open(utils.TIMECOST_PATH, encoding='utf-8') as timecost_f:
        # 从本地加载文件名列表 时间开销信息
        filenames = json.load(filenames_f)
        timecost_ = json.load(timecost_f)

        return render_template('timecost.html', timecost=timecost_, filenames=filenames)


# 运行flask项目
if __name__ == '__main__':
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    # 运行flask主程序
    app.run()
