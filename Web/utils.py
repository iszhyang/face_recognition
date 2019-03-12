import codecs
import json
import os
import time

# 时间花费日志
TIMECOST_PATH = "data/timecost.json"
FILENAMES_PATH = "data/filenames.json"
NAME_LIST = ["数据传输时间", "人脸检测时间", "人脸识别时间", "表情识别时间"]
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
EMOTIONS_CH = ['生气', '厌烦', '恐惧', '开心', '伤心', '惊喜', '平淡']


def push_timecost(file_name, timecosts):
    try:
        # 本地存在文件 从本地加载后更改
        if os.path.exists(TIMECOST_PATH) and os.path.exists(FILENAMES_PATH):
            with open(FILENAMES_PATH, encoding='utf-8') as filenames_f, open(TIMECOST_PATH, encoding='utf-8') as timecost_f:
                # 从本地加载文件名列表 时间开销信息
                filenames = json.load(filenames_f)
                timecost = json.load(timecost_f)

            # 仅有一个时间时 转成json文件再转回来会被认为是float类型而非list类型
            if not isinstance(timecost[0]['data'], list):
                for i in range(4):
                    timecost[i]['data'] = [timecost[i]['data']]

            # 附加本次调用时间
            filenames.append(file_name)
            for i in range(4):
                timecost[i]['data'].append(timecosts[i])

            # 重新写入文件名列表和时间开销信息    
            with open(FILENAMES_PATH, "w", encoding='utf-8') as filenames_f, open(TIMECOST_PATH, "w", encoding='utf-8') as timecost_f:
                json.dump(filenames, filenames_f, indent=4, separators=(',', ': '), ensure_ascii=False)
                json.dump(timecost, timecost_f, indent=4, separators=(',', ': '), ensure_ascii=False)

        else:
            with open(FILENAMES_PATH, "w", encoding='utf-8') as filenames_f, open(TIMECOST_PATH, "w", encoding='utf-8') as timecost_f:
                # 以本次调用时间创建文件名列表和时间开销信息   
                filenames = [file_name]
                timecost = []
                for i in range(4):
                    temp_dict = {'name':NAME_LIST[i], 'data':timecosts[i]}
                    timecost.append(temp_dict)
                
                # 写入文件名列表和时间开销信息
                json.dump(filenames, filenames_f, indent=4, separators=(',', ': '), ensure_ascii=False)
                json.dump(timecost, timecost_f, indent=4, separators=(',', ': '), ensure_ascii=False)
                
    except IndexError as e:
        print(e)
        print("[-]保存timecost.data和filenames.data失败")
