# -*- coding: utf-8 -*-

__version__ = '2.0'
__author__ = 'doudouban'
__QQ__ = 'deepcoding'
__url__ = 'https://deepcode.blog.csdn.net/'

AUTHOR_INFO = ("基于YOLOv8/v5的疲劳驾驶检测系统v1.0\n"
               "作者：逗逗班学Python\n"
               "面包多: https://mbd.pub/o/deepcode\n"
               "CSDN: https://deepcode.blog.csdn.net/\n"
               "BiliBili: https://space.bilibili.com/582341464\n"
               "博客园: https://www.cnblogs.com/deeppython\n"
               "知乎: https://www.zhihu.com/people/deeppython\n\n"
               "原创代码，仅用于学习交流，禁止他人商用和盗卖，盗版必究！\n\n"
               "\n"
               "")

ENV_CONFIG = ("[配置环境]\n"
              "请按照给定的python版本配置环境，否则可能会因依赖不兼容而出错\n"
              "(1)使用anaconda新建python3.10环境:\n"
              "conda create -n env_rec python=3.10\n"
              "(2)激活创建的环境:\n"
              "conda activate env_rec\n"
              "(3)使用pip安装所需的依赖，可通过requirements.txt:\n"
              "pip install -r requirements.txt\n")

with open('./环境配置.txt', 'w', encoding='utf-8') as f:
    f.writelines(ENV_CONFIG + "\n\n" + AUTHOR_INFO)
