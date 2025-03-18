import random
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from QtFusion.path import abs_path
from QtFusion.utils import drawRectBox

from LoggerRes import ResultLogger, LogTable
from YOLOv8v5Model import YOLOv8v5Detector
from datasets.Drowsiness.label_name import Label_list
from style_css import def_css_hitml
from utils_web import save_uploaded_file, concat_results, load_default_image, get_camera_names


class Detection_UI:
    """
    检测系统类。

    Attributes:
        model_type (str): 模型类型。
        conf_threshold (float): 置信度阈值。
        iou_threshold (float): IOU阈值。
        selected_camera (str): 选定的摄像头。
        file_type (str): 文件类型。
        uploaded_file (FileUploader): 上传的文件。
        detection_result (str): 检测结果。
        detection_location (str): 检测位置。
        detection_confidence (str): 检测置信度。
        detection_time (str): 检测用时。
    """

    def __init__(self):
        """
        初始化行人跌倒检测系统的参数。
        """
        # 初始化类别标签列表和为每个类别随机分配颜色
        self.cls_name = Label_list
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                       range(len(self.cls_name))]

        # 设置页面标题
        self.title = "基于YOLOv8/v5的疲劳驾驶检测系统"
        self.setup_page()  # 初始化页面布局
        def_css_hitml()  # 应用 CSS 样式

        # 初始化检测相关的配置参数
        self.model_type = None
        self.conf_threshold = 0.25  # 默认置信度阈值
        self.iou_threshold = 0.5  # 默认IOU阈值

        # 初始化相机和文件相关的变量
        self.selected_camera = None
        self.file_type = None
        self.uploaded_file = None
        self.uploaded_video = None
        self.custom_model_file = None  # 自定义的模型文件

        # 初始化检测结果相关的变量
        self.detection_result = None
        self.detection_location = None
        self.detection_confidence = None
        self.detection_time = None

        # 初始化UI显示相关的变量
        self.display_mode = None  # 设置显示模式
        self.close_flag = None  # 控制图像显示结束的标志
        self.close_placeholder = None  # 关闭按钮区域
        self.image_placeholder = None  # 用于显示图像的区域
        self.image_placeholder_res = None  # 图像显示区域
        self.table_placeholder = None  # 表格显示区域
        self.log_table_placeholder = None  # 完整结果表格显示区域
        self.selectbox_placeholder = None  # 下拉框显示区域
        self.selectbox_target = None  # 下拉框选中项
        self.progress_bar = None  # 用于显示的进度条

        # 初始化日志数据保存路径
        self.saved_log_data = abs_path("tempDir/log_table_data.csv", path_type="current")

        # 如果在 session state 中不存在logTable，创建一个新的LogTable实例
        if 'logTable' not in st.session_state:
            st.session_state['logTable'] = LogTable(self.saved_log_data)

        # 获取或更新可用摄像头列表
        if 'available_cameras' not in st.session_state:
            st.session_state['available_cameras'] = get_camera_names()
        self.available_cameras = st.session_state['available_cameras']

        # 初始化或获取识别结果的表格
        self.logTable = st.session_state['logTable']

        # 加载或创建模型实例
        if 'model' not in st.session_state:
            st.session_state['model'] = YOLOv8v5Detector()  # 创建YOLOv8/v5Detector模型实例

        self.model = st.session_state['model']
        # 加载训练的模型权重
        self.model.load_model(model_path=abs_path("weights/best-yolov8n.pt", path_type="current"))
        # 为模型中的类别重新分配颜色
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                       range(len(self.model.names))]
        self.setup_sidebar()  # 初始化侧边栏布局

    def setup_page(self):
        # 设置页面布局
        st.set_page_config(
            page_title=self.title,
            page_icon="🚀",
            initial_sidebar_state="expanded"
        )

    def setup_sidebar(self):
        """
        设置 Streamlit 侧边栏。

        在侧边栏中配置模型设置、摄像头选择以及识别项目设置等选项。
        """
        # 设置侧边栏的模型设置部分
        st.sidebar.header("模型设置")
        # 选择模型类型的下拉菜单
        self.model_type = st.sidebar.selectbox("选择模型类型", ["YOLOv8/v5", "其他模型"])

        # 选择模型文件类型，可以是默认的或者自定义的
        model_file_option = st.sidebar.radio("模型文件", ["默认", "自定义"])
        if model_file_option == "自定义":
            # 如果选择自定义模型文件，则提供文件上传器
            model_file = st.sidebar.file_uploader("选择.pt文件", type="pt")

            # 如果上传了模型文件，则保存并加载该模型
            if model_file is not None:
                self.custom_model_file = save_uploaded_file(model_file)
                self.model.load_model(model_path=self.custom_model_file)
                self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                               range(len(self.model.names))]
        elif model_file_option == "默认":
            self.model.load_model(model_path=abs_path("weights/best-yolov8n.pt", path_type="current"))
            # 为模型中的类别重新分配颜色
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                           range(len(self.model.names))]

        # 置信度阈值的滑动条
        self.conf_threshold = float(st.sidebar.slider("置信度阈值", min_value=0.0, max_value=1.0, value=0.25))
        # IOU阈值的滑动条
        self.iou_threshold = float(st.sidebar.slider("IOU阈值", min_value=0.0, max_value=1.0, value=0.5))

        # 设置侧边栏的摄像头配置部分
        st.sidebar.header("摄像头配置")
        # 选择摄像头的下拉菜单
        self.selected_camera = st.sidebar.selectbox("选择摄像头", self.available_cameras)

        # 设置侧边栏的识别项目设置部分
        st.sidebar.header("识别项目设置")
        # 选择文件类型的下拉菜单
        self.file_type = st.sidebar.selectbox("选择文件类型", ["图片文件", "视频文件"])
        # 根据所选的文件类型，提供对应的文件上传器
        if self.file_type == "图片文件":
            self.uploaded_file = st.sidebar.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
        elif self.file_type == "视频文件":
            self.uploaded_video = st.sidebar.file_uploader("上传视频文件", type=["mp4"])

        # 提供相关提示信息，根据所选摄像头和文件类型的不同情况
        if self.selected_camera == "未启用摄像头":
            if self.file_type == "图片文件":
                st.sidebar.write("请选择图片并点击'开始运行'按钮，进行图片检测！")
            if self.file_type == "视频文件":
                st.sidebar.write("请选择视频并点击'开始运行'按钮，进行视频检测！")
        else:
            st.sidebar.write("请点击'开始运行'按钮，启动摄像头检测！")

    def load_model_file(self):
        if self.custom_model_file:
            self.model.load_model(self.custom_model_file)
        else:
            pass  # 载入

    def process_camera_or_file(self):
        """
        处理摄像头或文件输入。

        根据用户选择的输入源（摄像头、图片文件或视频文件），处理并显示检测结果。
        """
        # 如果选择了摄像头输入
        if self.selected_camera != "未启用摄像头":
            self.logTable.clear_frames()  # 清除之前的帧记录
            # 创建一个结束按钮
            self.close_flag = self.close_placeholder.button(label="停止")

            # 使用 OpenCV 捕获摄像头画面
            cap = cv2.VideoCapture(int(self.selected_camera))

            # 设置总帧数为1000
            total_frames = 1000
            current_frame = 0
            self.progress_bar.progress(0)  # 初始化进度条
            while cap.isOpened() and not self.close_flag:
                ret, frame = cap.read()
                if ret:
                    # 显示画面并处理结果
                    image, detInfo, _ = self.frame_process(frame, "Camera: " + self.selected_camera)

                    # 设置新的尺寸
                    new_width = 1080
                    new_height = int(new_width * (9 / 16))
                    resized_image = cv2.resize(image, (new_width, new_height))  # 调整图像尺寸
                    resized_frame = cv2.resize(frame, (new_width, new_height))

                    # 根据显示模式显示处理后的图像或原始图像
                    if self.display_mode == "单画面显示":
                        self.image_placeholder.image(resized_image, channels="BGR", caption="摄像头画面")
                    else:
                        self.image_placeholder.image(resized_frame, channels="BGR", caption="原始画面")
                        self.image_placeholder_res.image(resized_image, channels="BGR", caption="识别画面")
                    # 将帧信息添加到日志表格中
                    self.logTable.add_frames(image, detInfo, cv2.resize(frame, (640, 640)))

                    # 更新进度条
                    progress_percentage = int((current_frame / total_frames) * 100)
                    self.progress_bar.progress(progress_percentage)
                    current_frame = (current_frame + 1) % total_frames  # 重置进度条
                else:
                    st.error("无法获取图像。")
                    break
                # time.sleep(0.01)  # 控制帧率

            # 保存结果到CSV并更新日志表格
            if self.close_flag:
                self.logTable.save_to_csv()
                self.logTable.update_table(self.log_table_placeholder)
                cap.release()

            self.logTable.save_to_csv()
            self.logTable.update_table(self.log_table_placeholder)
            cap.release()
        else:
            # 如果上传了图片文件
            if self.uploaded_file is not None:
                self.logTable.clear_frames()
                self.progress_bar.progress(0)
                # 显示上传的图片
                source_img = self.uploaded_file.read()
                file_bytes = np.asarray(bytearray(source_img), dtype=np.uint8)
                image_ini = cv2.imdecode(file_bytes, 1)

                image, detInfo, select_info = self.frame_process(image_ini, self.uploaded_file.name)

                # self.selectbox_placeholder = st.empty()
                self.selectbox_target = self.selectbox_placeholder.selectbox("目标过滤", select_info, key="22113")

                self.logTable.save_to_csv()
                self.logTable.update_table(self.log_table_placeholder)  # 更新所有结果记录的表格

                # 设置新的尺寸
                new_width = 1080
                new_height = int(new_width * (9 / 16))
                # 调整图像尺寸
                resized_image = cv2.resize(image, (new_width, new_height))
                resized_frame = cv2.resize(image_ini, (new_width, new_height))
                if self.display_mode == "单画面显示":
                    self.image_placeholder.image(resized_image, channels="BGR", caption="图片显示")
                else:
                    self.image_placeholder.image(resized_frame, channels="BGR", caption="原始画面")
                    self.image_placeholder_res.image(resized_image, channels="BGR", caption="识别画面")

                self.logTable.add_frames(image, detInfo, cv2.resize(image_ini, (640, 640)))
                self.progress_bar.progress(100)

            # 如果上传了视频文件
            elif self.uploaded_video is not None:
                # 处理上传的视频
                self.logTable.clear_frames()
                self.close_flag = self.close_placeholder.button(label="停止")

                video_file = self.uploaded_video
                tfile = tempfile.NamedTemporaryFile()
                tfile.write(video_file.read())
                cap = cv2.VideoCapture(tfile.name)

                # 获取视频总帧数和帧率
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # 计算视频总长度（秒）
                total_length = total_frames / fps if fps > 0 else 0

                # 创建进度条
                self.progress_bar.progress(0)

                current_frame = 0
                while cap.isOpened() and not self.close_flag:
                    ret, frame = cap.read()
                    if ret:
                        image, detInfo, _ = self.frame_process(frame, self.uploaded_video.name)

                        # 设置新的尺寸
                        new_width = 1080
                        new_height = int(new_width * (9 / 16))
                        # 调整图像尺寸
                        resized_image = cv2.resize(image, (new_width, new_height))
                        resized_frame = cv2.resize(frame, (new_width, new_height))
                        if self.display_mode == "单画面显示":
                            self.image_placeholder.image(resized_image, channels="BGR", caption="视频画面")
                        else:
                            self.image_placeholder.image(resized_frame, channels="BGR", caption="原始画面")
                            self.image_placeholder_res.image(resized_image, channels="BGR", caption="识别画面")

                        self.logTable.add_frames(image, detInfo, cv2.resize(frame, (640, 640)))

                        # 更新进度条
                        if total_length > 0:
                            progress_percentage = int(((current_frame + 1) / total_frames) * 100)
                            self.progress_bar.progress(progress_percentage)

                        current_frame += 1
                    else:
                        break
                if self.close_flag:
                    self.logTable.save_to_csv()
                    self.logTable.update_table(self.log_table_placeholder)
                    cap.release()

                self.logTable.save_to_csv()
                self.logTable.update_table(self.log_table_placeholder)
                cap.release()

            else:
                st.warning("请选择摄像头或上传文件。")

    def toggle_comboBox(self, frame_id):
        """
        处理并显示指定帧的检测结果。

        Args:
            frame_id (int): 指定要显示检测结果的帧ID。

        根据用户选择的帧ID，显示该帧的检测结果和图像。
        """
        # 确保已经保存了检测结果
        if len(self.logTable.saved_results) > 0:
            frame = self.logTable.saved_images_ini[-1]  # 获取最近一帧的图像
            image = frame  # 将其设为当前图像

            # 遍历所有保存的检测结果
            for i, detInfo in enumerate(self.logTable.saved_results):
                if frame_id != -1:
                    # 如果指定了帧ID，只处理该帧的结果
                    if frame_id != i:
                        continue

                if len(detInfo) > 0:
                    name, bbox, conf, use_time, cls_id = detInfo  # 获取检测信息
                    label = '%s %.0f%%' % (name, conf * 100)  # 构造标签文本

                    disp_res = ResultLogger()  # 创建结果记录器
                    res = disp_res.concat_results(name, bbox, str(round(conf, 2)), str(round(use_time, 2)))  # 合并结果
                    self.table_placeholder.table(res)  # 在表格中显示结果

                    # 如果有保存的初始图像
                    if len(self.logTable.saved_images_ini) > 0:
                        if len(self.colors) < cls_id:
                            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in
                                           range(cls_id+1)]
                        image = drawRectBox(image, bbox, alpha=0.2, addText=label,
                                            color=self.colors[cls_id])  # 绘制检测框和标签

            # 设置新的尺寸并调整图像尺寸
            new_width = 1080
            new_height = int(new_width * (9 / 16))
            resized_image = cv2.resize(image, (new_width, new_height))
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # 根据显示模式显示处理后的图像或原始图像
            if self.display_mode == "单画面显示":
                self.image_placeholder.image(resized_image, channels="BGR", caption="识别画面")
            else:
                self.image_placeholder.image(resized_frame, channels="BGR", caption="原始画面")
                self.image_placeholder_res.image(resized_image, channels="BGR", caption="识别画面")

    def frame_process(self, image, file_name):
        """
        处理并预测单个图像帧的内容。

        Args:
            image (numpy.ndarray): 输入的图像。
            file_name (str): 处理的文件名。

        Returns:
            tuple: 处理后的图像，检测信息，选择信息列表。

        对输入图像进行预处理，使用模型进行预测，并处理预测结果。
        """
        image = cv2.resize(image, (640, 640))  # 调整图像大小以适应模型
        pre_img = self.model.preprocess(image)  # 对图像进行预处理

        # 更新模型参数
        params = {'conf': self.conf_threshold, 'iou': self.iou_threshold}
        self.model.set_param(params)

        t1 = time.time()
        pred = self.model.predict(pre_img)  # 使用模型进行预测
        t2 = time.time()
        use_time = t2 - t1  # 计算单张图片推理时间

        det = pred[0]  # 获取预测结果

        # 初始化检测信息和选择信息列表
        detInfo = []
        select_info = ["全部目标"]

        # 如果有有效的检测结果
        if det is not None and len(det):
            det_info = self.model.postprocess(pred)  # 后处理预测结果
            if len(det_info):
                disp_res = ResultLogger()
                res = None
                cnt = 0

                # 遍历检测到的对象
                for info in det_info:
                    name, bbox, conf, cls_id = info['class_name'], info['bbox'], info['score'], info['class_id']
                    label = '%s %.0f%%' % (name, conf * 100)

                    res = disp_res.concat_results(name, bbox, str(round(conf, 2)), str(round(use_time, 2)))

                    # 绘制检测框和标签
                    image = drawRectBox(image, bbox, alpha=0.2, addText=label, color=self.colors[cls_id])
                    # 添加日志条目
                    self.logTable.add_log_entry(file_name, name, bbox, conf, use_time)
                    # 记录检测信息
                    detInfo.append([name, bbox, conf, use_time, cls_id])
                    # 添加到选择信息列表
                    select_info.append(name + "-" + str(cnt))
                    cnt += 1

                # 在表格中显示检测结果
                self.table_placeholder.table(res)

        return image, detInfo, select_info

    def frame_table_process(self, frame, caption):
        # 显示画面并更新结果
        self.image_placeholder.image(frame, channels="BGR", caption=caption)

        # 更新检测结果
        detection_result = "None"
        detection_location = "[0, 0, 0, 0]"
        detection_confidence = str(random.random())
        detection_time = "0.00s"

        # 使用 display_detection_results 函数显示结果
        res = concat_results(detection_result, detection_location, detection_confidence, detection_time)
        self.table_placeholder.table(res)
        # 添加适当的延迟
        cv2.waitKey(1)

    def setupMainWindow(self):
        """
        运行行人跌倒检测系统。

        构建并显示行人跌倒检测系统的主界面，包括图像显示、控制选项、结果展示等。
        """
        st.title(self.title)  # 显示系统标题
        st.write("--------")
        st.write("CSDN/知乎/B站/博客园：逗逗班学Python-------------https://space.bilibili.com/582341464")
        st.write("--------")  # 插入一条分割线

        # 创建列布局
        col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 2, 1])

        # 在第一列设置显示模式的选择
        with col1:
            self.display_mode = st.radio("显示模式", ["单画面显示", "双画面显示"])

        # 根据显示模式创建用于显示视频画面的空容器
        if self.display_mode == "单画面显示":
            self.image_placeholder = st.empty()
            if not self.logTable.saved_images_ini:
                self.image_placeholder.image(load_default_image(), caption="原始画面")
        else:  # "双画面显示"
            self.image_placeholder = st.empty()
            self.image_placeholder_res = st.empty()
            if not self.logTable.saved_images_ini:
                self.image_placeholder.image(load_default_image(), caption="原始画面")
                self.image_placeholder_res.image(load_default_image(), caption="识别画面")

        # 显示用的进度条
        self.progress_bar = st.progress(0)

        # 创建一个空的结果表格
        res = concat_results("None", "[0, 0, 0, 0]", "0.00", "0.00s")
        self.table_placeholder = st.empty()
        self.table_placeholder.table(res)

        # 创建一个导出结果的按钮
        st.write("---------------------")
        if st.button("导出结果"):
            self.logTable.save_to_csv()
            res = self.logTable.save_frames_file()
            st.write("🚀识别结果文件已经保存：" + self.saved_log_data)
            if res:
                st.write(f"🚀结果的视频/图片文件已经保存：{res}")
            self.logTable.clear_data()

        # 显示所有结果记录的空白表格
        self.log_table_placeholder = st.empty()
        self.logTable.update_table(self.log_table_placeholder)

        # 在第五列设置一个空的停止按钮占位符
        with col5:
            st.write("")
            self.close_placeholder = st.empty()

        # 在第二列处理目标过滤
        with col2:
            self.selectbox_placeholder = st.empty()
            detected_targets = ["全部目标"]  # 初始化目标列表

            # 遍历并显示检测结果
            for i, info in enumerate(self.logTable.saved_results):
                name, bbox, conf, use_time, cls_id = info
                detected_targets.append(name + "-" + str(i))
            self.selectbox_target = self.selectbox_placeholder.selectbox("目标过滤", detected_targets)

            # 处理目标过滤的选择
            for i, info in enumerate(self.logTable.saved_results):
                name, bbox, conf, use_time, cls_id = info
                if self.selectbox_target == name + "-" + str(i):
                    self.toggle_comboBox(i)
                elif self.selectbox_target == "全部目标":
                    self.toggle_comboBox(-1)

        # 在第四列设置一个开始运行的按钮
        with col4:
            st.write("")
            run_button = st.button("开始运行")

            if run_button:
                self.process_camera_or_file()  # 运行摄像头或文件处理
            else:
                # 如果没有保存的图像，则显示默认图像
                if not self.logTable.saved_images_ini:
                    self.image_placeholder.image(load_default_image(), caption="原始画面")
                    if self.display_mode == "双画面显示":
                        self.image_placeholder_res.image(load_default_image(), caption="识别画面")


# 实例化并运行应用
if __name__ == "__main__":
    app = Detection_UI()
    app.setupMainWindow()
