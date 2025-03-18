import os
import time
import cv2
import pandas as pd
from QtFusion.path import abs_path


class ResultLogger:
    def __init__(self):
        """
        初始化ResultLogger类。
        """
        self.results_df = pd.DataFrame(columns=["识别结果", "位置", "置信度", "用时"])

    def concat_results(self, result, location, confidence, time):
        """
        显示检测结果，并将结果添加到结果DataFrame中。

        Args:
            result (str): 检测结果。
            location (str): 检测位置。
            confidence (str): 置信度。
            time (str): 检测用时。

        Returns:
            pd.DataFrame: 更新后的DataFrame。
        """
        # 创建一个包含这些信息的字典
        result_data = {
            "识别结果": [result],
            "位置": [location],
            "置信度": [confidence],
            "用时": [time]
        }

        # 创建一个新的DataFrame并将其添加到实例的DataFrame
        new_row = pd.DataFrame(result_data)
        self.results_df = pd.concat([self.results_df, new_row], ignore_index=True)

        return self.results_df


class LogTable:
    def __init__(self, csv_file_path=None):
        """
        初始化类实例。

        Args:
            csv_file_path (str): 保存初始数据的CSV文件路径。
        """
        self.csv_file_path = csv_file_path
        self.saved_images = []
        self.saved_images_ini = []
        self.saved_results = []

        columns = ['文件路径', '识别结果', '位置', '置信度', '用时']

        # 尝试从CSV文件加载数据，如果失败则创建一个空的DataFrame
        try:
            # 检查CSV文件是否存在
            if not os.path.exists(csv_file_path):
                # 如果文件不存在，创建一个带有初始表头的空DataFrame并保存为CSV文件
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(csv_file_path, index=False, header=True)

            self.data = pd.DataFrame(columns=columns)
            # self.data = pd.read_csv(csv_file_path, encoding='utf-8')
        except (FileNotFoundError, pd.errors.EmptyDataError):
            columns = ['文件路径', '识别结果', '位置', '置信度', '用时']
            self.data = pd.DataFrame(columns=columns)

    def add_frames(self, image, detInfo, img_ini):
        self.saved_images.append(image)
        self.saved_images_ini.append(img_ini)
        self.saved_results = detInfo

    def clear_frames(self):
        self.saved_images = []
        self.saved_images_ini = []
        self.saved_results = []

    def save_frames_file(self):
        if self.saved_images:  # 检查列表是否不为空
            # 执行保存操作
            now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            if len(self.saved_images) == 1:
                # 只有一张图像时，保存为图片
                file_name = abs_path('tempDir/pic_' + str(now_time) + '.png', path_type="current")
                cv2.imwrite(file_name, self.saved_images[0])
                return file_name
            else:
                # 为图像序列时，保存为视频
                height, width, layers = self.saved_images[0].shape
                size = (width, height)
                file_name = abs_path('tempDir/video_' + str(now_time) + '.avi', path_type="current")
                out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
                for img in self.saved_images:
                    out.write(img)
                out.release()
                return file_name
        return False

    def add_log_entry(self, file_path, recognition_result, position, confidence, time_spent):
        """
        向日志中添加一条新记录。

        Args:
            file_path (str): 文件路径
            recognition_result (str): 识别结果
            position (str): 位置
            confidence (float): 置信度
            time_spent (float): 用时（通常是秒或毫秒）

        Returns:
            None
        """
        # 创建新的数据行
        position_str = str(position)
        file_path = str(file_path)
        new_entry = pd.DataFrame([[file_path, recognition_result, position_str, confidence, time_spent]],
                                 columns=['文件路径', '识别结果', '位置', '置信度', '用时'])

        # 将新行添加到DataFrame中
        self.data = pd.concat([new_entry, self.data]).reset_index(drop=True)

        return self.data

    def clear_data(self):
        columns = ['文件路径', '识别结果', '位置', '置信度', '用时']
        self.data = pd.DataFrame(columns=columns)

    def save_to_csv(self):
        # 将更新后的DataFrame保存到CSV文件
        self.data.to_csv(self.csv_file_path, index=False, encoding='utf-8', mode='a', header=False)

    def update_table(self, log_table_placeholder):
        """
        更新表格，显示最新的500条记录。

        Args:
            log_table_placeholder: Streamlit的表格占位符

        Returns:
            None
        """
        # 判断DataFrame的长度是否超过500
        if len(self.data) > 500:
            # 如果超过500，仅显示最新的500条记录
            display_data = self.data.head(500)
        else:
            # 如果不超过，显示全部数据
            display_data = self.data

        log_table_placeholder.table(display_data)