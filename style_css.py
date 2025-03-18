import base64

import streamlit as st


# 读取图片并转换为 Base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as file:
        data = file.read()
    return base64.b64encode(data).decode()


def def_css_hitml():
    st.markdown("""
        <style>
        /* 全局样式 */
        .css-2trqyj, .css-1d391kg, .st-bb, .st-at {
            font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
            background-color: #cadefc;
            color: #21618C;
        }

        /* 按钮样式 */
        .stButton > button {
            border: none;
            color: white;
            padding: 10px 20px; /* 减小 padding */
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px; /* 减小字体大小 */
            margin: 2px 1px; /* 调整边距 */
            cursor: pointer;
            border-radius: 8px; /* 调整边框圆角 */
            background-color: #9896f1;
            box-shadow: 0 2px 4px 0 rgba(0,0,0,0.2);
            transition-duration: 0.4s;
        }
        .stButton > button:hover {
            background-color: #5499C7;
            color: white;
            box-shadow: 0 8px 12px 0 rgba(0,0,0,0.24);
        }

        /* 侧边栏样式 */
        .css-1lcbmhc.e1fqkh3o0 {
            background-color: #154360;
            color: #FDFEFE;
            border-right: 2px solid #DDD;
        }

        /* Radio 按钮样式 */
        .stRadio > label {
            display: inline-flex;
            align-items: center;
            cursor: pointer;
        }
        .stRadio > label > span:first-child {
            background-color: #FFF;
            border: 1px solid #CCC;
            width: 1em;
            height: 1em;
            border-radius: 50%;
            margin-right: 10px;
            display: inline-block;
        }

        /* 滑块样式 */
        .stSlider .thumb {
            background-color: #2E86C1;
        }
        .stSlider .track {
            background-color: #DDD;
        }

        /* 全屏显示图标样式 */
        /* .fullscreen-button {
            filter: invert(1);
        } */

        /* 表格样式 */
        table {
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 18px;
            font-family: sans-serif;
            min-width: 400px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        thead tr {
            background-color: #a8d8ea;
            color: #ffcef3;
            text-align: left;
        }
        th, td {
            padding: 15px 18px;
        }
        tbody tr {
            border-bottom: 2px solid #ddd;
        }
        tbody tr:nth-of-type(even) {
            background-color: #D6EAF8;
        }
        tbody tr:last-of-type {
            border-bottom: 3px solid #5499C7;
        }
        tbody tr:hover {
            background-color: #AED6F1;
        }
        </style>
        """, unsafe_allow_html=True)