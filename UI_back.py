import subprocess
import io
import sys
import queue
import numpy as np
import gradio as gr
from PIL import Image
from test1 import test_presention1

# 功能函数
def traj_gen(generate_type, num, progress=gr.Progress()):
    if generate_type == "预训练模型":
        generate_type = 'pred'
    elif generate_type == "模拟器":
        generate_type = 'dj'
    else:
        raise ValueError(f"Invalid type selected: {generate_type}")
    
    video_path = f"./task1/video/{generate_type}/video.mp4"

    # 如果编码不兼容，就转码为 h264
    if not is_browser_compatible(video_path):
        print(f"视频编码不兼容，正在转码为 h264 编码...")
        h264_path = video_path.replace(".mp4", "_h264.mp4")
        video_path = convert_to_h264(video_path, h264_path)
        print(f"转码完成，保存为 {h264_path}")

    return video_path

def vol_pred(text):
    return f"预测结果（占位）：文本长度为 {len(text.split())}"

def shortest_path(text):
    return "最短路预测功能尚未实现。"

def signal_control(text):
    return "信号灯控制功能尚未实现。"

def show_map(city):
    if city == "济南":
        img = Image.open("./UI_element/citymap/jinan.png")
    else:
        raise ValueError("City not supported.")
    return img

def is_browser_compatible(video_path):
    """使用 ffprobe 判断视频编码是否为 h264。"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1',
             video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        codec = result.stdout.strip()
        return codec == 'h264'
    except Exception as e:
        print(f"[ffprobe error]: {e}")
        return False

def convert_to_h264(input_path, output_path):
    """使用 ffmpeg 转码为 h264 编码。"""
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', input_path, '-vcodec', 'libx264', '-acodec', 'aac', output_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return output_path
    except Exception as e:
        print(f"[ffmpeg error]: {e}")
        return input_path

# 创建 Gradio 界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1) as left_panel:
            with gr.Group(visible=True) as pre_panel:
                gr.Markdown("## 选择城市和功能")
                city = gr.Radio(choices=["济南"], label="选择城市")
                func_selector = gr.Dropdown(
                    choices=["轨迹生成", "在途量预测", "最短路预测", "信号灯控制"],
                    label="选择功能",
                    value="轨迹生成"
                )
        with gr.Column(scale=3) as right_panel:
            city_img = gr.Image(label="城市地图", type="pil")
            city.change(fn=show_map, inputs=city, outputs=city_img)

    with gr.Row():
        with gr.Column(scale=1) as left_panel:
            with gr.Group(visible=True) as traj_panel_left:
                gr.Markdown("## 轨迹生成")
                task1_type = gr.Radio(choices=["预训练模型", "模拟器"], label="选择轨迹生成方式", value="预训练模型")
                task1_num = gr.Number(label="轨迹数量", value=1)
                task1_btn = gr.Button("开始生成")
        with gr.Column(scale=3) as right_panel:
            with gr.Group(visible=True) as traj_panel_right:
                task1_output = gr.Video(label="生成的视频")
                task1_btn.click(fn=traj_gen, inputs=[task1_type, task1_num], outputs=task1_output)

        with gr.Group(visible=False) as vol_panel:
            gr.Markdown("## 在途量预测")
            vol_input = gr.Textbox(label="输入文本描述")
            vol_output = gr.Textbox(label="预测结果")
            vol_input.change(fn=vol_pred, inputs=vol_input, outputs=vol_output)

        with gr.Group(visible=False) as path_panel:
            gr.Markdown("## 最短路预测")
            path_input = gr.Textbox(label="输入起点和终点")
            path_output = gr.Textbox(label="最短路径结果")
            path_input.change(fn=shortest_path, inputs=path_input, outputs=path_output)

        with gr.Group(visible=False) as signal_panel:
            gr.Markdown("## 信号灯控制")
            signal_input = gr.Textbox(label="输入交叉口信息")
            signal_output = gr.Textbox(label="信号灯控制结果")
            signal_input.change(fn=signal_control, inputs=signal_input, outputs=signal_output)

    def update_panels(selected):
        return {
            traj_panel_left: gr.update(visible=selected == "轨迹生成"),
            traj_panel_right: gr.update(visible=selected == "轨迹生成"),
            vol_panel: gr.update(visible=selected == "在途量预测"),
            path_panel: gr.update(visible=selected == "最短路预测"),
            signal_panel: gr.update(visible=selected == "信号灯控制"),
        }

    func_selector.change(fn=update_panels, inputs=func_selector, outputs=[
        traj_panel_left, traj_panel_right, vol_panel, path_panel, signal_panel
    ])

# 启动
demo.launch()
