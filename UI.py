import subprocess
import multiprocessing
import io
import sys
import queue
import numpy as np
import gradio as gr
from PIL import Image

from device_selection import global_device
from test_script.test1 import test_presention1 as fun_1
from test_script.test2 import task2_test as fun_3
from test_script.test3 import test_presentation_lightning as fun_2
from test_script.test3 import test_presentation as fun_2_musa
from test_script.test4 import test_presention as fun_4

# 功能函数
def traj_gen(generate_type, num):
    if generate_type == "虚实结合算法":
        generate_type = 'pred'
    elif generate_type == "传统算法":
        generate_type = 'dj'
    else:
        raise ValueError(f"Invalid type selected: {generate_type}")
    
    # video_path = f'./UI_element/task1/{generate_type}/video.mp4'
    video_path = fun_1(num, generate_type, save_path = f'./UI_element/task1')

    video_path = refine_video(video_path)

    return video_path

def vol_pred(num, observation_ratio):
    assert 0 < num <= 100000
    assert 0 < observation_ratio <= 1

    def run_task2(num, observation_ratio, save_path):
        print("runing subprocess...")
        cmd = f"python -m test_script.test3 --num {num} --observe_ratio {observation_ratio} --save_path {save_path}"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error: {stderr.decode()}")
        else:
            print(f"Output: {stdout.decode()}")
        print("subprocess finished.")

    # 运行任务2
    if 'musa' in global_device:
        vol_real_path, vol_pred_path = fun_2_musa(num, observation_ratio, save_path = f'./UI_element/task2')
    else:
        run_task2(num, observation_ratio, save_path = f'./UI_element/task2')
        vol_real_path = f'./UI_element/task2/videos/real/video.mp4'
        vol_pred_path = f'./UI_element/task2/videos/pred/video.mp4'

    return vol_real_path, vol_pred_path

def shortest_path(num, generate_type):
    if generate_type == "神经网络":
        generate_type = 'pred'
    elif generate_type == "Dijkstra":
        generate_type = 'dj'
    else:
        raise ValueError(f"Invalid type selected: {generate_type}")
    
    # image_path, time = f"./UI_element/task3/{generate_type}/image.png", 0
    image_path, time = fun_3(num, generate_type, save_path = f'./UI_element/task3')

    image = Image.open(image_path)
    return image, time

def signal_control(num, generate_type):
    if generate_type == "AI信号灯":
        generate_type = 1
    elif generate_type == "传统算法":
        generate_type = 0
    else:
        raise ValueError(f"Invalid type selected: {generate_type}")
    
    # video_path, wait_time = f"./UI_element/task4/{generate_type}/video.mp4", 0
    video_path, orical, pred = fun_4(num, generate_type, save_path = f'./UI_element/task4')

    video_path = refine_video(video_path)
    return video_path, pred


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
    
def refine_video(video_path):
    if not is_browser_compatible(video_path):
        print(f"视频编码不兼容，正在转码为 h264 编码...")
        video_path = convert_to_h264(video_path, video_path)
        print(f"视频转码完成，保存为 {video_path}")
    else:
        print(f"视频编码已兼容，无需转码。")
    return video_path

if __name__ == "__main__":
    # 创建 Gradio 界面
    with gr.Blocks() as demo:
        # 设置页面标题和描述
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
            # 轨迹生成
            with gr.Column(scale=1) as left_panel:
                with gr.Group(visible=True) as task1_panel_left:
                    gr.Markdown("## 轨迹生成")
                    task1_type = gr.Radio(choices=["虚实结合算法", "传统算法"], label="选择轨迹生成方式", value="虚实结合算法")
                    task1_num = gr.Number(label="轨迹数量", value=1)
                    task1_btn = gr.Button("开始生成")
            with gr.Column(scale=3) as right_panel:
                with gr.Group(visible=True) as task1_panel_right:
                    task1_output = gr.Video(label="生成的轨迹")
                    task1_btn.click(fn=traj_gen, inputs=[task1_type, task1_num], outputs=task1_output)

        with gr.Row():
            # 在途量预测
            with gr.Column(scale=1) as left_panel:
                with gr.Group(visible=False) as task2_panel_left:
                    gr.Markdown("## 在途量预测")
                    task2_num = gr.Slider(label="车辆数量", minimum=0, maximum=100000, value=10000, step = 1)
                    task2_ratio = gr.Slider(label="卡口比例", minimum=0.0, maximum=1.0, value=0.5)
                    task2_btn = gr.Button("开始预测")
            with gr.Column(scale=3) as right_panel:
                with gr.Group(visible=False) as task2_panel_right:
                    task2_realvideo = gr.Video(label="真实的在途量")
                    task2_predvideo = gr.Video(label="预测的在途量")
                    task2_btn.click(fn=vol_pred, inputs=[task2_num, task2_ratio], outputs=[task2_realvideo, task2_predvideo])

        with gr.Row():
            # 最短路预测
            with gr.Column(scale=1) as left_panel:
                with gr.Group(visible=False) as task3_panel_left:
                    gr.Markdown("## 最短路预测")
                    task3_num = gr.Slider(label="车辆数量", minimum=0, maximum=10000, value=1000, step = 1)
                    task3_type = gr.Radio(choices=["神经网络", "Dijkstra"], label="选择最短路预测方式", value="神经网络")
                    task3_btn = gr.Button("开始预测")
            with gr.Column(scale=3) as right_panel:
                with gr.Group(visible=False) as task3_panel_right:
                    task3_image = gr.Image(label="预测的最短路")
                    task3_time = gr.Textbox(label="预测时间")
                    task3_btn.click(fn=shortest_path, inputs=[task3_num, task3_type], outputs=[task3_image, task3_time])

        with gr.Row():
            # 信号灯控制
            with gr.Column(scale=1) as left_panel:
                with gr.Group(visible=False) as task4_panel_left:
                    gr.Markdown("## 信号灯控制")
                    task4_num = gr.Slider(label="车辆数量", minimum=0, maximum=400_000, value=1000, step = 1)
                    task4_type = gr.Radio(choices=["AI信号灯", "传统算法"], label="选择信号灯控制方式", value="AI信号灯")
                    task4_btn = gr.Button("开始模拟")
            with gr.Column(scale=3) as right_panel:
                with gr.Group(visible=False) as task4_panel_right:
                    task4_video = gr.Video(label="相邻道路状态")
                    # task4_orical = gr.Textbox(label="理论通行率")
                    task4_pass = gr.Textbox(label="通行效率：通过车辆/等待车辆")
                    task4_btn.click(fn=signal_control, inputs=[task4_num, task4_type], outputs=[task4_video, task4_pass])
            
            

        def update_panels(selected):
            return {
                task1_panel_left: gr.update(visible=selected == "轨迹生成"),
                task1_panel_right: gr.update(visible=selected == "轨迹生成"),
                task2_panel_left: gr.update(visible=selected == "在途量预测"),
                task2_panel_right: gr.update(visible=selected == "在途量预测"),
                task3_panel_left: gr.update(visible=selected == "最短路预测"),
                task3_panel_right: gr.update(visible=selected == "最短路预测"),
                task4_panel_left: gr.update(visible=selected == "信号灯控制"),
                task4_panel_right: gr.update(visible=selected == "信号灯控制"),
            }

        func_selector.change(fn=update_panels, inputs=func_selector, outputs=[
            task1_panel_left, task1_panel_right,
            task2_panel_left, task2_panel_right,
            task3_panel_left, task3_panel_right,
            task4_panel_left, task4_panel_right
        ])

    # 启动
    #demo.launch(server_port=8388,auth=("admin", "admin123"))
    demo.launch(server_port=8388,share=True)
