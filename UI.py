import gradio as gr
import threading
import time
from PIL import Image
import numpy as np

from test1 import test_presention1

# 定义多个功能函数
def traj_gen(generate_type, num, stop_flag):
    if generate_type == "预训练模型":
        generate_type = 'pred'
    elif generate_type == "模拟器":
        generate_type = 'dj'
    else:
        raise ValueError(f"Invalid type selected: {generate_type}")
    
    print(f"Generating video with type: {generate_type}")  # 调试输出
    
    def generate_video():
        nonlocal stop_flag
        video_path = None
        for i in range(10):  # 假设生成过程持续10秒
            if stop_flag.is_set():  # 如果停止标志被设置，终止生成
                print("Generation stopped.")
                return None
            time.sleep(1)  # 模拟视频生成过程
            print(f"Generating video... {i + 1} seconds")
        video_path = test_presention1(num, generate_type)
        return video_path
    
    # 启动生成过程
    thread = threading.Thread(target=generate_video)
    thread.start()
    
    # 等待生成完成，模拟视频路径返回
    thread.join()
    if stop_flag.is_set():
        return None
    return "generated_video_path.mp4"  # 返回视频路径（这里用一个占位符）


def vol_pred(text):
    return len(text.split())

def shortest_path(text):
    return "Shortest path is not implemented yet."

def signal_control(text):
    return "Signal control is not implemented yet."

def show_map(city):
    print(city)
    if city == "济南":
        img = Image.open("./UI_element/citymap/jinan.png")
    else:
        raise ValueError("City not supported.")
    return img


with gr.Blocks() as demo:
    with gr.Row():  # 使用 Row 来创建左右布局
        # 左半边 - 地图选择和功能选择
        with gr.Column(scale=1):  # 左半边占比
            # 地图选择
            city = gr.Radio(choices=["济南"], label="选择城市")
            city_img = gr.Image(label="城市地图", type="pil")
            city.change(fn=show_map, inputs=city, outputs=city_img)

            # 功能选择
            func_selector = gr.Dropdown(choices=["轨迹生成", "在途量预测", "最短路预测", "信号灯控制"], label="选择功能", value="轨迹生成")

        # 右半边 - 轨迹生成任务
        with gr.Column(scale=2):  # 右半边占比更大
            task1_type = gr.Radio(choices=["预训练模型", "模拟器"], label="选择轨迹生成方式", value="预训练模型")
            task1_num = gr.Number(label="轨迹数量", value=1)
            task1_btn = gr.Button("开始/停止")  # 初始按钮文本为“开始”
            video_output = gr.Video(label="生成的视频")

            stop_flag = threading.Event()  # 定义停止标志

            # 生成按钮点击事件
            def on_generate_click(generate_type, num):
                print(f"Selected generate type: {generate_type}")  # 调试输出
                if generate_type is None:
                    raise ValueError("没有选择生成类型")
                
                # 开始生成，根据情况更新按钮
                video_path = traj_gen(generate_type, num, stop_flag)
                
                if video_path:
                    return gr.update(visible=True, value=video_path), gr.update(value="开始", disabled=False)  # 生成完成后恢复为“开始”按钮
                else:
                    return gr.update(visible=False), gr.update(value="开始", disabled=False)  # 生成失败时，恢复按钮为“开始”

            # 按钮点击事件处理（初始化按钮后使用 `gr.update()`）
            task1_btn.click(fn=on_generate_click, inputs=[task1_type, task1_num], outputs=[video_output, task1_btn])

            # 在按钮点击时，根据状态控制生成过程停止
            def on_toggle_click():
                # 判断按钮是否处于“开始”状态
                if task1_btn.text == "开始":
                    # 启动生成
                    task1_btn.update(value="停止")  # 切换为“停止”
                    stop_flag.clear()  # 清除停止标志，继续生成
                else:
                    # 停止生成
                    task1_btn.update(value="开始")  # 恢复为“开始”
                    stop_flag.set()  # 设置停止标志，终止生成

                return task1_btn  # 更新按钮状态

            task1_btn.click(on_toggle_click)

    # 切换显示哪个 Block
    def toggle_blocks(choice):
        return {
            task1: gr.update(visible=(choice == "轨迹生成")),
        }

    func_selector.change(fn=toggle_blocks, inputs=func_selector, outputs=[task1])
    demo.load(fn=toggle_blocks, inputs=func_selector, outputs=[task1])

demo.launch()
