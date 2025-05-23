import json
import base64
import os
import sys

# 设置环境变量以模拟 Lambda 环境
os.environ['LAMBDA_TASK_ROOT'] = os.path.dirname(os.path.abspath(__file__))

# 导入 Lambda 处理函数
from lambda_function import lambda_handler

def read_file_as_base64(file_path):
    """读取文件并转换为 base64 编码"""
    with open(file_path, 'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

def test_image():
    """测试图像处理功能"""
    # 图像路径 - 使用相对路径或环境变量
    image_path = os.environ.get("TEST_IMAGE_PATH", "test_image.jpg")  # 默认使用当前目录下的test_image.jpg

    print(f"Testing image processing with: {image_path}")

    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"Warning: Test image not found at {image_path}. Please provide a valid image path.")
        return

    # 读取图像文件并转换为 base64
    image_data = read_file_as_base64(image_path)

    # 构造模拟的 Lambda 事件
    event = {
        'file_type': 'image',
        'file_data': image_data
    }

    # 调用 Lambda 处理函数
    response = lambda_handler(event, None)

    # 打印响应状态码
    print(f"Response status code: {response['statusCode']}")

    # 如果成功，保存处理后的图像
    if response['statusCode'] == 200:
        result = json.loads(response['body'])
        print(f"Detected {result['defect_count']} defects")
        print(f"Defect summary: {result['defect_summary']}")

        # 保存处理后的图像
        processed_image_data = result['processed_image']
        output_path = "processed_image.jpg"
        with open(output_path, 'wb') as f:
            f.write(base64.b64decode(processed_image_data))
        print(f"Processed image saved to: {output_path}")
    else:
        print(f"Error: {response['body']}")

def test_video():
    """测试视频处理功能"""
    # 视频路径 - 使用相对路径或环境变量
    video_path = os.environ.get("TEST_VIDEO_PATH", "test_video.mp4")  # 默认使用当前目录下的test_video.mp4

    print(f"Testing video processing with: {video_path}")

    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"Warning: Test video not found at {video_path}. Please provide a valid video path.")
        return

    # 读取视频文件并转换为 base64
    video_data = read_file_as_base64(video_path)

    # 构造模拟的 Lambda 事件
    event = {
        'file_type': 'video',
        'file_data': video_data
    }

    # 调用 Lambda 处理函数
    response = lambda_handler(event, None)

    # 打印响应状态码
    print(f"Response status code: {response['statusCode']}")

    # 如果成功，保存样本帧
    if response['statusCode'] == 200:
        result = json.loads(response['body'])
        print(f"Detected {result['defect_count']} defects across {result['total_frames']} frames")
        print(f"Defect summary: {result['defect_summary']}")

        # 保存样本帧
        for i, frame_data in enumerate(result['sample_frames']):
            output_path = f"frame_{i}.jpg"
            with open(output_path, 'wb') as f:
                f.write(base64.b64decode(frame_data))
            print(f"Sample frame saved to: {output_path}")
    else:
        print(f"Error: {response['body']}")

if __name__ == "__main__":
    # 选择要测试的功能
    test_type = "image"  # 可以是 "image" 或 "video"

    if test_type == "image":
        test_image()
    elif test_type == "video":
        test_video()
    else:
        print("请指定正确的测试类型: 'image' 或 'video'")
