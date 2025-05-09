import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock, mock_open

# 设置环境变量以模拟 Lambda 环境
os.environ['LAMBDA_TASK_ROOT'] = os.path.dirname(os.path.abspath(__file__))

# 导入 Lambda 处理函数
from lambda_function import lambda_handler, process_image, process_video

class TestPCBDefectDetection(unittest.TestCase):
    
    @patch('lambda_function.YOLO')
    @patch('lambda_function.Image')
    @patch('lambda_function.base64.b64decode')
    def test_process_image_with_confidence_threshold(self, mock_b64decode, mock_image, mock_yolo):
        """测试图像处理函数是否使用了正确的置信度阈值"""
        # 模拟图像数据
        mock_image_data = "fake_base64_data"
        mock_b64decode.return_value = b"fake_image_bytes"
        
        # 模拟PIL图像
        mock_img = MagicMock()
        mock_image.open.return_value = mock_img
        
        # 模拟YOLO模型和预测结果
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model
        
        mock_boxes = MagicMock()
        mock_boxes.cls = MagicMock()
        mock_boxes.cls.cpu.return_value.numpy.return_value = []
        mock_boxes.conf = MagicMock()
        mock_boxes.conf.cpu.return_value.numpy.return_value = []
        
        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_result.plot.return_value = "fake_plotted_img"
        
        mock_model.predict.return_value = [mock_result]
        
        # 调用process_image函数
        process_image(mock_image_data)
        
        # 验证predict是否使用了conf=0.3参数
        mock_model.predict.assert_called_once()
        args, kwargs = mock_model.predict.call_args
        self.assertIn('conf', kwargs)
        self.assertEqual(kwargs['conf'], 0.3)
    
    @patch('lambda_function.YOLO')
    @patch('lambda_function.cv2')
    @patch('lambda_function.tempfile.NamedTemporaryFile')
    @patch('lambda_function.base64.b64decode')
    def test_process_video_no_duplicate_model_loading(self, mock_b64decode, mock_tempfile, mock_cv2, mock_yolo):
        """测试视频处理函数是否不再重复加载模型"""
        # 模拟视频数据
        mock_video_data = "fake_base64_data"
        mock_b64decode.return_value = b"fake_video_bytes"
        
        # 模拟临时文件
        mock_temp_file = MagicMock()
        mock_tempfile.return_value = mock_temp_file
        mock_temp_file.name = "fake_temp_file.mp4"
        
        # 模拟OpenCV视频捕获
        mock_cap = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: 30 if x == mock_cv2.CAP_PROP_FPS else 10
        mock_cap.read.side_effect = [(True, "fake_frame"), (False, None)]
        
        # 模拟YOLO模型和预测结果
        # 注意：这里我们不需要模拟模型加载，因为我们已经在模块级别加载了模型
        
        mock_boxes = MagicMock()
        mock_boxes.cls = MagicMock()
        mock_boxes.cls.cpu.return_value.numpy.return_value = []
        mock_boxes.conf = MagicMock()
        mock_boxes.conf.cpu.return_value.numpy.return_value = []
        
        mock_result = MagicMock()
        mock_result.boxes = mock_boxes
        mock_result.plot.return_value = "fake_plotted_frame"
        
        # 使用全局模型而不是重新加载
        mock_model = mock_yolo.return_value
        mock_model.predict.return_value = [mock_result]
        
        # 调用process_video函数
        with patch('lambda_function.os.unlink'):  # 模拟文件删除
            process_video(mock_video_data)
        
        # 验证YOLO只被实例化一次（在模块级别）
        mock_yolo.assert_called_once()
        
        # 验证predict是否使用了conf=0.75参数
        mock_model.predict.assert_called()
        args, kwargs = mock_model.predict.call_args
        self.assertIn('conf', kwargs)
        self.assertEqual(kwargs['conf'], 0.75)

    @patch('lambda_function.process_image')
    def test_lambda_handler_image(self, mock_process_image):
        """测试Lambda处理函数处理图像请求"""
        # 模拟图像处理结果
        mock_process_image.return_value = {
            "processed_image": "fake_processed_image",
            "defects": [{"type": "Missing hole", "confidence": 0.95}],
            "defect_count": 1,
            "defect_summary": {"Missing hole": 1}
        }
        
        # 构造模拟的Lambda事件
        event = {
            'file_type': 'image',
            'file_data': 'fake_image_data'
        }
        
        # 调用Lambda处理函数
        response = lambda_handler(event, None)
        
        # 验证响应
        self.assertEqual(response['statusCode'], 200)
        self.assertEqual(json.loads(response['body'])['defect_count'], 1)
        mock_process_image.assert_called_once_with('fake_image_data')

    @patch('lambda_function.process_video')
    def test_lambda_handler_video(self, mock_process_video):
        """测试Lambda处理函数处理视频请求"""
        # 模拟视频处理结果
        mock_process_video.return_value = {
            "sample_frames": ["fake_frame1", "fake_frame2"],
            "defects": [{"type": "Short", "confidence": 0.85, "frame": 5}],
            "defect_count": 1,
            "defect_summary": {"Short": 1},
            "total_frames": 10,
            "fps": 30
        }
        
        # 构造模拟的Lambda事件
        event = {
            'file_type': 'video',
            'file_data': 'fake_video_data'
        }
        
        # 调用Lambda处理函数
        response = lambda_handler(event, None)
        
        # 验证响应
        self.assertEqual(response['statusCode'], 200)
        self.assertEqual(json.loads(response['body'])['defect_count'], 1)
        mock_process_video.assert_called_once_with('fake_video_data')

if __name__ == '__main__':
    unittest.main()