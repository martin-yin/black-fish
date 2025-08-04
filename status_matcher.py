import numpy as np
import mss
import os
import time
import json
import threading
import random
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from PIL import Image
from pynput import keyboard

try:
    import keyboard
except ImportError:
    print("警告: keyboard库未安装，请运行: pip install keyboard")
    keyboard = None

try:
    import cv2
except ImportError:
    print("错误: opencv-python库未安装，请运行: pip install opencv-python")
    exit(1)


class StatusMatcher:
    def __init__(self, template_dir: str = "template_images/tips"):
        """
        初始化状态匹配器
        :param template_dir: 状态模板图片目录
        """
        self.template_dir = template_dir
        self.templates = self.load_templates()
        # self.key_templates = self.load_key_templates()  # 注释掉原来的按键模板加载
        self.discard_fish_templates = self.load_discard_fish_templates()
        # 加载YOLO模型用于按键检测
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("best.pt")
            print("YOLO模型加载成功")
        except ImportError:
            print("错误: ultralytics库未安装，请运行: pip install ultralytics")
            self.yolo_model = None
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")
            self.yolo_model = None
        
        # 当前状态
        self.current_state = "monitoring"  # monitoring, fishing, key_input
        
        # 不同状态的检测区域
        self.regions = {
            "status": {
                "left": 1640,
                "top": 442,
                "width": 560,
                "height": 50
            },
            "blue_detection": {
                "left": 1998,
                "top": 880,
                "width": 172,  # 2170 - 1998
                "height": 22   # 902 - 880
            },
            "key_input": {
                "left": 1678,
                "top": 790,
                "width": 502,
                "height": 120
            }
        }
        
        # 主循环控制
        self.is_running = False
        self.main_thread = None
        
        
    def load_templates(self) -> Dict[str, np.ndarray]:
        """
        加载状态模板图片
        :return: 模板字典 {文件名: 图片数组}
        """
        templates = {}
        
        if not os.path.exists(self.template_dir):
            print(f"警告: 模板目录不存在 {self.template_dir}")
            return templates
            
        # 获取目录中的所有PNG文件
        for file in os.listdir(self.template_dir):
            if file.lower().endswith('.png'):
                template_path = os.path.join(self.template_dir, file)
                template = cv2.imread(template_path, cv2.IMREAD_COLOR)
                if template is not None:
                    templates[file] = template
                    print(f"已加载状态模板: {file}, 尺寸: {template.shape}")
                else:
                    print(f"警告: 无法读取模板 {file}")
                    
        return templates
    
    def load_key_templates(self) -> Dict[str, np.ndarray]:
        """
        加载按键模板
        """
        templates = {}
        keys_dir = os.path.join(self.template_dir, "keys")
        
        if not os.path.exists(keys_dir):
            print(f"警告: 按键模板目录不存在: {keys_dir}")
            return templates
        
        for filename in os.listdir(keys_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                template_path = os.path.join(keys_dir, filename)
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[filename] = template
                    print(f"已加载按键模板: {filename}, 尺寸: {template.shape}")
                else:
                    print(f"警告: 无法加载按键模板: {filename}")
        
        return templates
    
    def load_discard_fish_templates(self) -> Dict[str, np.ndarray]:
        """
        加载丢弃鱼类模板
        """
        templates = {}
        discard_dir = "template_images/discard-fish-icon"
        
        if not os.path.exists(discard_dir):
            print(f"警告: 丢弃鱼类模板目录不存在: {discard_dir}")
            return templates
        
        for filename in os.listdir(discard_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                template_path = os.path.join(discard_dir, filename)
                template = cv2.imread(template_path, cv2.IMREAD_COLOR)
                if template is not None:
                    templates[filename] = template
                    print(f"已加载丢弃鱼类模板: {filename}, 尺寸: {template.shape}")
                else:
                    print(f"警告: 无法加载丢弃鱼类模板: {filename}")
        
        return templates
    
    def capture_region(self, region_name: str, save_path: str = None) -> Optional[np.ndarray]:
        """
        截取指定区域的屏幕截图
        :param region_name: 区域名称 (status, blue_detection, key_input)
        :param save_path: 保存截图的路径
        :return: 截图的numpy数组
        """
        if region_name not in self.regions:
            print(f"错误: 未知的区域名称 {region_name}")
            return None
            
        region = self.regions[region_name]
        
        try:
            with mss.mss() as sct:
                monitor = {
                    "left": region["left"],
                    "top": region["top"],
                    "width": region["width"],
                    "height": region["height"]
                }
                
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                
                if save_path:
                    img.save(save_path)
                    print(f"截图已保存: {save_path}")
                
                img_array = np.array(img)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return img_bgr
            
        except Exception as e:
            print(f"截图失败: {e}")
            return None
    
    def detect_status(self, image: np.ndarray, threshold: float = 0.8) -> Dict:
        """
        检测当前状态
        :param image: 输入图像
        :param threshold: 匹配阈值
        :return: 检测结果字典
        """
        results = {
            "matches": {},
            "best_match": None,
            "detected_status": None
        }
        
        best_confidence = 0
        best_match_info = None
        
        # 对每个模板进行匹配
        for template_name, template in self.templates.items():
            matches = self.match_template(image, template, threshold)
            results["matches"][template_name] = matches
            
            # 记录最佳匹配
            if matches and matches[0][2] > best_confidence:
                best_confidence = matches[0][2]
                best_match_info = {
                    "template": template_name,
                    "position": (matches[0][0], matches[0][1]),
                    "confidence": matches[0][2]
                }
        
        if best_match_info:
            results["best_match"] = best_match_info
            results["detected_status"] = best_match_info["template"].replace(".png", "")
        
        return results
    
    def match_template(self, image: np.ndarray, template: np.ndarray, threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """
        在图像中匹配模板
        :param image: 目标图像
        :param template: 模板图像
        :param threshold: 匹配阈值
        :return: 匹配结果列表 [(x, y, confidence), ...]
        """
        # 检查图像和模板尺寸
        img_h, img_w = image.shape[:2]
        tmpl_h, tmpl_w = template.shape[:2]
        
        # 如果模板比图像大，返回空结果
        if tmpl_h > img_h or tmpl_w > img_w:
            return []
        
        # 转换为灰度图像进行匹配
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
            
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template
        
        # 模板匹配
        result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # 找到所有匹配位置
        locations = np.where(result >= threshold)
        matches = []
        
        for pt in zip(*locations[::-1]):
            confidence = result[pt[1], pt[0]]
            matches.append((pt[0], pt[1], confidence))
        
        # 按置信度排序
        matches.sort(key=lambda x: x[2], reverse=True)
        
        return matches
    
    def detect_blue_region(self, image: np.ndarray) -> bool:
        """
        检测图像中是否有蓝色区域
        :param image: 输入图像
        :return: 是否检测到蓝色
        """
        try:
            # 转换为HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 定义蓝色的HSV范围
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # 创建蓝色掩码
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # 计算蓝色像素的数量
            blue_pixels = cv2.countNonZero(mask)
            total_pixels = image.shape[0] * image.shape[1]
            blue_ratio = blue_pixels / total_pixels
            
            # 如果蓝色像素占比超过5%，认为检测到蓝色区域
            return blue_ratio > 0.05
            
        except Exception as e:
            print(f"蓝色检测失败: {e}")
            return False
    
    def match_key_templates(self, image: np.ndarray) -> List[Dict]:
        """
        匹配按键模板
        :param image: 输入图像（灰度图）
        :return: 匹配结果列表
        """
        all_matches = []
        
        for template_name, template in self.key_templates.items():
            # 模板匹配
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= 0.8)  # 提高匹配阈值到0.85
            
            for pt in zip(*locations[::-1]):
                confidence = result[pt[1], pt[0]]
                all_matches.append({
                    "template": template_name,
                    "x": pt[0],
                    "y": pt[1],
                    "confidence": confidence,
                    "width": template.shape[1],
                    "height": template.shape[0]
                })
        
        # 去除重叠的检测结果（非极大值抑制）
        filtered_matches = self._non_max_suppression(all_matches)
        
        # 按x坐标排序
        filtered_matches.sort(key=lambda x: x["x"])
        
        return filtered_matches
    
    def _non_max_suppression(self, matches: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
        """
        非极大值抑制，去除重叠的检测结果
        :param matches: 匹配结果列表
        :param overlap_threshold: 重叠阈值
        :return: 过滤后的匹配结果
        """
        if not matches:
            return []
        
        # 按置信度降序排序
        matches = sorted(matches, key=lambda x: x["confidence"], reverse=True)
        
        filtered = []
        
        for current in matches:
            # 检查当前匹配是否与已选择的匹配重叠
            is_overlapping = False
            
            for selected in filtered:
                # 计算重叠区域
                x1 = max(current["x"], selected["x"])
                y1 = max(current["y"], selected["y"])
                x2 = min(current["x"] + current["width"], selected["x"] + selected["width"])
                y2 = min(current["y"] + current["height"], selected["y"] + selected["height"])
                
                if x2 > x1 and y2 > y1:
                    # 有重叠
                    overlap_area = (x2 - x1) * (y2 - y1)
                    current_area = current["width"] * current["height"]
                    overlap_ratio = overlap_area / current_area
                    
                    if overlap_ratio > overlap_threshold:
                        is_overlapping = True
                        break
            
            if not is_overlapping:
                filtered.append(current)
        
        return filtered
    
    def get_pixel_color(self, x: int, y: int) -> Tuple[int, int, int]:
        """
        获取指定坐标的像素颜色
        :param x: x坐标
        :param y: y坐标
        :return: RGB颜色值元组
        """
        try:
            with mss.mss() as sct:
                # 截取1x1像素的区域
                monitor = {
                    "left": x,
                    "top": y,
                    "width": 1,
                    "height": 1
                }
                screenshot = sct.grab(monitor)
                # 转换为PIL Image
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                # 获取像素颜色
                pixel_color = img.getpixel((0, 0))
                return pixel_color
        except Exception as e:
            print(f"获取像素颜色失败: {e}")
            return (0, 0, 0)
    
    def capture_specific_region(self, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
        """
        截取指定坐标和尺寸的区域
        :param x: 左上角x坐标
        :param y: 左上角y坐标
        :param width: 宽度
        :param height: 高度
        :return: 截图的numpy数组
        """
        try:
            with mss.mss() as sct:
                monitor = {
                    "left": x,
                    "top": y,
                    "width": width,
                    "height": height
                }
                
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                img_array = np.array(img)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return img_bgr
            
        except Exception as e:
            print(f"截取指定区域失败: {e}")
            return None
    
    def detect_discard_fish(self, image: np.ndarray, threshold: float = 0.9) -> bool:
        """
        检测图像中是否包含需要丢弃的鱼类图标
        :param image: 输入图像
        :param threshold: 匹配阈值
        :return: 是否检测到需要丢弃的鱼类
        """
        if not self.discard_fish_templates:
            print("警告: 没有加载丢弃鱼类模板")
            return False
        
        best_confidence = 0
        
        # 对每个丢弃鱼类模板进行匹配
        for template_name, template in self.discard_fish_templates.items():
            matches = self.match_template(image, template, threshold)
            if matches:
                confidence = matches[0][2]
                if confidence > best_confidence:
                    best_confidence = confidence
                print(f"模板 {template_name} 匹配置信度: {confidence:.3f}")
        
        if best_confidence >= threshold:
            print(f"检测到需要丢弃的鱼类，最高置信度: {best_confidence:.3f}")
            return True
        else:
            print(f"未检测到需要丢弃的鱼类，最高置信度: {best_confidence:.3f}")
            return False
    
    def press_space_key(self):
        """
        按下空格键
        """
        if keyboard is None:
            print("keyboard库未安装，无法自动按键")
            return False
        
        try:
            keyboard.press_and_release('space')
            print("已按下空格键")
            return True
        except Exception as e:
            print(f"按键失败: {e}")
            return False
    
    def input_key_sequence(self, matches: List[Dict]):
        """
        输入按键序列
        :param matches: 匹配结果列表
        """
        if not keyboard:
            print("keyboard库未安装，无法自动输入")
            return
        
        print(f"开始输入 {len(matches)} 个按键...")
        
        for match in matches:
            template_name = match["template"]
            # 提取字母（去掉文件扩展名）
            letter = template_name.split('.')[0].upper()
            
            print(f"输入按键: {letter}")
            try:
                keyboard.press_and_release(letter)
                delay = random.uniform(0.1, 0.4)
                time.sleep(delay)
            except Exception as e:
                print(f"输入按键 {letter} 时出错: {e}")
        
        print("按键序列输入完成")
    
    def main_loop(self, interval: float = 1.0):
        """
        主循环：根据当前状态进行不同的检测
        :param interval: 基础检测间隔
        """
        detection_count = 0
        
        while self.is_running:
            try:
                detection_count += 1
                current_time = datetime.now().strftime('%H:%M:%S')
                
                if self.current_state == "monitoring":
                    # 状态监控模式
                    print(f"\n=== 状态监控 第{detection_count}次 [{current_time}] ===")
                    
                    # 截取状态区域
                    status_image = self.capture_region("status")
                    if status_image is not None:
                        # 检测状态
                        results = self.detect_status(status_image)
                        
                        detected_status = results.get("detected_status")
                        if detected_status:
                            best_match = results.get("best_match")
                            confidence = best_match.get('confidence', 0)
                            print(f"检测到状态: {detected_status}, 置信度: {confidence:.3f}")
                            
                            # 根据状态执行相应动作
                            if detected_status == "find_some_one" and confidence >= 0.95:
                                print("检测到find_some_one状态，按下空格键进入拉扯状态")
                                if self.press_space_key():
                                    print("等待1.5秒进入拉扯状态...")
                                    time.sleep(1.5)
                                    self.current_state = "fishing"
                                    print("切换到拉扯检测模式")
                                    continue
                            
                            elif detected_status == "start":
                                print("检测到start状态，按下空格键开始游戏")
                                if self.press_space_key():
                                    print("等待10秒开始游戏...")
                                    time.sleep(10)
                            
                            elif detected_status == "waiting":
                                print("检测到waiting状态，使用较长检测间隔")
                                # waiting状态使用5秒间隔，跳过默认的interval延迟
                                time.sleep(2)
                                continue
                        else:
                            print("未检测到已知状态")
                    
                    time.sleep(interval)
                
                elif self.current_state == "fishing":
                    # 拉扯检测模式 - 进行多次尝试
                    max_attempts = 100  # 最大尝试次数
                    attempt = 0
                    blue_detected = False
                    
                    print(f"[{current_time}] 开始蓝色区域检测，最多尝试 {max_attempts} 次")
                    
                    while attempt < max_attempts and not blue_detected:
                        attempt += 1
                        print(f"第 {attempt} 次蓝色检测...")
                        # 截取蓝色检测区域
                        blue_image = self.capture_region("blue_detection")
                        if blue_image is not None:
                            has_blue = self.detect_blue_region(blue_image)
                            if has_blue:
                                print(f"[{current_time}] 检测到蓝色区域！")
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                                filename = f"blue_detection_{timestamp}.png"
                                print(f"蓝色区域截图已保存: {filename}")
                                
                                if self.press_space_key():
                                    print("等待2.1秒进入按键输入状态...")
                                    time.sleep(2.1)
                                    self.current_state = "key_input"
                                    print("切换到按键输入模式")
                                    blue_detected = True
                            else:
                                print(f"第 {attempt} 次未检测到蓝色区域")
                                if attempt < max_attempts:
                                    print("等待0.05秒后重试...")
                                    time.sleep(0.05)
                    
                    if not blue_detected:
                        print(f"经过 {max_attempts} 次尝试仍未检测到蓝色区域，继续监控...")
                        # 继续在fishing状态进行快速检测
                        time.sleep(0.1)
                    
                    # 如果检测到蓝色并切换了状态，continue会跳过这里
                    if not blue_detected:
                        continue
                
                elif self.current_state == "key_input":
                    # 按键输入模式 - 进行多次尝试
                    max_attempts = 2  # 最大尝试次数
                    attempt = 0
                    key_detected = False
                    
                    print(f"[{current_time}] 开始按键输入检测，最多尝试 {max_attempts} 次")
                    
                    while attempt < max_attempts and not key_detected:
                        attempt += 1
                        print(f"第 {attempt} 次按键检测...")
                        
                        # 截取按键区域
                        key_image = self.capture_region("key_input")
                        if key_image is not None:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            filename = f"key_area_{timestamp}.png"
                            cv2.imwrite(f"./pos_color-3/{filename}", key_image)
                            print(f"按键区域截图已保存: {filename}")
                            
                            # 使用YOLO检测按键
                            matches = self.detect_keys_with_yolo(key_image, confidence_threshold=0.6)
                            
                            if matches:
                                print(f"检测到 {len(matches)} 个按键")
                                for match in matches:
                                    print(f"  - {match['class_name']}: 置信度 {match['confidence']:.3f}")
                                
                                # 输入按键序列
                                self.input_key_sequence(matches)
                                key_detected = True
                                print("按键输入成功")
                            else:
                                print(f"第 {attempt} 次未检测到按键")
                                if attempt < max_attempts:
                                    print("等待0.5秒后重试...")
                                    time.sleep(0.5)
                    
                    if not key_detected:
                        print(f"经过 {max_attempts} 次尝试仍未检测到按键")
                        # 保存key_input区域截图用于调试
                        key_image = self.capture_region("key_input")
                        if key_image is not None:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            key_debug_filename = f"./pos_color/key_input_failed_{timestamp}.png"
                            cv2.imwrite(key_debug_filename, key_image)
                            print(f"按键检测失败时的key_input截图已保存: {key_debug_filename}")
                    
                    # 按键输入完成后回到监控状态
                    print("按键输入阶段完成，回到状态监控模式")
                    time.sleep(3)  # 等待3秒再继续监控
                    
                    # 按下r键之前进行区域截图和模板匹配
                    print("按下r键前进行区域截图和模板匹配...")
                    
                    # 截取指定区域 (x: 2566, y: 1108, width: 52, height: 52)
                    region_x, region_y, region_width, region_height = 2566, 1108, 52, 52
                    region_image = self.capture_specific_region(region_x, region_y, region_width, region_height)
                    
                    if region_image is not None:
                        # 生成文件名
                        # 检测是否匹配discard-fish-icon模板
                        should_discard = self.detect_discard_fish(region_image, threshold=0.85)
                        if should_discard:
                            print("检测到需要丢弃的鱼类，不按R键，开始新的循环")
                        else:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            screenshot_filename = f"./pos_color/region_{timestamp}.png"
                            # 保存区域截图
                            cv2.imwrite(screenshot_filename, region_image)
                            print(f"区域截图已保存: {screenshot_filename}")
                            print("未检测到需要丢弃的鱼类, 按下R键")
                            keyboard.press_and_release("r")
                            time.sleep(1)
                    else:
                        print("区域截图失败，跳过本次检测")
                    
                    # 回到监控状态
                    self.current_state = "monitoring"
                
            except Exception as e:
                print(f"主循环出错: {e}")
                time.sleep(1)
    
    def start_monitoring(self, interval: float = 1.0):
        """
        开始监控
        :param interval: 检测间隔（秒）
        """
        if self.is_running:
            print("监控已在运行中")
            return
        
        self.is_running = True
        self.current_state = "monitoring"
        self.main_thread = threading.Thread(target=self.main_loop, args=(interval,))
        self.main_thread.daemon = True
        self.main_thread.start()
        print(f"开始监控，检测间隔: {interval}秒")
        print("按 Home 键停止监控")
    
    def stop_monitoring(self):
        """
        停止监控
        """
        if not self.is_running:
            print("监控未在运行")
            return
        
        self.is_running = False
        if self.main_thread:
            self.main_thread.join(timeout=2)
        
        print("监控已停止")

    
    def setup_hotkey(self):
        """
        设置热键
        """
        if keyboard is None:
            print("keyboard库未安装，无法使用热键功能")
            return False
        
        try:
            keyboard.add_hotkey('home', self.toggle_monitoring)
            print("热键已设置: Home 键切换监控状态")
            return True
        except Exception as e:
            print(f"设置热键失败: {e}")
            return False
    
    def toggle_monitoring(self):
        """
        切换监控状态
        """
        if self.is_running:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    def detect_keys_with_yolo(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        使用YOLO模型检测按键
        :param image: 输入图像
        :param confidence_threshold: 置信度阈值
        :return: 检测结果列表
        """
        if self.yolo_model is None:
            print("YOLO模型未加载，无法进行按键检测")
            return []
        
        try:
            # 使用YOLO模型进行推理
            results = self.yolo_model(image)
            
            detection_data = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        # 获取边界框坐标 (x1, y1, x2, y2)
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        
                        # 计算中心点坐标
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # 获取置信度
                        confidence = float(boxes.conf[i].cpu().numpy())
                        
                        # 获取类别ID和名称
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]  # 对应 A, D, S, W
                        
                        # 只保留置信度高于阈值的检测结果
                        if confidence >= confidence_threshold:
                            detection_info = {
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'center_x': float(center_x),
                                'center_y': float(center_y),
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': class_name,
                                'template': f"{class_name}.png"  # 为了兼容原有的input_key_sequence方法
                            }
                            detection_data.append(detection_info)
            
            # 按x轴坐标（center_x）从小到大排序
            detection_data_sorted = sorted(detection_data, key=lambda x: x['center_x'])
            
            print(f"YOLO检测到 {len(detection_data_sorted)} 个按键目标")
            for detection in detection_data_sorted:
                print(f"  - {detection['class_name']}: 置信度 {detection['confidence']:.3f}, 位置 ({detection['center_x']:.1f}, {detection['center_y']:.1f})")
            
            return detection_data_sorted
            
        except Exception as e:
            print(f"YOLO按键检测失败: {e}")
            return []


def main():
    # 创建状态匹配器
    matcher = StatusMatcher()
    
    if not matcher.templates:
        print("没有找到状态模板，请检查template_images/tips目录")
        return
    
    # 设置热键
    hotkey_enabled = matcher.setup_hotkey()
    
    while True:
        print("\n=== 状态匹配器 (重构版) ===")
        print("1. 开始/停止监控")
        print(f"2. 当前状态: {matcher.current_state}")
        print("3. 设置检测区域")
        print("4. 手动检测状态")
        print("5. 退出")
        
        if hotkey_enabled:
            print("\n提示: 按 Home 键可随时开始/停止监控")
        
        choice = input("请选择操作 (1-5): ").strip()
        
        if choice == "1":
            if matcher.is_running:
                matcher.stop_monitoring()
            else:
                try:
                    interval = float(input("请输入检测间隔(秒，默认1.0): ") or "1.0")
                    matcher.start_monitoring(interval)
                except ValueError:
                    print("输入无效，使用默认间隔1.0秒")
                    matcher.start_monitoring()
        
        elif choice == "2":
            print(f"当前状态: {matcher.current_state}")
            print(f"监控运行中: {matcher.is_running}")
            print(f"检测区域: {matcher.regions}")
        
        elif choice == "3":
            print("当前检测区域:")
            for name, region in matcher.regions.items():
                print(f"  {name}: {region}")
            
            region_name = input("请输入要修改的区域名称 (status/blue_detection/key_input): ").strip()
            if region_name in matcher.regions:
                try:
                    left = int(input(f"请输入{region_name}左边距: "))
                    top = int(input(f"请输入{region_name}上边距: "))
                    width = int(input(f"请输入{region_name}宽度: "))
                    height = int(input(f"请输入{region_name}高度: "))
                    
                    matcher.regions[region_name] = {
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height
                    }
                    print(f"{region_name}区域已更新")
                except ValueError:
                    print("输入无效，请输入数字")
            else:
                print("无效的区域名称")
        
        elif choice == "4":
            print("手动检测当前状态...")
            status_image = matcher.capture_region("status", "manual_status_check.png")
            if status_image is not None:
                results = matcher.detect_status(status_image)
                detected_status = results.get("detected_status")
                if detected_status:
                    best_match = results.get("best_match")
                    confidence = best_match.get('confidence', 0)
                    print(f"检测到状态: {detected_status}")
                    print(f"置信度: {confidence:.3f}")
                    print(f"位置: {best_match.get('position')}")
                else:
                    print("未检测到已知状态")
        
        elif choice == "5":
            if matcher.is_running:
                matcher.stop_monitoring()
            print("退出程序")
            break
        
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()
