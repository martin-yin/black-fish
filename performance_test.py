import time
import ctypes
from ctypes import windll, byref, c_ubyte, c_int, c_void_p, c_wchar_p
from ctypes.wintypes import RECT, HWND, DWORD, BOOL
import psutil
import win32gui
import win32process

GetDC = windll.user32.GetDC
CreateCompatibleDC = windll.gdi32.CreateCompatibleDC
GetClientRect = windll.user32.GetClientRect
CreateCompatibleBitmap = windll.gdi32.CreateCompatibleBitmap
SelectObject = windll.gdi32.SelectObject
BitBlt = windll.gdi32.BitBlt
SRCCOPY = 0x00CC0020
DeleteObject = windll.gdi32.DeleteObject
ReleaseDC = windll.user32.ReleaseDC
windll.user32.SetProcessDPIAware()

def find_window_by_process_name(process_name):
    """通过进程名查找窗口"""
    def enum_windows_proc(hwnd, lParam):
        if win32gui.IsWindowVisible(hwnd):
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                process = psutil.Process(pid)
                if process.name().lower() == process_name.lower():
                    window_title = win32gui.GetWindowText(hwnd)
                    lParam.append((hwnd, window_title, pid))
            except:
                pass
        return True
    
    windows = []
    win32gui.EnumWindows(enum_windows_proc, windows)
    return windows

def find_game_window():
    """查找游戏窗口"""
    print("开始查找游戏窗口...")
    
    # 方法1: 精确匹配
    print("方法1: 尝试精确匹配窗口标题 'BlackDesert64'")
    hwnd = win32gui.FindWindow(None, "BlackDesert64")
    if hwnd:
        print(f"方法1成功: 找到窗口句柄 {hwnd}")
        return hwnd
    else:
        print("方法1失败: 未找到精确匹配的窗口")
    
    # 方法3: 通过进程名
    print("方法3: 尝试通过进程名 'BlackDesert64.exe' 查找")
    windows = find_window_by_process_name("BlackDesert64.exe")
    if windows:
        print(f"方法3成功: 找到 {len(windows)} 个匹配进程的窗口")
        for i, (hwnd, title, pid) in enumerate(windows):
            print(f"  窗口{i+1}: 句柄={hwnd}, 标题='{title}', PID={pid}")
        return windows[0][0]
    else:
        print("方法3失败: 未找到 'BlackDesert64.exe' 进程的窗口")
    
    print("所有方法都失败，未找到游戏窗口")
    return None

def simple_capture(handle: HWND):
    """简化版窗口截图，只测试性能，不返回图像数据
    
    Args:
        handle (HWND): 要截图的窗口句柄
    
    Returns:
        tuple: (width, height, success) - 宽度、高度和是否成功
    """
    try:
        # 获取窗口客户区的大小
        r = RECT()
        result = GetClientRect(handle, byref(r))
        if not result:
            return 0, 0, False
            
        width, height = r.right, r.bottom
        
        if width <= 0 or height <= 0:
            return 0, 0, False
        
        # 开始截图
        dc = GetDC(handle)
        if not dc:
            return width, height, False
            
        cdc = CreateCompatibleDC(dc)
        if not cdc:
            ReleaseDC(handle, dc)
            return width, height, False
            
        bitmap = CreateCompatibleBitmap(dc, width, height)
        if not bitmap:
            DeleteObject(cdc)
            ReleaseDC(handle, dc)
            return width, height, False
            
        SelectObject(cdc, bitmap)
        blt_result = BitBlt(cdc, 0, 0, 800, 100, dc, 0, 0, SRCCOPY)
        
        # 清理资源
        DeleteObject(bitmap)
        DeleteObject(cdc)
        ReleaseDC(handle, dc)
        
        return width, height, bool(blt_result)
        
    except Exception as e:
        print(f"截图过程中发生错误: {e}")
        return 0, 0, False

def test_capture_performance(num_captures=10):
    """测试截图性能
    
    Args:
        num_captures (int): 截图次数，默认10次
    
    Returns:
        tuple: (总耗时, 平均耗时, 每次截图耗时列表)
    """
    print("开始性能测试...")
    
    # 获取窗口句柄（只获取一次，复用）
    print("获取窗口句柄...")
    hwnd = find_game_window()
    if not hwnd:
        print("错误: 未找到游戏窗口")
        print("\n提示: 请确保游戏已启动且窗口未最小化")
        print("支持的游戏窗口标题: BlackDesert64, 或包含 'BlackDesert' 的窗口")
        return None, None, None
    
    print(f"找到窗口句柄: {hwnd}")
    print(f"开始进行 {num_captures} 次截图测试...")
    
    capture_times = []
    total_start_time = time.time()
    
    for i in range(num_captures):
        start_time = time.time()
        
        # 执行截图
        width, height, success = simple_capture(hwnd)
        
        end_time = time.time()
        capture_time = end_time - start_time
        
        if success:
            capture_times.append(capture_time)
            print(f"第 {i+1} 次截图完成: {capture_time:.4f}秒 (窗口尺寸: {width}x{height})")
        else:
            print(f"第 {i+1} 次截图失败: 窗口尺寸: {width}x{height}")
            if width == 0 and height == 0:
                print("  可能原因: 窗口已最小化或不可见")
            else:
                print("  可能原因: BitBlt操作失败")
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    if not capture_times:
        print("错误: 没有成功的截图")
        return None, None, None
    
    avg_time = sum(capture_times) / len(capture_times)
    
    return total_time, avg_time, capture_times

def print_performance_report(total_time, avg_time, capture_times):
    """打印性能报告"""
    if total_time is None:
        print("性能测试失败")
        return
    
    print("\n" + "="*60)
    print("截图性能测试报告")
    print("="*60)
    print(f"总测试时间: {total_time:.4f}秒")
    print(f"成功截图次数: {len(capture_times)}次")
    print(f"平均每次截图耗时: {avg_time:.4f}秒 ({avg_time*1000:.2f}毫秒)")
    print(f"最快截图耗时: {min(capture_times):.4f}秒 ({min(capture_times)*1000:.2f}毫秒)")
    print(f"最慢截图耗时: {max(capture_times):.4f}秒 ({max(capture_times)*1000:.2f}毫秒)")
    print(f"截图频率: {len(capture_times)/total_time:.2f} FPS")
    
    # 计算标准差
    variance = sum((t - avg_time) ** 2 for t in capture_times) / len(capture_times)
    std_dev = variance ** 0.5
    print(f"耗时标准差: {std_dev:.4f}秒 ({std_dev*1000:.2f}毫秒)")
    
    print("\n详细耗时列表:")
    for i, t in enumerate(capture_times, 1):
        print(f"  第{i:2d}次: {t:.4f}秒 ({t*1000:6.2f}毫秒)")
    
    print("\n" + "-"*60)
    print("性能分析:")
    print("-"*60)
    
    # 性能评级
    if avg_time < 0.005:  # 5ms
        print("✓ 截图性能优秀 (< 5ms) - 适合高频实时应用")
    elif avg_time < 0.01:  # 10ms
        print("✓ 截图性能很好 (< 10ms) - 适合实时应用")
    elif avg_time < 0.02:  # 20ms
        print("✓ 截图性能良好 (< 20ms) - 适合一般应用")
    elif avg_time < 0.05:  # 50ms
        print("⚠ 截图性能一般 (< 50ms) - 可用但不够理想")
    else:
        print("✗ 截图性能较差 (> 50ms) - 需要优化")
    
    # FPS评估
    fps = len(capture_times) / total_time
    if fps > 100:
        print("✓ 可支持超高频率截图 (> 100 FPS)")
    elif fps > 60:
        print("✓ 可支持高频率截图 (> 60 FPS)")
    elif fps > 30:
        print("✓ 可支持中等频率截图 (> 30 FPS)")
    elif fps > 15:
        print("⚠ 截图频率较低 (> 15 FPS)")
    else:
        print("✗ 截图频率很低 (< 15 FPS)")
    
    # 稳定性评估
    if std_dev < avg_time * 0.1:
        print("✓ 性能稳定 (标准差 < 10% 平均值)")
    elif std_dev < avg_time * 0.2:
        print("⚠ 性能较稳定 (标准差 < 20% 平均值)")
    else:
        print("✗ 性能不稳定 (标准差 > 20% 平均值)")
    
    print("="*60)

def test_different_scenarios():
    """测试不同场景下的性能"""
    scenarios = [
        ("快速测试", 5),
        ("标准测试", 10),
        ("压力测试", 50),
        ("长时间测试", 100)
    ]
    
    print("开始多场景性能测试...\n")
    
    for name, count in scenarios:
        print(f"\n{'='*20} {name} ({'='*20})")
        print(f"测试次数: {count}")
        
        total_time, avg_time, capture_times = test_capture_performance(count)
        
        if total_time is not None:
            print(f"\n简要结果:")
            print(f"  总耗时: {total_time:.4f}秒")
            print(f"  平均耗时: {avg_time:.4f}秒 ({avg_time*1000:.2f}毫秒)")
            print(f"  截图频率: {len(capture_times)/total_time:.2f} FPS")
        else:
            print("测试失败")
        
        # 等待一秒再进行下一组测试
        time.sleep(1)

if __name__ == "__main__":
    print("截图性能测试工具 (纯ctypes版本)")
    print("请确保游戏窗口已打开且未最小化")
    print("支持的游戏: BlackDesert64 或包含 'BlackDesert' 的窗口")
    
    print("\n选择测试模式:")
    print("1. 快速测试 (10次截图)")
    print("2. 多场景测试 (5/10/50/100次)")
    print("3. 自定义次数测试")
    
    try:
        choice = input("\n请输入选择 (1-3, 默认1): ").strip()
        
        if choice == "2":
            test_different_scenarios()
        elif choice == "3":
            try:
                num_captures = int(input("请输入测试次数: "))
                if num_captures <= 0:
                    print("无效次数，使用默认值10")
                    num_captures = 10
            except ValueError:
                print("输入无效，使用默认值10")
                num_captures = 10
            
            total_time, avg_time, capture_times = test_capture_performance(num_captures)
            print_performance_report(total_time, avg_time, capture_times)
        else:
            # 默认快速测试
            total_time, avg_time, capture_times = test_capture_performance(10)
            print_performance_report(total_time, avg_time, capture_times)
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
    
    input("\n按回车键退出...")